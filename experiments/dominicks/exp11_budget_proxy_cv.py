"""
Experiment 11: Budget-proxy replacement for CV calculations (Dominick's).

Implements:
1) per-customer budget construction from ccount store-week data,
2) income replacement in train/test arrays with scaled y_percustomer,
3) per-customer and aggregate CV computation with fixed pre-shock income,
4) welfare tables in three panels (A/B/C),
5) robustness checks across alternative budget definitions.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

from experiments.dominicks.data import G, GOODS, build_arrays, load_panel, prepare_splits
from experiments.dominicks.exp01_main_runs import run_once
from experiments.dominicks.utils import pred
from run_neural_demand_dominicks import BASE_CFG


MODEL_ORDER = [
    "Neural Demand (static)",
    "Neural Demand (habit)",
    "Neural Demand (habit, FE)",
    "Neural Demand (habit, CF)",
]


@dataclass
class Cfg:
    out_dir: str
    n_seeds: int
    cv_steps: int
    annual_freq_low: float
    annual_freq_high: float


def _seed_list(n: int) -> list[int]:
    return [42 + i * 15 for i in range(n)]


def _build_cfg(args) -> tuple[dict, Cfg]:
    cfg = dict(BASE_CFG)
    cfg["weekly_path"] = args.weekly
    cfg["upc_path"] = args.upc
    cfg["device"] = "cpu"
    # Budget proxy changed => force full retraining for Exp 11.
    cfg["force_retrain"] = True
    cfg["verbose"] = False
    cfg["model_cache_dir"] = (
        "results/neural_demand/dominicks/models/fast"
        if args.fast
        else "results/neural_demand/dominicks/models/full"
    )
    if args.fast:
        cfg["nirl_epochs"] = 100
        cfg["mdp_epochs"] = 100
        cfg["mdp_e2e_epochs"] = 100
        cfg["lirl_epochs"] = 100
    cfg["cv_steps"] = int(args.cv_steps)
    cfg["N_RUNS"] = int(args.n_seeds)

    out_dir = os.path.join(cfg["out_dir"], "budget_proxy_cv")
    os.makedirs(out_dir, exist_ok=True)

    pcfg = Cfg(
        out_dir=out_dir,
        n_seeds=int(args.n_seeds),
        cv_steps=int(args.cv_steps),
        annual_freq_low=float(args.annual_freq_low),
        annual_freq_high=float(args.annual_freq_high),
    )
    return cfg, pcfg


def _load_aux_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ccount = pd.read_csv(os.path.join(data_dir, "ccount.csv"), low_memory=False)
    demo = pd.read_csv(os.path.join(data_dir, "demo.csv"), low_memory=False)
    ccount.columns = [c.lower() for c in ccount.columns]
    demo.columns = [c.lower() for c in demo.columns]

    demo = demo.copy()
    demo["shopavid"] = demo.get("shpavid", np.nan)
    demo["shophurr"] = demo.get("shphurr", np.nan)
    demo["shopcons"] = demo.get("shpcons", np.nan)
    keep = [
        "store",
        "income",
        "poverty",
        "shopavid",
        "shopcons",
        "shophurr",
        "lat",
        "long",
    ]
    keep = [k for k in keep if k in demo.columns]
    demo = demo[keep].drop_duplicates(subset=["store"])
    return ccount, demo


def _build_budget_weekly(ccount: pd.DataFrame) -> pd.DataFrame:
    cols = ccount.columns
    for req in ["store", "week", "custcoun"]:
        if req not in cols:
            raise ValueError(f"ccount missing required column: {req}")

    dept_cols = [c for c in ["grocery", "haba", "frozen", "dairy", "convfood"] if c in cols]
    if len(dept_cols) < 2:
        raise ValueError("ccount is missing nondurable revenue columns for Exp 11")

    grp_cols = ["custcoun"] + dept_cols
    c_week = ccount.groupby(["store", "week"], as_index=False)[grp_cols].sum()
    c_week["nondurable_revenue"] = c_week[dept_cols].sum(axis=1)
    c_week["y_percustomer_nondurable"] = (
        c_week["nondurable_revenue"] / c_week["custcoun"].clip(lower=1.0)
    )

    # Robustness #1: broad budget (all numeric department revenue columns).
    numeric_cols = ccount.select_dtypes(include=[np.number]).columns.tolist()
    rev_candidates = [c for c in numeric_cols if c not in ["store", "week", "custcoun"]]
    if rev_candidates:
        all_rev = ccount.groupby(["store", "week"], as_index=False)[rev_candidates].sum()
        all_rev["total_revenue"] = all_rev[rev_candidates].sum(axis=1)
        c_week = c_week.merge(
            all_rev[["store", "week", "total_revenue"]],
            on=["store", "week"],
            how="left",
        )
    else:
        c_week["total_revenue"] = np.nan
    c_week["y_percustomer_total"] = c_week["total_revenue"] / c_week["custcoun"].clip(lower=1.0)

    # Robustness #2: narrow HABA-only budget.
    if "haba" in c_week.columns:
        c_week["y_percustomer_haba"] = c_week["haba"] / c_week["custcoun"].clip(lower=1.0)
    else:
        c_week["y_percustomer_haba"] = np.nan
    return c_week


def _merge_panel_with_aux(panel: pd.DataFrame, c_week: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy().rename(columns={"STORE": "store", "WEEK": "week"})
    out["store"] = out["store"].astype(int)
    out["week"] = out["week"].astype(int)
    out = out.merge(
        c_week[
            [
                "store",
                "week",
                "custcoun",
                "y_percustomer_nondurable",
                "y_percustomer_total",
                "y_percustomer_haba",
            ]
        ],
        on=["store", "week"],
        how="left",
    )
    out = out.merge(demo, on="store", how="left")
    out["custcoun"] = pd.to_numeric(out["custcoun"], errors="coerce")
    out["invalid_custcoun"] = out["custcoun"].isna() | (out["custcoun"] <= 0)
    return out


def _winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    x = s.astype(float).copy()
    mask = np.isfinite(x.values)
    if mask.sum() < 10:
        return x
    q1, q2 = np.nanquantile(x[mask], [lo, hi])
    return x.clip(lower=q1, upper=q2)


def _scale_budget_to_income_units(
    panel_sw: pd.DataFrame,
    y_col_raw: str,
    old_income_mean: float,
    out_col_model: str,
    winsorize: bool = False,
) -> dict:
    y_raw = panel_sw[y_col_raw].astype(float)
    if winsorize:
        y_raw = _winsorize(y_raw, 0.01, 0.99)
    valid = np.isfinite(y_raw.values) & (y_raw.values > 0)
    new_mean = float(np.nanmean(y_raw[valid])) if valid.any() else np.nan
    if np.isfinite(new_mean) and new_mean > 0:
        scale = float(old_income_mean) / new_mean
    else:
        scale = 1.0
    panel_sw[out_col_model] = y_raw * scale
    panel_sw[out_col_model] = panel_sw[out_col_model].where(panel_sw[out_col_model] > 0)
    return {
        "raw_mean": new_mean,
        "old_income_mean": float(old_income_mean),
        "scale_factor": scale,
    }


def _filter_and_replace_income(
    splits: dict,
    panel_sw: pd.DataFrame,
    y_col_model: str,
) -> tuple[dict, pd.DataFrame, dict]:
    panel_sorted = panel_sw.sort_values(["store", "week"]).reset_index(drop=True).copy()
    y_all = panel_sorted[y_col_model].to_numpy(dtype=float)
    valid_income = np.isfinite(y_all) & (y_all > 0)

    tr0 = splits["tr"]
    te0 = splits["te"]
    keep_tr = valid_income[tr0]
    keep_te = valid_income[te0]

    drops = {
        "train_total": int(len(tr0)),
        "test_total": int(len(te0)),
        "train_dropped_invalid_income_or_custcoun": int((~keep_tr).sum()),
        "test_dropped_invalid_income_or_custcoun": int((~keep_te).sum()),
    }

    out = dict(splits)
    tr_new = tr0[keep_tr]
    te_new = te0[keep_te]
    out["tr"] = np.arange(len(tr_new))
    out["te"] = np.arange(len(te_new))

    out["y_tr"] = y_all[tr_new]
    out["y_te"] = y_all[te_new]

    tr_keys = ["p_tr", "w_tr", "mw_tr", "xb_tr", "qp_tr", "ls_tr", "s_tr", "wk_tr", "s_tr_idx", "Z_tr"]
    te_keys = ["p_te", "w_te", "mw_te", "xb_te", "qp_te", "ls_te", "s_te", "wk_te", "s_te_idx"]
    for k in tr_keys:
        out[k] = splits[k][keep_tr]
    for k in te_keys:
        out[k] = splits[k][keep_te]

    out["p_mn"] = out["p_te"].mean(axis=0)
    out["y_mn"] = float(np.nanmean(out["y_te"]))
    out["xb_mn"] = out["xb_te"].mean(axis=0)
    out["qp_mn"] = out["qp_te"].mean(axis=0)
    out["p0w"] = out["p_mn"].copy()
    out["p1w"] = out["p_mn"].copy()
    out["p1w"][int(out.get("shock_good", 0))] *= 1.0 + float(out.get("shock_pct", 0.10))

    sg = int(out.get("shock_good", 0))
    n_gr = 80
    pgr_all = []
    tpx_all = []
    for g in range(G):
        plo = float(np.percentile(out["p_te"][:, g], 5))
        phi = float(np.percentile(out["p_te"][:, g], 95))
        pgr = np.linspace(plo, phi, n_gr)
        tpx = np.tile(out["p_te"].mean(0), (n_gr, 1))
        tpx[:, g] = pgr
        pgr_all.append(pgr)
        tpx_all.append(tpx)
    out["pgr_all"] = pgr_all
    out["tpx_all"] = tpx_all
    out["pgr"] = pgr_all[sg]
    out["tpx"] = tpx_all[sg]
    out["fy"] = np.full(n_gr, float(np.nanmean(out["y_te"])))

    te_panel = panel_sorted.iloc[te_new].copy().reset_index(drop=True)
    te_panel["store_idx"] = out["s_te_idx"]
    return out, te_panel, drops


def _cluster_mean_se(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    x = pd.Series(values).astype(float)
    g = pd.Series(groups)
    m = x.notna() & g.notna() & np.isfinite(x.values)
    x = x[m]
    g = g[m]
    if len(x) == 0:
        return np.nan, np.nan
    if g.nunique() < 2:
        return float(x.mean()), np.nan
    try:
        fit = sm.OLS(x.values, np.ones((len(x), 1))).fit(cov_type="cluster", cov_kwds={"groups": g.values})
        return float(fit.params[0]), float(fit.bse[0])
    except Exception:
        return float(x.mean()), np.nan


def _compute_cv_for_obs(
    spec: str,
    p0: np.ndarray,
    p1: np.ndarray,
    y_percustomer: float,
    cfg: dict,
    custcoun: float | None = None,
    xb_prev0: np.ndarray | None = None,
    q_prev0: np.ndarray | None = None,
    **kw,
) -> dict:
    cv = 0.0
    n_steps = int(cfg["cv_steps"])
    dp = (p1 - p0) / float(n_steps)
    for step in range(n_steps):
        # Midpoint rule and fixed pre-shock y over the full integration path.
        p_tau = p0 + (step + 0.5) * dp
        w_hat = pred(
            spec,
            p_tau[None, :],
            np.array([y_percustomer]),
            cfg,
            xb_prev=xb_prev0[None, :] if xb_prev0 is not None else None,
            q_prev=q_prev0[None, :] if q_prev0 is not None else None,
            **kw,
        )[0]
        q_hat = w_hat * y_percustomer / np.maximum(p_tau, 1e-8)
        cv -= float(np.sum(q_hat * dp))

    cv_percustomer = cv * 100.0
    out = {"cv_percustomer": cv_percustomer}
    if custcoun is not None and np.isfinite(custcoun):
        out["cv_aggregate"] = cv_percustomer * float(custcoun)
    return out


def _compute_xbar_variants(run: dict, splits: dict, cfg: dict) -> dict:
    from src.models.dominicks import compute_xbar_e2e

    dev = cfg["device"]
    ls_te_t = torch.tensor(splits["ls_te"], dtype=torch.float32, device=dev)
    s_te = splits["s_te"]
    out = {}
    with torch.no_grad():
        d_habit = torch.tensor(float(run["delta_mdp"]), dtype=torch.float32, device=dev)
        out["xb_habit"] = compute_xbar_e2e(d_habit, ls_te_t, store_ids=s_te).cpu().numpy()
        d_fe = torch.tensor(float(run["delta_mdp_fe"]), dtype=torch.float32, device=dev)
        out["xb_habit_fe"] = compute_xbar_e2e(d_fe, ls_te_t, store_ids=s_te).cpu().numpy()
        d_cf = torch.tensor(float(run.get("delta_mdp_cf", run["delta_mdp"])), dtype=torch.float32, device=dev)
        out["xb_habit_cf"] = compute_xbar_e2e(d_cf, ls_te_t, store_ids=s_te).cpu().numpy()
    out["q_prev"] = splits["qp_te"]
    return out


def run(args):
    cfg, pcfg = _build_cfg(args)

    panel = load_panel(cfg)
    data = build_arrays(panel, cfg)
    base_splits = prepare_splits(data, cfg)
    old_income_mean = float(np.nanmean(base_splits["y_tr"]))

    ccount, demo = _load_aux_data(args.data_dir)
    c_week = _build_budget_weekly(ccount)
    panel_sw = _merge_panel_with_aux(panel, c_week, demo)

    scale_meta = _scale_budget_to_income_units(
        panel_sw=panel_sw,
        y_col_raw="y_percustomer_nondurable",
        old_income_mean=old_income_mean,
        out_col_model="y_model_nondurable",
        winsorize=False,
    )

    splits_main, te_panel, drops = _filter_and_replace_income(base_splits, panel_sw, "y_model_nondurable")
    print(
        "[exp11] Dropped due to invalid/missing custcoun or y_percustomer: "
        f"train={drops['train_dropped_invalid_income_or_custcoun']}/{drops['train_total']}, "
        f"test={drops['test_dropped_invalid_income_or_custcoun']}/{drops['test_total']}"
    )
    print(
        "[exp11] y_percustomer scaling: "
        f"old_mean={scale_meta['old_income_mean']:.4f}, "
        f"new_mean_raw={scale_meta['raw_mean']:.4f}, "
        f"factor={scale_meta['scale_factor']:.6f}"
    )

    seeds = _seed_list(pcfg.n_seeds)
    runs = []
    for i, seed in enumerate(seeds, 1):
        print(f"[exp11] retraining with budget proxy seed={seed} ({i}/{len(seeds)})")
        runs.append(run_once(seed, splits_main, cfg))

    base_run = runs[0]
    xbars = _compute_xbar_variants(base_run, splits_main, cfg)

    sg = int(cfg["shock_good"])
    ss = float(cfg["shock_pct"])

    cv_rows = []
    for i, row in te_panel.iterrows():
        p0 = np.array([row["ASP"], row["ACET"], row["IBU"]], dtype=float)
        p1 = p0.copy()
        p1[sg] *= 1.0 + ss
        y = float(row["y_model_nondurable"])
        n = float(row["custcoun"]) if np.isfinite(row["custcoun"]) else np.nan
        for model in MODEL_ORDER:
            if model == "Neural Demand (static)":
                res = _compute_cv_for_obs("nirl", p0, p1, y, cfg, custcoun=n, **base_run["KW"])
            elif model == "Neural Demand (habit)":
                res = _compute_cv_for_obs(
                    "mdp",
                    p0,
                    p1,
                    y,
                    cfg,
                    custcoun=n,
                    xb_prev0=xbars["xb_habit"][i],
                    q_prev0=xbars["q_prev"][i],
                    **base_run["KW"],
                )
            elif model == "Neural Demand (habit, FE)":
                res = _compute_cv_for_obs(
                    "mdp-fe",
                    p0,
                    p1,
                    y,
                    cfg,
                    custcoun=n,
                    xb_prev0=xbars["xb_habit_fe"][i],
                    q_prev0=xbars["q_prev"][i],
                    store_idx=np.array([int(row["store_idx"])]),
                    s_te_mode_idx=int(splits_main["s_te_mode_idx"]),
                    **base_run["KW"],
                )
            elif model == "Neural Demand (habit, CF)":
                res = _compute_cv_for_obs(
                    "mdp-cf",
                    p0,
                    p1,
                    y,
                    cfg,
                    custcoun=n,
                    xb_prev0=xbars["xb_habit_cf"][i],
                    q_prev0=xbars["q_prev"][i],
                    **base_run["KW"],
                )
            else:
                continue

            cv_rows.append(
                {
                    "store": int(row["store"]),
                    "week": int(row["week"]),
                    "model": model,
                    "cv_percustomer": res["cv_percustomer"],
                    "cv_aggregate": res.get("cv_aggregate", np.nan),
                    "custcoun": n,
                    "y_percustomer_raw_nondurable": row["y_percustomer_nondurable"],
                    "y_model_scaled": y,
                    "income": row.get("income", np.nan),
                    "poverty": row.get("poverty", np.nan),
                    "shopavid": row.get("shopavid", np.nan),
                }
            )

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join(pcfg.out_dir, "exp11_cv_store_week_model.csv"), index=False)

    # Panel A: per-customer CV loss.
    rows_a = []
    for m in MODEL_ORDER:
        x = cv_df.loc[cv_df["model"] == m]
        mu, se = _cluster_mean_se(x["cv_percustomer"].values, x["store"].values)
        rows_a.append({"model": m, "cv_percustomer_mean": mu, "cv_percustomer_se_store_cluster": se})
    table_a = pd.DataFrame(rows_a)
    s_mu = float(table_a.loc[table_a["model"] == "Neural Demand (static)", "cv_percustomer_mean"].iloc[0])
    h_mu = float(table_a.loc[table_a["model"] == "Neural Demand (habit)", "cv_percustomer_mean"].iloc[0])
    gap = h_mu - s_mu
    gap_pct = 100.0 * gap / abs(s_mu) if s_mu != 0 else np.nan
    table_a = pd.concat(
        [
            table_a,
            pd.DataFrame(
                [
                    {
                        "model": "Static-Habit gap",
                        "cv_percustomer_mean": gap,
                        "cv_percustomer_se_store_cluster": np.nan,
                        "gap_pct_vs_static": gap_pct,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    table_a.to_csv(os.path.join(pcfg.out_dir, "table11_panel_A_per_customer_cv.csv"), index=False)

    # Panel B: annualized CV loss per household at 4 and 6 visits/year.
    rows_b = []
    for _, r in table_a[table_a["model"].isin(MODEL_ORDER)].iterrows():
        rows_b.append(
            {
                "model": r["model"],
                "annualized_cv_freq4": r["cv_percustomer_mean"] * pcfg.annual_freq_low,
                "annualized_cv_freq6": r["cv_percustomer_mean"] * pcfg.annual_freq_high,
            }
        )
    table_b = pd.DataFrame(rows_b)
    b_s = float(table_b.loc[table_b["model"] == "Neural Demand (static)", "annualized_cv_freq4"].iloc[0])
    b_h = float(table_b.loc[table_b["model"] == "Neural Demand (habit)", "annualized_cv_freq4"].iloc[0])
    b_gap = b_h - b_s
    b_gap_pct = 100.0 * b_gap / abs(b_s) if b_s != 0 else np.nan
    table_b = pd.concat(
        [
            table_b,
            pd.DataFrame(
                [
                    {
                        "model": "Static-Habit gap",
                        "annualized_cv_freq4": b_gap,
                        "annualized_cv_freq6": (h_mu - s_mu) * pcfg.annual_freq_high,
                        "gap_pct_vs_static_freq4": b_gap_pct,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    table_b.to_csv(os.path.join(pcfg.out_dir, "table11_panel_B_annualized_cv.csv"), index=False)

    # Panel C: aggregate CV loss per store-week.
    rows_c = []
    for m in MODEL_ORDER:
        x = cv_df.loc[cv_df["model"] == m]
        mu, se = _cluster_mean_se(x["cv_aggregate"].values, x["store"].values)
        rows_c.append({"model": m, "cv_aggregate_mean": mu, "cv_aggregate_se_store_cluster": se})
    table_c = pd.DataFrame(rows_c)
    c_s = float(table_c.loc[table_c["model"] == "Neural Demand (static)", "cv_aggregate_mean"].iloc[0])
    c_h = float(table_c.loc[table_c["model"] == "Neural Demand (habit)", "cv_aggregate_mean"].iloc[0])
    c_gap = c_h - c_s
    c_gap_pct = 100.0 * c_gap / abs(c_s) if c_s != 0 else np.nan
    table_c = pd.concat(
        [
            table_c,
            pd.DataFrame(
                [{"model": "Static-Habit gap", "cv_aggregate_mean": c_gap, "cv_aggregate_se_store_cluster": np.nan, "gap_pct_vs_static": c_gap_pct}]
            ),
        ],
        ignore_index=True,
    )
    table_c.to_csv(os.path.join(pcfg.out_dir, "table11_panel_C_aggregate_cv.csv"), index=False)

    # Robustness variants (static/habit only, one seed each for tractability).
    robust_defs = [
        ("nondurable", "y_percustomer_nondurable", False),
        ("total_revenue", "y_percustomer_total", False),
        ("haba_only", "y_percustomer_haba", False),
        ("nondurable_winsor_1_99", "y_percustomer_nondurable", True),
    ]
    robust_rows = []
    for tag, y_col, do_winsor in robust_defs:
        panel_tmp = panel_sw.copy()
        scale_tmp = _scale_budget_to_income_units(
            panel_sw=panel_tmp,
            y_col_raw=y_col,
            old_income_mean=old_income_mean,
            out_col_model="y_model_tmp",
            winsorize=do_winsor,
        )
        sp, tep, dr = _filter_and_replace_income(base_splits, panel_tmp, "y_model_tmp")
        rr = run_once(42, sp, cfg)
        xb = _compute_xbar_variants(rr, sp, cfg)
        vals = {"Neural Demand (static)": [], "Neural Demand (habit)": []}
        for i, row in tep.iterrows():
            p0 = np.array([row["ASP"], row["ACET"], row["IBU"]], dtype=float)
            p1 = p0.copy()
            p1[sg] *= 1.0 + ss
            y = float(row["y_model_tmp"])
            v_s = _compute_cv_for_obs("nirl", p0, p1, y, cfg, **rr["KW"])["cv_percustomer"]
            v_h = _compute_cv_for_obs(
                "mdp",
                p0,
                p1,
                y,
                cfg,
                xb_prev0=xb["xb_habit"][i],
                q_prev0=xb["q_prev"][i],
                **rr["KW"],
            )["cv_percustomer"]
            vals["Neural Demand (static)"].append(v_s)
            vals["Neural Demand (habit)"].append(v_h)
        cv_s = float(np.nanmean(vals["Neural Demand (static)"]))
        cv_h = float(np.nanmean(vals["Neural Demand (habit)"]))
        gap = cv_h - cv_s
        gap_pct = 100.0 * gap / abs(cv_s) if cv_s != 0 else np.nan
        premium_ratio = abs(cv_h) / abs(cv_s) if cv_s != 0 else np.nan
        robust_rows.append(
            {
                "budget_definition": tag,
                "cv_percustomer_static": cv_s,
                "cv_percustomer_habit": cv_h,
                "habit_minus_static": gap,
                "habit_minus_static_pct_of_static": gap_pct,
                "habit_to_static_ratio_abs": premium_ratio,
                "scale_factor": scale_tmp["scale_factor"],
                "test_dropped_invalid_income_or_custcoun": dr["test_dropped_invalid_income_or_custcoun"],
            }
        )
    pd.DataFrame(robust_rows).to_csv(
        os.path.join(pcfg.out_dir, "table11_appendix_budget_definition_robustness.csv"),
        index=False,
    )

    meta_df = pd.DataFrame(
        [
            {
                "old_income_mean": scale_meta["old_income_mean"],
                "new_nondurable_mean_raw": scale_meta["raw_mean"],
                "new_nondurable_scale_factor": scale_meta["scale_factor"],
                **drops,
                "note": "Income proxy changed to per-customer budget; models retrained from scratch.",
            }
        ]
    )
    meta_path = os.path.join(pcfg.out_dir, "exp11_budget_proxy_metadata.csv")
    meta_df.to_csv(meta_path, index=False)

    print(f"[exp11] done. Outputs in: {pcfg.out_dir}")
    return {
        "status": "ok",
        "out_dir": pcfg.out_dir,
        "n_test_rows": int(len(te_panel)),
        "n_cv_rows": int(len(cv_df)),
        "drop_summary": drops,
        "scale_factor": float(scale_meta["scale_factor"]),
        "metadata_path": meta_path,
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Dominick's Exp 11: budget-proxy CV replacement")
    p.add_argument("--weekly", type=str, default="data/wana.csv")
    p.add_argument("--upc", type=str, default="data/upcana.csv")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--fast", action="store_true", help="Use fast model cache/epochs")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--cv-steps", type=int, default=100)
    p.add_argument("--annual-freq-low", type=float, default=4.0)
    p.add_argument("--annual-freq-high", type=float, default=6.0)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
