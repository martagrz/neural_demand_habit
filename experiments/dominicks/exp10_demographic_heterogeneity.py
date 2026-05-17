"""
Demographic heterogeneity and per-buyer welfare normalization (Dominick's).

This script implements:
1) store-week merges with ccount/demo files,
2) per-buyer CV normalization (and HABA-denominator robustness),
3) demographic regressions for welfare/rmse habit gaps,
4) subgroup summaries by demographic terciles,
5) profile-delta heterogeneity by demographic groups,
6) requested figures A-E and tables A-D.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

from experiments.dominicks.data import load_panel, build_arrays, prepare_splits
from experiments.dominicks.exp01_main_runs import run_once
from experiments.dominicks.utils import pred, fit_nd_delta_grid_dom
from run_neural_demand_dominicks import BASE_CFG


MODEL_ORDER = [
    "Neural Demand (static)",
    "Neural Demand (habit)",
    "Neural Demand (habit, FE)",
    "Neural Demand (habit, CF)",
]

DEMOGRAPHIC_KEYS = [
    "income",
    "age60",
    "hsizeavg",
    "hsize1",
    "ethnic",
    "educ",
    "workwom",
    "poverty",
    "shopavid",
    "shophurr",
    "shopcons",
    "density",
]

REGRESSION_KEYS = ["income", "age60", "hsizeavg", "workwom", "poverty", "shopavid"]


@dataclass
class Cfg:
    out_dir: str
    fig_dir: str
    purchase_freq: float
    cv_steps: int
    n_seeds: int
    profile_epochs: int
    profile_delta_grid: np.ndarray


def _seed_list(n: int) -> list[int]:
    return [42 + i * 15 for i in range(n)]


def _build_cfg(args) -> tuple[dict, Cfg]:
    cfg = dict(BASE_CFG)
    cfg["weekly_path"] = args.weekly
    cfg["upc_path"] = args.upc
    cfg["device"] = "cpu"
    cfg["force_retrain"] = False
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
    cfg["cv_steps"] = args.cv_steps
    cfg["N_RUNS"] = args.n_seeds

    out_dir = os.path.join(cfg["out_dir"], "heterogeneity")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    pcfg = Cfg(
        out_dir=out_dir,
        fig_dir=fig_dir,
        purchase_freq=float(args.purchase_freq),
        cv_steps=int(args.cv_steps),
        n_seeds=int(args.n_seeds),
        profile_epochs=int(args.profile_epochs),
        profile_delta_grid=np.asarray(args.profile_delta_grid, dtype=float),
    )
    return cfg, pcfg


def _load_aux_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ccount = pd.read_csv(os.path.join(data_dir, "ccount.csv"), low_memory=False)
    demo = pd.read_csv(os.path.join(data_dir, "demo.csv"), low_memory=False)

    ccount.columns = [c.lower() for c in ccount.columns]
    demo.columns = [c.lower() for c in demo.columns]

    c_week = (
        ccount.groupby(["store", "week"], as_index=False)[["custcoun", "haba"]]
        .sum()
        .rename(columns={"custcoun": "CUSTCOUN", "haba": "HABA"})
    )

    demo = demo.copy()
    demo["shopavid"] = demo.get("shpavid", np.nan)
    demo["shophurr"] = demo.get("shphurr", np.nan)
    demo["shopcons"] = demo.get("shpcons", np.nan)
    keep = ["store", "lat", "long", "priclow", "pricmed", "prichigh"] + DEMOGRAPHIC_KEYS
    keep = [k for k in keep if k in demo.columns]
    demo = demo[keep].drop_duplicates(subset=["store"])
    return c_week, demo


def _merge_panel_with_aux(panel: pd.DataFrame, c_week: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out = out.rename(columns={"STORE": "store", "WEEK": "week"})
    out["store"] = out["store"].astype(int)
    out["week"] = out["week"].astype(int)
    out = out.merge(c_week, on=["store", "week"], how="left")
    out = out.merge(demo, on="store", how="left")
    out["CUSTCOUN"] = out["CUSTCOUN"].fillna(0.0)
    out["HABA"] = out["HABA"].replace(0, np.nan)
    out["analgesic_revenue"] = out[["R_ASP", "R_ACET", "R_IBU"]].sum(axis=1)
    out["pen_it"] = out["analgesic_revenue"] / out["HABA"]
    out["buyers_it"] = out["CUSTCOUN"] * out["pen_it"]
    out.loc[~np.isfinite(out["buyers_it"]), "buyers_it"] = np.nan
    out.loc[out["buyers_it"] <= 0, "buyers_it"] = np.nan
    return out


def _build_price_tier(df: pd.DataFrame) -> pd.Series:
    if all(c in df.columns for c in ["prichigh", "pricmed", "priclow"]):
        return np.select(
            [df["prichigh"] > 0, df["pricmed"] > 0, df["priclow"] > 0],
            ["High", "Medium", "Low"],
            default="CubFighter",
        )
    return pd.Series(["Unknown"] * len(df), index=df.index)


def _predict_test_matrix(run: dict, splits: dict, cfg: dict, spec_name: str) -> np.ndarray:
    p_te = splits["p_te"]
    y_te = splits["y_te"]
    ls_te = splits["ls_te"]
    s_te = splits["s_te"]
    s_te_idx = splits["s_te_idx"]
    q_prev = splits["qp_te"]

    from src.models.dominicks import compute_xbar_e2e
    import torch

    dev = cfg["device"]
    ls_te_t = torch.tensor(ls_te, dtype=torch.float32, device=dev)

    if spec_name == "Neural Demand (static)":
        return pred("nirl", p_te, y_te, cfg, **run["KW"])
    if spec_name == "Neural Demand (habit)":
        d = torch.tensor(float(run["delta_mdp"]), dtype=torch.float32, device=dev)
        xb = compute_xbar_e2e(d, ls_te_t, store_ids=s_te).cpu().numpy()
        return pred("mdp", p_te, y_te, cfg, xb_prev=xb, q_prev=q_prev, **run["KW"])
    if spec_name == "Neural Demand (habit, FE)":
        d = torch.tensor(float(run["delta_mdp_fe"]), dtype=torch.float32, device=dev)
        xb = compute_xbar_e2e(d, ls_te_t, store_ids=s_te).cpu().numpy()
        return pred(
            "mdp-fe",
            p_te,
            y_te,
            cfg,
            xb_prev=xb,
            q_prev=q_prev,
            store_idx=s_te_idx,
            s_te_mode_idx=int(splits["s_te_mode_idx"]),
            **run["KW"],
        )
    if spec_name == "Neural Demand (habit, CF)":
        d = torch.tensor(float(run.get("delta_mdp_cf", run["delta_mdp"])), dtype=torch.float32, device=dev)
        xb = compute_xbar_e2e(d, ls_te_t, store_ids=s_te).cpu().numpy()
        return pred("mdp-cf", p_te, y_te, cfg, xb_prev=xb, q_prev=q_prev, **run["KW"])
    raise ValueError(f"Unknown spec_name={spec_name}")


def _cv_per_obs(run: dict, splits: dict, cfg: dict, spec_name: str) -> np.ndarray:
    p0 = splits["p_te"].copy()
    y = splits["y_te"].copy()
    n, g = p0.shape
    sg = int(cfg["shock_good"])
    ss = float(cfg["shock_pct"])
    p1 = p0.copy()
    p1[:, sg] *= (1.0 + ss)
    dp = (p1 - p0) / float(cfg["cv_steps"])

    cv = np.zeros(n, dtype=float)
    for t in range(cfg["cv_steps"]):
        pt = p0 + (t / float(cfg["cv_steps"])) * (p1 - p0)
        wp = _predict_test_matrix(run, {**splits, "p_te": pt, "y_te": y}, cfg, spec_name)
        cv -= np.sum((wp * y[:, None] / np.maximum(pt, 1e-8)) * dp, axis=1)
    return cv * 100.0


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
    X = np.ones((len(x), 1))
    try:
        fit = sm.OLS(x.values, X).fit(cov_type="cluster", cov_kwds={"groups": g.values})
        return float(fit.params[0]), float(fit.bse[0])
    except Exception:
        return float(x.mean()), np.nan


def _reg_table(df_store: pd.DataFrame, dep: str, rhs: list[str]) -> pd.DataFrame:
    use = df_store[[dep] + rhs].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 10:
        return pd.DataFrame(columns=["variable", "coef", "se_robust", "pvalue", "n", "r2"])
    X = sm.add_constant(use[rhs])
    y = use[dep]
    try:
        fit = sm.OLS(y, X).fit(cov_type="HC1")
    except Exception:
        return pd.DataFrame(columns=["variable", "coef", "se_robust", "pvalue", "n", "r2"])
    out = pd.DataFrame(
        {
            "variable": fit.params.index,
            "coef": fit.params.values,
            "se_robust": fit.bse.values,
            "pvalue": fit.pvalues.values,
            "n": len(use),
            "r2": fit.rsquared,
        }
    )
    return out


def _lasso_select(df_store: pd.DataFrame, dep: str, rhs: list[str]) -> pd.DataFrame:
    use = df_store[[dep] + rhs].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 10:
        return pd.DataFrame(columns=["variable", "coef"])
    X = use[rhs].values
    y = use[dep].values
    model = make_pipeline(
        StandardScaler(),
        LassoCV(
            cv=min(5, max(2, len(use) // 10)),
            random_state=42,
            max_iter=50000,
            tol=1e-4,
            n_jobs=-1,
        ),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, y)
    lasso = model.named_steps["lassocv"]
    coef = pd.Series(lasso.coef_, index=rhs)
    sel = coef[coef.abs() > 1e-8].sort_values(key=np.abs, ascending=False)
    return sel.rename("coef").reset_index().rename(columns={"index": "variable"})


def _tercile_labels(x: pd.Series) -> pd.Series:
    q = pd.qcut(x, q=3, labels=["Low", "Mid", "High"], duplicates="drop")
    q = q.astype(object).where(~q.isna(), "Missing")
    return q.astype(str)


def _compute_group_profile_delta(
    splits: dict, cfg: dict, group_store_ids: np.ndarray, delta_grid: np.ndarray, epochs: int, tag: str
) -> dict:
    tr_mask = np.isin(splits["s_tr"], group_store_ids)
    te_mask = np.isin(splits["s_te"], group_store_ids)
    out = {"delta_hat": np.nan, "id_set_l": np.nan, "id_set_u": np.nan, "kl_grid": np.nan * delta_grid}
    if tr_mask.sum() < 50 or te_mask.sum() < 30:
        return out

    sub_cfg = dict(cfg)
    sub_cfg["mdp_epochs"] = int(epochs)
    sub_cfg["mdp_e2e_epochs"] = int(epochs)
    sw = fit_nd_delta_grid_dom(
        splits["p_tr"][tr_mask],
        splits["y_tr"][tr_mask],
        splits["w_tr"][tr_mask],
        splits["ls_tr"][tr_mask],
        splits["p_te"][te_mask],
        splits["y_te"][te_mask],
        splits["w_te"][te_mask],
        splits["ls_te"][te_mask],
        sub_cfg,
        delta_grid=delta_grid,
        with_fe=False,
        store_ids_tr=splits["s_tr"][tr_mask],
        store_ids_val=splits["s_te"][te_mask],
        hidden=sub_cfg.get("mdp_hidden", 128),
        tag=tag,
    )
    out["delta_hat"] = float(sw["delta_hat"])
    out["id_set_l"] = float(sw["id_set"][0])
    out["id_set_u"] = float(sw["id_set"][1])
    out["kl_grid"] = sw["kl_grid"]
    return out


def run(args):
    cfg, pcfg = _build_cfg(args)

    panel = load_panel(cfg)
    data = build_arrays(panel, cfg)
    splits = prepare_splits(data, cfg)

    c_week, demo = _load_aux_data(args.data_dir)
    merged = _merge_panel_with_aux(panel, c_week, demo)
    merged = merged.sort_values(["store", "week"]).reset_index(drop=True)

    seeds = _seed_list(pcfg.n_seeds)
    runs = []
    for i, seed in enumerate(seeds, 1):
        print(f"[exp10] run_once seed {seed} ({i}/{len(seeds)})")
        runs.append(run_once(seed, splits, cfg))

    te_idx = splits["te"]
    te_panel = merged.iloc[te_idx].copy().reset_index(drop=True)
    te_panel["price_tier"] = _build_price_tier(te_panel)
    te_panel["buyers_it"] = te_panel["buyers_it"].replace([np.inf, -np.inf], np.nan)
    te_panel.loc[te_panel["buyers_it"] <= 0, "buyers_it"] = np.nan

    # CV per observation (seed 1 as baseline)
    base_run = runs[0]
    for mn in MODEL_ORDER:
        te_panel[f"cv_{mn}"] = _cv_per_obs(base_run, splits, cfg, mn)
        te_panel[f"cv_per_buyer_{mn}"] = te_panel[f"cv_{mn}"] / te_panel["buyers_it"]
        te_panel[f"cv_per_haba_{mn}"] = te_panel[f"cv_{mn}"] / te_panel["HABA"]

    # Seed-level RMSE gaps by store for Figure C error bars
    seed_rmse_by_store = []
    for si, run in enumerate(runs):
        wp_static = _predict_test_matrix(run, splits, cfg, "Neural Demand (static)")
        wp_habit = _predict_test_matrix(run, splits, cfg, "Neural Demand (habit)")
        err_static = ((splits["w_te"] - wp_static) ** 2).sum(axis=1)
        err_habit = ((splits["w_te"] - wp_habit) ** 2).sum(axis=1)
        tmp = pd.DataFrame(
            {
                "store": te_panel["store"].values,
                "rmse_static_store": np.sqrt(err_static),
                "rmse_habit_store": np.sqrt(err_habit),
            }
        )
        sb = tmp.groupby("store", as_index=False).mean()
        sb["rmse_gap"] = sb["rmse_static_store"] - sb["rmse_habit_store"]
        # run_once in exp01_main_runs does not carry a seed field; set it explicitly.
        sb["seed"] = seeds[si] if si < len(seeds) else (42 + 15 * si)
        seed_rmse_by_store.append(sb)
    seed_rmse_by_store = pd.concat(seed_rmse_by_store, ignore_index=True)

    # Store-level outcomes (baseline seed, CV and RMSE gaps)
    wp_static = _predict_test_matrix(base_run, splits, cfg, "Neural Demand (static)")
    wp_habit = _predict_test_matrix(base_run, splits, cfg, "Neural Demand (habit)")
    err_static = ((splits["w_te"] - wp_static) ** 2).sum(axis=1)
    err_habit = ((splits["w_te"] - wp_habit) ** 2).sum(axis=1)
    te_panel["rmse_static_obs"] = np.sqrt(err_static)
    te_panel["rmse_habit_obs"] = np.sqrt(err_habit)
    te_panel["welfare_gap_obs"] = te_panel["cv_Neural Demand (habit)"] - te_panel["cv_Neural Demand (static)"]
    te_panel["welfare_gap_per_buyer_obs"] = (
        te_panel["cv_per_buyer_Neural Demand (habit)"] - te_panel["cv_per_buyer_Neural Demand (static)"]
    )

    store_cols = ["store"] + [c for c in DEMOGRAPHIC_KEYS + ["lat", "long", "price_tier"] if c in te_panel.columns]
    store_df = te_panel[store_cols].drop_duplicates(subset=["store"])
    store_metrics = (
        te_panel.groupby("store", as_index=False)
        .agg(
            welfare_gap=("welfare_gap_obs", "mean"),
            welfare_gap_per_buyer=("welfare_gap_per_buyer_obs", "mean"),
            rmse_static=("rmse_static_obs", "mean"),
            rmse_habit=("rmse_habit_obs", "mean"),
            cv_static=("cv_Neural Demand (static)", "mean"),
            cv_habit=("cv_Neural Demand (habit)", "mean"),
        )
    )
    store_metrics["rmse_gap"] = store_metrics["rmse_static"] - store_metrics["rmse_habit"]
    store_metrics = store_metrics.merge(store_df, on="store", how="left")

    # Table A: per-buyer normalization
    rows_a = []
    for mn in MODEL_ORDER:
        m, se = _cluster_mean_se(te_panel[f"cv_per_buyer_{mn}"].values, te_panel["store"].values)
        mean_cv_sw = float(np.nanmean(te_panel[f"cv_{mn}"]))
        mean_buyers = float(np.nanmean(te_panel["buyers_it"]))
        ann = m * pcfg.purchase_freq if np.isfinite(m) else np.nan
        rows_a.append(
            {
                "Model": mn,
                "CV_per_store_week": mean_cv_sw,
                "Implied_buyers": mean_buyers,
                "CV_per_buyer": m,
                "SE_cluster_store": se,
                f"Annualized_x{pcfg.purchase_freq:g}": ann,
            }
        )
    table_a = pd.DataFrame(rows_a)
    static_pb = table_a.loc[table_a["Model"] == "Neural Demand (static)", "CV_per_buyer"].iloc[0]
    habit_pb = table_a.loc[table_a["Model"] == "Neural Demand (habit)", "CV_per_buyer"].iloc[0]
    gap_pb = (habit_pb - static_pb) if np.isfinite(habit_pb) and np.isfinite(static_pb) else np.nan
    table_a = pd.concat(
        [
            table_a,
            pd.DataFrame(
                [
                    {
                        "Model": "Gap (habit-static)",
                        "CV_per_store_week": np.nan,
                        "Implied_buyers": np.nan,
                        "CV_per_buyer": gap_pb,
                        "SE_cluster_store": np.nan,
                        f"Annualized_x{pcfg.purchase_freq:g}": gap_pb * pcfg.purchase_freq
                        if np.isfinite(gap_pb)
                        else np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    table_a.to_csv(os.path.join(pcfg.out_dir, "table_new_A_per_buyer_cv.csv"), index=False)

    # Robustness with HABA denominator
    rows_a_h = []
    for mn in MODEL_ORDER:
        m, se = _cluster_mean_se(te_panel[f"cv_per_haba_{mn}"].values, te_panel["store"].values)
        rows_a_h.append({"Model": mn, "CV_per_HABA": m, "SE_cluster_store": se})
    pd.DataFrame(rows_a_h).to_csv(os.path.join(pcfg.out_dir, "table_new_A_per_haba_robustness.csv"), index=False)

    # Table B: demographic predictors
    reg_rhs = [k for k in REGRESSION_KEYS if k in store_metrics.columns]
    reg_w = _reg_table(store_metrics, "welfare_gap", reg_rhs)
    reg_r = _reg_table(store_metrics, "rmse_gap", reg_rhs)
    reg_w["depvar"] = "welfare_gap"
    reg_r["depvar"] = "rmse_gap"
    table_b = pd.concat([reg_w, reg_r], ignore_index=True)
    table_b.to_csv(os.path.join(pcfg.out_dir, "table_new_B_demographic_predictors.csv"), index=False)

    # LASSO sparse model
    lasso_rhs = [k for k in DEMOGRAPHIC_KEYS if k in store_metrics.columns]
    lasso_w = _lasso_select(store_metrics, "welfare_gap", lasso_rhs)
    lasso_w["depvar"] = "welfare_gap"
    lasso_r = _lasso_select(store_metrics, "rmse_gap", lasso_rhs)
    lasso_r["depvar"] = "rmse_gap"
    pd.concat([lasso_w, lasso_r], ignore_index=True).to_csv(
        os.path.join(pcfg.out_dir, "table_new_B_lasso_selected_predictors.csv"), index=False
    )

    # Table C + subgroup summaries
    for k in ["income", "poverty", "shopavid"]:
        if k in store_metrics.columns:
            store_metrics[f"{k}_tercile"] = _tercile_labels(store_metrics[k])

    sub_rows = []
    subgroup_defs = []
    if "income_tercile" in store_metrics.columns:
        subgroup_defs += [("income", "Low"), ("income", "Mid"), ("income", "High")]
    if "poverty_tercile" in store_metrics.columns:
        subgroup_defs += [("poverty", "Low"), ("poverty", "High")]
    if "shopavid_tercile" in store_metrics.columns:
        subgroup_defs += [("shopavid", "High"), ("shopavid", "Low")]

    for dim, lvl in subgroup_defs:
        gcol = f"{dim}_tercile"
        gstores = store_metrics.loc[store_metrics[gcol] == lvl, "store"]
        idx = te_panel["store"].isin(gstores)
        cv_s = te_panel.loc[idx, "cv_Neural Demand (static)"].mean()
        cv_h = te_panel.loc[idx, "cv_Neural Demand (habit)"].mean()
        gap = cv_h - cv_s
        gap_pct = 100.0 * gap / abs(cv_s) if np.isfinite(cv_s) and cv_s != 0 else np.nan
        sub_rows.append(
            {
                "Demographic_group": f"{dim}:{lvl}",
                "CV_static": cv_s,
                "CV_habit": cv_h,
                "Gap_$": gap,
                "Gap_%": gap_pct,
            }
        )
    table_c = pd.DataFrame(sub_rows)
    table_c.to_csv(os.path.join(pcfg.out_dir, "table_new_C_subgroup_welfare.csv"), index=False)

    # Profile delta by subgroup
    profile_rows = []
    prof = {}
    if "income_tercile" in store_metrics.columns:
        low_income_st = store_metrics.loc[store_metrics["income_tercile"] == "Low", "store"].values
        high_income_st = store_metrics.loc[store_metrics["income_tercile"] == "High", "store"].values
        prof["Low income"] = _compute_group_profile_delta(
            splits, cfg, low_income_st, pcfg.profile_delta_grid, pcfg.profile_epochs, "exp10-low-income"
        )
        prof["High income"] = _compute_group_profile_delta(
            splits, cfg, high_income_st, pcfg.profile_delta_grid, pcfg.profile_epochs, "exp10-high-income"
        )
    if "poverty_tercile" in store_metrics.columns:
        high_pov_st = store_metrics.loc[store_metrics["poverty_tercile"] == "High", "store"].values
        prof["High poverty"] = _compute_group_profile_delta(
            splits, cfg, high_pov_st, pcfg.profile_delta_grid, pcfg.profile_epochs, "exp10-high-poverty"
        )
    if "shopavid_tercile" in store_metrics.columns:
        avid_st = store_metrics.loc[store_metrics["shopavid_tercile"] == "High", "store"].values
        prof["Avid shoppers"] = _compute_group_profile_delta(
            splits, cfg, avid_st, pcfg.profile_delta_grid, pcfg.profile_epochs, "exp10-avid-shoppers"
        )
    for gname, d in prof.items():
        profile_rows.append(
            {
                "Group": gname,
                "delta_hat": d["delta_hat"],
                "IS_lower": d["id_set_l"],
                "IS_upper": d["id_set_u"],
                "IS_width": d["id_set_u"] - d["id_set_l"] if np.isfinite(d["id_set_u"]) else np.nan,
            }
        )
    table_d = pd.DataFrame(profile_rows)
    table_d.to_csv(os.path.join(pcfg.out_dir, "table_new_D_profile_delta_subgroup.csv"), index=False)

    # Figure A: per-buyer welfare bars
    fa = table_a[table_a["Model"].isin(MODEL_ORDER)].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fa))
    ax.bar(x, fa["CV_per_buyer"], yerr=fa["SE_cluster_store"], capsize=4, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(fa["Model"], rotation=20, ha="right")
    ax.set_ylabel("CV loss per implied analgesics buyer ($)")
    if {"Neural Demand (static)", "Neural Demand (habit)"} <= set(fa["Model"]):
        s = fa.loc[fa["Model"] == "Neural Demand (static)", "CV_per_buyer"].iloc[0]
        h = fa.loc[fa["Model"] == "Neural Demand (habit)", "CV_per_buyer"].iloc[0]
        ax.text(0.02, 0.95, f"Static-Habit gap: {(s-h):.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(pcfg.fig_dir, "figure_A_per_buyer_welfare_gap.png"), dpi=160)
    plt.close(fig)

    # Figure B: welfare gap vs income
    if "income" in store_metrics.columns:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        colors = {
            "High": "#D62728",
            "Medium": "#1F77B4",
            "Low": "#2CA02C",
            "CubFighter": "#9467BD",
            "Unknown": "#7F7F7F",
        }
        for tier, gdf in store_metrics.groupby("price_tier"):
            ax.scatter(
                gdf["income"],
                gdf["welfare_gap_per_buyer"],
                s=25,
                alpha=0.75,
                color=colors.get(tier, "#7F7F7F"),
                label=tier,
            )
        lw = lowess(store_metrics["welfare_gap_per_buyer"], store_metrics["income"], frac=0.5, return_sorted=True)
        ax.plot(lw[:, 0], lw[:, 1], color="black", lw=2, label="LOWESS")
        ax.set_xlabel("Store median income")
        ax.set_ylabel("Welfare gap per buyer (habit-static)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(pcfg.fig_dir, "figure_B_welfare_gap_vs_income.png"), dpi=160)
        plt.close(fig)

    # Figure C: RMSE reduction by terciles with seed-based error bars
    panel_dims = [d for d in ["income", "poverty", "shopavid"] if f"{d}_tercile" in store_metrics.columns]
    if panel_dims:
        fig, axes = plt.subplots(1, len(panel_dims), figsize=(5 * len(panel_dims), 4), sharey=True)
        if len(panel_dims) == 1:
            axes = [axes]
        for ax, dim in zip(axes, panel_dims):
            tmp = store_metrics[["store", f"{dim}_tercile"]].dropna().rename(columns={f"{dim}_tercile": "tercile"})
            sr = seed_rmse_by_store.merge(tmp, on="store", how="inner")
            grp = sr.groupby(["seed", "tercile"], as_index=False)["rmse_gap"].mean()
            agg = grp.groupby("tercile")["rmse_gap"].agg(["mean", "std", "count"]).reset_index()
            if len(agg) == 0:
                ax.text(0.5, 0.5, "No valid seed-level data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
            else:
                agg["se"] = agg["std"] / np.sqrt(np.maximum(agg["count"], 1))
                order = [o for o in ["Low", "Mid", "High"] if o in agg["tercile"].values]
                agg = agg.set_index("tercile").loc[order].reset_index()
                xx = np.arange(len(agg))
                ax.bar(xx, agg["mean"], yerr=agg["se"], capsize=4, color="#72B7B2")
                ax.set_xticks(xx)
                ax.set_xticklabels(agg["tercile"])
            ax.set_ylabel("RMSE reduction (static - habit)")

        fig.tight_layout()
        fig.savefig(os.path.join(pcfg.fig_dir, "figure_C_habit_intensity_terciles.png"), dpi=160)
        plt.close(fig)

    # Figure D: profile-delta curves by income group
    if {"Low income", "High income"} <= set(prof.keys()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for gname, col in [("Low income", "#1F77B4"), ("High income", "#D62728")]:
            d = prof[gname]
            ax.plot(pcfg.profile_delta_grid, d["kl_grid"], marker="o", color=col, label=gname)
            if np.isfinite(d["id_set_l"]):
                ax.axvspan(d["id_set_l"], d["id_set_u"], color=col, alpha=0.12)
        ax.set_xlabel("delta grid")
        ax.set_ylabel("validation KL")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(pcfg.fig_dir, "figure_D_profile_delta_income_groups.png"), dpi=160)
        plt.close(fig)

    # Figure E: distributional welfare map (lat/long scatter)
    if {"lat", "long"} <= set(store_metrics.columns):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        sc = ax.scatter(
            store_metrics["long"],
            store_metrics["lat"],
            c=store_metrics["welfare_gap_per_buyer"],
            cmap="viridis",
            s=45,
            alpha=0.9,
        )
        plt.colorbar(sc, ax=ax, label="Welfare gap per buyer")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        fig.savefig(os.path.join(pcfg.fig_dir, "figure_E_distributional_welfare_map.png"), dpi=160)
        plt.close(fig)

    # Save merged panel + store metrics for reproducibility
    te_panel.to_csv(os.path.join(pcfg.out_dir, "panel_store_week_demographics_test.csv"), index=False)
    store_metrics.to_csv(os.path.join(pcfg.out_dir, "store_level_habit_intensity.csv"), index=False)

    print(f"[exp10] done. Outputs in: {pcfg.out_dir}")
    return {
        "out_dir": pcfg.out_dir,
        "fig_dir": pcfg.fig_dir,
        "n_stores": int(store_metrics["store"].nunique()),
        "n_test_rows": int(len(te_panel)),
        "tables": {
            "A": os.path.join(pcfg.out_dir, "table_new_A_per_buyer_cv.csv"),
            "A_robust": os.path.join(pcfg.out_dir, "table_new_A_per_haba_robustness.csv"),
            "B": os.path.join(pcfg.out_dir, "table_new_B_demographic_predictors.csv"),
            "B_lasso": os.path.join(pcfg.out_dir, "table_new_B_lasso_selected_predictors.csv"),
            "C": os.path.join(pcfg.out_dir, "table_new_C_subgroup_welfare.csv"),
            "D": os.path.join(pcfg.out_dir, "table_new_D_profile_delta_subgroup.csv"),
        },
        "figures": {
            "A": os.path.join(pcfg.fig_dir, "figure_A_per_buyer_welfare_gap.png"),
            "B": os.path.join(pcfg.fig_dir, "figure_B_welfare_gap_vs_income.png"),
            "C": os.path.join(pcfg.fig_dir, "figure_C_habit_intensity_terciles.png"),
            "D": os.path.join(pcfg.fig_dir, "figure_D_profile_delta_income_groups.png"),
            "E": os.path.join(pcfg.fig_dir, "figure_E_distributional_welfare_map.png"),
        },
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Dominick's demographic heterogeneity experiment")
    p.add_argument("--weekly", type=str, default="data/wana.csv")
    p.add_argument("--upc", type=str, default="data/upcana.csv")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--fast", action="store_true", help="Use fast model cache/epochs")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--cv-steps", type=int, default=25)
    p.add_argument("--purchase-freq", type=float, default=5.0)
    p.add_argument("--profile-epochs", type=int, default=120)
    p.add_argument("--profile-delta-grid", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())

