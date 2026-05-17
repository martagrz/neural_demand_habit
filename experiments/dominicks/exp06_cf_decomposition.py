"""
experiments/neural_demand/dominicks/exp06_cf_decomposition.py
=============================================================
Section 2.4 — Endogeneity Correction via Control Function (Dominick's)

Implements the three-way decomposition described in the paper:

  (1) Neural Demand (static)       — uncorrected; mixes structural + endogeneity
  (2) Neural Demand (CF)         — CF-corrected; isolates causal preference
  (3) Neural Demand (habit, CF)  — adds structural habit persistence

The gap (1)-(2) captures endogeneity bias removed by the Hausman IV control
function.  The gap (2)-(3) captures structural habit persistence.  Model (3)
provides the cleanest welfare / elasticity estimates.

We report:
  • First-stage R² (Hausman instruments relevance)
  • RMSE / MAE / KL comparison across the three models
  • Own-price and cross-price elasticity matrices for each model
  • Compensating variation for a 10% Aspirin price shock
  • Demand curves with and without CF correction

Produces
--------
results/neural_demand/dominicks/
  table_cf_rsq.csv / .tex               First-stage R²
  table_cf_accuracy.csv / .tex          RMSE / MAE / KL comparison
  table_cf_elasticities.csv / .tex      Elasticity comparison
  table_cf_welfare.csv / .tex           CV comparison
  fig_cf_accuracy.{pdf,png}             Bar chart: RMSE reduction
  fig_cf_demand_curves.{pdf,png}        Demand curves — with/without CF
  fig_cf_elasticity_heatmap.{pdf,png}   Elasticity matrix heatmap
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.dominicks import (
    LAAIDS, BLPLogitIV, QUAIDS, SeriesDemand,
    StaticND,
    HabitND,
    _train,
    cf_first_stage,
    compute_xbar_e2e,
    feat_good_specific, feat_orth, feat_shared,
    run_lirl,
)
from experiments.dominicks.data import G, GOODS

from experiments.dominicks.utils import (
    predict,
    elasticity_matrix as _elast_mat,
    compensating_variation as _comp_var,
    fit_nd_delta_grid_dom,
    STYLE, BAND,
    bar_chart,
    make_performance_table,
    make_elasticity_table,
    aggregate_runs,
    ALL_MODEL_NAMES,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Model names for this experiment (all benchmarks + CF models)
# ─────────────────────────────────────────────────────────────────────────────

CF_MODEL_NAMES = [
    "LA-AIDS", "Logit-IV", "QUAIDS", "Series Est.",
    "Linear Demand (Shared)", "Linear Demand (GoodSpec)", "Linear Demand (Orth)",
    "Neural Demand (static)",
    "Neural Demand (habit)",
    "Neural Demand (CF)",
    "Neural Demand (habit, CF)",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Single-run pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_once(seed: int, splits: dict, cfg: dict) -> dict:
    """Train all paper models and produce CF-decomposition metrics for one seed.

    Returns
    -------
    dict with keys: perf, rsq, elast_rows, welf_rows, curves, seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    p_tr  = splits["p_tr"];  p_te  = splits["p_te"]
    w_tr  = splits["w_tr"];  w_te  = splits["w_te"]
    mw_tr = splits.get("mw_tr", w_tr)
    y_tr  = splits["y_tr"];  y_te  = splits["y_te"]
    xb_tr = splits["xb_tr"]; xb_te = splits["xb_te"]
    qp_tr = splits["qp_tr"]; qp_te = splits["qp_te"]
    ls_tr = splits["ls_tr"]; ls_te = splits["ls_te"]
    s_tr  = splits["s_tr"];  s_te  = splits["s_te"]
    Z_tr  = splits["Z_tr"]
    p_grid = splits.get("p_grid")

    dev = cfg["device"]

    # ── Static benchmarks ─────────────────────────────────────────────────────
    aids_m   = LAAIDS().fit(p_tr, w_tr, y_tr)
    blp_m    = BLPLogitIV().fit(p_tr, mw_tr, Z_tr, y_tr)
    quaids_m = QUAIDS().fit(p_tr, w_tr, y_tr)
    series_m = SeriesDemand().fit(p_tr, w_tr, y_tr)
    th_sh   = run_lirl(feat_shared,        p_tr, y_tr, w_tr, cfg)
    th_gs   = run_lirl(feat_good_specific, p_tr, y_tr, w_tr, cfg)
    th_orth = run_lirl(feat_orth,          p_tr, y_tr, w_tr, cfg)

    # ── Neural Demand (static) ─────────────────────────────────────────────────
    nirl_m, _ = _train(
        StaticND(cfg["nirl_hidden"]),
        p_tr, y_tr, w_tr, "nirl", cfg,
        tag=f"StaticND s={seed}")

    # ── Neural Demand (habit) — δ sweep ───────────────────────────────────────
    sw_hab = fit_nd_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg, with_fe=False,
        store_ids_tr=s_tr, store_ids_val=s_te,
        hidden=cfg["mdp_e2e_hidden"],
        tag=f"ND-Habit s={seed}")
    nd_hab_m  = sw_hab["best_model"]
    delta_hat = sw_hab["delta_hat"]
    d_t       = torch.tensor(float(delta_hat), dtype=torch.float32, device=dev)
    ls_te_t   = torch.tensor(ls_te, dtype=torch.float32, device=dev)
    with torch.no_grad():
        xb_e2e_te = compute_xbar_e2e(d_t, ls_te_t, store_ids=s_te).cpu().numpy()

    # ── CF first stage (Hausman instruments) ──────────────────────────────────
    _log_p_tr  = np.log(np.maximum(p_tr, 1e-8))
    v_hat_tr, cf_rsq = cf_first_stage(_log_p_tr, Z_tr)
    print(f"   CF first-stage R²: {cf_rsq.round(3)}")

    # ── Neural Demand (CF) ─────────────────────────────────────────────────────
    nirl_cf_m, _ = _train(
        StaticND(cfg["nirl_hidden"], n_cf=G),
        p_tr, y_tr, w_tr, "nirl", cfg,
        v_hat_tr=v_hat_tr,
        tag=f"StaticND-CF s={seed}")

    # ── Neural Demand (habit, CF) — HabitND with CF residuals ────────────
    mdp_hidden = cfg.get("mdp_hidden", cfg["nirl_hidden"])
    mdp_cf_m, _ = _train(
        HabitND(mdp_hidden, n_cf=G),
        p_tr, y_tr, w_tr, "mdp", cfg,
        xb_prev_tr=xb_tr, q_prev_tr=qp_tr,
        v_hat_tr=v_hat_tr,
        tag=f"ND-Habit-CF s={seed}")

    # ── Unified KW bundle ─────────────────────────────────────────────────────
    KW = dict(
        aids=aids_m, blp=blp_m, quaids=quaids_m, series=series_m,
        ff=feat_shared, theta=th_sh,
        nirl=nirl_m,
        nirl_cf=nirl_cf_m,
        mdp_cf=mdp_cf_m,
        mdp=nd_hab_m,
    )

    # helper: xb_prev for each model
    def _xb(nm):
        if "CF" in nm and "habit" in nm:
            return xb_te
        if "habit" in nm:
            return xb_e2e_te
        return None

    def _kw_extra(nm):
        if nm == "Linear Demand (GoodSpec)":
            return {"ff": feat_good_specific, "theta": th_gs}
        if nm == "Linear Demand (Orth)":
            return {"ff": feat_orth, "theta": th_orth}
        return {}

    # ── Accuracy (RMSE / MAE / KL) ────────────────────────────────────────────
    perf = {}
    for nm in CF_MODEL_NAMES:
        try:
            w_pred = predict(nm, p_te, y_te, cfg,
                             xb_prev=_xb(nm), q_prev=qp_te,
                             **{**KW, **_kw_extra(nm)})
            valid  = ~(np.isnan(w_pred).any(1) | np.isnan(w_te).any(1))
            if valid.sum() == 0:
                raise ValueError("All predictions NaN")
            rmse  = float(np.sqrt(np.mean((w_pred[valid] - w_te[valid]) ** 2)))
            mae   = float(np.mean(np.abs(w_pred[valid] - w_te[valid])))
            kl_v  = float(np.mean(
                np.sum(w_te[valid] * np.log(
                    np.maximum(w_te[valid], 1e-10)
                    / np.maximum(w_pred[valid], 1e-10)), axis=1)))
            perf[nm] = {"RMSE": rmse, "MAE": mae, "KL": kl_v}
        except Exception as exc:
            print(f"    [{nm} accuracy failed: {exc}]")
            perf[nm] = {"RMSE": np.nan, "MAE": np.nan, "KL": np.nan}

    # ── First-stage R² table ──────────────────────────────────────────────────
    rsq_row = {GOODS[j]: float(cf_rsq[j]) for j in range(G)}
    rsq_row["seed"] = seed

    # ── Elasticities at test-mean prices ──────────────────────────────────────
    p0_te  = p_te.mean(0)
    y0_te  = float(y_te.mean())
    xb0_te = xb_te.mean(0)
    xb0_e2e = xb_e2e_te.mean(0)
    qp0_te = qp_te.mean(0)

    elast_rows = []
    for nm in CF_MODEL_NAMES:
        try:
            xb0 = xb0_e2e if ("habit" in nm and "CF" not in nm) else (
                  xb0_te  if "habit" in nm else None)
            mat = _elast_mat(nm, p0_te, y0_te, cfg,
                             xb_prev0=xb0, q_prev0=qp0_te,
                             **{**KW, **_kw_extra(nm)})   # shape (G, G)
            row = {"Model": nm, "seed": seed, "mat": mat}
            # Also store flattened for table: eps_{i}_{j}
            for i in range(G):
                for j in range(G):
                    row[f"eps_{i}_{j}"] = float(mat[i, j])
            elast_rows.append(row)
        except Exception as exc:
            print(f"    [{nm} elasticity failed: {exc}]")
            nan_mat = np.full((G, G), np.nan)
            row = {"Model": nm, "seed": seed, "mat": nan_mat}
            for i in range(G):
                for j in range(G):
                    row[f"eps_{i}_{j}"] = np.nan
            elast_rows.append(row)

    # ── Welfare: CV for 10% Aspirin (good-0) price shock ─────────────────────
    p1_welf = p0_te.copy(); p1_welf[0] *= 1.10

    welf_rows = []
    for nm in CF_MODEL_NAMES:
        try:
            xb0 = xb0_e2e if ("habit" in nm and "CF" not in nm) else (
                  xb0_te  if "habit" in nm else None)
            cv = _comp_var(nm, p0_te, p1_welf, y0_te, cfg,
                           xb_prev0=xb0, q_prev0=qp0_te,
                           **{**KW, **_kw_extra(nm)})
            welf_rows.append({"Model": nm, "seed": seed, "CV": float(cv)})
        except Exception as exc:
            print(f"    [{nm} welfare failed: {exc}]")
            welf_rows.append({"Model": nm, "seed": seed, "CV": np.nan})

    # ── Demand curves (vary good-0 price) ─────────────────────────────────────
    curves = {}
    pgr    = None
    if p_grid is not None:
        pgr   = p_grid[0] if isinstance(p_grid, (list, tuple)) else p_grid
        N_gr  = len(pgr)
        p_cv  = np.tile(p0_te, (N_gr, 1))
        p_cv[:, 0] = pgr
        y_gr  = np.full(N_gr, y0_te)
        xb_gr_e2e = np.tile(xb0_e2e, (N_gr, 1))
        xb_gr_ewm = np.tile(xb0_te,  (N_gr, 1))
        qp_gr     = np.tile(qp0_te,  (N_gr, 1))
        for nm in CF_MODEL_NAMES:
            try:
                xb_g = xb_gr_e2e if ("habit" in nm and "CF" not in nm) else (
                       xb_gr_ewm  if "habit" in nm else None)
                w_cr = predict(nm, p_cv, y_gr, cfg,
                               xb_prev=xb_g, q_prev=qp_gr if xb_g is not None else None,
                               **{**KW, **_kw_extra(nm)})
                curves[nm] = w_cr[:, 0]
            except Exception as exc:
                print(f"    [{nm} demand curve failed: {exc}]")
                curves[nm] = np.full(N_gr, np.nan)

    return dict(
        perf=perf,
        rsq_row=rsq_row,
        elast_rows=elast_rows,
        welf_rows=welf_rows,
        curves=curves,
        p_grid_0=pgr,
        cf_rsq=cf_rsq,
        delta_hat=delta_hat,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _se(arr):
    a = np.asarray([x for x in arr if x is not None and not np.isnan(float(x))], float)
    if len(a) < 2:
        return 0.0
    return float(np.std(a, ddof=1) / np.sqrt(len(a)))


def aggregate(all_results: list) -> dict:
    n_runs = len(all_results)

    # ── Performance ──────────────────────────────────────────────────────────
    perf_agg = {}
    for nm in CF_MODEL_NAMES:
        rmses = [r["perf"].get(nm, {}).get("RMSE", np.nan) for r in all_results]
        maes  = [r["perf"].get(nm, {}).get("MAE",  np.nan) for r in all_results]
        kls   = [r["perf"].get(nm, {}).get("KL",   np.nan) for r in all_results]
        perf_agg[nm] = {
            "RMSE_mean": float(np.nanmean(rmses)), "RMSE_se": _se(rmses),
            "MAE_mean":  float(np.nanmean(maes)),  "MAE_se":  _se(maes),
            "KL_mean":   float(np.nanmean(kls)),   "KL_se":   _se(kls),
        }

    # ── First-stage R² ───────────────────────────────────────────────────────
    rsq_rows_all = [r["rsq_row"] for r in all_results]
    rsq_df = pd.DataFrame(rsq_rows_all)

    # ── Elasticities (full G×G matrices) ─────────────────────────────────────
    elast_agg = []
    for nm in CF_MODEL_NAMES:
        mats = []
        for r in all_results:
            matched = [row for row in r["elast_rows"] if row["Model"] == nm]
            if matched and matched[0].get("mat") is not None:
                mats.append(matched[0]["mat"])
        if mats:
            mat_mean = np.nanmean(mats, axis=0)
            mat_se   = (np.nanstd(mats, axis=0, ddof=1) / np.sqrt(len(mats))
                        if len(mats) > 1 else np.zeros((G, G)))
        else:
            mat_mean = np.full((G, G), np.nan)
            mat_se   = np.zeros((G, G))
        row = {"Model": nm, "mat_mean": mat_mean, "mat_se": mat_se}
        for i in range(G):
            for j in range(G):
                row[f"eps_{i}_{j}"]    = float(mat_mean[i, j])
                row[f"eps_{i}_{j}_se"] = float(mat_se[i, j])
        elast_agg.append(row)

    # ── Welfare ──────────────────────────────────────────────────────────────
    welf_all = []
    for r in all_results:
        welf_all.extend(r["welf_rows"])
    wf_df   = pd.DataFrame(welf_all)
    wf_mean = wf_df.groupby("Model", sort=False)[["CV"]].mean().reset_index()
    wf_se   = wf_df.groupby("Model", sort=False)[["CV"]].sem().reset_index()
    welf_agg = []
    for nm in CF_MODEL_NAMES:
        sub_m = wf_mean.loc[wf_mean["Model"] == nm, "CV"]
        sub_s = wf_se.loc[wf_se["Model"]   == nm, "CV"]
        welf_agg.append({
            "Model": nm,
            "CV_mean": float(sub_m.iloc[0]) if len(sub_m) else np.nan,
            "CV_se":   float(sub_s.iloc[0]) if len(sub_s) else 0.0,
        })

    # ── Demand curves ─────────────────────────────────────────────────────────
    curve_keys = list(all_results[0]["curves"].keys())
    curves_mean = {k: np.nanmean([r["curves"].get(k, np.full(1, np.nan))
                                  for r in all_results], axis=0)
                   for k in curve_keys}
    curves_se   = {}
    for k in curve_keys:
        stack = np.array([r["curves"].get(k, np.full(1, np.nan))
                          for r in all_results])
        curves_se[k] = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n_runs)

    return dict(
        perf_agg=perf_agg,
        rsq_df=rsq_df,
        elast_agg=elast_agg,
        welf_agg=welf_agg,
        curves_mean=curves_mean,
        curves_se=curves_se,
        p_grid_0=all_results[0].get("p_grid_0"),
        n_runs=n_runs,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(agg: dict, cfg: dict) -> None:
    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    n_runs = agg["n_runs"]
    se_note = f"  ({n_runs} runs, ±1 SE)" if n_runs > 1 else ""

    perf_agg = agg["perf_agg"]

    # ── Fig A: RMSE / KL bar chart ────────────────────────────────────────────
    rmse_means = {nm: perf_agg[nm]["RMSE_mean"] for nm in CF_MODEL_NAMES}
    rmse_ses   = {nm: perf_agg[nm]["RMSE_se"]   for nm in CF_MODEL_NAMES}
    kl_means   = {nm: perf_agg[nm]["KL_mean"]   for nm in CF_MODEL_NAMES}
    kl_ses     = {nm: perf_agg[nm]["KL_se"]     for nm in CF_MODEL_NAMES}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bar_chart(rmse_means, rmse_ses, "Out-of-Sample RMSE",
              f"CF Correction — RMSE{se_note}", ax=axes[0], n_runs=n_runs)
    bar_chart(kl_means, kl_ses, "KL Divergence",
              f"CF Correction — KL{se_note}", ax=axes[1], n_runs=n_runs)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_dir}/fig_cf_accuracy.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig_cf_accuracy")

    # ── Fig B: Demand curves (good-0 budget share vs. p_0) ───────────────────
    pgr0 = agg.get("p_grid_0")
    if pgr0 is not None and agg["curves_mean"]:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for nm in CF_MODEL_NAMES:
            if nm not in agg["curves_mean"]:
                continue
            mu  = agg["curves_mean"][nm]
            sig = agg["curves_se"].get(nm, np.zeros_like(mu))
            sty = STYLE.get(nm, {})
            ax2.plot(pgr0, mu, label=nm, **sty)
            if n_runs > 1:
                ax2.fill_between(pgr0, mu - sig, mu + sig,
                                 color=sty.get("color", "#888"), alpha=BAND)
        ax2.set_xlabel(f"Aspirin price $p_0$", fontsize=13)
        ax2.set_ylabel("Aspirin budget share $w_0$", fontsize=13)
        # ax2.set_title(
        #     f"Demand Curves — CF Decomposition{se_note}",
        #     fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(f"{fig_dir}/fig_cf_demand_curves.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {fig_dir}/fig_cf_demand_curves")

    # ── Fig C: Full G×G elasticity heatmaps (grid layout) ────────────────────
    elast_agg = agg["elast_agg"]
    valid_els  = [row for row in elast_agg
                  if not np.all(np.isnan(row.get("mat_mean", np.full((G, G), np.nan))))]
    if valid_els:
        import matplotlib.colors as mcolors
        n_models = len(valid_els)
        n_cols = min(4, n_models)
        n_rows = int(np.ceil(n_models / n_cols))
        fig3, axes3 = plt.subplots(n_rows, n_cols,
                                   figsize=(4.5 * n_cols, 4.5 * n_rows),
                                   squeeze=False)
        axes3_flat = axes3.flatten()

        # shared colour scale: symmetric around 0, clipped to 95th percentile
        all_vals = np.concatenate([row["mat_mean"].flatten() for row in valid_els])
        vabs = max(min(float(np.nanpercentile(np.abs(all_vals), 95)), 3.0), 0.1)
        norm3 = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

        last_im3 = None
        for idx, row in enumerate(valid_els):
            ax  = axes3_flat[idx]
            mat = row["mat_mean"]   # (G, G)
            im  = ax.imshow(mat, cmap="RdBu_r", norm=norm3, aspect="equal")
            last_im3 = im
            ax.set_xticks(range(G)); ax.set_yticks(range(G))
            ax.set_xticklabels([f"{GOODS[g]}\n($w_{{{g}}}$)" for g in range(G)], fontsize=9)
            ax.set_yticklabels([f"{GOODS[g]}\n($p_{{{g}}}$)" for g in range(G)], fontsize=9)
            for i in range(G):
                for j in range(G):
                    val = mat[i, j]
                    if not np.isnan(val):
                        sign = "+" if val >= 0 else ""
                        txt = f"{sign}{val:.2f}"
                    else:
                        txt = "—"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")
            if idx == n_models - 1 and last_im3 is not None:
                cbar3 = fig3.colorbar(last_im3, ax=ax, fraction=0.046, pad=0.04)
                cbar3.set_label(r"$\varepsilon_{ij} = \partial\log w_j/\partial\log p_i$",
                                fontsize=8)

        for idx in range(n_models, len(axes3_flat)):
            axes3_flat[idx].set_visible(False)

        fig3.tight_layout()
        for ext in ("pdf", "png"):
            fig3.savefig(f"{fig_dir}/fig_cf_elasticity_heatmap.{ext}",
                         dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved: {fig_dir}/fig_cf_elasticity_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
#  LaTeX tables
# ─────────────────────────────────────────────────────────────────────────────

def make_tables(agg: dict, cfg: dict) -> None:
    out_dir = cfg["out_dir"]
    fig_dir = cfg["fig_dir"]
    n_runs  = agg["n_runs"]
    os.makedirs(out_dir, exist_ok=True)

    def _c(m, s=None, d=5):
        if m is None or np.isnan(float(m)):
            return "{---}"
        if s is not None and n_runs > 1 and not np.isnan(float(s)) and float(s) > 0:
            return f"${float(m):.{d}f} \\pm {float(s):.{d}f}$"
        return f"${float(m):.{d}f}$"

    # ── First-stage R² ───────────────────────────────────────────────────────
    rsq_df = agg["rsq_df"]
    rsq_mean = rsq_df[[g for g in GOODS]].mean()
    rsq_std  = rsq_df[[g for g in GOODS]].std()

    rsq_df.round(4).to_csv(f"{out_dir}/table_cf_rsq.csv", index=False)

    lines = [
        r"% ================================================================",
        r"% Neural Demand Paper — CF Tables (auto-generated)",
        f"% N_RUNS = {n_runs}",
        r"% ================================================================", "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Control-Function First-Stage $R^2$ (Hausman Instruments)}",
        r"  \label{tab:cf_rsq}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    & \textbf{Aspirin} & \textbf{Acetaminophen} & \textbf{Ibuprofen} \\",
        r"    \midrule",
        "    $R^2$ & " + " & ".join(
            _c(float(rsq_mean[g]), float(rsq_std[g]) if n_runs > 1 else None, d=3)
            for g in GOODS
        ) + r" \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}\small",
        r"    \item Hausman mean-price instruments. First-stage OLS of log price on instruments.",
        r"  \end{tablenotes}",
        r"\end{table}", "",
    ]

    # ── Performance ──────────────────────────────────────────────────────────
    make_performance_table(
        agg["perf_agg"],
        out_dir=out_dir,
        label="table_cf_accuracy",
        caption=(r"CF Correction --- Predictive Accuracy. "
                 r"Comparison of uncorrected and CF-corrected Neural Demand models. "
                 r"Out-of-sample RMSE, MAE, and KL on held-out test weeks."),
        n_runs=n_runs,
    )

    # ── Elasticities (own-price diagonal for compact table) ───────────────────
    elast_rows_tex = []
    for row in agg["elast_agg"]:
        elast_rows_tex.append({
            "Model":    row["Model"],
            "eps_0":    row.get("eps_0_0"),    "eps_0_se": row.get("eps_0_0_se"),
            "eps_1":    row.get("eps_1_1"),    "eps_1_se": row.get("eps_1_1_se"),
            "eps_2":    row.get("eps_2_2"),    "eps_2_se": row.get("eps_2_2_se"),
        })
    make_elasticity_table(
        elast_rows_tex, out_dir=out_dir,
        label="table_cf_elasticities",
        caption=(r"Own-Price Elasticities — CF Decomposition. "
                 r"Diagonal of the $G\times G$ elasticity matrix at test-mean prices. "
                 r"Full cross-price matrices shown in Figure~\ref{fig:cf_elasticity_heatmap}."),
        n_runs=n_runs,
    )

    # ── Full elasticity CSV ───────────────────────────────────────────────────
    el_csv_rows = []
    for row in agg["elast_agg"]:
        r_out = {"Model": row["Model"]}
        for i in range(G):
            for j in range(G):
                r_out[f"eps_{i}_{j}"]    = row.get(f"eps_{i}_{j}", np.nan)
                r_out[f"eps_{i}_{j}_se"] = row.get(f"eps_{i}_{j}_se", 0.0)
        el_csv_rows.append(r_out)
    pd.DataFrame(el_csv_rows).round(4).to_csv(
        f"{out_dir}/table_cf_full_elasticities.csv", index=False)
    print(f"  Saved: {out_dir}/table_cf_full_elasticities.csv")

    # ── Welfare ──────────────────────────────────────────────────────────────
    wf = agg["welf_agg"]
    pd.DataFrame(wf).round(6).to_csv(f"{out_dir}/table_cf_welfare.csv", index=False)
    lines += [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Compensating Variation --- CF Decomposition (10\% Aspirin shock)}",
        r"  \label{tab:cf_welfare}",
        r"  \begin{tabular}{lc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{CV (£ per household)} \\",
        r"    \midrule",
    ]
    for row in wf:
        lines.append(f"    {row['Model']} & "
                     f"{_c(row['CV_mean'], row.get('CV_se'), d=4)} \\\\")
    lines += [
        r"    \bottomrule", r"  \end{tabular}",
        r"  \begin{tablenotes}\small",
        r"    \item CV for 10\% increase in Aspirin price at test-mean prices.",
        r"    Values at structural evaluation ($\hat{v}=0$).",
        r"  \end{tablenotes}",
        r"\end{table}", "",
    ]

    # Figure environments
    for fn, cap, lbl in [
        ("fig_cf_accuracy",
         r"CF Correction: RMSE and KL Divergence. "
         r"Columns show the three-way decomposition: uncorrected, CF-corrected static, "
         r"and CF-corrected with habit.",
         "fig:cf_accuracy"),
        ("fig_cf_demand_curves",
         r"Demand Curves — CF Decomposition. Aspirin budget share as its price varies. "
         r"CF-corrected curves recover the structural preference component.",
         "fig:cf_demand_curves"),
        ("fig_cf_elasticity_heatmap",
         r"Own-Price Elasticity Heatmaps for the three CF-decomposition models.",
         "fig:cf_elasticity_heatmap"),
    ]:
        lines += [
            r"\begin{figure}[htbp]", r"  \centering",
            f"  \\includegraphics[width=0.85\\linewidth]{{{fig_dir}/{fn}.pdf}}",
            f"  \\caption{{{cap}}}", f"  \\label{{{lbl}}}",
            r"\end{figure}", "",
        ]

    tex_path = f"{out_dir}/cf_decomposition_tables.tex"
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved LaTeX to {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run(splits: dict, cfg: dict) -> tuple:
    """Orchestrate multiple seeds, aggregate, produce outputs.

    Parameters
    ----------
    splits : from experiments.dominicks.data.load()
    cfg    : config dict (must include 'N_RUNS', 'out_dir', 'fig_dir')
    """
    N_RUNS = cfg.get("N_RUNS", 1)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand — Dominick's Exp 06: CF Decomposition")
    print("=" * 68)

    pre = cfg.get("_precomputed_dom_exp06")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed exp06_cf_decomposition runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15
            t0   = time.time()
            print(f"  Run {ri+1}/{N_RUNS}  seed={seed}")
            r = run_once(seed, splits, cfg)
            all_results.append(r)
            print(f"    Done in {time.time()-t0:.0f}s  "
                  f"NDS_RMSE={r['perf'].get('Neural Demand (static)', {}).get('RMSE', np.nan):.5f}  "
                  f"CF_RMSE={r['perf'].get('Neural Demand (CF)', {}).get('RMSE', np.nan):.5f}")

    agg = aggregate(all_results)
    make_figures(agg, cfg)
    make_tables(agg, cfg)

    print("\n── CF Decomposition Summary ─────────────────────────────────────────")
    for nm, d in agg["perf_agg"].items():
        print(f"  {nm:45s}  RMSE={d['RMSE_mean']:.5f}±{d['RMSE_se']:.5f}  "
              f"KL={d['KL_mean']:.5f}±{d['KL_se']:.5f}")

    return all_results, agg
