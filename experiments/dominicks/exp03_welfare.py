"""
experiments/neural_demand/dominicks/exp03_welfare.py
=====================================================
Section 5.3 — Compensating Variation on Dominick's Analgesics.

Evaluates welfare estimates (compensating variation) for a 10% price
increase in aspirin (good 0) at the test-set mean price vector.

Produces
--------
results/neural_demand/dominicks/
  table_dom_welfare.csv / .tex
  fig_dom_welfare.{pdf,png}
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
    feat_shared, feat_good_specific, feat_orth,
    run_lirl,
)
from experiments.dominicks.data import G, GOODS
from experiments.dominicks.data import G as _G
from experiments.dominicks.utils import (
    compensating_variation,
    fit_nd_delta_grid_dom,
    ALL_MODEL_NAMES,
    STYLE,
    BAND,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_once(seed: int, splits: dict, cfg: dict) -> dict:
    """Compute compensating variation for one seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    p_tr  = splits["p_tr"];  p_te  = splits["p_te"]
    w_tr  = splits["w_tr"];  w_te  = splits["w_te"]
    mw_tr = splits["mw_tr"]
    y_tr  = splits["y_tr"];  y_te  = splits["y_te"]
    xb_tr = splits["xb_tr"]; xb_te = splits["xb_te"]
    qp_tr = splits["qp_tr"]; qp_te = splits["qp_te"]
    ls_tr = splits["ls_tr"]; ls_te = splits["ls_te"]
    s_tr  = splits["s_tr"];  s_te  = splits["s_te"]
    Z_tr  = splits["Z_tr"]
    dev   = cfg["device"]

    p_mn  = splits["p_mn"]
    y_mn  = float(splits["y_mn"])
    xb_mn = splits["xb_mn"]
    qp_mn = splits["qp_mn"]
    sg    = cfg["shock_good"]
    ss    = cfg["shock_pct"]
    p0w   = splits["p0w"]
    p1w   = splits["p1w"]

    # ── Train ──────────────────────────────────────────────────────────────────
    aids_m   = LAAIDS().fit(p_tr, w_tr, y_tr)
    blp_m    = BLPLogitIV().fit(p_tr, mw_tr, Z_tr)
    quaids_m = QUAIDS().fit(p_tr, w_tr, y_tr)
    series_m = SeriesDemand().fit(p_tr, w_tr, y_tr)
    th_sh   = run_lirl(feat_shared,        p_tr, y_tr, w_tr, cfg)
    th_gs   = run_lirl(feat_good_specific, p_tr, y_tr, w_tr, cfg)
    th_orth = run_lirl(feat_orth,          p_tr, y_tr, w_tr, cfg)

    nirl_m, _ = _train(StaticND(cfg["nirl_hidden"]),
                       p_tr, y_tr, w_tr, "nirl", cfg,
                       tag=f"StaticND s={seed}")

    sw = fit_nd_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg, with_fe=False,
        store_ids_tr=s_tr, store_ids_val=s_te,
        hidden=cfg["mdp_e2e_hidden"],
        tag=f"ND-Habit s={seed}")
    nd_hab_m  = sw["best_model"]
    delta_hat = sw["delta_hat"]

    d_t     = torch.tensor(float(delta_hat), dtype=torch.float32, device=dev)
    ls_te_t = torch.tensor(ls_te, dtype=torch.float32, device=dev)
    with torch.no_grad():
        xb_e2e_te = compute_xbar_e2e(d_t, ls_te_t, store_ids=s_te).cpu().numpy()
    xb_e2e_mn = xb_e2e_te.mean(0)

    # ── CF first-stage + CF models ─────────────────────────────────────────────
    v_hat_tr, _ = cf_first_stage(
        np.log(np.maximum(p_tr, 1e-8)), Z_tr)
    nirl_cf_m, _ = _train(
        StaticND(cfg["nirl_hidden"], n_cf=_G),
        p_tr, y_tr, w_tr, "nirl", cfg,
        v_hat_tr=v_hat_tr,
        tag=f"StaticND-CF s={seed}")
    mdp_hidden = cfg.get("mdp_hidden", cfg["nirl_hidden"])
    mdp_cf_m, _ = _train(
        HabitND(mdp_hidden, n_cf=_G),
        p_tr, y_tr, w_tr, "mdp", cfg,
        xb_prev_tr=xb_tr, q_prev_tr=qp_tr,
        v_hat_tr=v_hat_tr,
        tag=f"ND-Habit-CF s={seed}")

    KW = dict(
        aids=aids_m, blp=blp_m, quaids=quaids_m, series=series_m,
        ff=feat_shared, theta=th_sh,
        nirl=nirl_m,
        nirl_cf=nirl_cf_m,
        mdp_cf=mdp_cf_m,
        mdp=nd_hab_m,
    )

    def _xb(nm):
        if "CF" in nm and "habit" in nm:
            return xb_mn          # precomputed EWMA for HabitND-CF
        return xb_e2e_mn if "habit" in nm else None

    def _kw(nm):
        if nm == "Linear Demand (GoodSpec)":
            return {"ff": feat_good_specific, "theta": th_gs}
        if nm == "Linear Demand (Orth)":
            return {"ff": feat_orth, "theta": th_orth}
        return {}

    # ── Compute CV ─────────────────────────────────────────────────────────────
    # Exclude FE and WindowND models from welfare to keep scope manageable.
    # Exp03 does not train WindowND, so evaluating it here is undefined.
    cv_models = [
        nm for nm in ALL_MODEL_NAMES
        if ("FE" not in nm) and (nm != "Neural Demand (window)")
    ]
    cv = {}
    
    # Scale factor: Model units are "hundreds of dollars" (income/100).
    # We multiply by 100 to report CV in real dollars ($).
    SCALE_TO_DOLLARS = 100.0
    
    for nm in cv_models:
        try:
            v = compensating_variation(nm, p0w, p1w, y_mn, cfg,
                                       xb_prev0=_xb(nm), q_prev0=qp_mn,
                                       **{**KW, **_kw(nm)})
            cv[nm] = float(v) * SCALE_TO_DOLLARS
        except Exception as exc:
            print(f"    Warning: CV {nm} failed — {exc}")
            cv[nm] = np.nan

    # Multiple shock magnitudes (for robustness plot)
    cv_by_pct = {}
    for pct in [0.05, 0.10, 0.15, 0.20, 0.25]:
        p1_tmp = p_mn.copy()
        p1_tmp[sg] *= (1.0 + pct)
        cv_by_pct[pct] = {}
        for nm in ["Neural Demand (static)", "Neural Demand (habit)",
                   "Neural Demand (CF)", "Neural Demand (habit, CF)",
                   "LA-AIDS"]:
            try:
                v = compensating_variation(nm, p_mn, p1_tmp, y_mn, cfg,
                                           xb_prev0=_xb(nm), q_prev0=qp_mn,
                                           **KW)
                cv_by_pct[pct][nm] = float(v) * SCALE_TO_DOLLARS
            except Exception:
                cv_by_pct[pct][nm] = np.nan

    return dict(cv=cv, cv_by_pct=cv_by_pct,
                p0w=p0w, p1w=p1w, shock_good=sg, shock_pct=ss,
                delta_hat=delta_hat, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _se(arr):
    a = np.asarray([x for x in arr if not np.isnan(float(x))], float)
    return float(np.std(a, ddof=1) / np.sqrt(max(len(a), 1))) if len(a) > 1 else 0.0


def aggregate(all_results: list) -> dict:
    n = len(all_results)
    model_names = list(all_results[0]["cv"].keys())
    cv_agg = {nm: {"mean": float(np.nanmean([r["cv"][nm] for r in all_results])),
                   "se":   _se([r["cv"][nm] for r in all_results])}
              for nm in model_names}

    pcts  = sorted(all_results[0]["cv_by_pct"].keys())
    rob_m = list(all_results[0]["cv_by_pct"][pcts[0]].keys())
    cv_by_pct_agg = {
        pct: {nm: {"mean": float(np.nanmean([r["cv_by_pct"][pct][nm] for r in all_results])),
                   "se":   _se([r["cv_by_pct"][pct][nm] for r in all_results])}
              for nm in rob_m}
        for pct in pcts
    }
    return dict(cv_agg=cv_agg, cv_by_pct_agg=cv_by_pct_agg, pcts=pcts,
                rob_models=rob_m, n_runs=n,
                shock_good=all_results[0]["shock_good"],
                shock_pct=all_results[0]["shock_pct"])


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(agg: dict, cfg: dict) -> None:
    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    n_runs  = agg["n_runs"]
    se_note = f"  ({n_runs} runs, ±1 SE)" if n_runs > 1 else ""
    sg      = agg["shock_good"]
    ss      = agg["shock_pct"]

    # ── Bar chart ─────────────────────────────────────────────────────────────
    model_names = list(agg["cv_agg"].keys())
    cv_means    = [agg["cv_agg"][nm]["mean"] for nm in model_names]
    cv_ses      = [agg["cv_agg"][nm]["se"]   for nm in model_names]
    bar_colors  = [STYLE.get(nm, {}).get("color", "#888") for nm in model_names]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(model_names))
    ax.bar(x, cv_means, yerr=cv_ses if n_runs > 1 else None, capsize=5,
           color=bar_colors, edgecolor="k", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel(f"Compensating Variation (CV)", fontsize=11)
    # ax.set_title(f"CV for {int(ss*100)}% Price Increase in {GOODS[sg]}{se_note}",
                #  fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_dir}/fig_dom_welfare.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig_dom_welfare.pdf/png")

    # ── Robustness: CV vs shock magnitude ─────────────────────────────────────
    figR, axR = plt.subplots(figsize=(10, 6))
    pcts = agg["pcts"]
    for nm in agg["rob_models"]:
        mu   = [agg["cv_by_pct_agg"][p][nm]["mean"] for p in pcts]
        se   = [agg["cv_by_pct_agg"][p][nm]["se"]   for p in pcts]
        sty  = STYLE.get(nm, {})
        axR.plot([100 * p for p in pcts], mu, label=nm, **sty)
        if n_runs > 1:
            axR.fill_between(
                [100 * p for p in pcts],
                [m - s for m, s in zip(mu, se)],
                [m + s for m, s in zip(mu, se)],
                color=sty.get("color", "#888"), alpha=BAND)
    axR.set_xlabel(f"Price shock on {GOODS[sg]} (%)", fontsize=12)
    axR.set_ylabel("Compensating Variation", fontsize=12)
    # axR.set_title(f"CV Robustness{se_note}", fontsize=12, fontweight="bold")
    axR.legend(fontsize=10)
    axR.grid(True, alpha=0.3)
    figR.tight_layout()
    for ext in ("pdf", "png"):
        figR.savefig(f"{fig_dir}/fig_dom_welfare_robustness.{ext}",
                     dpi=150, bbox_inches="tight")
    plt.close(figR)
    print(f"  Saved: {fig_dir}/fig_dom_welfare_robustness.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
#  Tables
# ─────────────────────────────────────────────────────────────────────────────

def make_tables(agg: dict, cfg: dict) -> None:
    out_dir = cfg["out_dir"]
    n_runs  = agg["n_runs"]
    os.makedirs(out_dir, exist_ok=True)

    rows = [{"Model": nm,
             "CV_mean": d["mean"],
             "CV_se":   d["se"]}
            for nm, d in agg["cv_agg"].items()]
    pd.DataFrame(rows).round(6).to_csv(
        f"{out_dir}/table_dom_welfare.csv", index=False)
    print(f"  Saved: {out_dir}/table_dom_welfare.csv")

    def _c(m, s, d=4):
        if np.isnan(float(m)):
            return "{---}"
        if n_runs > 1 and s > 0:
            return f"${float(m):.{d}f} \\pm {float(s):.{d}f}$"
        return f"${float(m):.{d}f}$"

    sg = agg["shock_good"]
    ss = agg["shock_pct"]
    lines = [
        r"% ============================================================",
        r"% Neural Demand — Compensating Variation (auto-generated)",
        r"% ============================================================", "",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{Compensating Variation for a {int(ss*100)}\% price increase "
        rf"in {GOODS[sg]} --- Dominick's Analgesics{' (mean $\\pm$ SE)' if n_runs>1 else ''}}}",
        r"  \label{tab:dom_welfare}",
        r"  \begin{tabular}{lc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{CV} \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(f"    {row['Model']} & {_c(row['CV_mean'], row['CV_se'])} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    tex_path = f"{out_dir}/table_dom_welfare.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(splits: dict, cfg: dict) -> tuple:
    N_RUNS = cfg.get("N_RUNS", 1)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand — Dominick's Exp 03: Welfare")
    print("=" * 68)

    pre = cfg.get("_precomputed_dom_exp03")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed exp03_welfare runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15
            t0   = time.time()
            print(f"  Run {ri+1}/{N_RUNS}  seed={seed}")
            r = run_once(seed, splits, cfg)
            all_results.append(r)
            nd = r["cv"].get("Neural Demand (static)", np.nan)
            nh = r["cv"].get("Neural Demand (habit)",  np.nan)
            la = r["cv"].get("LA-AIDS", np.nan)
            print(f"    Done in {time.time()-t0:.0f}s  "
                  f"CV: LA-AIDS={la:.4f}  NDS={nd:.4f}  ND+Habit={nh:.4f}")

    agg = aggregate(all_results)
    make_figures(agg, cfg)
    make_tables(agg, cfg)
    return all_results, agg
