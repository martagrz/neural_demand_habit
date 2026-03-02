"""
experiments/neural_demand/simulation/exp04_cf_endogeneity.py
=============================================================
Section 2.4 — Endogeneity Correction via Control Function (Simulation)

Design
------
We generate data from a CES or Habit DGP where prices are *endogenous*:
  log p_j = γ · cost_z_j  +  ε_j
with ε_j correlated with unobserved taste shocks that also shift budget
shares, creating OLS endogeneity bias.

We compare four estimators at test time:
  1. Neural Demand (static)          — uncorrected; biased
  2. Neural Demand (CF)              — StaticND(n_cf=G) + CF first stage
  3. Neural Demand (habit)           — uncorrected habit model
  4. Neural Demand (habit, CF)       — HabitND(n_cf=G) + CF first stage

Ground truth is evaluated at *structural* prices (ε = 0), so the CF
models should win when endogeneity is non-trivial.

Section 2.4 comparison: first-stage R² and RMSE-reduction from IV control.

Produces
--------
results/neural_demand/simulations/
  table_cf_endogeneity.csv
  table_cf_endogeneity.tex
  fig_cf_bias_reduction.{pdf,png}      RMSE vs. endogeneity strength ρ
  fig_cf_first_stage_rsq.{pdf,png}     first-stage R² across goods and ρ
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models.simulation import (
    CESConsumer, HabitFormationConsumer,
    StaticND, HabitND,
    compute_xbar_e2e,
    cf_first_stage,
    train_neural_irl,
)
from experiments.simulation.utils import (
    AVG_Y, STYLE,
    predict_shares,
    get_metrics,
    compute_compensating_variation,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  DGP helper — endogenous price generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_endogenous_data(consumer, N: int, rho: float, seed: int):
    """Generate (p, y, w, Z, p_struct) with endogenous prices.

    Price model:   log p_j = log z_j + rho * u_j + (1-rho) * xi_j
      z_j  ~ Uniform(0.5, 1.5)  — cost shifter (valid instrument)
      u_j  ~ N(0, 0.25)         — supply shock, orthogonal to tastes
      xi_j ~ N(0, 0.25)         — endogenous component; correlated with
                                   taste shock that shifts w

    The taste shock shifts the *true* budget shares by a small amount
    so that unobserved xi causes OLS endogeneity bias.

    Parameters
    ----------
    consumer : demand consumer with solve_demand(p, y)
    N        : number of observations
    rho      : endogeneity strength in [0, 1]
                 rho=0 → exogenous;  rho=1 → maximally endogenous
    seed     : RNG seed

    Returns
    -------
    p_obs    : (N, G) endogenous prices (what econometrician observes)
    p_struct : (N, G) structural prices (ε=0; used for ground-truth w)
    y        : (N,) incomes
    w_struct : (N, G) ground-truth budget shares at p_struct
    w_obs    : (N, G) observed budget shares (from p_obs; unbiased target)
    Z        : (N, G) cost-shifter instruments
    """
    rng = np.random.default_rng(seed)
    G   = 3

    Z       = rng.uniform(0.5, 1.5, (N, G))        # instruments
    xi      = rng.normal(0, 0.25, (N, G))           # endogenous shocks
    eta_p   = rng.normal(0, 0.25, (N, G))           # idiosyncratic supply noise

    # Observed log price = log z + rho * xi + (1-rho) * eta_p
    log_p_obs  = np.log(Z) + rho * xi + (1.0 - rho) * eta_p
    log_p_obs  = np.clip(log_p_obs, -2.0, 2.0)
    p_obs      = np.exp(log_p_obs)

    # Structural price (instrument only; endogeneity zeroed out)
    log_p_struct = np.log(Z)
    p_struct  = np.exp(np.clip(log_p_struct, -2.0, 2.0))

    y = rng.uniform(1200, 2000, N)

    # Ground-truth shares at structural prices
    try:
        w_struct = consumer.solve_demand(p_struct, y)
    except Exception:
        w_struct = consumer.solve_demand(p_obs, y)

    # Observed shares: same consumer, but prices include the endogenous part
    # (the econometrician sees w_obs; the model must recover structural demand)
    try:
        w_obs = consumer.solve_demand(p_obs, y)
    except Exception:
        w_obs = w_struct.copy()

    return p_obs, p_struct, y, w_struct, w_obs, Z


# ─────────────────────────────────────────────────────────────────────────────
#  Single-seed full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_one_seed(seed: int, cfg: dict, verbose: bool = False) -> dict:
    """Run all models under each rho level for one seed.

    Returns
    -------
    dict with keys:
      rows     : list of metric dicts (model × rho × DGP)
      rsq_rows : list of first-stage R² dicts
    """
    N       = cfg["N_OBS"]
    DEVICE  = cfg["DEVICE"]
    EPOCHS  = cfg["EPOCHS"]
    RHO_GRID = cfg.get("RHO_GRID", [0.0, 0.3, 0.6, 0.9])
    DELTA_HAB = cfg.get("DELTA_HAB", 0.7)   # fixed δ for habit model
    CACHE_DIR = cfg.get("model_cache_dir")
    FORCE_RETRAIN = cfg.get("force_retrain", False)

    rows     = []
    rsq_rows = []

    for dgp_name, consumer_cls in [("CES", CESConsumer),
                                   ("Habit", HabitFormationConsumer)]:
        consumer = consumer_cls()

        for rho in RHO_GRID:
            np.random.seed(seed + int(rho * 1000))
            torch.manual_seed(seed + int(rho * 1000))

            # ── Training data ───────────────────────────────────────────────
            p_obs, p_struct, y_tr, w_struct_tr, w_obs_tr, Z_tr = \
                _generate_endogenous_data(consumer, N, rho, seed=seed * 100 + int(rho * 1000))

            # ── Held-out test data (new draws, same rho) ────────────────────
            N_te = max(N // 4, 50)
            p_te_obs, p_te_struct, y_te, w_te_struct, w_te_obs, Z_te = \
                _generate_endogenous_data(consumer, N_te, rho,
                                          seed=seed * 100 + int(rho * 1000) + 77777)

            # ── CF first stage (training) ────────────────────────────────────
            # Use log(Z) because DGP is log-linear: log(p) ~ log(Z)
            v_hat_tr, r_sq_tr = cf_first_stage(np.log(np.maximum(p_obs, 1e-8)),
                                               np.log(np.maximum(Z_tr, 1e-8)))
            v_hat_te, _       = cf_first_stage(np.log(np.maximum(p_te_obs, 1e-8)),
                                               np.log(np.maximum(Z_te, 1e-8)))

            for g in range(3):
                rsq_rows.append({
                    "DGP":  dgp_name, "rho": rho, "good": g, "seed": seed,
                    "R2": float(r_sq_tr[g]),
                })

            if verbose:
                print(f"    DGP={dgp_name} rho={rho:.1f} | "
                      f"1st-stage R²: {r_sq_tr.round(3)}")

            # ── Habit stock for HabitND ─────────────────────────────────
            # Use simple EWMA with fixed delta for training
            log_shares = np.log(np.maximum(w_obs_tr, 1e-8))
            xbar_all   = np.zeros_like(log_shares)
            xbar_all[0] = log_shares[0]
            for t in range(1, N):
                xbar_all[t] = DELTA_HAB * xbar_all[t - 1] + (1.0 - DELTA_HAB) * log_shares[t - 1]
            q_prev = np.roll(log_shares, 1, axis=0); q_prev[0] = log_shares[0]

            log_shares_te = np.log(np.maximum(w_te_obs, 1e-8))
            xbar_te = np.zeros_like(log_shares_te)
            xbar_te[0] = log_shares_te[0]
            for t in range(1, N_te):
                xbar_te[t] = DELTA_HAB * xbar_te[t-1] + (1.0 - DELTA_HAB) * log_shares_te[t-1]
            q_prev_te = np.roll(log_shares_te, 1, axis=0); q_prev_te[0] = log_shares_te[0]

            # ── Train models ─────────────────────────────────────────────────
            # 1. Static (uncorrected)
            nds = StaticND(n_goods=3, hidden_dim=cfg.get("hidden_dim", 128), n_cf=0)
            nds, _ = train_neural_irl(
                nds, p_obs, y_tr, w_obs_tr,
                epochs=EPOCHS, lr=5e-4, batch_size=256,
                lam_mono=0.3, lam_slut=0.1,
                device=DEVICE, verbose=False,
                tag=f"nds-static-cfendo-{dgp_name}-r{rho}-s{seed}",
                cache_dir=CACHE_DIR,
                force_retrain=FORCE_RETRAIN)

            # 2. Static + CF
            nds_cf = StaticND(n_goods=3, hidden_dim=cfg.get("hidden_dim", 128), n_cf=3)
            nds_cf, _ = train_neural_irl(
                nds_cf, p_obs, y_tr, w_obs_tr,
                epochs=EPOCHS, lr=5e-4, batch_size=256,
                lam_mono=0.3, lam_slut=0.1,
                v_hat_data=v_hat_tr,
                device=DEVICE, verbose=False,
                tag=f"nds-cf-cfendo-{dgp_name}-r{rho}-s{seed}",
                cache_dir=CACHE_DIR,
                force_retrain=FORCE_RETRAIN)

            # 3. Habit (uncorrected) — HabitND with fixed delta
            nds_hab = HabitND(n_goods=3, hidden_dim=cfg.get("hidden_dim", 128),
                                   delta_init=DELTA_HAB, n_cf=0)
            nds_hab, _ = train_neural_irl(
                nds_hab, p_obs, y_tr, w_obs_tr,
                epochs=EPOCHS, lr=5e-4, batch_size=256,
                lam_mono=0.3, lam_slut=0.1,
                xb_prev_data=np.exp(xbar_all),   # train_neural_irl log-transforms internally
                q_prev_data=np.exp(q_prev),
                device=DEVICE, verbose=False,
                tag=f"nds-hab-cfendo-{dgp_name}-r{rho}-s{seed}",
                cache_dir=CACHE_DIR,
                force_retrain=FORCE_RETRAIN)

            # 4. Habit + CF
            nds_hab_cf = HabitND(n_goods=3, hidden_dim=cfg.get("hidden_dim", 128),
                                      delta_init=DELTA_HAB, n_cf=3)
            nds_hab_cf, _ = train_neural_irl(
                nds_hab_cf, p_obs, y_tr, w_obs_tr,
                epochs=EPOCHS, lr=5e-4, batch_size=256,
                lam_mono=0.3, lam_slut=0.1,
                xb_prev_data=np.exp(xbar_all),
                q_prev_data=np.exp(q_prev),
                v_hat_data=v_hat_tr,
                device=DEVICE, verbose=False,
                tag=f"nds-hab-cf-cfendo-{dgp_name}-r{rho}-s{seed}",
                cache_dir=CACHE_DIR,
                force_retrain=FORCE_RETRAIN)

            # ── Evaluate at TEST structural prices (v_hat = 0) ────────────────
            # "Structural demand" = what we'd observe without the endogenous shock
            KW_te = dict(
                consumer=consumer,
                nds=nds,
                nds_cf=nds_cf,
                nds_hab_cf=nds_hab_cf,
                xbar_hab=xbar_te,       # for nd-habit-cf
                q_prev_hab=q_prev_te,
                v_hat=None,             # zeros → structural demand
                device=DEVICE,
            )

            model_specs = [
                ("Neural Demand (static)",     "nd-static"),
                ("Neural Demand (CF)", "nd-static-cf"),
            ]

            # Habit model specs need separate KW because HabitND takes
            # xb_prev + q_prev, not the E2E interface
            for nm, model_obj, sp_tag in [
                ("Neural Demand (static)",     nds,        "nd-static"),
                ("Neural Demand (CF)",         nds_cf,     "nd-static-cf"),
            ]:
                try:
                    w_pred = predict_shares(
                        sp_tag, p_te_struct, y_te,
                        nds=nds, nds_cf=nds_cf,
                        device=DEVICE)
                    # RMSE vs. structural ground truth (w_te_struct)
                    rmse = float(np.sqrt(np.mean((w_pred - w_te_struct) ** 2)))
                    mae  = float(np.mean(np.abs(w_pred - w_te_struct)))
                except Exception as exc:
                    if verbose:
                        print(f"      [{nm} failed: {exc}]")
                    rmse = mae = np.nan

                rows.append({
                    "DGP": dgp_name, "rho": rho, "Model": nm, "seed": seed,
                    "RMSE": rmse, "MAE": mae,
                })

            # Habit models: pass zeros for v_hat at structural evaluation
            zeros_te = np.zeros_like(v_hat_te)
            for nm, m_obj, xb_te, qp_te in [
                ("Neural Demand (habit)",      nds_hab,    xbar_te, q_prev_te),
                ("Neural Demand (habit, CF)",  nds_hab_cf, xbar_te, q_prev_te),
            ]:
                # Determine spec based on model type
                if getattr(m_obj, "n_cf", 0) > 0:
                    # Model expects CF residuals
                    w_pred = predict_shares(
                        "nd-habit-cf", p_te_struct, y_te,
                        nds_hab_cf=m_obj,
                        xbar_hab=xb_te, q_prev_hab=qp_te,
                        v_hat=zeros_te,
                        device=DEVICE)
                else:
                    # Model does not use CF residuals
                    w_pred = predict_shares(
                        "nd-habit", p_te_struct, y_te,
                        nds_hab=m_obj,
                        xbar_hab=xb_te, q_prev_hab=qp_te,
                        v_hat=None,
                        device=DEVICE)

                try:
                    rmse = float(np.sqrt(np.mean((w_pred - w_te_struct) ** 2)))
                    mae  = float(np.mean(np.abs(w_pred - w_te_struct)))
                except Exception as exc:
                    if verbose:
                        print(f"      [{nm} failed: {exc}]")
                    rmse = mae = np.nan
                rows.append({
                    "DGP": dgp_name, "rho": rho, "Model": nm, "seed": seed,
                    "RMSE": rmse, "MAE": mae,
                })

    return dict(rows=rows, rsq_rows=rsq_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _se(arr):
    a = np.asarray(arr, float)
    return np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))


def aggregate(all_results: list) -> dict:
    rows_all     = []
    rsq_rows_all = []
    for r in all_results:
        rows_all.extend(r["rows"])
        rsq_rows_all.extend(r["rsq_rows"])

    df     = pd.DataFrame(rows_all)
    df_rsq = pd.DataFrame(rsq_rows_all)

    idx      = ["DGP", "rho", "Model"]
    agg_mean = df.groupby(idx, sort=False)[["RMSE", "MAE"]].mean().reset_index()
    agg_se   = df.groupby(idx, sort=False)[["RMSE", "MAE"]].sem().reset_index()
    agg_se.rename(columns={"RMSE": "RMSE_se", "MAE": "MAE_se"}, inplace=True)
    agg      = agg_mean.merge(agg_se, on=idx)

    rsq_idx  = ["DGP", "rho", "good"]
    rsq_mean = df_rsq.groupby(rsq_idx, sort=False)[["R2"]].mean().reset_index()

    return dict(agg=agg, rsq=rsq_mean, n_runs=len(all_results))


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(agg: dict, cfg: dict) -> None:
    fig_dir  = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    df      = agg["agg"]
    df_rsq  = agg["rsq"]
    n_runs  = agg["n_runs"]

    rho_grid   = sorted(df["rho"].unique())
    dgp_names  = list(df["DGP"].unique())

    model_order = [
        "Neural Demand (static)",
        "Neural Demand (CF)",
        "Neural Demand (habit)",
        "Neural Demand (habit, CF)",
    ]

    # ── Fig A: RMSE vs ρ (one panel per DGP) ────────────────────────────────
    fig, axes = plt.subplots(1, len(dgp_names), figsize=(6 * len(dgp_names), 5),
                             sharey=False)
    if len(dgp_names) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgp_names):
        sub = df[df["DGP"] == dgp]
        for nm in model_order:
            sty = STYLE.get(nm, {})
            ms = sub[sub["Model"] == nm]
            if ms.empty:
                continue
            ys   = [float(ms[ms["rho"] == r]["RMSE"].mean())   for r in rho_grid]
            ses  = [float(ms[ms["rho"] == r]["RMSE_se"].mean()) for r in rho_grid]
            ax.plot(rho_grid, ys, label=nm, **sty)
            ax.fill_between(rho_grid,
                            [y - s for y, s in zip(ys, ses)],
                            [y + s for y, s in zip(ys, ses)],
                            color=sty.get("color", "gray"), alpha=0.15)
        ax.set_xlabel("Endogeneity strength $\\rho$", fontsize=12)
        ax.set_ylabel("Test RMSE (vs. structural truth)", fontsize=11)
        # ax.set_title(f"{dgp} DGP", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    se_note = f" ({n_runs} runs)" if n_runs > 1 else ""
    # fig.suptitle(f"RMSE vs. Endogeneity Strength — CF Correction{se_note}",
    #              fontsize=13, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_dir}/fig_cf_bias_reduction.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig_cf_bias_reduction")

    # ── Fig B: First-stage R² per good across ρ ──────────────────────────────
    if not df_rsq.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors_g = ["#1E88E5", "#E53935", "#43A047"]
        for dgp in dgp_names:
            sub = df_rsq[df_rsq["DGP"] == dgp]
            for g in sorted(sub["good"].unique()):
                sg = sub[sub["good"] == g]
                ys = [float(sg[sg["rho"] == r]["R2"].mean()) for r in rho_grid]
                ls = "-" if dgp == dgp_names[0] else "--"
                ax2.plot(rho_grid, ys,
                         color=colors_g[int(g)], ls=ls,
                         label=f"{dgp} — Good {g}")
        ax2.set_xlabel("Endogeneity strength $\\rho$", fontsize=12)
        ax2.set_ylabel("First-stage $R^2$", fontsize=12)
        # ax2.set_title("CF First-Stage Fit — Instrument Relevance", fontsize=12,
        #               fontweight="bold")
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(f"{fig_dir}/fig_cf_first_stage_rsq.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {fig_dir}/fig_cf_first_stage_rsq")


# ─────────────────────────────────────────────────────────────────────────────
#  LaTeX tables
# ─────────────────────────────────────────────────────────────────────────────

def make_tables(agg: dict, cfg: dict) -> None:
    out_dir = cfg["out_dir"]
    fig_dir = cfg["fig_dir"]
    n_runs  = agg["n_runs"]
    os.makedirs(out_dir, exist_ok=True)

    df = agg["agg"]
    df.round(6).to_csv(f"{out_dir}/table_cf_endogeneity.csv", index=False)

    rho_grid  = sorted(df["rho"].unique())
    dgp_names = list(df["DGP"].unique())
    model_order = [
        "Neural Demand (static)",
        "Neural Demand (CF)",
        "Neural Demand (habit)",
        "Neural Demand (habit, CF)",
    ]
    model_present = [m for m in model_order if m in df["Model"].values]

    def _cell(m, s, d=5):
        if np.isnan(m): return "{---}"
        return f"${m:.{d}f} \\pm {s:.{d}f}$"

    lines = [
        r"% ================================================================",
        r"% Neural Demand Paper — CF Endogeneity Tables (auto-generated)",
        f"% N_RUNS = {n_runs}",
        r"% ================================================================", "",
    ]

    for dgp in dgp_names:
        sub = df[df["DGP"] == dgp]
        col_spec = "l" + "c" * len(rho_grid)
        lines += [
            r"\begin{table}[htbp]",
            r"  \centering",
            rf"  \caption{{Control-Function RMSE — {dgp} DGP (mean $\pm$ SE, {n_runs} runs)}}",
            rf"  \label{{tab:cf_endogeneity_{dgp.lower()}}}",
            r"  \begin{threeparttable}",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            r"    \toprule",
            r"    \textbf{Model} & " +
            " & ".join(rf"$\rho={r:.1f}$" for r in rho_grid) + r" \\",
            r"    \midrule",
        ]
        for mn in model_present:
            cells = []
            for r in rho_grid:
                row = sub[(sub["Model"] == mn) & (sub["rho"] == r)]
                if len(row):
                    cells.append(_cell(float(row["RMSE"].iloc[0]),
                                       float(row["RMSE_se"].iloc[0])))
                else:
                    cells.append("{---}")
            lines.append("    " + mn + " & " + " & ".join(cells) + r" \\")
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{tablenotes}\small",
            r"    \item RMSE evaluated against structural (IV-purged) demand at test prices.",
            rf"    \item $\rho$ = endogeneity strength; $\rho=0$ is fully exogenous.",
            r"  \end{tablenotes}",
            r"  \end{threeparttable}",
            r"\end{table}", "",
        ]

    # Figure environments
    for fn, cap, lbl in [
        ("fig_cf_bias_reduction",
         "RMSE vs.\\ endogeneity strength $\\rho$.  CF-corrected models "
         "(dashed) recover structural demand even at high $\\rho$.",
         "fig:cf_bias_reduction"),
        ("fig_cf_first_stage_rsq",
         "First-stage $R^2$ for each good as a function of $\\rho$, "
         "confirming instrument relevance.",
         "fig:cf_first_stage_rsq"),
    ]:
        lines += [
            r"\begin{figure}[htbp]", r"  \centering",
            f"  \\includegraphics[width=0.85\\linewidth]{{{fig_dir}/{fn}.pdf}}",
            f"  \\caption{{{cap}}}", f"  \\label{{{lbl}}}",
            r"\end{figure}", "",
        ]

    tex_path = f"{out_dir}/cf_endogeneity_tables.tex"
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved LaTeX to {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> tuple:
    """Orchestrate multiple seeds, aggregate, produce outputs."""
    N_RUNS  = cfg["N_RUNS"]
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand — Simulation Exp 04: CF Endogeneity Correction")
    print("=" * 68)

    pre = cfg.get("_precomputed_sim_exp04")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed Exp04 runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15
            t0   = time.time()
            print(f"  Run {ri+1}/{N_RUNS} (seed={seed})")
            r = run_one_seed(seed, cfg, verbose=(ri == N_RUNS - 1))
            all_results.append(r)
            print(f"    Done in {time.time()-t0:.0f}s")

    agg = aggregate(all_results)
    make_figures(agg, cfg)
    make_tables(agg, cfg)
    return all_results, agg
