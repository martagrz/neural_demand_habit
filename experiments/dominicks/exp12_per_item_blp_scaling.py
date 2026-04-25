"""
Experiment 12: Dominick's per-item (UPC-level) Logit-IV vs Neural Demand benchmarks.

This experiment implements:
1) UPC-level analgesics panel construction (all UPCs available in upcana/wana),
2) train/test evaluation using a week split,
3) Logit-IV with Hausman instruments and a footfall-based market size
   (M_it = footfall_it * spend_scale) that identifies the outside-option share
   as category non-purchase, consistent with Chintagunta (2002),
4) Neural Demand benchmarks (static, habit, FE, CF, etc.),
5) scaling comparison across number of UPCs, reporting fit-time, test RMSE, and
   first-stage R² for the Logit-IV model.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.dominicks.data import _classify, _parse_tablets
from run_neural_demand_dominicks import BASE_CFG, FAST_CFG
from src.models.blp_logit_iv import BLPLogitIV, NestedLogitIV
from src.models.dominicks import cf_first_stage, hausman_iv
from src.models.dominicks_train import train_dominicks as _train
from src.models.mdp_neural_irl import HabitND_FE
from src.models.neural_irl import StaticND


@dataclass
class Exp12Cfg:
    out_dir: str
    fig_dir: str
    goods_grid: list[int]
    n_upc_cap: int
    seed: int
    test_cutoff: int
    min_store_weeks: int
    force_retrain: bool
    verbose: bool
    full_only: bool


def _build_cfg(args) -> tuple[dict, Exp12Cfg]:
    cfg = dict(BASE_CFG)
    cfg["weekly_path"] = args.weekly
    cfg["upc_path"] = args.upc
    cfg["device"] = "cpu"
    cfg["force_retrain"] = bool(args.force_retrain)
    cfg["verbose"] = bool(args.verbose)
    cfg["test_cutoff"] = int(args.test_cutoff)
    cfg["min_store_wks"] = int(args.min_store_weeks)
    cfg["model_cache_dir"] = "results/neural_demand/dominicks/models/full"
    if args.fast:
        cfg.update(FAST_CFG)
        cfg["nirl_epochs"] = min(int(cfg["nirl_epochs"]), 200)
        cfg["mdp_epochs"] = min(int(cfg["mdp_epochs"]), 200)

    out_dir = os.path.join(cfg["out_dir"], "per_item_blp_scaling")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    goods_grid = [int(x) for x in args.goods_grid.split(",") if x.strip()]
    ecfg = Exp12Cfg(
        out_dir=out_dir,
        fig_dir=fig_dir,
        goods_grid=sorted(set(goods_grid)),
        n_upc_cap=int(args.n_upc_cap),
        seed=int(args.seed),
        test_cutoff=int(args.test_cutoff),
        min_store_weeks=int(args.min_store_weeks),
        force_retrain=bool(args.force_retrain),
        verbose=bool(args.verbose),
        full_only=bool(args.full_only),
    )
    return cfg, ecfg


def _build_upc_nest_maps(upc_path: str, selected_upcs: list[int]) -> dict:
    """Return therapeutic and binary nest-id arrays for selected_upcs.

    Therapeutic nests (3 nests):
        0 = Aspirin (ASP), 1 = Acetaminophen (ACET), 2 = Ibuprofen (IBU)
        3 = unclassified catch-all

    Binary nests (2 nests):
        0 = Ibuprofen (IBU, the anti-inflammatory)
        1 = Aspirin + Acetaminophen + other (analgesics)
    """
    K = len(selected_upcs)
    udf = pd.read_csv(upc_path, low_memory=False)
    udf.columns = [c.upper() for c in udf.columns]
    if "DESCRIP" not in udf.columns:
        return {
            "nest_therapeutic": np.zeros(K, dtype=int),
            "nest_binary": np.zeros(K, dtype=int),
        }
    udf = udf[["UPC", "DESCRIP"]].drop_duplicates(subset=["UPC"])
    upc_to_descrip = dict(zip(udf["UPC"].astype(int), udf["DESCRIP"].astype(str)))

    cat_to_idx = {"ASP": 0, "ACET": 1, "IBU": 2, "OTHER": 3}
    nest_th = np.zeros(K, dtype=int)
    nest_bi = np.zeros(K, dtype=int)
    for i, upc in enumerate(selected_upcs):
        descrip = upc_to_descrip.get(int(upc), "")
        cat = _classify(descrip)
        nest_th[i] = cat_to_idx[cat]
        nest_bi[i] = 0 if cat == "IBU" else 1   # IBU=0 vs rest=1
    return {"nest_therapeutic": nest_th, "nest_binary": nest_bi}


def _load_footfall(data_dir: str) -> pd.DataFrame:
    fp = os.path.join(data_dir, "ccount.csv")
    if not os.path.isfile(fp):
        raise FileNotFoundError(
            f"ccount.csv not found in {data_dir!r}; required for footfall market size."
        )
    ccount = pd.read_csv(fp, low_memory=False)
    ccount.columns = [c.lower() for c in ccount.columns]
    need = ["store", "week", "custcoun"]
    miss = [c for c in need if c not in ccount.columns]
    if miss:
        raise ValueError(f"ccount.csv missing columns: {miss}")
    ff = (
        ccount.groupby(["store", "week"], as_index=False)["custcoun"]
        .sum()
        .rename(columns={"store": "STORE", "week": "WEEK", "custcoun": "FOOTFALL"})
    )
    ff["STORE"] = ff["STORE"].astype(int)
    ff["WEEK"] = ff["WEEK"].astype(int)
    return ff


def _build_upc_panel(cfg: dict, footfall_sw: pd.DataFrame) -> pd.DataFrame:
    print("[exp12] Loading weekly and UPC files...", flush=True)
    wdf = pd.read_csv(cfg["weekly_path"])
    udf = pd.read_csv(cfg["upc_path"])
    print(
        f"[exp12] Raw rows: weekly={len(wdf):,}, upc_catalog={len(udf):,}",
        flush=True,
    )

    keep_upc_cols = [c for c in ["UPC", "DESCRIP", "SIZE"] if c in udf.columns]
    udf = udf[keep_upc_cols].drop_duplicates(subset=["UPC"])
    if "SIZE" not in udf.columns:
        udf["SIZE"] = np.nan

    merged = wdf.merge(udf, on="UPC", how="left")
    merged["TABLETS"] = merged["SIZE"].apply(_parse_tablets)
    merged["UNITS"] = merged["MOVE"] * merged["QTY"]
    merged["REVENUE"] = merged["UNITS"] * merged["PRICE"]
    merged["UNIT_PX"] = np.where(
        (merged["TABLETS"] > 0) & np.isfinite(merged["TABLETS"]),
        merged["PRICE"] * float(cfg["std_tablets"]) / merged["TABLETS"],
        merged["PRICE"],
    )
    merged = merged[np.isfinite(merged["UNIT_PX"])].copy()

    def _rev_weighted_px(g):
        pos = g[g["UNITS"] > 0]
        if len(pos) > 0 and np.isfinite(pos["UNIT_PX"]).any():
            return float(np.average(pos["UNIT_PX"], weights=np.maximum(pos["UNITS"], 1e-8)))
        return float(np.nanmean(g["UNIT_PX"]))

    print("[exp12] Aggregating to store-week-UPC panel...", flush=True)
    upc_sw = (
        merged.groupby(["STORE", "WEEK", "UPC"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "PX": _rev_weighted_px(g),
                    "UNITS": float(g["UNITS"].sum()),
                    "REV": float(g["REVENUE"].sum()),
                }
            )
        )
        .reset_index(drop=True)
    )
    upc_sw = upc_sw.merge(footfall_sw, on=["STORE", "WEEK"], how="left")
    upc_sw["FOOTFALL"] = pd.to_numeric(upc_sw["FOOTFALL"], errors="coerce")
    upc_sw = upc_sw[np.isfinite(upc_sw["FOOTFALL"]) & (upc_sw["FOOTFALL"] > 0)].copy()

    # Keep stores with enough observations for stable Hausman IV and FE.
    store_n = upc_sw.groupby("STORE")["WEEK"].nunique()
    good_stores = store_n[store_n >= int(cfg["min_store_wks"])].index
    upc_sw = upc_sw[upc_sw["STORE"].isin(good_stores)].copy()
    print(
        f"[exp12] Panel ready: rows={len(upc_sw):,}, stores={upc_sw['STORE'].nunique()}, "
        f"weeks={upc_sw['WEEK'].nunique()}, upcs={upc_sw['UPC'].nunique()}",
        flush=True,
    )
    return upc_sw


def _train_test_index(weeks: np.ndarray, test_cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    tr = np.where(weeks < test_cutoff)[0]
    te = np.where(weeks >= test_cutoff)[0]
    if len(tr) < 50 or len(te) < 30:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(weeks))
        cut = int(0.75 * len(idx))
        tr, te = idx[:cut], idx[cut:]
    return tr, te


def _build_selected_arrays(
    upc_sw: pd.DataFrame,
    selected_upcs: list[int],
    cfg: dict,
    nest_maps: dict | None = None,
) -> dict:
    # Build complete store-week grid including footfall for market-size estimation.
    sw = upc_sw[["STORE", "WEEK", "FOOTFALL"]].drop_duplicates()
    sw = sw.sort_values(["STORE", "WEEK"]).reset_index(drop=True)
    base = sw.assign(key=1).merge(
        pd.DataFrame({"UPC": selected_upcs, "key": 1}),
        on="key",
        how="inner",
    ).drop(columns=["key"])

    dat = base.merge(
        upc_sw[["STORE", "WEEK", "UPC", "PX", "UNITS", "REV"]],
        on=["STORE", "WEEK", "UPC"],
        how="left",
    )
    dat["UNITS"] = dat["UNITS"].fillna(0.0)
    dat["REV"] = dat["REV"].fillna(0.0)

    dat = dat.sort_values(["STORE", "UPC", "WEEK"])
    dat["PX"] = dat.groupby(["STORE", "UPC"])["PX"].transform(lambda s: s.ffill().bfill())
    global_upc_median = dat.groupby("UPC")["PX"].transform("median")
    dat["PX"] = dat["PX"].fillna(global_upc_median)
    dat["PX"] = dat["PX"].fillna(float(np.nanmedian(dat["PX"].values)))
    dat["PX"] = np.maximum(dat["PX"], 1e-6)

    prices = dat.pivot_table(index=["STORE", "WEEK"], columns="UPC", values="PX", aggfunc="first")
    revs = dat.pivot_table(index=["STORE", "WEEK"], columns="UPC", values="REV", aggfunc="sum").fillna(0.0)
    foot = sw.set_index(["STORE", "WEEK"])["FOOTFALL"].reindex(prices.index)

    prices = prices[selected_upcs]
    revs = revs[selected_upcs]

    store = prices.index.get_level_values("STORE").to_numpy(dtype=int)
    week = prices.index.get_level_values("WEEK").to_numpy(dtype=int)
    p = prices.to_numpy(dtype=float)
    r = revs.to_numpy(dtype=float)
    y = np.maximum(r.sum(axis=1), 1.0) / 100.0
    w = r / np.maximum(r.sum(axis=1, keepdims=True), 1e-8)
    w = np.clip(w, 1e-8, 1.0)
    w /= w.sum(axis=1, keepdims=True)

    # Market size from footfall: M_it = footfall_it × spend_scale, where
    # spend_scale is estimated on the training split only (98th percentile of
    # per-visit category spend) to avoid look-ahead bias.
    # The outside option s₀ = 1 − Σ s_g represents category non-purchase,
    # consistent with the BLP/Chintagunta (2002) treatment of scanner data.
    foot = np.maximum(foot.to_numpy(dtype=float), 1.0)
    tr_idx, te_idx = _train_test_index(week, int(cfg["test_cutoff"]))
    spend_per_visit = r.sum(axis=1)[tr_idx] / np.maximum(foot[tr_idx], 1.0)
    spend_scale = float(np.nanquantile(spend_per_visit[np.isfinite(spend_per_visit)], 0.98))
    spend_scale = max(spend_scale, 1.0)

    mkt_size = foot * spend_scale
    s_inside = r / np.maximum(mkt_size[:, None], 1e-8)
    s_inside = np.clip(s_inside, 1e-10, 1.0)
    s_sum = s_inside.sum(axis=1, keepdims=True)
    overflow = s_sum[:, 0] >= 0.98
    if np.any(overflow):
        # Rescale rows that violate the simplex for numerical stability.
        s_inside[overflow] = s_inside[overflow] / (s_sum[overflow] / 0.98)
        s_sum = s_inside.sum(axis=1, keepdims=True)
    s0 = np.maximum(1.0 - s_sum[:, 0], 1e-8)[:, None]
    mw = np.concatenate([s_inside, s0], axis=1)
    mw = mw / np.maximum(mw.sum(axis=1, keepdims=True), 1e-8)

    log_w = np.log(np.maximum(w, 1e-8))
    delta0 = float(cfg.get("habit_decay", 0.7))
    xb = np.zeros_like(log_w)
    qp = np.zeros_like(log_w)
    gm = log_w.mean(axis=0)
    prev = gm.copy()
    prev_q = gm.copy()
    for i in range(len(log_w)):
        if i > 0 and store[i] != store[i - 1]:
            prev = gm.copy()
            prev_q = gm.copy()
        xb[i] = prev
        qp[i] = prev_q
        prev_q = log_w[i]
        prev = delta0 * prev + (1.0 - delta0) * log_w[i]

    store_uniq = np.sort(np.unique(store))
    store_map = {int(s): i for i, s in enumerate(store_uniq)}
    s_idx = np.array([store_map[int(s)] for s in store], dtype=np.int64)

    z = hausman_iv(p, store, week)
    return dict(
        p=p,
        y=y,
        w=w,
        mw=mw,
        xb=xb,
        qp=qp,
        log_w=log_w,
        store=store,
        week=week,
        store_idx=s_idx,
        n_stores=len(store_uniq),
        tr_idx=tr_idx,
        te_idx=te_idx,
        z=z,
        spend_scale=spend_scale,
        nest_therapeutic=(
            nest_maps["nest_therapeutic"] if nest_maps else
            np.zeros(len(selected_upcs), dtype=int)
        ),
        nest_binary=(
            nest_maps["nest_binary"] if nest_maps else
            np.zeros(len(selected_upcs), dtype=int)
        ),
    )


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _pred_static(model: StaticND, p: np.ndarray, y: np.ndarray, cfg: dict, v_hat=None) -> np.ndarray:
    dev = cfg["device"]
    with torch.no_grad():
        lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32, device=dev))
        ly = torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32, device=dev)).unsqueeze(1)
        if v_hat is None:
            wp = model(lp, ly)
        else:
            vh = torch.tensor(v_hat, dtype=torch.float32, device=dev)
            wp = model(lp, ly, v_hat=vh)
    return wp.cpu().numpy()



def _pred_habit_fe(
    model: HabitND_FE,
    p: np.ndarray,
    y: np.ndarray,
    xb: np.ndarray,
    qp: np.ndarray,
    store_idx: np.ndarray,
    cfg: dict,
    v_hat=None,
) -> np.ndarray:
    dev = cfg["device"]
    with torch.no_grad():
        lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32, device=dev))
        ly = torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32, device=dev)).unsqueeze(1)
        xbt = torch.tensor(xb, dtype=torch.float32, device=dev)
        qpt = torch.tensor(qp, dtype=torch.float32, device=dev)
        si = torch.tensor(store_idx, dtype=torch.long, device=dev)
        if v_hat is None:
            wp = model(lp, ly, xbt, qpt, si)
        else:
            vh = torch.tensor(v_hat, dtype=torch.float32, device=dev)
            wp = model(lp, ly, xbt, qpt, si, v_hat=vh)
    return wp.cpu().numpy()


def _fit_models(arr: dict, cfg: dict, seed: int) -> dict:
    """Fit Neural Demand (static) and Neural Demand (habit, FE, CF) benchmarks."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    tr = arr["tr_idx"]
    te = arr["te_idx"]
    p_tr, p_te = arr["p"][tr], arr["p"][te]
    y_tr, y_te = arr["y"][tr], arr["y"][te]
    w_tr, w_te = arr["w"][tr], arr["w"][te]
    mw_tr = arr["mw"][tr]
    z_tr = arr["z"][tr]
    xb_tr, xb_te = arr["xb"][tr], arr["xb"][te]
    qp_tr, qp_te = arr["qp"][tr], arr["qp"][te]
    sidx_tr, sidx_te = arr["store_idx"][tr], arr["store_idx"][te]
    n_goods = p_tr.shape[1]
    model_prefix = f"[exp12][K={n_goods}]"

    log_p_tr = np.log(np.maximum(p_tr, 1e-8))
    v_hat_tr, _ = cf_first_stage(log_p_tr, z_tr)
    vh_te_zeros = np.zeros_like(p_te, dtype=np.float32)

    out: dict = {}

    # --- BLP (IV, Chintagunta 2002 spec) ---
    print(f"{model_prefix} fitting BLP (IV, Chintagunta 2002 spec)...", flush=True)
    t0 = time.perf_counter()
    blp = BLPLogitIV().fit(p_tr, mw_tr, z_tr, y_tr, verbose=False)
    t_blp = time.perf_counter() - t0
    blp_pred_mkt = blp.predict(p_te, y_te)[:, :n_goods]
    blp_cond = blp_pred_mkt / np.maximum(blp_pred_mkt.sum(axis=1, keepdims=True), 1e-8)
    out["BLP (IV, Chintagunta 2002 spec)"] = {
        "rmse": _rmse(w_te, blp_cond), "fit_time_sec": float(t_blp)
    }
    print(
        f"{model_prefix} done BLP (IV, Chintagunta 2002 spec): "
        f"time={t_blp:.1f}s rmse={out['BLP (IV, Chintagunta 2002 spec)']['rmse']:.6f}",
        flush=True,
    )

    # --- BLP (IV, Chintagunta 2002 spec, demeaned) ---
    # Within-store and within-week demeaning via pandas groupby (vectorised).
    print(f"{model_prefix} fitting BLP (IV, Chintagunta 2002 spec, demeaned)...", flush=True)
    t0 = time.perf_counter()
    store_tr, week_tr = arr["store"][tr], arr["week"][tr]

    def _within_demean(arr2d: np.ndarray, group: np.ndarray) -> np.ndarray:
        """Subtract group means from each column."""
        df_tmp = pd.DataFrame(arr2d)
        df_tmp["_g"] = group
        return arr2d - df_tmp.groupby("_g").transform("mean").values

    p_dm = _within_demean(p_tr, store_tr)
    p_dm = _within_demean(p_dm, week_tr)
    z_dm = _within_demean(z_tr, store_tr)
    z_dm = _within_demean(z_dm, week_tr)
    mw_dm = mw_tr.copy()
    blp_dm = BLPLogitIV().fit(p_dm, mw_dm, z_dm, y_tr, verbose=False)
    t_blp_dm = time.perf_counter() - t0
    # Predict on raw test prices (demeaning is only an estimation device)
    blp_dm_pred = blp_dm.predict(p_te, y_te)[:, :n_goods]
    blp_dm_cond = blp_dm_pred / np.maximum(blp_dm_pred.sum(axis=1, keepdims=True), 1e-8)
    out["BLP (IV, Chintagunta 2002 spec, demeaned)"] = {
        "rmse": _rmse(w_te, blp_dm_cond), "fit_time_sec": float(t_blp_dm)
    }
    print(
        f"{model_prefix} done BLP (IV, Chintagunta 2002 spec, demeaned): "
        f"time={t_blp_dm:.1f}s rmse={out['BLP (IV, Chintagunta 2002 spec, demeaned)']['rmse']:.6f}",
        flush=True,
    )

    # --- Nested Logit (2SLS, therapeutic nests: ASP / ACET / IBU) ---
    print(f"{model_prefix} fitting Nested Logit (2SLS, therapeutic nests)...", flush=True)
    t0 = time.perf_counter()
    nest_th = arr["nest_therapeutic"]
    nlogit_th = NestedLogitIV().fit(p_tr, mw_tr, z_tr, nest_th, y_tr, verbose=False)
    t_nl_th = time.perf_counter() - t0
    nl_th_pred = nlogit_th.predict(p_te, y_te)[:, :n_goods]
    nl_th_cond = nl_th_pred / np.maximum(nl_th_pred.sum(axis=1, keepdims=True), 1e-8)
    out["Nested Logit (2SLS, therapeutic nests)"] = {
        "rmse": _rmse(w_te, nl_th_cond), "fit_time_sec": float(t_nl_th)
    }
    print(
        f"{model_prefix} done Nested Logit (2SLS, therapeutic nests): "
        f"time={t_nl_th:.1f}s rmse={out['Nested Logit (2SLS, therapeutic nests)']['rmse']:.6f}",
        flush=True,
    )

    # --- Nested Logit (2SLS, binary nests: IBU vs ASP+ACET+other) ---
    print(f"{model_prefix} fitting Nested Logit (2SLS, binary nests)...", flush=True)
    t0 = time.perf_counter()
    nest_bi = arr["nest_binary"]
    nlogit_bi = NestedLogitIV().fit(p_tr, mw_tr, z_tr, nest_bi, y_tr, verbose=False)
    t_nl_bi = time.perf_counter() - t0
    nl_bi_pred = nlogit_bi.predict(p_te, y_te)[:, :n_goods]
    nl_bi_cond = nl_bi_pred / np.maximum(nl_bi_pred.sum(axis=1, keepdims=True), 1e-8)
    out["Nested Logit (2SLS, binary nests)"] = {
        "rmse": _rmse(w_te, nl_bi_cond), "fit_time_sec": float(t_nl_bi)
    }
    print(
        f"{model_prefix} done Nested Logit (2SLS, binary nests): "
        f"time={t_nl_bi:.1f}s rmse={out['Nested Logit (2SLS, binary nests)']['rmse']:.6f}",
        flush=True,
    )

    # --- Neural Demand (static) ---
    print(f"{model_prefix} fitting Neural Demand (static)...", flush=True)
    t0 = time.perf_counter()
    nirl = StaticND(hidden_dim=cfg["nirl_hidden"], n_goods=n_goods)
    nirl, _ = _train(
        nirl, p_tr, y_tr, w_tr, "nirl", cfg,
        tag=f"Exp12_StaticND_G{n_goods}_seed{seed}",
    )
    t_nirl = time.perf_counter() - t0
    nirl_pred = _pred_static(nirl, p_te, y_te, cfg)
    out["Neural Demand (static)"] = {"rmse": _rmse(w_te, nirl_pred), "fit_time_sec": float(t_nirl)}
    print(
        f"{model_prefix} done Neural Demand (static): "
        f"time={t_nirl:.1f}s rmse={out['Neural Demand (static)']['rmse']:.6f}",
        flush=True,
    )

    # --- Neural Demand (habit, FE, CF) ---
    print(f"{model_prefix} fitting Neural Demand (habit, FE, CF)...", flush=True)
    t0 = time.perf_counter()
    mdp_fe_cf = HabitND_FE(
        hidden_dim=cfg["mdp_hidden"],
        n_goods=n_goods,
        delta_init=float(cfg.get("habit_decay", 0.7)),
        n_stores=int(arr["n_stores"]),
        emb_dim=8,
        n_cf=n_goods,
    )
    mdp_fe_cf, _ = _train(
        mdp_fe_cf, p_tr, y_tr, w_tr, "mdp", cfg,
        xb_prev_tr=xb_tr, q_prev_tr=qp_tr,
        store_idx_tr=sidx_tr, v_hat_tr=v_hat_tr,
        tag=f"Exp12_HabitND_FE_CF_G{n_goods}_seed{seed}",
    )
    t_mdp_fe_cf = time.perf_counter() - t0
    mdp_fe_cf_pred = _pred_habit_fe(
        mdp_fe_cf, p_te, y_te, xb_te, qp_te, sidx_te, cfg, v_hat=vh_te_zeros
    )
    out["Neural Demand (habit, FE, CF)"] = {
        "rmse": _rmse(w_te, mdp_fe_cf_pred),
        "fit_time_sec": float(t_mdp_fe_cf),
    }
    print(
        f"{model_prefix} done Neural Demand (habit, FE, CF): "
        f"time={t_mdp_fe_cf:.1f}s rmse={out['Neural Demand (habit, FE, CF)']['rmse']:.6f}",
        flush=True,
    )
    return out


_COLOR_MAP = {
    "Neural Demand (static)": "#1E88E5",
    "Neural Demand (habit, FE, CF)": "#8E24AA",
    "BLP (IV, Chintagunta 2002 spec)": "#E53935",
    "BLP (IV, Chintagunta 2002 spec, demeaned)": "#FB8C00",
    "Nested Logit (2SLS, therapeutic nests)": "#00897B",
    "Nested Logit (2SLS, binary nests)": "#43A047",
}
_MARKER_MAP = {
    "Neural Demand (static)": "o",
    "Neural Demand (habit, FE, CF)": "s",
    "BLP (IV, Chintagunta 2002 spec)": "^",
    "BLP (IV, Chintagunta 2002 spec, demeaned)": "D",
    "Nested Logit (2SLS, therapeutic nests)": "v",
    "Nested Logit (2SLS, binary nests)": "P",
}
_LS_MAP = {
    "Neural Demand (static)": "-",
    "Neural Demand (habit, FE, CF)": "-",
    "BLP (IV, Chintagunta 2002 spec)": "--",
    "BLP (IV, Chintagunta 2002 spec, demeaned)": "--",
    "Nested Logit (2SLS, therapeutic nests)": ":",
    "Nested Logit (2SLS, binary nests)": ":",
}

# IV/Logit models whose compute time grows as O(K³) or faster.
_IV_MODELS = {
    "BLP (IV, Chintagunta 2002 spec)",
    "BLP (IV, Chintagunta 2002 spec, demeaned)",
    "Nested Logit (2SLS, therapeutic nests)",
    "Nested Logit (2SLS, binary nests)",
}


def _fit_extrapolation(df: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit a power-law t = a * K^b to the observed (K, time) pairs for `model`.

    Returns (K_fine, t_hat) arrays for plotting, or None if fewer than 2 points.
    Uses log-log OLS: log t = log a + b * log K.
    """
    d = df[df["model"] == model].sort_values("n_goods")
    d = d[(d["n_goods"] > 0) & (d["fit_time_sec"] > 0)]
    if len(d) < 2:
        return None
    log_k = np.log(d["n_goods"].to_numpy(float))
    log_t = np.log(d["fit_time_sec"].to_numpy(float))
    b, log_a = np.polyfit(log_k, log_t, 1)
    a = np.exp(log_a)
    k_max = d["n_goods"].max()
    k_fine = np.logspace(np.log10(d["n_goods"].min()), np.log10(k_max * 3), 120)
    return k_fine, a * k_fine ** b, float(b)


def _breakpoint_k(df: pd.DataFrame, model: str, time_limit_s: float = 3600.0) -> float | None:
    """Extrapolate observed fit-time scaling to find K where time exceeds `time_limit_s`.

    Returns the estimated K, or None if extrapolation is not possible.
    """
    result = _fit_extrapolation(df, model)
    if result is None:
        return None
    k_fine, t_hat, _ = result
    idx = np.searchsorted(t_hat, time_limit_s)
    if idx >= len(k_fine):
        return None
    return float(k_fine[idx])


def _plot_time_vs_goods(df: pd.DataFrame, fig_dir: str, n_train: int = 0) -> None:
    """Separate figure: fit time vs K (log-log), with extrapolated scaling and breakpoint."""
    if df.empty:
        return
    model_order = [m for m in _COLOR_MAP if m in df["model"].values]
    if not model_order:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in model_order:
        d = df[df["model"] == model].sort_values("n_goods")
        if d.empty:
            continue
        ax.plot(
            d["n_goods"], d["fit_time_sec"],
            marker=_MARKER_MAP[model], lw=2.0, ms=7,
            color=_COLOR_MAP[model], ls=_LS_MAP[model], label=model,
        )
        # Dashed extrapolation for IV / Logit models (multiple K points needed)
        if model in _IV_MODELS:
            result = _fit_extrapolation(d if len(d) > 1 else df, model)
            if result is not None:
                k_ext, t_ext, b_exp = result
                # Only plot the extrapolated region (beyond observed range)
                k_obs_max = d["n_goods"].max()
                mask = k_ext > k_obs_max
                if mask.any():
                    ax.plot(
                        k_ext[mask], t_ext[mask],
                        color=_COLOR_MAP[model], ls="--", lw=1.0, alpha=0.45,
                    )

    # Mark the 1-hour threshold
    ax.axhline(3600, color="black", ls=":", lw=1.2, alpha=0.6, label="1-hour wall time")

    # Mark √N — the point where the number of IV instruments per good equals
    # √(training obs), a conventional weak-IV signal.  Beyond this, first-stage
    # R² for Logit-IV typically collapses even with ridge regularisation.
    if n_train > 0:
        k_weak_iv = int(np.sqrt(n_train))
        ax.axvline(k_weak_iv, color="#E53935", ls=(0, (3, 5, 1, 5)), lw=1.4, alpha=0.7,
                   label=rf"$K = \sqrt{{N_{{train}}}} \approx {k_weak_iv}$ (weak-IV signal)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of UPCs ($K$)", fontsize=12)
    ax.set_ylabel("Fit time (seconds)", fontsize=12)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="upper left", ncol=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(fig_dir, f"fig12_fit_time_vs_goods.{ext}"), dpi=150, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig12_fit_time_vs_goods")


def _plot_rmse_vs_goods(df: pd.DataFrame, fig_dir: str, n_train: int = 0) -> None:
    """Separate figure: test RMSE vs K."""
    if df.empty:
        return
    model_order = [m for m in _COLOR_MAP if m in df["model"].values]
    if not model_order:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in model_order:
        d = df[df["model"] == model].sort_values("n_goods")
        if d.empty:
            continue
        ax.plot(
            d["n_goods"], d["rmse_test"],
            marker=_MARKER_MAP[model], lw=2.0, ms=7,
            color=_COLOR_MAP[model], ls=_LS_MAP[model], label=model,
        )

    if n_train > 0:
        k_weak_iv = int(np.sqrt(n_train))
        ax.axvline(k_weak_iv, color="#E53935", ls=(0, (3, 5, 1, 5)), lw=1.4, alpha=0.7,
                   label=rf"$K = \sqrt{{N_{{train}}}} \approx {k_weak_iv}$ (weak-IV signal)")

    ax.set_xlabel("Number of UPCs ($K$)", fontsize=12)
    ax.set_ylabel("Test RMSE (conditional share)", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(fig_dir, f"fig12_rmse_vs_goods.{ext}"), dpi=150, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig12_rmse_vs_goods")


def _plot_scaling(df: pd.DataFrame, fig_dir: str, n_train: int = 0) -> None:
    """Combined 2-panel figure (fit time + RMSE) plus separate per-metric figures."""
    if df.empty:
        return
    model_order = [m for m in _COLOR_MAP if m in df["model"].values]
    if not model_order:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: fit time (log-log)
    ax = axes[0]
    for model in model_order:
        d = df[df["model"] == model].sort_values("n_goods")
        if d.empty:
            continue
        ax.plot(
            d["n_goods"], d["fit_time_sec"],
            marker=_MARKER_MAP[model], lw=2.0, ms=7,
            color=_COLOR_MAP[model], ls=_LS_MAP[model], label=model,
        )
    ax.axhline(3600, color="black", ls=":", lw=1.0, alpha=0.5, label="1-hour wall time")
    if n_train > 0:
        ax.axvline(int(np.sqrt(n_train)), color="#E53935", ls=(0, (3, 5, 1, 5)),
                   lw=1.2, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of UPCs ($K$)")
    ax.set_ylabel("Fit time (seconds)")
    ax.set_title("Fit Time vs $K$ (log–log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)

    # Right: test RMSE
    ax = axes[1]
    for model in model_order:
        d = df[df["model"] == model].sort_values("n_goods")
        if d.empty:
            continue
        ax.plot(
            d["n_goods"], d["rmse_test"],
            marker=_MARKER_MAP[model], lw=2.0, ms=7,
            color=_COLOR_MAP[model], ls=_LS_MAP[model], label=model,
        )
    if n_train > 0:
        ax.axvline(int(np.sqrt(n_train)), color="#E53935", ls=(0, (3, 5, 1, 5)),
                   lw=1.2, alpha=0.6)
    ax.set_xlabel("Number of UPCs ($K$)")
    ax.set_ylabel("Test RMSE (conditional share)")
    ax.set_title("Test RMSE vs $K$")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(fig_dir, f"fig12_scaling.{ext}"), dpi=150, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig12_scaling")

    # Produce separate single-metric figures as well
    _plot_time_vs_goods(df, fig_dir, n_train=n_train)
    _plot_rmse_vs_goods(df, fig_dir, n_train=n_train)


def run(args):
    cfg, ecfg = _build_cfg(args)
    np.random.seed(ecfg.seed)
    torch.manual_seed(ecfg.seed)

    print("[exp12] Starting experiment...", flush=True)
    print(
        f"[exp12] settings: full_only={ecfg.full_only}, "
        f"seed={ecfg.seed}, test_cutoff={ecfg.test_cutoff}",
        flush=True,
    )
    footfall_sw = _load_footfall(args.data_dir)
    print(
        f"[exp12] Footfall loaded: rows={len(footfall_sw):,}, "
        f"stores={footfall_sw['STORE'].nunique()}, weeks={footfall_sw['WEEK'].nunique()}",
        flush=True,
    )
    upc_sw = _build_upc_panel(cfg, footfall_sw)

    # UPC ranking by train-period revenue.
    rev_rank = (
        upc_sw[upc_sw["WEEK"] < ecfg.test_cutoff]
        .groupby("UPC", as_index=False)["REV"]
        .sum()
        .sort_values("REV", ascending=False)
    )
    ranked_upcs = rev_rank["UPC"].tolist()
    if ecfg.n_upc_cap > 0:
        ranked_upcs = ranked_upcs[: ecfg.n_upc_cap]
    if len(ranked_upcs) < 5:
        raise ValueError("Not enough UPCs with train-period sales for Exp 12.")
    print(f"[exp12] Ranked UPCs available: {len(ranked_upcs)}", flush=True)

    full_goods = len(ranked_upcs)
    if ecfg.full_only:
        goods_grid = [full_goods]
    else:
        goods_grid = [g for g in ecfg.goods_grid if g <= len(ranked_upcs) and g >= 5]
        if full_goods not in goods_grid:
            goods_grid.append(full_goods)
        goods_grid = sorted(set(goods_grid))

    # Build UPC→nest maps once (needs UPC catalogue for DESCRIP lookup).
    print("[exp12] Building UPC nest maps...", flush=True)

    scaling_rows = []
    print(f"[exp12] Goods grid to run: {goods_grid}", flush=True)

    for k in goods_grid:
        print(f"[exp12] Preparing arrays for K={k}...", flush=True)
        selected = ranked_upcs[:k]
        nest_maps = _build_upc_nest_maps(cfg["upc_path"], selected)
        arr = _build_selected_arrays(upc_sw, selected, cfg, nest_maps=nest_maps)
        fit = _fit_models(arr, cfg, seed=ecfg.seed)

        for model_name, vals in fit.items():
            row = {
                "n_goods": int(k),
                "model": model_name,
                "fit_time_sec": float(vals["fit_time_sec"]),
                "rmse_test": float(vals["rmse"]),
                "n_train": int(len(arr["tr_idx"])),
                "n_test": int(len(arr["te_idx"])),
                "spend_scale_from_footfall": float(arr["spend_scale"]),
                "blp_fs_rsq_mean": float(vals["fs_rsq_mean"]) if "fs_rsq_mean" in vals else float("nan"),
                "blp_fs_rsq_min": float(vals["fs_rsq_min"]) if "fs_rsq_min" in vals else float("nan"),
            }
            scaling_rows.append(row)

        _s_rmse = fit.get("Neural Demand (static)", {}).get("rmse", float("nan"))
        _h_rmse = fit.get("Neural Demand (habit, FE, CF)", {}).get("rmse", float("nan"))
        _b_rmse = fit.get("BLP (IV, Chintagunta 2002 spec)", {}).get("rmse", float("nan"))
        print(
            f"[exp12] K={k:>3d}  "
            f"BLP_RMSE={_b_rmse:.6f}  "
            f"static_RMSE={_s_rmse:.6f}  "
            f"habit_FE_CF_RMSE={_h_rmse:.6f}"
        )

    scaling_df = pd.DataFrame(scaling_rows).sort_values(["n_goods", "model"]).reset_index(drop=True)
    bench_df = (
        scaling_df[scaling_df["n_goods"] == max(goods_grid)]
        .sort_values("model")
        .reset_index(drop=True)
    )

    scaling_csv = os.path.join(ecfg.out_dir, "table12_scaling_time_rmse.csv")
    benchmark_csv = os.path.join(ecfg.out_dir, "table12_full_benchmark_at_max_goods.csv")
    scaling_df.to_csv(scaling_csv, index=False)
    bench_df.to_csv(benchmark_csv, index=False)
    # Pass n_train from the largest K run so breakpoint lines are data-driven.
    _n_train_max = int(scaling_df["n_train"].max()) if not scaling_df.empty else 0
    _plot_scaling(scaling_df, ecfg.fig_dir, n_train=_n_train_max)

    meta = pd.DataFrame(
        [
            {
                "n_total_ranked_upcs": int(len(ranked_upcs)),
                "goods_grid": ",".join(str(x) for x in goods_grid),
                "max_goods_for_full_benchmark": int(max(goods_grid)),
                "full_only": bool(ecfg.full_only),
                "test_cutoff": int(ecfg.test_cutoff),
                "min_store_weeks": int(ecfg.min_store_weeks),
                "seed": int(ecfg.seed),
                "market_size_note": "M_it = FOOTFALL_it * spend_scale(train p98 of category spend per visit)",
            }
        ]
    )
    meta_csv = os.path.join(ecfg.out_dir, "exp12_metadata.csv")
    meta.to_csv(meta_csv, index=False)

    print(f"[exp12] done. Outputs in: {ecfg.out_dir}")
    return {
        "status": "ok",
        "out_dir": ecfg.out_dir,
        "scaling_csv": scaling_csv,
        "benchmark_csv": benchmark_csv,
        "metadata_csv": meta_csv,
        "n_ranked_upcs": int(len(ranked_upcs)),
        "goods_grid": goods_grid,
    }


def _parse_args():
    p = argparse.ArgumentParser(
        description="Dominick's Exp 12: UPC-level Logit-IV vs Neural Demand scaling benchmark"
    )
    p.add_argument("--weekly", type=str, default="data/wana.csv")
    p.add_argument("--upc", type=str, default="data/upcana.csv")
    p.add_argument("--data-dir", type=str, default="data", help="Directory containing ccount.csv")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--force-retrain", action="store_true", help="Ignore model cache and retrain")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--test-cutoff", type=int, default=351)
    p.add_argument("--min-store-weeks", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--goods-grid",
        type=str,
        default="5,10,20,40,80,160,320",
        help=(
            "Comma-separated UPC counts for the scaling sweep. "
            "The full-goods count is always appended automatically. "
            "Default gives a log-spaced grid from 5 to 320 plus all-goods."
        ),
    )
    p.add_argument(
        "--n-upc-cap",
        type=int,
        default=0,
        help="Max top-ranked UPCs to consider; 0 means all UPCs",
    )
    p.add_argument(
        "--full-only",
        action="store_true",
        help="Run only the full-UPC benchmark (skip goods-grid sweep)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
