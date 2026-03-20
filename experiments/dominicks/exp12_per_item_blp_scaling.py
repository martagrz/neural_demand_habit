"""
Experiment 12: Dominick's per-item (UPC-level) BLP vs Neural Demand benchmarks.

This experiment implements:
1) UPC-level analgesics panel construction (all UPCs available in upcana/wana),
2) train/test evaluation using a week split,
3) BLP logit-IV with Hausman instruments and market size estimated from footfall,
4) Neural Demand benchmarks (static, habit, FE, CF, habit FE, habit CF, habit FE CF),
5) scaling comparison across number of UPCs, reporting fit-time and test RMSE.
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

from experiments.dominicks.data import _parse_tablets
from run_neural_demand_dominicks import BASE_CFG, FAST_CFG
from src.models.blp_logit_iv import BLPLogitIV
from src.models.dominicks import cf_first_stage, hausman_iv
from src.models.dominicks_train import train_dominicks as _train
from src.models.mdp_neural_irl import HabitND, HabitND_FE
from src.models.neural_irl import StaticND, StaticND_FE


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
    use_fast: bool
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
    cfg["model_cache_dir"] = (
        "results/neural_demand/dominicks/models/fast"
        if args.fast
        else "results/neural_demand/dominicks/models/full"
    )
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
        use_fast=bool(args.fast),
        full_only=bool(args.full_only),
    )
    return cfg, ecfg


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


def _build_selected_arrays(upc_sw: pd.DataFrame, selected_upcs: list[int], cfg: dict) -> dict:
    # Build complete store-week grid.
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

    # Market size from footfall:
    # M_it = footfall_it * spend_scale, where spend_scale is estimated on train.
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
        # Rescale rows that violate simplex for numerical stability.
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


def _pred_fe(model: StaticND_FE, p: np.ndarray, y: np.ndarray, store_idx: np.ndarray, cfg: dict) -> np.ndarray:
    dev = cfg["device"]
    with torch.no_grad():
        lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32, device=dev))
        ly = torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32, device=dev)).unsqueeze(1)
        si = torch.tensor(store_idx, dtype=torch.long, device=dev)
        wp = model(lp, ly, si)
    return wp.cpu().numpy()


def _pred_habit(model: HabitND, p: np.ndarray, y: np.ndarray, xb: np.ndarray, qp: np.ndarray, cfg: dict, v_hat=None) -> np.ndarray:
    dev = cfg["device"]
    with torch.no_grad():
        lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32, device=dev))
        ly = torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32, device=dev)).unsqueeze(1)
        xbt = torch.tensor(xb, dtype=torch.float32, device=dev)
        qpt = torch.tensor(qp, dtype=torch.float32, device=dev)
        if v_hat is None:
            wp = model(lp, ly, xbt, qpt)
        else:
            vh = torch.tensor(v_hat, dtype=torch.float32, device=dev)
            wp = model(lp, ly, xbt, qpt, v_hat=vh)
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


def _fit_models(
    arr: dict,
    cfg: dict,
    seed: int,
    run_extended: bool,
    fast_only_habit_fe_cf: bool = False,
) -> dict:
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

    if fast_only_habit_fe_cf:
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
            mdp_fe_cf,
            p_tr,
            y_tr,
            w_tr,
            "mdp",
            cfg,
            xb_prev_tr=xb_tr,
            q_prev_tr=qp_tr,
            store_idx_tr=sidx_tr,
            v_hat_tr=v_hat_tr,
            tag=f"Exp12_HabitND_FE_CF_G{n_goods}_seed{seed}",
        )
        t_fit = time.perf_counter() - t0
        mdp_fe_cf_pred = _pred_habit_fe(
            mdp_fe_cf, p_te, y_te, xb_te, qp_te, sidx_te, cfg, v_hat=vh_te_zeros
        )
        rmse = _rmse(w_te, mdp_fe_cf_pred)
        print(
            f"{model_prefix} done Neural Demand (habit, FE, CF): "
            f"time={t_fit:.1f}s rmse={rmse:.6f}",
            flush=True,
        )
        return {
            "Neural Demand (habit, FE, CF)": {
                "rmse": rmse,
                "fit_time_sec": float(t_fit),
            }
        }

    # BLP (IV) benchmark.
    print(f"{model_prefix} fitting BLP (IV)...", flush=True)
    t0 = time.perf_counter()
    blp = BLPLogitIV().fit(p_tr, mw_tr, z_tr, verbose=False)
    t_blp = time.perf_counter() - t0
    blp_pred_mkt = blp.predict(p_te)[:, :n_goods]
    blp_pred_cond = blp_pred_mkt / np.maximum(blp_pred_mkt.sum(axis=1, keepdims=True), 1e-8)
    out = {
        "BLP (IV)": {"rmse": _rmse(w_te, blp_pred_cond), "fit_time_sec": float(t_blp)}
    }
    print(
        f"{model_prefix} done BLP (IV): time={t_blp:.1f}s rmse={out['BLP (IV)']['rmse']:.6f}",
        flush=True,
    )

    # Neural Demand (static) used in scaling comparison.
    print(f"{model_prefix} fitting Neural Demand (static)...", flush=True)
    t0 = time.perf_counter()
    nirl = StaticND(hidden_dim=cfg["nirl_hidden"], n_goods=n_goods)
    nirl, _ = _train(
        nirl,
        p_tr,
        y_tr,
        w_tr,
        "nirl",
        cfg,
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

    if not run_extended:
        return out

    # Habit
    print(f"{model_prefix} fitting Neural Demand (habit)...", flush=True)
    t0 = time.perf_counter()
    mdp = HabitND(hidden_dim=cfg["mdp_hidden"], n_goods=n_goods, delta_init=float(cfg.get("habit_decay", 0.7)))
    mdp, _ = _train(
        mdp,
        p_tr,
        y_tr,
        w_tr,
        "mdp",
        cfg,
        xb_prev_tr=xb_tr,
        q_prev_tr=qp_tr,
        tag=f"Exp12_HabitND_G{n_goods}_seed{seed}",
    )
    t_mdp = time.perf_counter() - t0
    mdp_pred = _pred_habit(mdp, p_te, y_te, xb_te, qp_te, cfg)
    out["Neural Demand (habit)"] = {"rmse": _rmse(w_te, mdp_pred), "fit_time_sec": float(t_mdp)}
    print(
        f"{model_prefix} done Neural Demand (habit): "
        f"time={t_mdp:.1f}s rmse={out['Neural Demand (habit)']['rmse']:.6f}",
        flush=True,
    )

    # FE
    print(f"{model_prefix} fitting Neural Demand (FE)...", flush=True)
    t0 = time.perf_counter()
    nirl_fe = StaticND_FE(
        hidden_dim=cfg["nirl_hidden"],
        n_goods=n_goods,
        n_stores=int(arr["n_stores"]),
        emb_dim=8,
    )
    nirl_fe, _ = _train(
        nirl_fe,
        p_tr,
        y_tr,
        w_tr,
        "nirl",
        cfg,
        store_idx_tr=sidx_tr,
        tag=f"Exp12_StaticND_FE_G{n_goods}_seed{seed}",
    )
    t_fe = time.perf_counter() - t0
    fe_pred = _pred_fe(nirl_fe, p_te, y_te, sidx_te, cfg)
    out["Neural Demand (FE)"] = {"rmse": _rmse(w_te, fe_pred), "fit_time_sec": float(t_fe)}
    print(
        f"{model_prefix} done Neural Demand (FE): "
        f"time={t_fe:.1f}s rmse={out['Neural Demand (FE)']['rmse']:.6f}",
        flush=True,
    )

    # CF (static + control function residuals).
    print(f"{model_prefix} fitting Neural Demand (CF)...", flush=True)
    t0 = time.perf_counter()
    nirl_cf = StaticND(hidden_dim=cfg["nirl_hidden"], n_goods=n_goods, n_cf=n_goods)
    nirl_cf, _ = _train(
        nirl_cf,
        p_tr,
        y_tr,
        w_tr,
        "nirl",
        cfg,
        v_hat_tr=v_hat_tr,
        tag=f"Exp12_StaticND_CF_G{n_goods}_seed{seed}",
    )
    t_cf = time.perf_counter() - t0
    cf_pred = _pred_static(nirl_cf, p_te, y_te, cfg, v_hat=vh_te_zeros)
    out["Neural Demand (CF)"] = {"rmse": _rmse(w_te, cf_pred), "fit_time_sec": float(t_cf)}
    print(
        f"{model_prefix} done Neural Demand (CF): "
        f"time={t_cf:.1f}s rmse={out['Neural Demand (CF)']['rmse']:.6f}",
        flush=True,
    )

    # Habit + FE
    print(f"{model_prefix} fitting Neural Demand (habit, FE)...", flush=True)
    t0 = time.perf_counter()
    mdp_fe = HabitND_FE(
        hidden_dim=cfg["mdp_hidden"],
        n_goods=n_goods,
        delta_init=float(cfg.get("habit_decay", 0.7)),
        n_stores=int(arr["n_stores"]),
        emb_dim=8,
    )
    mdp_fe, _ = _train(
        mdp_fe,
        p_tr,
        y_tr,
        w_tr,
        "mdp",
        cfg,
        xb_prev_tr=xb_tr,
        q_prev_tr=qp_tr,
        store_idx_tr=sidx_tr,
        tag=f"Exp12_HabitND_FE_G{n_goods}_seed{seed}",
    )
    t_mdp_fe = time.perf_counter() - t0
    mdp_fe_pred = _pred_habit_fe(mdp_fe, p_te, y_te, xb_te, qp_te, sidx_te, cfg)
    out["Neural Demand (habit, FE)"] = {"rmse": _rmse(w_te, mdp_fe_pred), "fit_time_sec": float(t_mdp_fe)}
    print(
        f"{model_prefix} done Neural Demand (habit, FE): "
        f"time={t_mdp_fe:.1f}s rmse={out['Neural Demand (habit, FE)']['rmse']:.6f}",
        flush=True,
    )

    # Habit + CF
    print(f"{model_prefix} fitting Neural Demand (habit, CF)...", flush=True)
    t0 = time.perf_counter()
    mdp_cf = HabitND(
        hidden_dim=cfg["mdp_hidden"],
        n_goods=n_goods,
        delta_init=float(cfg.get("habit_decay", 0.7)),
        n_cf=n_goods,
    )
    mdp_cf, _ = _train(
        mdp_cf,
        p_tr,
        y_tr,
        w_tr,
        "mdp",
        cfg,
        xb_prev_tr=xb_tr,
        q_prev_tr=qp_tr,
        v_hat_tr=v_hat_tr,
        tag=f"Exp12_HabitND_CF_G{n_goods}_seed{seed}",
    )
    t_mdp_cf = time.perf_counter() - t0
    mdp_cf_pred = _pred_habit(mdp_cf, p_te, y_te, xb_te, qp_te, cfg, v_hat=vh_te_zeros)
    out["Neural Demand (habit, CF)"] = {"rmse": _rmse(w_te, mdp_cf_pred), "fit_time_sec": float(t_mdp_cf)}
    print(
        f"{model_prefix} done Neural Demand (habit, CF): "
        f"time={t_mdp_cf:.1f}s rmse={out['Neural Demand (habit, CF)']['rmse']:.6f}",
        flush=True,
    )

    # Habit + FE + CF
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
        mdp_fe_cf,
        p_tr,
        y_tr,
        w_tr,
        "mdp",
        cfg,
        xb_prev_tr=xb_tr,
        q_prev_tr=qp_tr,
        store_idx_tr=sidx_tr,
        v_hat_tr=v_hat_tr,
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


def _plot_scaling(df: pd.DataFrame, fig_dir: str) -> None:
    if df.empty:
        return
    model_order = list(df["model"].drop_duplicates())
    color_cycle = [
        "#8E24AA", "#1E88E5", "#00897B", "#1565C0", "#283593", "#1B5E20",
        "#E53935", "#FB8C00",
    ]
    color_map = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(model_order)}

    # Time plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in model_order:
        d = df[df["model"] == model].sort_values("n_goods")
        if len(d) == 0:
            continue
        ax.plot(
            d["n_goods"], d["fit_time_sec"], marker="o", lw=2.0,
            color=color_map[model], label=model
        )
    ax.set_xlabel("Number of UPCs (goods)")
    ax.set_ylabel("Fit time (seconds)")
    ax.set_title("Fit Time vs Number of Goods")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"fig12_fit_time_vs_goods.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # RMSE plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in model_order:
        d = df[df["model"] == model].sort_values("n_goods")
        if len(d) == 0:
            continue
        ax.plot(
            d["n_goods"], d["rmse_test"], marker="o", lw=2.0,
            color=color_map[model], label=model
        )
    ax.set_xlabel("Number of UPCs (goods)")
    ax.set_ylabel("Test RMSE (conditional share)")
    ax.set_title("Test RMSE vs Number of Goods")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"fig12_rmse_vs_goods.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args):
    cfg, ecfg = _build_cfg(args)
    np.random.seed(ecfg.seed)
    torch.manual_seed(ecfg.seed)

    print("[exp12] Starting experiment...", flush=True)
    print(
        f"[exp12] settings: fast={ecfg.use_fast}, full_only={ecfg.full_only}, "
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

    scaling_rows = []
    full_benchmark_rows = []
    print(f"[exp12] Goods grid to run: {goods_grid}", flush=True)
    if ecfg.use_fast:
        print("[exp12] Fast mode: fitting only Neural Demand (habit, FE, CF).", flush=True)

    for k in goods_grid:
        print(f"[exp12] Preparing arrays for K={k}...", flush=True)
        selected = ranked_upcs[:k]
        arr = _build_selected_arrays(upc_sw, selected, cfg)
        run_extended = (k == max(goods_grid))
        fit = _fit_models(
            arr,
            cfg,
            seed=ecfg.seed,
            run_extended=run_extended,
            fast_only_habit_fe_cf=bool(ecfg.use_fast),
        )

        for model_name, vals in fit.items():
            row = {
                "n_goods": int(k),
                "model": model_name,
                "fit_time_sec": float(vals["fit_time_sec"]),
                "rmse_test": float(vals["rmse"]),
                "n_train": int(len(arr["tr_idx"])),
                "n_test": int(len(arr["te_idx"])),
                "spend_scale_from_footfall": float(arr["spend_scale"]),
            }
            scaling_rows.append(row)
            if run_extended:
                full_benchmark_rows.append(row)

        print(
            f"[exp12] K={k:>3d}  "
            f"BLP_RMSE={fit['BLP (IV)']['rmse']:.6f}  "
            f"NDS_RMSE={fit['Neural Demand (static)']['rmse']:.6f}"
        )

    scaling_df = pd.DataFrame(scaling_rows).sort_values(["n_goods", "model"]).reset_index(drop=True)
    bench_df = pd.DataFrame(full_benchmark_rows).sort_values(["model"]).reset_index(drop=True)

    scaling_csv = os.path.join(ecfg.out_dir, "table12_scaling_time_rmse.csv")
    benchmark_csv = os.path.join(ecfg.out_dir, "table12_full_benchmark_at_max_goods.csv")
    scaling_df.to_csv(scaling_csv, index=False)
    bench_df.to_csv(benchmark_csv, index=False)
    _plot_scaling(scaling_df, ecfg.fig_dir)

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
        description="Dominick's Exp 12: UPC-level BLP (IV) vs Neural Demand scaling benchmark"
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
        default="8,12,16,24,32",
        help="Comma-separated UPC counts for scaling curves",
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
