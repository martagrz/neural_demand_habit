"""
experiments/neural_demand/dominicks/exp09_regularity_dashboard.py
=================================================================
Generates the "Near-Integrability Diagnostics" table (Regularity Dashboard).

Evaluates:
  - Static (no reg)
  - Static (reg)
  - Habit (no reg)
  - Habit (reg)
  - FE static (reg)
  - FE habit (reg)

Metrics:
  - Test KL
  - Mean D_sym (Slutsky symmetry violation)
  - Max D_sym
  - Prob(lambda_max(S_tilde) > 0) (Curvature violation freq)
  - Mean D_curv (Curvature violation magnitude)
  - Mean D_hom (Homogeneity violation)
"""

import numpy as np
import torch
import pandas as pd
import os
from copy import deepcopy

from src.models.dominicks import (
    StaticND, StaticND_FE,
    HabitND, HabitND_FE,
    _train
)
from experiments.dominicks.utils import get_metrics, kl_div

# ─────────────────────────────────────────────────────────────────────────────
#  SLUTSKY & DIAGNOSTICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_diagnostics(model, p, y, xb=None, qp=None, store_idx=None,
                        hom_c_values=[0.8, 1.2], device="cpu"):
    """Compute regularity diagnostics for a batch of data."""
    model.eval()
    model.to(device)
    
    # Prepare tensors
    p_t = torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32, device=device)
    y_t = torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32, device=device)
    
    log_p = torch.log(p_t).requires_grad_(True)
    log_y = torch.log(y_t).unsqueeze(1).requires_grad_(True)
    
    xb_t = torch.tensor(xb, dtype=torch.float32, device=device) if xb is not None else None
    qp_t = torch.tensor(qp, dtype=torch.float32, device=device) if qp is not None else None
    si_t = torch.tensor(store_idx, dtype=torch.long, device=device) if store_idx is not None else None
    
    # ── 1. Forward pass to get w and q ───────────────────────────────────────
    # Helper to call model with correct args
    def _fwd(lp, ly):
        if xb_t is not None and si_t is not None:
            return model(lp, ly, xb_t, qp_t, si_t)
        elif xb_t is not None:
            return model(lp, ly, xb_t, qp_t)
        elif si_t is not None:
            return model(lp, ly, si_t)
        else:
            return model(lp, ly)

    w = _fwd(log_p, log_y)
    # q_i = w_i * y / p_i
    # In log space: log q_i = log w_i + log y - log p_i
    # But we need gradients of q, so let's compute q explicitly
    q = w * torch.exp(log_y) / torch.exp(log_p)
    
    B, G = p.shape
    
    # ── 2. Slutsky Matrix ────────────────────────────────────────────────────
    # S_ij = dq_i/dp_j + q_j * dq_i/dy
    # We use autograd to get Jacobians.
    # Since we have a batch, we can't use standard jacobian easily without loop or vmap.
    # We'll use a loop over goods G (since G is small, e.g., 3) to get columns/rows.
    
    # Gradients w.r.t log_p and log_y are easier from model, but we need w.r.t p and y.
    # dq/dp = (dq/dlogp) * (1/p)
    # dq/dy = (dq/dlogy) * (1/y)
    
    grad_q_p = torch.zeros(B, G, G, device=device) # (B, i, j) -> dq_i / dp_j
    grad_q_y = torch.zeros(B, G, device=device)    # (B, i)    -> dq_i / dy
    
    for i in range(G):
        # Grad of q[:, i] sum w.r.t log_p
        # We need per-sample gradients. torch.autograd.grad sums over batch by default.
        # To get per-sample, we can use torch.func.vmap or just loop if batch is small?
        # Batch is large (test set).
        # Standard trick: grad(output.sum(), input) gives sum of grads.
        # But we need Jacobian.
        # Let's use the fact that q_i depends on p and y.
        
        # Actually, let's compute gradients of w first, then assemble q grads.
        # w = model(...)
        # dw_i/dlogp_j
        # dw_i/dlogy
        pass

    # Re-approach: Use torch.autograd.functional.jacobian? It might be slow for large batch.
    # Better: Use the analytical relation with elasticities if possible, or just efficient autograd.
    # Let's use a loop over G to compute gradients of w_i w.r.t inputs.
    
    # dw/dlogp: (B, G, G)
    # dw/dlogy: (B, G, 1)
    
    J_w_lp = torch.zeros(B, G, G, device=device)
    J_w_ly = torch.zeros(B, G, 1, device=device)
    
    for i in range(G):
        # We want grad of w[:, i] w.r.t log_p and log_y.
        # To get per-sample gradients efficiently:
        # grad_outputs = torch.eye(G)[i].expand(B, G) ? No.
        # We can use `torch.autograd.grad(w[:, i].sum(), inputs, create_graph=False)`
        # This gives sum_k (dw_{k,i}/dx). This is not what we want.
        # We want dw_{k,i}/dx_k for each k.
        # Actually, since samples are independent, d(sum_k w_{k,i}) / d x_k = dw_{k,i}/dx_k.
        # So yes, summing the output and taking grad w.r.t batch input gives the batch of gradients.
        
        g_lp, g_ly = torch.autograd.grad(
            w[:, i].sum(), [log_p, log_y], retain_graph=True
        )
        J_w_lp[:, i, :] = g_lp
        J_w_ly[:, i, :] = g_ly
        
    # Now convert to q derivatives
    # q_i = w_i * y / p_i
    # log q_i = log w_i + log y - log p_i
    # d log q_i / d log p_j = (1/w_i) * (dw_i/dlog p_j) - delta_{ij}
    # d log q_i / d log y   = (1/w_i) * (dw_i/dlog y) + 1
    
    # Elasticities E_ij = d log q_i / d log p_j
    # Income Elast eta_i = d log q_i / d log y
    
    # Avoid div by zero
    w_safe = torch.clamp(w, min=1e-9)
    
    E = (1.0 / w_safe.unsqueeze(2)) * J_w_lp - torch.eye(G, device=device).unsqueeze(0)
    eta = (1.0 / w_safe) * J_w_ly.squeeze(2) + 1.0
    
    # Slutsky S_ij = (q_i / p_j) * (E_ij + w_j * eta_i)
    # q_i / p_j = (w_i y / p_i) / p_j
    
    # Let's compute term (E_ij + w_j * eta_i)
    # w_j * eta_i: outer product for each batch
    # eta: (B, G), w: (B, G)
    # term2_ij = eta_i * w_j
    term2 = eta.unsqueeze(2) * w.unsqueeze(1) # (B, G, G)
    
    # Bracket term
    bracket = E + term2
    
    # Prefactor: q_i / p_j
    # q: (B, G)
    # p: (B, G)
    # prefactor_ij = q_i / p_j
    prefactor = q.unsqueeze(2) / p_t.unsqueeze(1)
    
    S = prefactor * bracket
    
    # ── Diagnostics ──────────────────────────────────────────────────────────
    
    # 1. Symmetry: ||S - S^T||_F
    S_T = S.transpose(1, 2)
    diff = S - S_T
    D_sym = torch.norm(diff, dim=(1, 2)) # Frobenius norm per sample
    
    # 2. Curvature: lambda_max((S + S^T)/2)
    S_tilde = 0.5 * (S + S_T)
    # torch.linalg.eigvalsh for symmetric matrices
    eigvals = torch.linalg.eigvalsh(S_tilde) # (B, G)
    lambda_max = eigvals[:, -1] # Largest eigenvalue
    
    is_pos = (lambda_max > 1e-6).float() # Tolerance for numerical noise
    D_curv = torch.clamp(lambda_max, min=0.0)
    
    # ── 3. Homogeneity ───────────────────────────────────────────────────────
    # D_hom(c) = || w(cp, cy) - w(p, y) ||_inf
    D_hom_list = []
    with torch.no_grad():
        ly_hom = torch.log(y_t).unsqueeze(1)
        w_base = _fwd(torch.log(p_t), ly_hom)
        for c in hom_c_values:
            # log(c*x) = log(c) + log(x)
            lc = np.log(c)
            w_c = _fwd(torch.log(p_t) + lc, ly_hom + lc)
            # Max abs diff per sample
            diff_c = torch.max(torch.abs(w_c - w_base), dim=1).values
            D_hom_list.append(diff_c)
            
    D_hom = torch.stack(D_hom_list).mean(0) # Average over c values for each sample
    
    return {
        "D_sym": D_sym.detach().cpu().numpy(),
        "is_pos": is_pos.detach().cpu().numpy(),
        "D_curv": D_curv.detach().cpu().numpy(),
        "D_hom": D_hom.detach().cpu().numpy()
    }


def run(splits, cfg):
    """Run Experiment 09: Regularity Dashboard."""
    
    # ── Setup ────────────────────────────────────────────────────────────────
    # We need to run 6 models.
    # Some are standard (reg), some are no-reg (lambda=0).
    # We will use the first seed (or loop over seeds? Table implies one set of numbers, 
    # but usually we report mean over runs or just one run. 
    # The table caption says "Test KL" which implies we might want to be consistent with other tables.
    # Let's assume we run for 1 seed (seed=42) or aggregate. 
    # Given the complexity, let's run for 1 seed (seed=42) as a "compact audit".
    # Or better, follow N_RUNS from cfg but that might be slow.
    # Let's stick to N_RUNS=1 (seed 42) for the dashboard unless specified otherwise.
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Unpack data
    p_tr, y_tr, w_tr = splits['p_tr'], splits['y_tr'], splits['w_tr']
    xb_tr, qp_tr     = splits['xb_tr'], splits['qp_tr']
    s_tr, s_tr_idx   = splits['s_tr'], splits['s_tr_idx']
    
    p_te, y_te, w_te = splits['p_te'], splits['y_te'], splits['w_te']
    xb_te, qp_te     = splits['xb_te'], splits['qp_te']
    s_te, s_te_idx   = splits['s_te'], splits['s_te_idx']
    
    N_STORES = splits['N_STORES']
    EMB_DIM  = splits['STORE_EMB_DIM']
    
    # Configs
    # "reg" means standard config.
    # "no reg" means lam_mono=0, lam_slut=0.
    
    # We need to construct configs for no-reg.
    cfg_reg = cfg.copy()
    cfg_noreg = cfg.copy()
    
    # Zero out regularization for no-reg
    for k in ["nirl_lam_mono", "nirl_lam_slut", 
              "mdp_lam_mono", "mdp_lam_slut",
              "mdp_e2e_lam_mono", "mdp_e2e_lam_slut"]:
        cfg_noreg[k] = 0.0
        
    # Define the 6 models
    # (Name, ModelClass, PFX, Config, ExtraArgs, TagSuffix)
    specs = [
        ("Static (no reg)", StaticND, 'nirl', cfg_noreg, {}, "no-reg"),
        ("Static (reg)",    StaticND, 'nirl', cfg_reg,   {}, "reg"),
        ("Habit (no reg)",  HabitND, 'mdp', cfg_noreg, 
         {'xb_prev_tr': xb_tr, 'q_prev_tr': qp_tr}, "no-reg"),
        ("Habit (reg)",     HabitND, 'mdp', cfg_reg,   
         {'xb_prev_tr': xb_tr, 'q_prev_tr': qp_tr}, "reg"),
        ("FE static (reg)", StaticND_FE, 'nirl', cfg_reg, 
         {'store_idx_tr': s_tr_idx, 'n_stores': N_STORES, 'emb_dim': EMB_DIM}, "reg"),
        ("FE habit (reg)",  HabitND_FE, 'mdp', cfg_reg, 
         {'xb_prev_tr': xb_tr, 'q_prev_tr': qp_tr, 'store_idx_tr': s_tr_idx, 
          'n_stores': N_STORES, 'emb_dim': EMB_DIM}, "reg"),
    ]
    
    results = []
    
    print("=" * 68)
    print("  Exp 09 — Near-Integrability Diagnostics (Regularity Dashboard)")
    print("=" * 68)
    
    for name, cls, pfx, run_cfg, train_kw, tag_suf in specs:
        print(f"\n  Running: {name} ...")
        
        # Init model
        # Need to handle FE args in init
        init_kw = {'hidden_dim': run_cfg[f'{pfx}_hidden']}
        if 'n_stores' in train_kw:
            init_kw['n_stores'] = train_kw['n_stores']
            init_kw['emb_dim'] = train_kw['emb_dim']
        
        model = cls(**init_kw)
        
        # Train
        tag = f"{name.replace(' ', '_')} s={seed}"
        # We need to ensure unique tags for no-reg to avoid loading reg models from cache
        # The 'name' variable already distinguishes them ("Static (no reg)" vs "Static (reg)")
        
        # Filter train_kw to remove init-only args (n_stores, emb_dim)
        # train_dominicks only accepts specific kwargs
        actual_train_kw = {k: v for k, v in train_kw.items() 
                           if k not in ['n_stores', 'emb_dim']}

        # Note: train_dominicks (_train) uses cfg for hyperparameters.
        # We pass the specific cfg (reg or noreg) to it.
        model, _ = _train(model, p_tr, y_tr, w_tr, pfx, run_cfg, 
                          tag=tag, **actual_train_kw)
        
        # Evaluate KL
        # Prepare eval kwargs
        eval_kw = {}
        if 'xb_prev_tr' in train_kw:
            eval_kw['xb_prev'] = xb_te
            eval_kw['q_prev'] = qp_te
        if 'store_idx_tr' in train_kw:
            eval_kw['store_idx'] = s_te_idx
            
        # Use existing kl_div helper? 
        # kl_div helper uses 'pred' which uses string specs.
        # We have the model object directly. Let's compute KL manually or wrap it.
        # Actually exp09 helper compute_diagnostics computes w.
        # Let's compute KL there or here.
        
        # Compute Diagnostics
        # Need to batch this if test set is huge, but Dominick's test is ~25k, 
        # might fit in memory or be slow.
        # Let's do it in one go if possible, or chunks.
        # 25k * 3 * 3 floats is small.
        
        # Prepare inputs for diagnostics
        diag_kw = {
            'xb': eval_kw.get('xb_prev'),
            'qp': eval_kw.get('q_prev'),
            'store_idx': eval_kw.get('store_idx'),
            'device': run_cfg['device']
        }
        
        diags = compute_diagnostics(model, p_te, y_te, **diag_kw)
        
        # KL
        # Re-run forward to get w for KL (or could have returned it)
        # Let's just use the w from diagnostics if we returned it, but we didn't.
        # Quick KL calc:
        model.eval()
        with torch.no_grad():
            # Minimal forward pass logic again...
            # Or just use kl_div from utils if we map model to spec string?
            # kl_div takes a spec string and looks up model in KW dict.
            # We can't easily use it without setting up KW.
            # Let's just do it manually.
            p_t = torch.tensor(np.maximum(p_te, 1e-8), dtype=torch.float32, device=run_cfg['device'])
            y_t = torch.tensor(np.maximum(y_te, 1e-8), dtype=torch.float32, device=run_cfg['device']).unsqueeze(1)
            
            if 'xb' in diag_kw and diag_kw['xb'] is not None:
                xb_t = torch.tensor(diag_kw['xb'], dtype=torch.float32, device=run_cfg['device'])
                qp_t = torch.tensor(diag_kw['qp'], dtype=torch.float32, device=run_cfg['device'])
                if 'store_idx' in diag_kw and diag_kw['store_idx'] is not None:
                    si_t = torch.tensor(diag_kw['store_idx'], dtype=torch.long, device=run_cfg['device'])
                    w_pred = model(torch.log(p_t), torch.log(y_t), xb_t, qp_t, si_t)
                else:
                    w_pred = model(torch.log(p_t), torch.log(y_t), xb_t, qp_t)
            elif 'store_idx' in diag_kw and diag_kw['store_idx'] is not None:
                si_t = torch.tensor(diag_kw['store_idx'], dtype=torch.long, device=run_cfg['device'])
                w_pred = model(torch.log(p_t), torch.log(y_t), si_t)
            else:
                w_pred = model(torch.log(p_t), torch.log(y_t))
                
            w_pred = w_pred.cpu().numpy()
            w_pred = np.clip(w_pred, 1e-8, 1.0)
            w_true = np.clip(w_te, 1e-8, 1.0)
            test_kl = np.sum(w_true * np.log(w_true / w_pred), axis=1).mean()

        # Aggregation
        res = {
            "Model": name,
            "Test KL": test_kl,
            "E[D_sym]": diags["D_sym"].mean(),
            "max D_sym": diags["D_sym"].max(),
            "Pr(pos curv)": diags["is_pos"].mean(),
            "E[D_curv]": diags["D_curv"].mean(),
            "E[D_hom]": diags["D_hom"].mean()
        }
        results.append(res)
        
        # Print row
        print(f"    KL={res['Test KL']:.4f}  "
              f"D_sym={res['E[D_sym]']:.4f} (max {res['max D_sym']:.4f})  "
              f"PosCurv={res['Pr(pos curv)']:.1%}  "
              f"D_hom={res['E[D_hom]']:.4f}")

    # ── Table Generation ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    
    # Format for LaTeX
    # Columns: Model, Test KL, E[D_sym], max D_sym, Pr(pos), E[D_curv], E[D_hom]
    
    print("\n" + "="*68)
    print("  Table: Near-Integrability Diagnostics")
    print("="*68)
    print(df.to_string(index=False))
    
    # Save CSV
    os.makedirs(cfg["out_dir"], exist_ok=True)
    df.to_csv(os.path.join(cfg["out_dir"], "table_regularity_dashboard.csv"), index=False)
    
    # Save LaTeX
    tex_path = os.path.join(cfg["out_dir"], "table_regularity_dashboard.tex")
    with open(tex_path, "w") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Near-integrability diagnostics (``regularity dashboard'').}" + "\n")
        f.write(r"\label{tab:regularity_dashboard}" + "\n")
        f.write(r"\begin{tabular}{lcccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Model & Test KL & $\mathbb E[D_{\mathrm{sym}}]$ & $\max D_{\mathrm{sym}}$ & "
                r"$\Pr(\lambda_{\max}(\tilde S)>0)$ & $\mathbb E[D_{\mathrm{curv}}]$ & "
                r"$\mathbb E[D_{\mathrm{hom}}]$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        
        for _, row in df.iterrows():
            # Format numbers
            kl = f"{row['Test KL']:.4f}"
            ds = f"{row['E[D_sym]']:.4f}"
            dm = f"{row['max D_sym']:.4f}"
            pp = f"{row['Pr(pos curv)']:.3f}"
            dc = f"{row['E[D_curv]']:.4f}"
            dh = f"{row['E[D_hom]']:.4f}"
            
            # Add line space before FE models
            if "FE" in row['Model'] and "FE" not in df.iloc[df.index.get_loc(row.name)-1]['Model']:
                f.write(r"\addlinespace" + "\n")
                
            f.write(f"{row['Model']} & {kl} & {ds} & {dm} & {pp} & {dc} & {dh} \\\\" + "\n")
            
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
        
    print(f"\n  Saved: {tex_path}")
    
    return {"regularity_dashboard": df}
