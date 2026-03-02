
import torch
import numpy as np

def compute_slutsky_matrix(model, p, y, xb=None, q_prev=None, store_idx=None, v_hat=None, device='cpu'):
    """
    Computes the Slutsky matrix S for a batch of inputs.
    S_ij = (y / (p_i * p_j)) * [ d(w_i)/d(ln p_j) + w_i * w_j - delta_ij * w_i + w_j * d(w_i)/d(ln y) ]
    
    Returns S of shape (B, G, G).
    """
    model.eval()
    
    # Prepare inputs
    p_t = torch.tensor(p, dtype=torch.float32, device=device).requires_grad_(True)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    
    log_p = torch.log(torch.clamp(p_t, min=1e-8))
    log_y = torch.log(torch.clamp(y_t, min=1e-8))
    
    # Handle different model signatures
    is_mdp = hasattr(model, 'delta') or 'MDP' in model.__class__.__name__
    is_fe = hasattr(model, 'store_emb')
    
    args = [log_p, log_y]
    kwargs = {}
    
    if is_mdp:
        if xb is None or q_prev is None:
            raise ValueError("MDP models require xb and q_prev")
        xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
        q_prev_t = torch.tensor(q_prev, dtype=torch.float32, device=device)
        lxb = torch.log(torch.clamp(xb_t, min=1e-8))
        lq = torch.log(torch.clamp(q_prev_t, min=1e-8))
        args.extend([lxb, lq])
        
    if is_fe:
        if store_idx is None:
             raise ValueError("FE models require store_idx")
        s_idx = torch.tensor(store_idx, dtype=torch.long, device=device)
        args.append(s_idx)
        
    if v_hat is not None:
        v_t = torch.tensor(v_hat, dtype=torch.float32, device=device)
        kwargs['v_hat'] = v_t
    
    # Forward pass
    w = model(*args, **kwargs)
        
    # Compute gradients
    B, G = w.shape
    
    # d(w_i)/d(log_p_j)
    dw_dlp = torch.zeros(B, G, G, device=device)
    for i in range(G):
        # We need to retain graph for subsequent gradients
        grad = torch.autograd.grad(w[:, i].sum(), log_p, create_graph=True, retain_graph=True)[0]
        dw_dlp[:, i, :] = grad
        
    # d(w_i)/d(log_y)
    dw_dly = torch.zeros(B, G, device=device)
    for i in range(G):
        grad = torch.autograd.grad(w[:, i].sum(), log_y, create_graph=True, retain_graph=True)[0]
        dw_dly[:, i] = grad.squeeze()
        
    # Convert to numpy for final calculation
    w_np = w.detach().cpu().numpy()
    p_np = p
    y_np = y.reshape(-1, 1)
    dw_dlp_np = dw_dlp.detach().cpu().numpy()
    dw_dly_np = dw_dly.detach().cpu().numpy()
    
    # S_ij calculation
    S = np.zeros((B, G, G))
    for b in range(B):
        # Outer product w_i * w_j
        w_outer = np.outer(w_np[b], w_np[b])
        # Diagonal matrix with w_i
        w_diag = np.diag(w_np[b])
        # w_j * dw_i/dlny (outer product of dw_dly and w)
        # dw_dly is (G,), w is (G,)
        # We want term4_ij = w_j * dw_i/dlny
        # So it's outer(dw_dly, w) ?
        # term4_ij = w_j * (dw_i/dlny) -> row i, col j
        term4 = np.outer(dw_dly_np[b], w_np[b])
        
        bracket = dw_dlp_np[b] + w_outer - w_diag + term4
        
        # factor = y / (p_i * p_j)
        p_outer = np.outer(p_np[b], p_np[b])
        factor = y_np[b, 0] / p_outer
        
        S[b] = factor * bracket
                
    return S

def check_symmetry(S):
    """
    Computes symmetry violation ||S - S^T||_F.
    Returns mean, median, max.
    """
    S_T = np.transpose(S, (0, 2, 1))
    diff = S - S_T
    # Frobenius norm per matrix in batch
    norms = np.linalg.norm(diff, axis=(1, 2), ord='fro')
    # Relative norm: ||S - S^T|| / ||S||
    s_norms = np.linalg.norm(S, axis=(1, 2), ord='fro')
    rel_norms = norms / (s_norms + 1e-8)
    
    return {
        'mean_fro': np.mean(norms),
        'median_fro': np.median(norms),
        'max_fro': np.max(norms),
        'mean_rel': np.mean(rel_norms)
    }

def check_curvature(S):
    """
    Computes curvature / negativity of Substitution matrix.
    Checks eigenvalues of (S + S^T)/2.
    """
    S_sym = (S + np.transpose(S, (0, 2, 1))) / 2
    eigvals = np.linalg.eigvalsh(S_sym)
    
    # Max eigenvalue (should be <= 0 for NSD)
    max_eig = np.max(eigvals, axis=1)
    
    # Share with any positive eigenvalue (tolerance 1e-6)
    share_positive = np.mean(max_eig > 1e-6)
    
    # Mean positive part
    mean_pos_part = np.mean(np.maximum(max_eig, 0))
    
    return {
        'share_positive': share_positive,
        'mean_pos_part': mean_pos_part,
        'max_eig_dist': max_eig
    }

def check_homogeneity(model, p, y, xb=None, q_prev=None, store_idx=None, v_hat=None, scalars=[0.8, 1.2], device='cpu'):
    """
    Checks homogeneity of degree 0 in prices and income.
    w(c*p, c*y) should equal w(p, y).
    """
    model.eval()
    
    # Helper to predict
    def predict(p_in, y_in):
        p_t = torch.tensor(p_in, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_in, dtype=torch.float32, device=device).view(-1, 1)
        
        log_p = torch.log(torch.clamp(p_t, min=1e-8))
        log_y = torch.log(torch.clamp(y_t, min=1e-8))
        
        is_mdp = hasattr(model, 'delta') or 'MDP' in model.__class__.__name__
        is_fe = hasattr(model, 'store_emb')
        
        args = [log_p, log_y]
        kwargs = {}
        
        if is_mdp:
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            q_prev_t = torch.tensor(q_prev, dtype=torch.float32, device=device)
            lxb = torch.log(torch.clamp(xb_t, min=1e-8))
            lq = torch.log(torch.clamp(q_prev_t, min=1e-8))
            args.extend([lxb, lq])
            
        if is_fe:
            s_idx = torch.tensor(store_idx, dtype=torch.long, device=device)
            args.append(s_idx)
            
        if v_hat is not None:
            v_t = torch.tensor(v_hat, dtype=torch.float32, device=device)
            kwargs['v_hat'] = v_t
            
        with torch.no_grad():
            w = model(*args, **kwargs)
        return w.cpu().numpy()

    w_base = predict(p, y)
    results = {}
    
    for c in scalars:
        w_scaled = predict(p * c, y * c)
        diff = np.abs(w_scaled - w_base)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        results[c] = {'max_diff': max_diff, 'mean_diff': mean_diff}
        
    return results

def regularity_dashboard(model, p, y, xb=None, q_prev=None, store_idx=None, v_hat=None, device='cpu'):
    """
    Runs all checks and returns a dictionary of results.
    """
    S = compute_slutsky_matrix(model, p, y, xb, q_prev, store_idx, v_hat, device)
    
    sym_res = check_symmetry(S)
    curv_res = check_curvature(S)
    hom_res = check_homogeneity(model, p, y, xb, q_prev, store_idx, v_hat, device=device)
    
    return {
        'symmetry': sym_res,
        'curvature': curv_res,
        'homogeneity': hom_res
    }
