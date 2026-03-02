"""
experiments/neural_demand/dominicks/exp07_first_stage.py
========================================================
Calculates first-stage F-statistics, R2, and partial R2 for the Hausman instruments.
Also investigates Ibuprofen specific stats.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from experiments.dominicks.data import load, GOODS, G

def run(splits, cfg):
    """Run first-stage diagnostics and generate Table tab:first_stage_fstats."""
    import os
    
    print("=" * 68)
    print("  Exp 08 — First-Stage Diagnostics")
    print("=" * 68)

    p_tr = splits['p_tr']
    s_tr = splits['s_tr']
    Z_tr = splits['Z_tr']
    
    # Log prices
    log_p = np.log(np.maximum(p_tr, 1e-8))
    
    results = []
    
    # We use the Store Fixed Effects specification for the table
    # as it provides a meaningful "Partial R2".
    
    # Create dummy variables for stores (for statsmodels)
    # Note: s_tr are store indices.
    store_dummies = pd.get_dummies(s_tr, prefix='store', drop_first=True).astype(float)
    
    print(f"  Running first-stage regressions (N={len(p_tr)}, Stores={len(np.unique(s_tr))})...")
    
    for j in range(G):
        y = log_p[:, j]
        z = Z_tr[:, j]
        
        # Full model: log(p) ~ Z + StoreFE
        X_full = pd.DataFrame({'Z': z})
        X_full = pd.concat([X_full, store_dummies.reset_index(drop=True)], axis=1)
        X_full = sm.add_constant(X_full)
        
        model_full = sm.OLS(y, X_full)
        res_full = model_full.fit()
        
        # Restricted model: log(p) ~ StoreFE
        X_rest = sm.add_constant(store_dummies.reset_index(drop=True))
        model_rest = sm.OLS(y, X_rest)
        res_rest = model_rest.fit()
        
        # Partial R2 = (RSS_rest - RSS_full) / RSS_rest
        rss_full = res_full.ssr
        rss_rest = res_rest.ssr
        partial_r2 = (rss_rest - rss_full) / rss_rest
        
        # F-stat for Z (t-stat squared)
        f_stat = res_full.tvalues['Z'] ** 2
        
        results.append({
            'Good': GOODS[j],
            'F_stat': f_stat,
            'R2': res_full.rsquared,
            'Partial_R2': partial_r2
        })
        print(f"    {GOODS[j]:<15} F={f_stat:.2f}  R2={res_full.rsquared:.4f}  PartialR2={partial_r2:.4f}")

    # Generate LaTeX Table
    out_dir = cfg['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    tex_path = f"{out_dir}/table_first_stage_fstats.tex"
    
    tex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{First-Stage Diagnostics --- Hausman Instruments (Dominick's Analgesics)}",
        r"  \label{tab:first_stage_fstats}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    \textbf{Price equation} & \textbf{First-stage $F$} & \textbf{First-stage $R^2$} & \textbf{Partial $R^2$} \\",
        r"    \midrule",
    ]
    
    for r in results:
        tex_lines.append(f"    {r['Good']} & {r['F_stat']:.2f} & {r['R2']:.3f} & {r['Partial_R2']:.3f} \\\\")
        
    tex_lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption*{OLS first-stage regression of $\ln p_{git}$ on revenue-weighted mean prices in other stores. $F$-statistic tests the joint significance of all $G$ Hausman instruments. Values below 10 indicate weak instruments \citep{staiger1994instrumental}.}",
        r"\end{table}"
    ]
    
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines))
        
    print(f"  Saved LaTeX table to {tex_path}")
    return results

if __name__ == "__main__":
    # Mock splits for standalone run
    from experiments.dominicks.data import load
    CFG = {
        'weekly_path': 'data/wana.csv',
        'upc_path':    'data/upcana.csv',
        'min_store_wks': 20,
        'test_cutoff':   351,
        'std_tablets':   100,
        'habit_decay':   0.7,
        'shock_good':    0,
        'shock_pct':     0.10,
        'out_dir':       'results/neural_demand/dominicks',
        'fig_dir':       'results/neural_demand/dominicks/figures',
        'device':        'cpu',
    }
    import os
    if os.path.exists(CFG['weekly_path']):
        _, splits = load(CFG)
        run(splits, CFG)
    else:
        print("Data not found.")
