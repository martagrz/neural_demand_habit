# Neural Demand Estimation with Habit Formation and Rationality Constraints

Codebase for neural demand estimation with habit formation, regularity penalties, and control-function corrections. The repository contains two end-to-end pipelines:

- `experiments/simulation`: synthetic DGP validation and delta profiling.
- `experiments/dominicks`: empirical application on Dominick's analgesics data.

## Setup

### Requirements
- Python 3.8+
- Packages in `requirements.txt` (`torch`, `numpy`, `pandas`, `matplotlib`, `statsmodels`, `scikit-learn`)

### Install
```bash
pip install -r requirements.txt
```

## Main Entry Points

### Simulation pipeline
Run all simulation experiments:
```bash
python run_neural_demand_simulation.py
```

Common options:
```bash
# Quick smoke test mode
python run_neural_demand_simulation.py --fast

# Run a subset of experiments
python run_neural_demand_simulation.py --exp 01 03

# Load exact-tag cached models instead of retraining
python run_neural_demand_simulation.py --load
```

Simulation experiments:
- `01` DGP recovery
- `02` habit advantage
- `03` delta identification
- `04` control-function endogeneity correction

Outputs are written under `results/neural_demand/simulations`.

### Dominick's pipeline
Run all Dominick's experiments:
```bash
python run_neural_demand_dominicks.py --weekly data/wana.csv --upc data/upcana.csv
```

Common options:
```bash
# Quick smoke test mode
python run_neural_demand_dominicks.py --weekly data/wana.csv --upc data/upcana.csv --fast

# Run a subset of experiments
python run_neural_demand_dominicks.py --weekly data/wana.csv --upc data/upcana.csv --exp 01 03 07

# Load exact-tag cached models instead of retraining
python run_neural_demand_dominicks.py --weekly data/wana.csv --upc data/upcana.csv --load
```

Dominick's experiments:
- `01` predictive accuracy
- `02` elasticities
- `03` welfare (CV)
- `04` demand curves
- `05` delta identification
- `06` CF decomposition
- `07` full model figures/tables
- `08` first-stage diagnostics
- `09` regularity dashboard

Outputs are written under `results/neural_demand/dominicks`.

## Model Taxonomy (Current Names)

Core model classes live in `src/models`:

- `StaticND` / `StaticND_FE`: static neural demand (with optional store fixed effects).
- `HabitND` / `HabitND_FE`: habit-augmented neural demand with fixed delta.
- `WindowND`: window-history neural demand model.
- Linear demand baselines via `linear_irl.py` and `linear_features.py`.
- Classical baselines: `LAAIDS`, `QUAIDS`, `SeriesDemand`, `BLP` variants.

Control-function variants are created by setting `n_cf` in `StaticND` / `HabitND` classes and passing first-stage residuals during training/evaluation.

## Caching and Reproducibility

- Both runners precompute seed-level results once and pass them into experiment modules to avoid retraining within a run.
- `--load` enables cache loading.
- Cache loading is strict exact-tag matching (legacy fallback aliases were removed).
- Without `--load`, models are retrained and overwritten.

## Repository Layout

```text
.
├── run_neural_demand_simulation.py
├── run_neural_demand_dominicks.py
├── experiments/
│   ├── simulation/
│   │   ├── exp01_dgp_recovery.py
│   │   ├── exp02_habit_advantage.py
│   │   ├── exp03_delta_identification.py
│   │   ├── exp04_cf_endogeneity.py
│   │   └── utils.py
│   └── dominicks/
│       ├── data.py
│       ├── exp01_main_runs.py
│       ├── exp01_predictive_accuracy.py
│       ├── exp02_elasticities.py
│       ├── exp03_welfare.py
│       ├── exp04_demand_curves.py
│       ├── exp05_delta_identification.py
│       ├── exp06_cf_decomposition.py
│       ├── exp07_first_stage.py
│       ├── exp09_regularity_dashboard.py
│       └── utils.py
├── src/models/
└── results/neural_demand/
```

## Citation
If you use this code, please cite:

Grzeskiewicz, M. (2026). Neural Demand Estimation with Habit Formation and Rationality Constraints. arXiv. 