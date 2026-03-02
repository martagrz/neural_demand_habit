"""Dominicks model facade with per-object module exports."""

from .blp_logit_iv import BLPLogitIV
from .dominicks_train import train_dominicks as _train
from .hausman_iv import hausman_iv, cf_first_stage
from .la_aids import LAAIDS, QUAIDS, SeriesDemand
from .linear_features import feat_good_specific, feat_orth, feat_shared
from .linear_irl import pred_lirl, run_lirl
from .mdp_neural_irl import HabitND, HabitND_FE
from .mdp_e2e_irl import (
    WindowND,
    compute_xbar_e2e,
    build_window_features,
    train_mdp_e2e,
    train_window_irl,
)

from .neural_irl import StaticND, StaticND_FE

