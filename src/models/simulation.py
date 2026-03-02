"""Simulation model facade with per-object module exports."""

from .hausman_iv import cf_first_stage
from .la_aids import AIDSBench, QUAIDS, SeriesDemand
from .blp_logit_iv import BLPBench
from .ces_consumer import CESConsumer
from .habit_formation_consumer import HabitFormationConsumer
from .leontief_consumer import LeontiefConsumer
from .linear_features import (
    features_good_specific,
    features_orthogonalised,
    features_shared,
)
from .linear_irl import predict_linear_irl, run_linear_irl
from .mdp_neural_irl import HabitND
from .mdp_e2e_irl import (
    WindowND,
    compute_xbar_e2e,
    build_window_features,
    train_mdp_e2e,
    train_window_irl,
)

from .neural_irl import StaticND
from .quasilinear_consumer import QuasilinearConsumer
from .simulation_train import train_neural_irl
from .stone_geary_consumer import StoneGearyConsumer

