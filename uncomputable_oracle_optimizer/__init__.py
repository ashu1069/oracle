"""
Uncomputable Optimizer - A Bayesian optimization framework for handling computationally expensive
and potentially uncomputable objective functions.
"""

from .core.optimization import run_uncomputable_optimizer
from .core.simulator import uncomputable_oracle, collect_initial_data
from .core.brain import (
    initialize_gp_regressor,
    initialize_gp_classifier,
    train_gp_regressor,
    train_gp_classifier,
    plot_models_and_data
)
from .core.decision_maker import (
    calculate_expected_improvement,
    calculate_risk_aware_acquisition
)

__version__ = "0.1.0"

__all__ = [
    "run_uncomputable_optimizer",
    "uncomputable_oracle",
    "collect_initial_data",
    "initialize_gp_regressor",
    "initialize_gp_classifier",
    "train_gp_regressor",
    "train_gp_classifier",
    "plot_models_and_data",
    "calculate_expected_improvement",
    "calculate_risk_aware_acquisition",
]
