"""CompuOpti - staffing collaborators on projects.

Decision Modelling project, MSc AI, CentraleSupelec 2026-27.
"""

from .instance import (load_instance, save_instance, generate_instance,
                       instance_stats, to_raw)
from .model import (OBJECTIVES, SENSE, LABELS, Solution,
                    build_model, solve, extract_solution, verify, model_size)
from .check import check_front, check_point_nondominated, search_for_missed_points
from .pareto import (dominates, pareto_filter, payoff_table,
                     epsilon_constraint, weighted_sum, front_table)

__all__ = [
    "load_instance", "save_instance", "generate_instance", "instance_stats", "to_raw",
    "OBJECTIVES", "SENSE", "LABELS", "Solution",
    "build_model", "solve", "extract_solution", "verify", "model_size",
    "dominates", "pareto_filter", "payoff_table", "check_front", "search_for_missed_points",
    "epsilon_constraint", "weighted_sum", "front_table",
]

from .viz import (plot_schedule, plot_front_3d, plot_front_projections,
                  plot_parallel, plot_method_comparison, plot_scaling)

__all__ += ["plot_schedule", "plot_front_3d", "plot_front_projections",
            "plot_parallel", "plot_method_comparison", "plot_scaling"]
