"""基线方法包：重新导出打分函数（分数越大越异常）。"""
from .fixed_rule import fixed_rules_scores
from .robust_mad import robust_stats_scores
from .isolation_forest import isolation_forest_scores
from .lof import lof_scores
from .forecast_residual import residual_scores
from .graph_baseline import run as graph_scores
from .forced import (
    random_uniform_scores,
    constant_score_scores,
    shared_payee_count_scores,
    shared_account_concentration_scores,
    ecod_scores,
    copod_scores,
    cblof_scores,
    mcd_scores,
    simple_ppr_scores,
    concat_if_scores,
)

__all__ = [
    "fixed_rules_scores",
    "robust_stats_scores",
    "isolation_forest_scores",
    "lof_scores",
    "residual_scores",
    "graph_scores",
    "random_uniform_scores",
    "constant_score_scores",
    "shared_payee_count_scores",
    "shared_account_concentration_scores",
    "ecod_scores",
    "copod_scores",
    "cblof_scores",
    "mcd_scores",
    "simple_ppr_scores",
    "concat_if_scores",
]
