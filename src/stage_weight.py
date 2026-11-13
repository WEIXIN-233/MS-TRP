"""
MS-TRP Step 5 —— 阶段感知边权
========================================
边权 u_ji(t) = a_r · b_r(stage_i(t)) · exp(−β_r · Δt_ji) · h_ji(t)
  · a_r       关系类型强度（graph_builder.REL_STRENGTH）
  · b_r(stage) 目标节点所处执行阶段因子（Early/Middle/Late）
  · exp(−β_r·Δt) 时间衰减（支付类边随 apply_date 距今越久权重越低）
  · h_ji      业务同质性（默认 1.0，预留同资金渠道等扩展）

所有参数（a_r、β_r、b_r）仅应在验证集确定；本文件给出可复算的默认值。
"""
import numpy as np
import pandas as pd

from src import graph_builder

# 时间衰减系数（按关系类型，单位 1/天）；静态关系无时间项
BETA = {
    "owns": 0.0, "allocates": 0.0, "used_by": 0.001,
    "executed_as": 0.001, "paid_to": 0.001, "through": 0.001,
    "owns_account": 0.0, "classified_as": 0.0, "belongs_to": 0.0,
}

# 阶段因子（执行阶段 → 边权放大系数）；Late 阶段对年末集中更敏感
STAGE_FACTOR = {"Early": 0.7, "Middle": 1.0, "Late": 1.5}


def compute_edge_weights(edges, node_type, node_stage, ref_date=None, use_stage=True,
                         node_ids=None):
    """
    计算每条边的权重 u。
      edges: graph_builder.build_edges 的输出（含 src/dst/rel/date）
      node_type: dict[node_id -> type]
      node_stage: dict[node_id -> stage]，支付类节点用其项目阶段
      node_ids: list[node_id]，与 edges 里 src/dst 的整数索引对应，用于反查端点类型
      ref_date: 参考时点（评分时点），用于时间衰减；None 时用边自身 date 距今的差距
      use_stage: False 时阶段因子恒为 1.0（消融：去阶段感知边权）
    返回与 edges 等长的 float 数组（未归一化权重）。

    阶段因子作用于「支付节点作为源头」的资金流向边（paid_to/through/executed_as 等）：
    阶段是支付所在项目的执行阶段，决定该支付资金流出的传播敏感度。此前取 dst 的阶段
    只会命中 Indicator→PaymentApplication 一条边，且被对称归一化抹平，导致消融无差异。
    """
    if ref_date is None:
        ref_date = pd.Timestamp.now()
    weights = np.zeros(len(edges), dtype=float)
    for k, e in enumerate(edges):
        a = graph_builder.REL_STRENGTH.get(e["rel"], 1.0)
        # 阶段因子：取源头节点（src）的阶段；node_stage 只含支付节点，
        # 因此仅支付节点作为源头的边（paid_to/through/executed_as/classified_as/belongs_to）
        # 受到阶段调节，其余边因子恒为 1.0。
        if use_stage and node_ids is not None:
            stage = node_stage.get(node_ids[e["src"]])
            b = STAGE_FACTOR.get(stage, 1.0)
        else:
            b = 1.0
        # 时间衰减
        beta = BETA.get(e["rel"], 0.0)
        decay = 1.0
        if beta > 0 and e.get("date") is not None and pd.notna(e["date"]):
            dt = max(0.0, (ref_date - pd.Timestamp(e["date"])).total_seconds() / 86400.0)
            decay = np.exp(-beta * dt)
        # 同质性（默认 1.0）
        h = 1.0
        weights[k] = a * b * decay * h
    return weights


# as-of 扫描用：评分时点 → 自 1970-01-01 的天数（日粒度，day_end 语义）
EPOCH = pd.Timestamp("1970-01-01")


def to_days(ts):
    """把 Timestamp/日期归一化到「自 1970-01-01 的天数」（整数），用于快照截止比较。"""
    return (pd.Timestamp(ts).normalize() - EPOCH).days


def prepare_edge_weights(edges, node_stage, use_stage=True, node_ids=None):
    """
    预计算与评分时点无关的每边常量（as-of 扫描只算一次）：
      a         关系类型强度 REL_STRENGTH
      beta      时间衰减系数 BETA
      b         源头节点阶段因子（use_stage=False 时恒 1）
      date_days 边业务日距 1970-01-01 的天数（静态边为 NaN）
    edges 需为 graph_builder.build_edges 输出（含 src/dst/rel/date）。
    """
    n = len(edges)
    a = np.array([graph_builder.REL_STRENGTH.get(e["rel"], 1.0) for e in edges], dtype=float)
    beta = np.array([BETA.get(e["rel"], 0.0) for e in edges], dtype=float)
    b = np.ones(n, dtype=float)
    if use_stage and node_ids is not None:
        for k, e in enumerate(edges):
            stage = node_stage.get(node_ids[e["src"]])
            b[k] = STAGE_FACTOR.get(stage, 1.0)
    date_days = np.full(n, np.nan, dtype=float)
    for k, e in enumerate(edges):
        d = e.get("date")
        if d is not None and pd.notna(d):
            date_days[k] = to_days(d)
    return a, beta, b, date_days


def compute_decay(a, beta, b, date_days, ref_date_days):
    """
    给定评分时点 ref_date_days（自 1970 的天数），计算边权 a·b·exp(−β·Δt)。
    静态边（date_days=NaN、beta=0）decay 恒 1；Δt 裁剪到 [0,∞)。
    返回与输入等长的权重数组。
    """
    age = np.where(np.isnan(date_days), 0.0,
                   np.maximum(ref_date_days - date_days, 0.0))
    return a * b * np.exp(-beta * age)
