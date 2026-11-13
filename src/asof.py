# -*- coding: utf-8 -*-
"""
MS-TRP P0-3 —— as-of 评分（day_end 语义）
==========================================
评分时点 = 每笔支付的 apply_date 当日日终。第 i 笔支付只用
「date <= apply_date_i」的图边 + 「apply_date <= apply_date_i」的支付节点 s_ts 种子，
逐评分日重建转移矩阵并跑 PPR；历史分数不依赖任何未来信息（验收判据②）。

同日多笔互不可见的落点：
  · 时序分支 s_ts 严格时间前向（rolling closed='left'，本日支付互不进入对方历史统计）；
  · 图分支采用「日终批量」语义：本日支付在日终快照中彼此可见（= 日终批量评价）。

等价性说明：全量节点集里未来支付节点在较早快照中是「孤立 + 零种子」，其行/列在
对称归一化矩阵中为零，不向历史支付扩散任何质量；故不必为每个评分日重建节点编号，
历史分数与「严格按快照重构图」完全一致（见 tests/test_future_invariance.py）。
"""
import numpy as np
import pandas as pd

from src import graph_builder, stage_weight, propagation


def score_asof_diffuse(edges, node_ids, node_stage, payments, s_ts, pay_idx,
                       rho=propagation.RHO_DEFAULT, use_stage=True):
    """
    按 as-of(day_end) 逐评分日快照 PPR，返回 (s_diffuse, deg_pay, mean_iters)。
      edges      graph_builder.build_edges 输出（src/dst 为全局整数索引，含 date）
      node_ids   list[node_id]，长度 = 节点数（src/dst 的上界）
      node_stage dict[PaymentApplication::id -> stage]，用于阶段边权
      payments   DataFrame（含 apply_date 列，已按时间升序）
      s_ts       时序风险，与 payments 等长、同序
      pay_idx    支付节点在全局图中的整数索引（与 payments 等长）
    s_diffuse / deg_pay 与 payments 等长、同序；历史分数只依赖其评分日可见信息。
    """
    n_nodes = len(node_ids)
    n_pay = len(payments)
    assert (np.asarray(pay_idx) >= 0).all(), "存在未入图的支付节点"

    static, timed = graph_builder.split_edges_by_time(edges)

    # 静态边：decay 恒 1，权重 = a·b，只算一次
    if static:
        a_s, _beta_s, b_s, _ = stage_weight.prepare_edge_weights(
            static, node_stage, use_stage=use_stage, node_ids=node_ids)
        s_src = np.array([e["src"] for e in static], dtype=np.int64)
        s_dst = np.array([e["dst"] for e in static], dtype=np.int64)
        s_w = a_s * b_s
    else:
        s_src = np.array([], dtype=np.int64)
        s_dst = np.array([], dtype=np.int64)
        s_w = np.array([], dtype=float)

    # 时变边：split_edges_by_time 已按 date 升序，date_days 单调不减
    if timed:
        a_t, beta_t, b_t, date_t = stage_weight.prepare_edge_weights(
            timed, node_stage, use_stage=use_stage, node_ids=node_ids)
        t_src = np.array([e["src"] for e in timed], dtype=np.int64)
        t_dst = np.array([e["dst"] for e in timed], dtype=np.int64)
    else:
        a_t = beta_t = b_t = date_t = np.array([], dtype=float)
        t_src = t_dst = np.array([], dtype=np.int64)

    apply_days = np.array([stage_weight.to_days(d) for d in payments["apply_date"]],
                          dtype=np.int64)
    unique_days = np.unique(apply_days)

    seed = np.zeros(n_nodes, dtype=float)
    s_diffuse = np.zeros(n_pay, dtype=float)
    deg_pay = np.zeros(n_pay, dtype=float)
    iters = []

    for day in unique_days:
        cur = np.where(apply_days == day)[0]
        cur_nodes = pay_idx[cur]
        # 1. 本日支付纳入种子（日终批量：本日支付在日终快照可见）
        seed[cur_nodes] = s_ts[cur]
        # 2. 本日可见时变边（date <= day）
        k = int(np.searchsorted(date_t, day, side="right"))
        src = np.concatenate([s_src, t_src[:k]])
        dst = np.concatenate([s_dst, t_dst[:k]])
        w_timed = stage_weight.compute_decay(a_t[:k], beta_t[:k], b_t[:k],
                                             date_t[:k], float(day))
        weights = np.concatenate([s_w, w_timed])
        # 3. 本日快照的转移矩阵 + PPR
        W, _ = propagation.build_transition_matrix_from_arrays(
            src, dst, weights, n_nodes)
        sg, it = propagation.personalized_ppr(seed, W, rho=rho)
        iters.append(it)
        # 4. 本日支付取值 + 未加权度（供可靠度门控 r_g）
        s_diffuse[cur] = sg[cur_nodes]
        deg = np.bincount(np.concatenate([src, dst]), minlength=n_nodes).astype(float)
        deg_pay[cur] = deg[cur_nodes]

    mean_iters = float(np.mean(iters)) if iters else 0.0
    return s_diffuse, deg_pay, mean_iters
