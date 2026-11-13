"""
基线 —— 图结构基线（DOMINANT-lite）
========================================
DOMINANT（Deep Anomaly Detection on Attributed Networks）的轻量化实现：
用图邻域聚合重建节点属性，重构残差作为异常分数。

  第 1 步：实体节点聚合其关联支付的金额（x̂ = W·x，行归一化随机游走）；
  第 2 步：支付节点再聚合关联实体的金额，得到"共享同一收款人/账户/指标
           的支付金额的邻域平均" x̂_2；
  残差 s = |x − x̂_2|，即"该笔金额偏离其图邻域平均"的程度，越大越异常。

相比 MS-TRP，本基线只用图拓扑 + 金额属性做一步重构，无多尺度时序特征、
无 Self/Peer 动态基线、无阶段感知边权、无可靠度门控，用于对照完整框架各分支的增益。
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import graph_builder

N_PROP_STEPS = 2       # 两步聚合：支付→实体→支付，等价于"共享实体的邻居平均"


def run(payments_df, tables=None):
    if tables is None:
        tables = graph_builder.load_graph_tables()
    node_ids, node_type, id2idx = graph_builder.build_nodes(tables)
    edges = graph_builder.build_edges(tables, id2idx)

    n = len(node_ids)
    # 节点属性：支付节点取 log(金额)，其余节点 0（仅作图拓扑传导）
    x = np.zeros(n, dtype=float)
    apply_ids = payments_df["apply_id"].astype(str)
    for aid, amt in zip(apply_ids, payments_df["amount_wan"].astype(float)):
        idx = id2idx.get(f"PaymentApplication::{aid}")
        if idx is not None:
            x[idx] = np.log1p(amt)

    # 无向邻接 + 行归一化（随机游走转移矩阵）
    src = np.array([e["src"] for e in edges], dtype=np.int64)
    dst = np.array([e["dst"] for e in edges], dtype=np.int64)
    v = np.ones(len(edges), dtype=float)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    val = np.concatenate([v, v])
    A = sp.csr_matrix((val, (row, col)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = 1.0 / deg[nz]
    W = (sp.diags(dinv) @ A).tocsr()

    # 两步邻域聚合重建
    xh = x.copy()
    for _ in range(N_PROP_STEPS):
        xh = W @ xh

    # 重构残差（仅支付节点），tanh 有界到 [0,1]
    pay_idx = np.asarray(
        apply_ids.map(lambda a: id2idx.get(f"PaymentApplication::{a}", -1)).tolist(),
        dtype=np.int64)
    valid = pay_idx >= 0
    resid = np.zeros(len(apply_ids), dtype=float)
    resid[valid] = np.abs(x[pay_idx[valid]] - xh[pay_idx[valid]])
    scale = np.std(resid[valid]) if valid.sum() > 0 else 1.0
    s = np.tanh(resid / max(scale, 1e-6))
    return pd.Series(s, index=payments_df.index)
