"""
MS-TRP Step 6 —— 个性化回跳风险传播（PPR）
========================================
图风险 s_g 通过个性化回跳传播得到：
    s_g^(k+1) = ρ·s_ts + (1−ρ)·W_t·s_g^k
  · s_ts 为重启向量（时序风险 + 结构关系风险，仅支付节点非零）；
  · W_t 为对称归一化邻接矩阵（D^{-1/2} A D^{-1/2}），每种关系内部按 a_r 赋权、
    全局按度数归一化，避免高连接节点仅因度数支配传播；
  · 停止条件：||s_g^(k+1)−s_g^k||_1 < 1e-6，最多 50 轮。

图分支的最终风险 s_g 是收敛后的向量在支付节点上的取值。
"""
import numpy as np
import scipy.sparse as sp

RHO_DEFAULT = 0.6      # 重启概率（验证集从 {0.2,0.4,0.6,0.8} 选择）
TOL = 1e-6
MAX_ITER = 50


def build_transition_matrix(n_nodes, edges, weights):
    """
    构建对称归一化邻接矩阵 W = D^{-1/2} A D^{-1/2}（无向，A 为加权邻接）。
    返回 scipy.sparse.csr_matrix (n_nodes × n_nodes)。
    """
    src = np.array([e["src"] for e in edges], dtype=np.int64)
    dst = np.array([e["dst"] for e in edges], dtype=np.int64)
    W, _ = build_transition_matrix_from_arrays(src, dst, weights, n_nodes)
    return W


def build_transition_matrix_from_arrays(src, dst, weights, n_nodes):
    """
    由 (src, dst, weights) 数组构建对称归一化邻接矩阵 W = D^{-1/2} A D^{-1/2}。
    返回 (W, deg)：deg 为加权度（A 行和），供归一化诊断/下游复用。
    as-of 逐快照重建时用此函数：元素级缩放（O(nnz)）比两次 D@A@D 稀疏矩阵乘更快。
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    w = np.asarray(weights, dtype=float)
    # 无向：上/下三角各一份（csr_matrix 构造时自动累加重复项）
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    val = np.concatenate([w, w])
    A = sp.csr_matrix((val, (row, col)), shape=(n_nodes, n_nodes))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = np.zeros_like(deg, dtype=float)
    nz = deg > 0
    deg_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    # W_ij = deg_i^{-1/2} · A_ij · deg_j^{-1/2}，按 CSR 内部结构做元素缩放
    W = A.copy()
    row_idx = np.repeat(np.arange(n_nodes), np.diff(W.indptr))
    W.data *= deg_inv_sqrt[row_idx] * deg_inv_sqrt[W.indices]
    return W.tocsr(), deg


def personalized_ppr(seed, W, rho=RHO_DEFAULT, max_iter=MAX_ITER, tol=TOL):
    """
    个性化回跳传播。seed: 长度 n_nodes 的重启向量（s_ts + 结构风险，已归一化）。
    返回收敛后的 s_g 向量与迭代轮数。
    """
    s = seed.copy()
    n_iters = 0
    for _ in range(max_iter):
        n_iters += 1
        s_new = rho * seed + (1.0 - rho) * (W @ s)
        delta = float(np.abs(s_new - s).sum())
        s = s_new
        if delta < tol:
            break
    return s, n_iters


def run_graph_branch(seed, node_type, id2idx, edges, weights, rho=RHO_DEFAULT):
    """
    端到端：构建转移矩阵 → PPR → 返回支付节点上的图风险。
    seed: dict[node_id -> 风险分数]（重启向量）。
    返回 dict[node_id -> s_g]（全部节点）与迭代信息。
    """
    n = len(id2idx)
    seed_vec = np.zeros(n, dtype=float)
    for nid, v in seed.items():
        idx = id2idx.get(nid)
        if idx is not None:
            seed_vec[idx] = v
    W = build_transition_matrix(n, edges, weights)
    s_g, n_iters = personalized_ppr(seed_vec, W, rho=rho)
    return {nid: float(s_g[id2idx[nid]]) for nid in id2idx}, n_iters
