"""
指导书 B5/Q2 九行强制基线 + concat_if 消融基线
================================================
分数越大越异常，接口统一：score(payments_df, ref_stats, train_idx) -> pd.Series(index=payments_df.index)。
评分时点语义与主模型一致（严格时间前向：rolling 窗口只回看 apply_date 之前的数据）。

九行强制基线（B5 表格中必须逐行出现）：
  random_uniform              U(0,1) 固定种子随机猜测（期望 AUPRC≈阳性率，随机下界）
  constant_score              全样本常数 0.5（无判别力，AUPRC=阳性率，排序退化）
  shared_payee_count          前 30 日同收款人跨项目数（Shared Payee 信号，纯计数）
  shared_account_concentration 前 30 日同账户跨收款人数（Shared Account 信号，纯计数）
  ecod / copod / mcd / cblof  PyOD 无监督密度/统计离群基线（特征=时序特征矩阵）
  simple_ppr                  同种子拓扑 + 均匀边权 PPR（重启向量=log金额）

B7 消融：
  concat_if                   时序特征 ⊕ 图邻域聚合金额，接 IsolationForest
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import feature_engineering, graph_builder, propagation
from baselines._common import _impute


# ── 1/2. 随机下界与常数基线 ─────────────────────────────────────────
def random_uniform_scores(payments_df, ref_stats=None, train_idx=None, seed=2026):
    """U(0,1) 固定种子。期望 AUPRC = 阳性率（随机下界参照）。"""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.uniform(0.0, 1.0, len(payments_df)), index=payments_df.index)


def constant_score_scores(payments_df, ref_stats=None, train_idx=None):
    """全样本常数 0.5。排序无信息，AUPRC = 阳性率，用作排序下界。"""
    return pd.Series(0.5, index=payments_df.index)


# ── 3/4. 前 30 日共享实体计数（纯业务计数，无泄漏）─────────────────
def _rolling_distinct(payments_df, entity_col, value_col, window_days):
    """对每笔支付，统计同一 entity_col 在过去 window_days 天（含当日）出现的
    不同 value_col 数量。两指针滑动窗口，O(n log n)。分数=不同值个数。"""
    n = len(payments_df)
    scores = np.zeros(n, dtype=float)
    date_arr = pd.to_datetime(payments_df["apply_date"]).values.astype("datetime64[ns]")
    ent = payments_df[entity_col].values
    val = payments_df[value_col].values
    win = np.timedelta64(window_days, "D")

    # 按 entity 分组，组内按日期升序做两指针
    groups = {}
    for i, e in enumerate(ent):
        if pd.isna(e):
            continue
        groups.setdefault(e, []).append(i)
    from collections import Counter
    for positions in groups.values():
        pos = sorted(positions, key=lambda p: date_arr[p])
        cnt = Counter()
        j = 0
        for k, i in enumerate(pos):
            if pd.notna(val[i]):
                cnt[val[i]] += 1
            while j <= k and (date_arr[i] - date_arr[pos[j]]) > win:
                vj = val[pos[j]]
                if pd.notna(vj):
                    cnt[vj] -= 1
                    if cnt[vj] <= 0:
                        del cnt[vj]
                j += 1
            scores[i] = len(cnt)
    return pd.Series(scores, index=payments_df.index)


def shared_payee_count_scores(payments_df, ref_stats=None, train_idx=None, window_days=30):
    """前 30 日同收款人跨项目数。跨项目越多越可疑（Shared Payee / 关联聚集）。"""
    return _rolling_distinct(payments_df, "payee_id", "project_id", window_days)


def shared_account_concentration_scores(payments_df, ref_stats=None, train_idx=None, window_days=30):
    """前 30 日同账户跨收款人数。多收款人共用一个账户越集中越可疑（Shared Account）。"""
    return _rolling_distinct(payments_df, "account_id", "payee_id", window_days)


# ── 5~8. PyOD 无监督基线（特征=时序特征矩阵，train 拟合 / test transform）──
def _pyod_score(payments_df, ref_stats, train_idx, Model, **kwargs):
    F = _impute(feature_engineering.build_feature_matrix(payments_df, ref_stats))
    X = F.values
    clf = Model(**kwargs)
    clf.fit(X[train_idx])
    return pd.Series(clf.decision_function(X), index=payments_df.index)


def ecod_scores(payments_df, ref_stats, train_idx):
    from pyod.models.ecod import ECOD
    return _pyod_score(payments_df, ref_stats, train_idx, ECOD)


def copod_scores(payments_df, ref_stats, train_idx):
    from pyod.models.copod import COPOD
    return _pyod_score(payments_df, ref_stats, train_idx, COPOD)


def cblof_scores(payments_df, ref_stats, train_idx):
    from pyod.models.cblof import CBLOF
    return _pyod_score(payments_df, ref_stats, train_idx, CBLOF,
                       n_clusters=8, contamination=0.1, random_state=2026)


def mcd_scores(payments_df, ref_stats, train_idx):
    from pyod.models.mcd import MCD
    return _pyod_score(payments_df, ref_stats, train_idx, MCD,
                       contamination=0.1)


# ── 9. simple_ppr：同种子拓扑 + 均匀边权 PPR ───────────────────────
def simple_ppr_scores(payments_df, ref_stats=None, train_idx=None, rho=propagation.RHO_DEFAULT):
    """同主模型拓扑 + 均匀边权（去 REL_STRENGTH / 阶段因子）的 PPR。
    重启向量 = 支付金额 log1p（自包含，不依赖 MS-TRP 时序分支）。
    分数 = 收敛后的 PPR 值在支付节点上的取值。"""
    tables = graph_builder.load_graph_tables()
    node_ids, node_type, id2idx = graph_builder.build_nodes(tables)
    edges = graph_builder.build_edges(tables, id2idx)
    n = len(node_ids)

    seed = np.zeros(n, dtype=float)
    apply_ids = payments_df["apply_id"].astype(str)
    for aid, amt in zip(apply_ids, payments_df["amount_wan"].astype(float)):
        idx = id2idx.get(f"PaymentApplication::{aid}")
        if idx is not None:
            seed[idx] = np.log1p(amt)

    weights = np.ones(len(edges), dtype=float)   # 均匀边权
    W = propagation.build_transition_matrix(n, edges, weights)
    s_g, _ = propagation.personalized_ppr(seed, W, rho=rho)

    pay_idx = np.asarray(
        [id2idx.get(f"PaymentApplication::{a}", -1) for a in apply_ids], dtype=np.int64)
    valid = pay_idx >= 0
    out = np.zeros(len(apply_ids), dtype=float)
    out[valid] = s_g[pay_idx[valid]]
    return pd.Series(out, index=payments_df.index)


# ── B7 消融：时序特征 ⊕ 图邻域聚合 → IF ─────────────────────────────
def _graph_neighbor_amount(payments_df, n_steps=2):
    """图邻域聚合金额：对 log(金额) 做 n_steps 步行归一化邻域聚合，取支付节点值。
    纯图拓扑信息（均匀边权），供 concat_if 拼接为「评分前图聚合」特征。"""
    tables = graph_builder.load_graph_tables()
    node_ids, node_type, id2idx = graph_builder.build_nodes(tables)
    edges = graph_builder.build_edges(tables, id2idx)
    n = len(node_ids)

    x = np.zeros(n, dtype=float)
    apply_ids = payments_df["apply_id"].astype(str)
    for aid, amt in zip(apply_ids, payments_df["amount_wan"].astype(float)):
        idx = id2idx.get(f"PaymentApplication::{aid}")
        if idx is not None:
            x[idx] = np.log1p(amt)

    src = np.array([e["src"] for e in edges], dtype=np.int64)
    dst = np.array([e["dst"] for e in edges], dtype=np.int64)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    val = np.ones(len(edges) * 2, dtype=float)
    A = sp.csr_matrix((val, (row, col)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.zeros_like(deg, dtype=float)
    nz = deg > 0
    dinv[nz] = 1.0 / deg[nz]
    W = (sp.diags(dinv) @ A).tocsr()

    xh = x.copy()
    for _ in range(n_steps):
        xh = W @ xh

    pay_idx = np.asarray(
        [id2idx.get(f"PaymentApplication::{a}", -1) for a in apply_ids], dtype=np.int64)
    valid = pay_idx >= 0
    out = np.zeros(len(apply_ids), dtype=float)
    out[valid] = xh[pay_idx[valid]]
    return out


def concat_if_scores(payments_df, ref_stats, train_idx, seed=2026):
    """时序特征矩阵 ⊕ 两步图邻域聚合金额 → IsolationForest。"""
    from sklearn.ensemble import IsolationForest
    F = _impute(feature_engineering.build_feature_matrix(payments_df, ref_stats))
    F = F.copy()
    F["f_graph_nb_amount"] = _graph_neighbor_amount(payments_df, n_steps=2)
    X = F.values
    clf = IsolationForest(random_state=seed, n_estimators=200, n_jobs=1)
    clf.fit(X[train_idx])
    return pd.Series(-clf.score_samples(X), index=payments_df.index)
