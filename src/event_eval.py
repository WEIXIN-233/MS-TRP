# -*- coding: utf-8 -*-
"""
MS-TRP P0-4 —— 事件级严格一对一评估
========================================
把点级报警聚成「报警事件」（同实体 + 时间间隔 ≤ Δ_event），再与真实事件做
「实体一致 + 时间交叠」的二分图最大匹配（一对一，杜绝重复命中虚高）。

指导书 §4.2：事件级 P = TP/(TP+FP)、R = TP/(TP+FN)。
  TP = 成功一对一匹配的真实事件数；FP = 未匹配的预测事件数；FN = 未匹配的真实事件数。
Δ_event 仅验证集确定；本文件给出可复算的默认实现。
"""
import numpy as np
import pandas as pd

# 事件 target_entity_type → 支付表上对应的实体字段（用于把支付映射到事件实体）
ENTITY_COL = {
    "PAYMENT": "apply_id",
    "ACCOUNT": "account_id",
    "PAYEE": "payee_id",
    "INDICATOR_DETAIL": "detail_id",
    "INDICATOR": "indicator_id",
    "UNIT": "unit_name",
}

# 报警事件聚类时间窗（天）：同实体相邻报警间隔 ≤ Δ_event 合并为一簇；验证集可调，默认 7 天
DELTA_EVENT_DEFAULT = 7.0


def _overlap(s1, e1, s2, e2):
    return s1 <= e2 and s2 <= e1


def _max_bipartite_matching(adj, n_left, n_right):
    """Kuhn 增广路求二分图最大匹配。adj[i] = 与左节点 i 相邻的右节点列表。
    返回 (match_right, tp)：match_right[j] = 匹配到右节点 j 的左节点（-1 未匹配）。"""
    match_right = [-1] * n_right

    def try_kuhn(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_right[v] == -1 or try_kuhn(match_right[v], seen):
                match_right[v] = u
                return True
        return False

    tp = 0
    for u in range(n_left):
        if try_kuhn(u, [False] * n_right):
            tp += 1
    return match_right, tp


def cluster_alerts(alerts, delta_event):
    """
    点级报警 → 报警事件。alerts: DataFrame，至少含 apply_date / entity_key；
    可选 apply_id（透传用于审计）。同 entity_key 且 apply_date 间隔 ≤ delta_event 天
    的报警合并为一簇。返回 DataFrame：entity_key / start / end / n_alerts / apply_ids。
    """
    delta = pd.Timedelta(days=float(delta_event))
    out = []
    for key, grp in alerts.groupby("entity_key", dropna=False):
        grp = grp.sort_values("apply_date")
        dates = pd.to_datetime(grp["apply_date"]).tolist()
        aids = (grp["apply_id"].astype(str).tolist() if "apply_id" in grp.columns
                else [None] * len(grp))
        runs = []
        for d, aid in zip(dates, aids):
            if runs and (d - runs[-1][1]) <= delta:
                runs[-1] = (runs[-1][0], d, runs[-1][2] + [aid])
            else:
                runs.append((d, d, [aid]))
        for cid, (s, e, ids) in enumerate(runs):
            out.append({"entity_key": key, "start": s, "end": e,
                        "n_alerts": len(ids), "apply_ids": ids})
    return pd.DataFrame(out)


def match_events(true_events, pred_events):
    """
    真实事件与预测事件的一对一最大匹配。
    true_events / pred_events: DataFrame，列 entity_key / start / end。
    匹配条件：entity_key 一致 且 时间区间交叠。
    返回 dict：tp / fn / fp / event_recall / event_precision / event_f1 / pairs。
    """
    t_key = true_events["entity_key"].astype(str).tolist()
    t_s = pd.to_datetime(true_events["start"]).tolist()
    t_e = pd.to_datetime(true_events["end"]).tolist()
    p_key = pred_events["entity_key"].astype(str).tolist()
    p_s = pd.to_datetime(pred_events["start"]).tolist()
    p_e = pd.to_datetime(pred_events["end"]).tolist()

    n_t = len(true_events)
    n_p = len(pred_events)
    adj = [[] for _ in range(n_t)]
    for i in range(n_t):
        for j in range(n_p):
            if t_key[i] == p_key[j] and _overlap(t_s[i], t_e[i], p_s[j], p_e[j]):
                adj[i].append(j)

    match_right, tp = _max_bipartite_matching(adj, n_t, n_p)
    fn = n_t - tp
    fp = n_p - tp
    rec = tp / n_t if n_t else 0.0
    prec = tp / n_p if n_p else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0
    pairs = [(u, j) for j, u in enumerate(match_right) if u != -1]
    return {"tp": tp, "fn": fn, "fp": fp,
            "event_recall": rec, "event_precision": prec, "event_f1": f1,
            "pairs": pairs}


def build_true_events(events):
    """
    真实事件 → 严格评估帧（entity_key / start / end）。
    直接用 ground_truth_event 的 target_entity_type + target_entity_id + start_time/end_time
    作为事件权威身份；缺失实体 id 或时间的事件跳过。
    """
    rows = []
    for _, ev in events.iterrows():
        et = ev.get("target_entity_type")
        eid = ev.get("target_entity_id")
        st = ev.get("start_time")
        en = ev.get("end_time")
        if et is None or eid is None or (isinstance(eid, float) and pd.isna(eid)):
            continue
        if st is None or en is None or pd.isna(st) or pd.isna(en):
            continue
        rows.append({"entity_key": f"{et}::{eid}",
                     "start": pd.Timestamp(st), "end": pd.Timestamp(en)})
    return pd.DataFrame(rows)


def build_payment_alerts(flagged_payments, entity_types=None):
    """
    把被报警的支付展开为 (支付, 实体类型) 行，供 cluster_alerts 按实体聚类。
    flagged_payments: DataFrame，含 apply_id/apply_date 及实体列（见 ENTITY_COL）。
    entity_types: 只展开这些实体类型（默认全部）。
    返回 DataFrame：apply_id / apply_date / entity_key。
    """
    if entity_types is None:
        entity_types = list(ENTITY_COL.keys())
    rows = []
    for _, p in flagged_payments.iterrows():
        for et in entity_types:
            col = ENTITY_COL.get(et)
            if col is None:
                continue
            val = p.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            s = str(val)
            if s in ("", "nan", "None"):
                continue
            rows.append({"apply_id": str(p["apply_id"]),
                         "apply_date": p["apply_date"],
                         "entity_key": f"{et}::{s}"})
    return pd.DataFrame(rows)
