"""
Phase 5 导出公共加载器 —— 统一读取标准数据 / 预测 / 图 / 特征，供各 run_*.py 复用。
==============================================================================
单一事实源：predictions.csv（run_main 已含全部中间信号）+ metrics.json（冻结阈值）
+ ground_truth_event / payment_apply / 图业务表。避免每个导出脚本各自重算、口径漂移。

支持 --data-seed N 切换到 data_fixed/seed{N}/（此时 predictions 应取 result_raw/seed/seed{N}/model1/）。
主库（data_fixed/budget_data_v5.db，data_seed=None）对应 result_raw/run_main/predictions.csv。
"""
import os
import sys
import json
import sqlite3

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, split, feature_engineering, temporal_baseline, graph_builder


def _pred_path(data_seed):
    if data_seed is None:
        return os.path.join(config.RESULT_DIR, "run_main", "predictions.csv")
    return os.path.join(config.RESULT_DIR, "seed", f"seed{data_seed}", "model1", "predictions.csv")


def _metrics_path(data_seed):
    if data_seed is None:
        return os.path.join(config.RESULT_DIR, "run_main", "metrics.json")
    return os.path.join(config.RESULT_DIR, "seed", f"seed{data_seed}", "model1", "metrics.json")


def set_seed(data_seed):
    """切换到指定数据种子（None=主库），返回 DB_PATH。"""
    if data_seed is not None:
        return config.set_data_seed(data_seed)
    return config.DB_PATH


def load_predictions(data_seed=None):
    """run_main 的预测帧（含 s_ts/s_g/s_gate/s_final/r_ts/r_g/... 全部中间信号）。"""
    return pd.read_csv(_pred_path(data_seed))


def load_thresholds(data_seed=None):
    """冻结阈值 thr_global / tau_fpr（验证集冻结）。"""
    with open(_metrics_path(data_seed), encoding="utf-8") as f:
        m = json.load(f)
    return {"thr_global": m.get("thr_global"), "tau_fpr": m.get("tau_fpr")}


def load_standard():
    """标准标签帧：payments / labels / events / point_labels。"""
    payments = data.load_payments()
    labels = data.load_labels()
    events = data.build_events(labels)
    point_labels = data.build_point_labels(payments, events)
    return payments, labels, events, point_labels


def load_feat(data_seed=None):
    """重跑特征管线得到完整 feat（含 baseline_mean/scale、lambda_self、peer_level 等长表字段）。"""
    set_seed(data_seed)
    payments, _, _, _ = load_standard()
    entities = data.load_entity_tables()
    enriched = temporal_baseline.enrich_payments(payments, entities["project"], entities["unit"])
    feat = temporal_baseline.compute_multi_scale_features(enriched)
    feat = temporal_baseline.compute_rhythm_features(enriched, feat)
    train_idx, _, _ = split.time_forward_split(payments)
    peer_stats = temporal_baseline.compute_peer_stats(enriched.loc[train_idx])
    feat = temporal_baseline.build_self_peer_baseline(feat, peer_stats)
    feat = temporal_baseline.compute_temporal_risk(feat)
    return payments, enriched, feat


def load_graph(data_seed=None):
    """图表 / 节点 / 边（用于 business_edges / propagation_edges / seed_diagnostics）。"""
    set_seed(data_seed)
    tables = graph_builder.load_graph_tables()
    node_ids, node_type, id2idx = graph_builder.build_nodes(tables)
    edges = graph_builder.build_edges(tables, id2idx)
    return tables, node_ids, node_type, id2idx, edges


def q(df, path):
    """便捷读取 CSV。"""
    return pd.read_csv(path)


def to_str_series(s):
    return s.astype(str).str.strip()


def event_strict_f1(payments, events, test_idx, scores, thr, delta_event=7.0):
    """
    严格一对一事件匹配的 event_f1（指导书 §8.3.1：实体一致 + 时间交叠）。
    scores: 与 payments 同索引（或同长 array）的分数。返回 match_events 的完整 dict。
    """
    import numpy as _np
    from src import event_eval as _ee

    s = _np.asarray(scores, dtype=float)
    if len(s) != len(payments):
        raise ValueError("scores 长度必须与 payments 一致")
    apply_test = payments["apply_id"].astype(str).iloc[test_idx].values
    test_set = set(apply_test)
    test_events = events[events["apply_ids"].apply(
        lambda ids: any(str(a) in test_set for a in ids))].reset_index(drop=True)
    true_ev = _ee.build_true_events(test_events)

    test_mask = _np.zeros(len(payments), dtype=bool)
    test_mask[test_idx] = True
    flagged_idx = _np.where(test_mask & (s > thr))[0]
    flagged = payments.iloc[flagged_idx].copy()
    flagged["s_final"] = s[flagged_idx]

    if len(true_ev) and len(flagged):
        alerts = _ee.build_payment_alerts(flagged)
        pred_ev = _ee.cluster_alerts(alerts, delta_event=delta_event)
        return _ee.match_events(true_ev, pred_ev)
    return {"tp": 0, "fn": int(len(true_ev)), "fp": 0,
            "event_recall": 0.0, "event_precision": 0.0, "event_f1": 0.0, "pairs": []}

