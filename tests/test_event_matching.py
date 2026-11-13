# -*- coding: utf-8 -*-
"""P0-4 事件级一对一匹配单元测试（match_events / cluster_alerts）。

学长验收：3 真 4 预测事件（含重复命中），断言一对一匹配全对。
"""
import pandas as pd

from src import event_eval


def _ts(s):
    return pd.Timestamp(s)


def test_match_events_one_to_one_with_duplicate():
    """3 真 4 预测：P3 与 P0 重复命中同一真事件 → 一对一匹配只算一次。"""
    true = pd.DataFrame({
        "entity_key": ["A", "B", "C"],
        "start": [_ts("2023-01-01"), _ts("2023-02-01"), _ts("2023-03-01")],
        "end":   [_ts("2023-01-05"), _ts("2023-02-05"), _ts("2023-03-05")],
    })
    pred = pd.DataFrame({
        "entity_key": ["A", "B", "C", "A"],           # 第 4 个与第 1 个重复命中 A
        "start": [_ts("2023-01-01"), _ts("2023-02-01"), _ts("2023-03-01"), _ts("2023-01-02")],
        "end":   [_ts("2023-01-03"), _ts("2023-02-03"), _ts("2023-03-03"), _ts("2023-01-04")],
    })
    r = event_eval.match_events(true, pred)
    assert r["tp"] == 3 and r["fn"] == 0 and r["fp"] == 1
    assert abs(r["event_recall"] - 1.0) < 1e-9
    assert abs(r["event_precision"] - 0.75) < 1e-9
    assert abs(r["event_f1"] - 2 * 1.0 * 0.75 / 1.75) < 1e-9
    assert len(r["pairs"]) == 3


def test_match_events_entity_mismatch():
    """实体不一致不匹配。"""
    true = pd.DataFrame({"entity_key": ["A"],
                         "start": [_ts("2023-01-01")], "end": [_ts("2023-01-05")]})
    pred = pd.DataFrame({"entity_key": ["B"],
                         "start": [_ts("2023-01-01")], "end": [_ts("2023-01-05")]})
    r = event_eval.match_events(true, pred)
    assert r["tp"] == 0 and r["fn"] == 1 and r["fp"] == 1


def test_match_events_time_no_overlap():
    """实体一致但时间不交叠不匹配。"""
    true = pd.DataFrame({"entity_key": ["A"],
                         "start": [_ts("2023-01-01")], "end": [_ts("2023-01-05")]})
    pred = pd.DataFrame({"entity_key": ["A"],
                         "start": [_ts("2023-02-01")], "end": [_ts("2023-02-05")]})
    r = event_eval.match_events(true, pred)
    assert r["tp"] == 0


def test_cluster_alerts_merges_same_entity_within_delta():
    alerts = pd.DataFrame({
        "apply_id": ["p1", "p2", "p3", "p4"],
        "apply_date": [_ts("2023-01-01"), _ts("2023-01-03"), _ts("2023-01-10"), _ts("2023-02-01")],
        "entity_key": ["A", "A", "A", "B"],
    })
    cl = event_eval.cluster_alerts(alerts, delta_event=7)
    # A: p1/p2 间隔 2 天合并；p3 距 p2 7 天（=7，含）也合并 → A 一簇；B 单独一簇
    assert len(cl) == 2
    a = cl[cl["entity_key"] == "A"].iloc[0]
    assert a["n_alerts"] == 3
    assert set(a["apply_ids"]) == {"p1", "p2", "p3"}


def test_build_payment_alerts():
    pay = pd.DataFrame({
        "apply_id": ["x1"],
        "apply_date": [_ts("2023-01-01")],
        "account_id": ["ACC01"],
        "payee_id": ["PY01"],
    })
    al = event_eval.build_payment_alerts(pay, entity_types=["ACCOUNT", "PAYEE"])
    assert len(al) == 2
    assert set(al["entity_key"]) == {"ACCOUNT::ACC01", "PAYEE::PY01"}


def test_build_true_events():
    """真实事件 → 严格评估帧：直接用 target_entity_type/id/time；缺失实体 id 跳过。"""
    events = pd.DataFrame({
        "target_entity_type": ["ACCOUNT", "PAYEE", "UNIT"],
        "target_entity_id": ["ACCT001", "PY001", None],
        "start_time": ["2023-01-01", "2023-02-01", "2023-03-01"],
        "end_time": ["2023-01-05", "2023-02-05", "2023-03-05"],
    })
    te = event_eval.build_true_events(events)
    assert len(te) == 2
    assert set(te["entity_key"]) == {"ACCOUNT::ACCT001", "PAYEE::PY001"}
