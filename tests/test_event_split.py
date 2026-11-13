# -*- coding: utf-8 -*-
"""异常事件完整性审计（P0-2 验收工具）。

指导书 §4.1：同一异常事件不得跨 train/val/test 集合。
学长整改判据①：事件跨集合数必须为 0。
当前数据（修复前）预期存在跨集合事件 → test_no_event_crosses_split 会 FAIL，作为失败审计记录；
P0-2 重生成数据后必须全绿。
"""
import pytest

from src import data, split


@pytest.fixture(scope="module")
def events_and_split_map():
    payments = data.load_payments()
    labels = data.load_labels()
    events = data.build_events(labels)
    tr, va, te = split.time_forward_split(payments)
    split_map = {}
    for name, ids in [("train", tr), ("val", va), ("test", te)]:
        for i in ids:
            split_map[str(payments.loc[i, "apply_id"])] = name
    return payments, events, split_map


def test_no_unknown_event_members(events_and_split_map):
    """事件成员必须都能在支付表里找到（无未知 ID）。"""
    _, events, split_map = events_and_split_map
    bad = []
    for _, ev in events.iterrows():
        for aid in ev["apply_ids"]:
            if aid not in split_map:
                bad.append((ev["event_id"], aid))
    assert not bad, f"存在 {len(bad)} 个未知支付ID成员（前几个）: {bad[:5]}"


def test_no_empty_events(events_and_split_map):
    """事件成员集合不得为空。"""
    _, events, _ = events_and_split_map
    empty = events[events["apply_ids"].map(len) == 0]
    assert len(empty) == 0, f"存在 {len(empty)} 个空事件: {empty['event_id'].tolist()[:10]}"


def test_no_event_crosses_split(events_and_split_map):
    """同一事件的所有成员必须落在同一个 split（P0-2 核心判据）。"""
    _, events, split_map = events_and_split_map
    cross = []
    for _, ev in events.iterrows():
        groups = {split_map[a] for a in ev["apply_ids"] if a in split_map}
        if len(groups) > 1:
            cross.append((ev["event_id"], ev["anomaly_type"], sorted(groups)))
    assert len(cross) == 0, (
        f"P0-2 未修复：存在 {len(cross)} 个跨集合事件。"
        f"（按类型/组合见 audit/event_split_audit.csv）"
    )
