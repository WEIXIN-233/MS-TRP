# -*- coding: utf-8 -*-
"""P0-3 图快照单元测试（make_asof_snapshot / split_edges_by_time）。"""
import pandas as pd

from src import graph_builder


def _edges():
    d0, d1, d2 = (pd.Timestamp(x) for x in ["2023-01-01", "2023-01-02", "2023-01-03"])
    return [
        {"src": 0, "dst": 1, "rel": "owns", "date": None},          # 静态
        {"src": 2, "dst": 3, "rel": "paid_to", "date": d1},         # 时变
        {"src": 4, "dst": 5, "rel": "through", "date": d0},         # 时变（更早）
        {"src": 6, "dst": 7, "rel": "executed_as", "date": d2},     # 时变（最晚）
    ]


def test_split_edges_by_time():
    static, timed = graph_builder.split_edges_by_time(_edges())
    assert [e["rel"] for e in static] == ["owns"]
    # 时变边按业务日期升序（through=d0 → paid_to=d1 → executed_as=d2）
    assert [e["rel"] for e in timed] == ["through", "paid_to", "executed_as"]


def test_make_asof_snapshot_cutoff():
    edges = _edges()
    cutoff = pd.Timestamp("2023-01-01")
    rels = [e["rel"] for e in graph_builder.make_asof_snapshot(edges, cutoff)]
    assert "owns" in rels          # 静态恒可见
    assert "through" in rels       # date == cutoff 可见（含当日）
    assert "paid_to" not in rels   # date > cutoff 不可见
    assert "executed_as" not in rels


def test_make_asof_snapshot_inclusive_same_day():
    """date == cutoff 的边可见（日终批量语义）。"""
    edges = [{"src": 0, "dst": 1, "rel": "paid_to", "date": pd.Timestamp("2023-05-10")}]
    snap = graph_builder.make_asof_snapshot(edges, pd.Timestamp("2023-05-10"))
    assert len(snap) == 1
