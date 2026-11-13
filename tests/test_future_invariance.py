# -*- coding: utf-8 -*-
"""P0-3 未来信息不变性验收测试。

学长整改判据：追加未来支付/凭证/关系后，历史评分逐条不变（数值容差内）。
用小型合成异构图验证 as-of 扫描：对同一批历史支付，额外追加「未来支付 + 其边」后，
历史支付的图扩散分数 s_diffuse 必须不变。
"""
import numpy as np
import pandas as pd

from src import asof


def _build_graph(include_future=True):
    """小型合成图：2 收款人 + 3 历史支付 + 1 未来支付（可选）。

    节点顺序固定，未来支付永远排在最后，保证历史节点索引不因追加而移动。"""
    node_ids = [
        "Payee::p0", "Payee::p1",
        "PaymentApplication::a0", "PaymentApplication::a1",
        "PaymentApplication::a2", "PaymentApplication::a3",
    ]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    d0, d1, d2, d3 = (pd.Timestamp(x) for x in
                      ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])

    edges = [
        {"src": idx["PaymentApplication::a0"], "dst": idx["Payee::p0"],
         "rel": "paid_to", "date": d0},
        {"src": idx["PaymentApplication::a1"], "dst": idx["Payee::p0"],
         "rel": "paid_to", "date": d1},
        {"src": idx["PaymentApplication::a2"], "dst": idx["Payee::p1"],
         "rel": "paid_to", "date": d2},
        # 静态关系边：date=None，恒可见
        {"src": idx["Payee::p0"], "dst": idx["Payee::p1"],
         "rel": "owns_account", "date": None},
    ]
    dates = [d0, d1, d2]
    ids = ["a0", "a1", "a2"]
    if include_future:
        edges.append({"src": idx["PaymentApplication::a3"], "dst": idx["Payee::p1"],
                      "rel": "paid_to", "date": d3})
        dates.append(d3)
        ids.append("a3")

    payments = pd.DataFrame({"apply_date": dates, "apply_id": ids})
    s_ts = np.array([0.9, 0.1, 0.8, 0.2][:len(ids)], dtype=float)
    pay_idx = np.array([idx[f"PaymentApplication::{a}"] for a in ids], dtype=np.int64)
    node_stage = {f"PaymentApplication::{a}": "Middle" for a in ids}
    return node_ids, edges, payments, s_ts, pay_idx, node_stage


def test_future_payment_does_not_change_history():
    """追加未来支付 + 其边后，历史支付 s_diffuse 逐条不变。"""
    node_ids, edges_hist, pay_hist, s_hist, idx_hist, stage_hist = _build_graph(
        include_future=False)
    _, edges_full, pay_full, s_full, idx_full, stage_full = _build_graph(
        include_future=True)

    s_hist_only, _, _ = asof.score_asof_diffuse(
        edges_hist, node_ids[:-1], stage_hist, pay_hist, s_hist, idx_hist)
    s_with_future, _, _ = asof.score_asof_diffuse(
        edges_full, node_ids, stage_full, pay_full, s_full, idx_full)

    np.testing.assert_allclose(s_with_future[:3], s_hist_only, rtol=1e-9, atol=1e-12)


def test_future_rel_does_not_change_history():
    """追加「未来日期」的关系边（rel）后，历史支付评分不变。"""
    node_ids, edges_hist, pay_hist, s_hist, idx_hist, stage_hist = _build_graph(
        include_future=False)
    # 在历史图上追加一条「未来日期的关系边」：a2(node_ids[4]) 在更晚日期新增到 p0(0) 的边
    edges_future = list(edges_hist) + [
        {"src": 4, "dst": 0, "rel": "paid_to",
         "date": pd.Timestamp("2023-01-05")}]
    s_hist_only, _, _ = asof.score_asof_diffuse(
        edges_hist, node_ids[:-1], stage_hist, pay_hist, s_hist, idx_hist)
    s_with_future_rel, _, _ = asof.score_asof_diffuse(
        edges_future, node_ids[:-1], stage_hist, pay_hist, s_hist, idx_hist)
    np.testing.assert_allclose(s_with_future_rel, s_hist_only, rtol=1e-9, atol=1e-12)


def test_snapshot_output_shape():
    node_ids, edges, payments, s_ts, pay_idx, node_stage = _build_graph(
        include_future=True)
    s_diffuse, deg_pay, mean_iters = asof.score_asof_diffuse(
        edges, node_ids, node_stage, payments, s_ts, pay_idx)
    assert s_diffuse.shape == (len(payments),)
    assert deg_pay.shape == (len(payments),)
    assert np.isfinite(s_diffuse).all()
    assert mean_iters > 0
