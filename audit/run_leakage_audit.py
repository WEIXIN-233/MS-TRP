# -*- coding: utf-8 -*-
"""P0-3 未来信息泄漏审计（可复现）。

加载真实图，用固定种子 s_ts，分别对「全量支付」与「截断到 cutoff 的历史支付」
跑 as-of 扫描，比较历史支付 s_diffuse 的逐条差异。
验收判据②：追加未来支付/凭证/边后，历史评分不变（max|Δ| < tol）。

写 audit/leakage_audit.json；退出码 0 当且仅当 max_abs_diff < tol。
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import graph_builder, data, asof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leakage_audit.json")
TOL = 1e-6
CUTOFF = pd.Timestamp("2026-06-30")   # 历史=截止前，未来=截止后（2026 下半年 test）


def main():
    t0 = time.time()
    payments = data.load_payments()
    tables = graph_builder.load_graph_tables()
    node_ids, node_type, id2idx = graph_builder.build_nodes(tables)
    edges = graph_builder.build_edges(tables, id2idx)

    apply_ids = payments["apply_id"].astype(str)
    pay_idx = np.asarray(
        apply_ids.map(lambda a: id2idx.get(f"PaymentApplication::{a}", -1)).tolist(),
        dtype=np.int64)
    assert (pay_idx >= 0).all(), "存在未入图的支付节点"

    # 阶段因子对不变性无关（静态），固定 Middle；s_ts 用固定种子随机向量，
    # 只检验图分支「未来节点/边是否反向影响历史评分」这一独立性质。
    node_stage = {f"PaymentApplication::{a}": "Middle" for a in apply_ids}
    rng = np.random.RandomState(2026)
    s_ts = rng.rand(len(payments)).astype(float)

    hist_mask = (payments["apply_date"] <= CUTOFF).to_numpy()
    n_hist = int(hist_mask.sum())
    n_future = len(payments) - n_hist
    # load_payments 已按 apply_date 升序，历史支付必须是时间前缀
    assert hist_mask[:n_hist].all() and not hist_mask[n_hist:].any(), "历史支付非时间前缀"

    print(f"构图耗时 {time.time()-t0:.1f}s；历史 {n_hist} 笔 / 未来 {n_future} 笔")

    t0 = time.time()
    s_full, _, _ = asof.score_asof_diffuse(
        edges, node_ids, node_stage, payments, s_ts, pay_idx)
    print(f"全量扫描 {time.time()-t0:.1f}s")

    t0 = time.time()
    s_trunc, _, _ = asof.score_asof_diffuse(
        edges, node_ids, node_stage,
        payments[hist_mask].reset_index(drop=True),
        s_ts[hist_mask], pay_idx[hist_mask])
    print(f"截断扫描 {time.time()-t0:.1f}s")

    diff = np.abs(s_full[:n_hist] - s_trunc)
    report = {
        "cutoff": str(CUTOFF.date()),
        "n_hist": n_hist,
        "n_future": n_future,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "tolerance": TOL,
        "pass": bool(diff.max() < TOL),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
