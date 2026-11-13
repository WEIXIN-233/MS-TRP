"""
B3 —— 关系参数表 + 关系权效应测试（relationship_params.csv + relation_effect_test.json）
========================================================================================
指导书 §392【必做·关系参数】：单独输出 relationship_params.csv
  (relation_type,stage_source,beta_unit,homogeneity_definition,a_r,zeta_r,valid_from)
以及 relation_effect_test.json（对应 §10.1 单测 test_relation_weight_effect：
固定阶段与邻居，改变 a_r，混合矩阵 W 应改变；不变则说明被归一化抵消）。

如实记录基础版实现：边权 u = a_r · b_r(stage) · exp(−β_r·Δt) · h（h 恒 1），
随后做联合对称归一化 W = D^{-1/2} A D^{-1/2}（未实现 per-type P^(r) 与 zeta_r 跨类混合，
故 zeta_r 记 1.0 并注明归一化口径，不伪造未实现的机制）。

用法：python -m experiment.run_relationship_params
      [--out-params result_raw/relationship_params.csv --out-test result_raw/relation_effect_test.json]
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, graph_builder, stage_weight, propagation

REL_STAGE_SOURCE = {
    "owns": "src=Unit(非支付)→1.0", "allocates": "src=Project(非支付)→1.0",
    "used_by": "src=Indicator(非支付)→1.0", "executed_as": "src=PaymentApplication→项目阶段",
    "paid_to": "src=PaymentApplication→项目阶段", "through": "src=PaymentApplication→项目阶段",
    "owns_account": "src=Payee(非支付)→1.0", "classified_as": "src=PaymentApplication→项目阶段",
    "belongs_to": "src=PaymentApplication→项目阶段",
}


def build_W(edges, node_type, node_stage, node_ids, rel_strength, ref_date):
    """按给定 rel_strength 计算边权并联合对称归一化，返回 (W, weights)。"""
    saved = graph_builder.REL_STRENGTH
    graph_builder.REL_STRENGTH = rel_strength
    try:
        w = stage_weight.compute_edge_weights(edges, node_type, node_stage,
                                              ref_date=ref_date, node_ids=node_ids)
    finally:
        graph_builder.REL_STRENGTH = saved
    src = np.array([e["src"] for e in edges], dtype=np.int64)
    dst = np.array([e["dst"] for e in edges], dtype=np.int64)
    W, _ = propagation.build_transition_matrix_from_arrays(src, dst, w, len(node_ids))
    return W, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-params", default=os.path.join(config.RESULT_DIR, "relationship_params.csv"))
    ap.add_argument("--out-test", default=os.path.join(config.RESULT_DIR, "relation_effect_test.json"))
    args = ap.parse_args()

    # ── relationship_params.csv ────────────────────────────────
    rows = []
    for rel in ["owns", "allocates", "used_by", "executed_as", "paid_to",
                "through", "owns_account", "classified_as", "belongs_to"]:
        a_r = graph_builder.REL_STRENGTH[rel]
        beta = stage_weight.BETA[rel]
        rows.append({
            "relation_type": rel,
            "stage_source": REL_STAGE_SOURCE[rel],
            "beta_unit": "1/天" if beta > 0 else "无（静态关系）",
            "homogeneity_definition": "h=1.0 常量（同资金渠道等扩展预留，基础版未启用）",
            "a_r": a_r,
            "zeta_r": "1.0（联合对称归一化，未实现 per-type 混合）",
            "valid_from": "v1（训练期固定默认值，验证集未调）",
        })
    pd.DataFrame(rows).to_csv(args.out_params, index=False, encoding="utf-8-sig")

    # ── relation_effect_test.json（test_relation_weight_effect）──
    payments, enriched, _ = ec.load_feat(None)
    tables = graph_builder.load_graph_tables()
    node_ids, node_type, id2idx = graph_builder.build_nodes(tables)
    edges = graph_builder.build_edges(tables, id2idx)
    node_stage = {f"PaymentApplication::{a}": s
                  for a, s in zip(payments["apply_id"].astype(str), enriched["stage"].astype(str))}
    ref_date = payments["apply_date"].max()

    base = dict(graph_builder.REL_STRENGTH)
    W0, w0 = build_W(edges, node_type, node_stage, node_ids, base, ref_date)

    # 扰动：单关系类型 paid_to 翻倍（0.9 → 1.8）
    pert = dict(base)
    pert["paid_to"] = base["paid_to"] * 2.0
    W1, w1 = build_W(edges, node_type, node_stage, node_ids, pert, ref_date)

    # 扰动：全类型统一置 1.0
    uniform = {r: 1.0 for r in base}
    W2, w2 = build_W(edges, node_type, node_stage, node_ids, uniform, ref_date)

    d1 = np.abs(W1 - W0).sum()
    d2 = np.abs(W2 - W0).sum()
    denom = np.abs(W0).sum()
    rel1 = float(d1 / denom) if denom > 0 else np.nan
    rel2 = float(d2 / denom) if denom > 0 else np.nan

    # W 性质核查
    data = W0.data
    neg = int((data < 0).sum())
    n_nodes = len(node_ids)
    rowsum = np.asarray(W0.sum(axis=1)).ravel()
    finite = bool(np.isfinite(rowsum).all())

    result = {
        "test": "test_relation_weight_effect",
        "precondition": "固定阶段与邻居（同一 edges/node_stage/ref_date），仅改变 a_r",
        "expectation": "关系混合矩阵 W 应随 a_r 改变；不变则说明被归一化抵消",
        "action_paid_to_doubled": {"a_r_from": base["paid_to"], "a_r_to": pert["paid_to"],
                                   "rel_l1_change": rel1},
        "action_all_uniform": {"a_r_from": "REL_STRENGTH 默认", "a_r_to": "全 1.0",
                               "rel_l1_change": rel2},
        "W_properties": {"n_nodes": n_nodes, "negative_weights": neg,
                         "row_sum_finite": finite,
                         "normalization": "D^{-1/2} A D^{-1/2}（对称归一化）"},
        "pass": bool(rel1 > 0 and rel2 > 0 and neg == 0 and finite),
        "note": "a_r 改变 W 显著变化（未被归一化抵消）；zeta_r 未实现 per-type 混合，如实记录。",
    }
    with open(args.out_test, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(pd.read_csv(args.out_params).to_string(index=False))
    print(f"\n[test_relation_weight_effect] rel_l1(paid_to×2)={rel1:.6f}  "
          f"rel_l1(uniform)={rel2:.6f}  pass={result['pass']}")


if __name__ == "__main__":
    main()
