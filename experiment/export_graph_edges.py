"""
B3 —— 业务边 / 传播边导出（business_edges.csv + propagation_edges.csv）
========================================================================
业务边保持真实业务方向与源数据时间；传播边按允许的业务关系显式给出两个消息方向
（例如 paid_to 同时导出 Payment→Payee 与 Payee→Payment）。

每条边字段（指导书 §6.1）：src_id、dst_id、edge_type、relation_family、
event_time、available_at、base_weight、stage_factor、time_decay、homogeneity。

用法：python -m experiment.export_graph_edges [--data-seed 10]
     [--out-business result_raw/business_edges.csv --out-propagation result_raw/propagation_edges.csv]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, graph_builder, stage_weight, data


FAMILY = {
    "owns": "organizational", "allocates": "organizational",
    "used_by": "payment", "executed_as": "payment",
    "paid_to": "payment", "through": "payment",
    "classified_as": "categorical", "belongs_to": "categorical",
    "owns_account": "account",
}


def _row(src, dst, rel, date, a, b, beta, ref_days):
    dt = 0.0
    if beta > 0 and date is not None and pd.notna(date):
        dt = max(0.0, ref_days - stage_weight.to_days(pd.Timestamp(date)))
    decay = float(np.exp(-beta * dt))
    return {
        "src_id": src, "dst_id": dst, "edge_type": rel,
        "relation_family": FAMILY.get(rel, "other"),
        "event_time": str(pd.Timestamp(date).date()) if (date is not None and pd.notna(date)) else "",
        "available_at": str(pd.Timestamp(date).date()) if (date is not None and pd.notna(date)) else "",
        "base_weight": round(float(a), 4),
        "stage_factor": round(float(b), 4),
        "time_decay": round(decay, 6),
        "homogeneity": 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out-business", default=os.path.join(config.RESULT_DIR, "business_edges.csv"))
    ap.add_argument("--out-propagation", default=os.path.join(config.RESULT_DIR, "propagation_edges.csv"))
    args = ap.parse_args()

    tables, node_ids, node_type, id2idx, edges = ec.load_graph(args.data_seed)
    pay = data.load_payments()
    ref_days = stage_weight.to_days(pay["apply_date"].max())

    # 阶段因子：支付节点作为源头的边取该项目阶段
    enriched = None
    try:
        entities = data.load_entity_tables()
        import src.temporal_baseline as tb
        enriched = tb.enrich_payments(pay, entities["project"], entities["unit"])
        node_stage = {f"PaymentApplication::{a}": str(s)
                      for a, s in zip(pay["apply_id"].astype(str), enriched["stage"].astype(str))}
    except Exception:
        node_stage = {}

    biz_rows, prop_rows = [], []
    for e in edges:
        src_id = node_ids[e["src"]]
        dst_id = node_ids[e["dst"]]
        rel = e["rel"]
        date = e.get("date")
        a = graph_builder.REL_STRENGTH.get(rel, 1.0)
        beta = stage_weight.BETA.get(rel, 0.0)
        b = stage_weight.STAGE_FACTOR.get(node_stage.get(src_id), 1.0) if src_id.startswith("PaymentApplication") else 1.0
        biz_rows.append(_row(src_id, dst_id, rel, date, a, b, beta, ref_days))
        # 传播边：两个方向都显式登记
        prop_rows.append(_row(src_id, dst_id, rel, date, a, b, beta, ref_days))
        prop_rows.append(_row(dst_id, src_id, rel, date, a, b, beta, ref_days))

    pd.DataFrame(biz_rows).to_csv(args.out_business, index=False, encoding="utf-8-sig")
    pd.DataFrame(prop_rows).to_csv(args.out_propagation, index=False, encoding="utf-8-sig")
    print(f"[B3] 业务边 {len(biz_rows)} 条 → {args.out_business}")
    print(f"[B3] 传播边 {len(prop_rows)} 条 → {args.out_propagation}")


if __name__ == "__main__":
    main()
