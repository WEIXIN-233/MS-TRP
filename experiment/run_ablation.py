"""
实验 2 —— 消融实验（正交：一次只改一个图信号分量）
========================================
  完整模型            MS-TRP（多尺度时序 + 阶段感知图 + 可靠度门控 + 硬关系 OR）
  Temporal-only        use_graph=False（s = s_ts，无任何图信号）
  +Rule                use_ppr=False, use_struct=False（账户一致性硬规则）
  +PPR                 use_struct=False, use_rule=False（PPR 图扩散）
  +Rule+PPR            use_struct=False（规则 + 图扩散）
  单尺度               use_multiscale=False（s_ts 只用金额长尺度，去 7日/30日）
  固定边权             use_stage_weight=False（阶段因子恒 1.0）
  固定 0.5/0.5 融合    use_gate=False（去可靠度门控）

输出点级 AUPRC（整体 + 各类别）对比表，写入 result_raw/run_ablation/。
"""
import os
import sys
import json

import numpy as np
import pandas as pd

# 让脚本在 PyCharm(脚本模式)或命令行(python -m)下都能从项目根 MS-TRP/ 导入 src、baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, pipeline

OUT_DIR = os.path.join(config.RESULT_DIR, "run_ablation")

VARIANTS = [
    ("完整模型", {}),
    ("Temporal-only(去图)", {"use_graph": False}),
    ("+Rule(账户一致性)", {"use_ppr": False, "use_struct": False}),
    ("+PPR(图扩散)", {"use_struct": False, "use_rule": False}),
    ("+Rule+PPR", {"use_struct": False}),
    ("单尺度(去多尺度)", {"use_multiscale": False}),
    ("固定边权(去阶段)", {"use_stage_weight": False}),
    ("固定0.5融合(去门控)", {"use_gate": False}),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 100)
    print("消融实验：各分支对点级 AUPRC 的贡献")
    print("=" * 100)

    rows = []
    summary = {}
    csv_rows = []
    for name, kw in VARIANTS:
        res, pred = pipeline.run_ms_trp(**kw)
        row = [name, f"{res['point_auprc']:.4f}"]
        for at in config.ANOMALY_TYPES:
            row.append(f"{res['per_type_auprc'][at]:.4f}")
        rows.append(row)
        summary[name] = {
            "overall_auprc": res["point_auprc"],
            "per_type_auprc": res["per_type_auprc"],
            "event_recall": res["event"]["event_recall"],
            "event_f1": res["event"]["event_f1"],
        }
        # 规范 ablation.csv：variant,seed,AP,F1,FAR_1000,entity_recall
        csv_rows.append({
            "variant": name,
            "seed": 2026,
            "AP": round(res["point_auprc"], 6),
            "F1": round(res["point"]["f1"], 6),
            "FAR_1000": round(res["far_per_1000"], 4),
            "entity_recall": round(res["per_type_event_recall"].get("ENTITY_AGGREGATION", 0.0), 6),
        })
        print(f"  {name:<12s} 整体 AUPRC = {res['point_auprc']:.4f}")

    headers = ["变体", "整体 AUPRC"] + config.ANOMALY_TYPES
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(headers))
    sep = "  " + "  ".join("-" * w[i] for i in range(len(headers)))
    body = "\n".join("  " + "  ".join(r[i].ljust(w[i]) for i in range(len(headers))) for r in rows)
    print("\n" + line + "\n" + sep + "\n" + body)

    with open(os.path.join(OUT_DIR, "ablation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    import pandas as _pd
    _pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, "ablation.csv"),
                                   index=False, encoding="utf-8-sig")
    print(f"\n结果已写入 {OUT_DIR}/ablation.json + ablation.csv")


if __name__ == "__main__":
    main()
