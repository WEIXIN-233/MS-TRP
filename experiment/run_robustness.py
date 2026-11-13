"""
实验 3 —— 鲁棒性实验
========================================
  缺失 10% / 30%   随机丢弃该比例的训练支付（模拟历史数据缺失）
  删边 20%         随机删除该比例的图边（模拟关系数据缺失）
  冷启动           只评估"指标历史笔数 = 0"的测试支付（首笔，无自身历史）

观察：完整框架在上述扰动下点级 AUPRC 的下降是否平缓。
输出 result_raw/run_robustness/robustness.json。
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# 让脚本在 PyCharm(脚本模式)或命令行(python -m)下都能从项目根 MS-TRP/ 导入 src、baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, pipeline

OUT_DIR = os.path.join(config.RESULT_DIR, "run_robustness")

VARIANTS = [
    ("完整(基线)", {}),
    ("缺失10%", {"train_drop_ratio": 0.10}),
    ("缺失30%", {"train_drop_ratio": 0.30}),
    ("删边20%", {"edge_drop_ratio": 0.20}),
]


def _report(name, res):
    return {
        "overall_auprc": round(res["point_auprc"], 4),
        "per_type_auprc": {k: round(v, 4) for k, v in res["per_type_auprc"].items()},
        "event_recall": round(res["event"]["event_recall"], 4),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 100)
    print("鲁棒性实验")
    print("=" * 100)

    summary = {}
    csv_rows = []
    full_pred = None
    for name, kw in VARIANTS:
        res, pred = pipeline.run_ms_trp(**kw)
        summary[name] = _report(name, res)
        csv_rows.append({
            "condition": name,
            "seed": 2026,
            "overall_auprc": round(res["point_auprc"], 6),
            "event_recall": round(res["event"]["event_recall"], 6),
            "entity_recall": round(res["per_type_event_recall"].get("ENTITY_AGGREGATION", 0.0), 6),
            "n_cold": None, "n_warm": None, "warm_auprc": None,
        })
        if name == "完整(基线)":
            full_pred = pred
        print(f"  {name:<10s} 整体 AUPRC = {res['point_auprc']:.4f}  "
              f"事件召回 = {res['event']['event_recall']:.4f}")

    # 冷启动：指标历史笔数为 0 的测试支付
    test = full_pred[full_pred["split"] == "test"]
    cold = test[test["hist_count"] == 0]
    warm = test[test["hist_count"] > 0]
    if len(cold) > 0:
        cold_auprc = average_precision_score(cold["is_anomaly"], cold["s_final"])
        warm_auprc = average_precision_score(warm["is_anomaly"], warm["s_final"])
        summary["冷启动(hist=0)"] = {
            "overall_auprc": round(cold_auprc, 4),
            "n_cold": int(len(cold)),
            "n_warm": int(len(warm)),
            "warm_auprc": round(warm_auprc, 4),
        }
        csv_rows.append({
            "condition": "冷启动(hist=0)",
            "seed": 2026,
            "overall_auprc": round(cold_auprc, 6),
            "event_recall": None, "entity_recall": None,
            "n_cold": int(len(cold)), "n_warm": int(len(warm)),
            "warm_auprc": round(warm_auprc, 6),
        })
        print(f"  冷启动(hist=0) 整体 AUPRC = {cold_auprc:.4f}  "
              f"(n={len(cold)}；有历史 n={len(warm)} AUPRC={warm_auprc:.4f})")
    else:
        summary["冷启动(hist=0)"] = {"overall_auprc": None, "n_cold": 0}

    with open(os.path.join(OUT_DIR, "robustness.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    import pandas as _pd
    _pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, "robustness.csv"),
                                   index=False, encoding="utf-8-sig")
    print(f"\n结果已写入 {OUT_DIR}/robustness.json + robustness.csv")


if __name__ == "__main__":
    main()
