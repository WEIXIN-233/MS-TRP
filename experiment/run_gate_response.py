"""
门控响应表 → result_raw/gate_response.csv
================================================================================
指导书 §8.3 / §9.1：若证明门控改善鲁棒性，须在同一条件下对比 MS-TRP 与 Fixed Fusion，
报告 r_ts / r_graph / g_ts / g_graph 变化与 AP。

实现：r_ts、r_graph 与 use_gate 无关，直接取自 run_main/predictions.csv（完整模型）。
      gated 条件的 g_ts/g_graph 为 softmax 门控实测均值；fixed 条件恒为 0.5/0.5。
      AP 取自 run_ablation/ablation.csv（完整模型 vs 固定0.5融合）。

用法：python -m experiment.run_gate_response
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config


def main():
    pred = pd.read_csv(os.path.join(config.RESULT_DIR, "run_main", "predictions.csv"))
    ab = pd.read_csv(os.path.join(config.RESULT_DIR, "run_ablation", "ablation.csv"))

    ap_full = float(ab.loc[ab["variant"] == "完整模型", "AP"].iloc[0])
    ap_fixed = float(ab.loc[ab["variant"] == "固定0.5融合(去门控)", "AP"].iloc[0])

    r_ts_mean = float(np.mean(pred["r_ts"]))
    r_g_mean = float(np.mean(pred["r_g"]))

    rows = [
        {
            "condition": "gated",
            "r_ts": round(r_ts_mean, 6),
            "r_graph": round(r_g_mean, 6),
            "g_ts": round(float(np.mean(pred["g_ts"])), 6),
            "g_graph": round(float(np.mean(pred["g_g"])), 6),
            "AP": ap_full,
        },
        {
            "condition": "fixed_fusion_0.5",
            "r_ts": round(r_ts_mean, 6),
            "r_graph": round(r_g_mean, 6),
            "g_ts": 0.5,
            "g_graph": 0.5,
            "AP": ap_fixed,
        },
    ]
    df = pd.DataFrame(rows)
    out = os.path.join(config.RESULT_DIR, "gate_response.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\n[gate_response] 写入 {out}")


if __name__ == "__main__":
    main()
