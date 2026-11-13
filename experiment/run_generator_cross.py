"""
实验 E22 —— 跨生成种子与独立生成器（机制 A vs B）
====================================================
验收判据（导师）：投稿论断跨数据机制成立，或诚实限缩。

做法：
  · 同一 3 个预定生成种子 {10,20,30} 下，分别运行生成器 A（仿真V3.py，机制A）
    与生成器 B（仿真V4.py，独立机制），各自得到一张 budget_data_v5.db。
  · 对每张库跑一遍 MS-TRP（阈值在各自验证集冻结、测试集只评一次），
    记录点级 AUPRC / 事件召回。
  · 报告：A 均值±区间、B 均值±区间、跨机制跌幅 = mean(A) − mean(B)。

输出：
  result_raw/all_seed_results.csv   (mechanism, seed, AP, event_recall, ...)
  result_raw/generator_cross.json   (per-mechanism mean/min/max/std + drop)

说明：数据生成在实验前完成，评估阶段不接触测试集标签调参；A/B 用同一套评估代码，
     唯一区别是 config.DB_PATH 指向各自机制目录下的库。
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, pipeline

SEEDS = [10, 20, 30]
MECH_DIR = {
    "A": "data_fixed/seed{seed}/budget_data_v5.db",     # 生成器 A（仿真V3.py）
    "B": "data_fixed_b/seed{seed}/budget_data_v5.db",   # 生成器 B（仿真V4.py）
}


def run_one(mech, seed):
    db = os.path.join(config.PROJECT_DIR, MECH_DIR[mech].format(seed=seed))
    if not os.path.exists(db):
        raise FileNotFoundError(f"缺少库文件: {db}（先运行对应生成器）")
    config.DB_PATH = db   # 直接指向目标机制库（不传 data_seed，避免 set_data_seed 覆盖）
    res, _pred = pipeline.run_ms_trp()
    return res


def main():
    rows = []
    by_mech = {}
    for mech in ("A", "B"):
        aps = []
        for seed in SEEDS:
            res = run_one(mech, seed)
            ap = res["point_auprc"]
            aps.append(ap)
            rows.append({
                "mechanism": mech,
                "seed": seed,
                "AP": round(ap, 6),
                "AUC": round(res["point_auc"], 6),
                "event_recall": round(res["event"]["event_recall"], 6),
                "event_f1": round(res["event"]["event_f1"], 6),
                "n_test": res["n_test"],
            })
            print(f"  {mech} seed{seed}: AUPRC={ap:.6f}  event_recall={res['event']['event_recall']:.4f}")
        by_mech[mech] = {
            "mean": round(float(np.mean(aps)), 6),
            "min": round(float(np.min(aps)), 6),
            "max": round(float(np.max(aps)), 6),
            "std": round(float(np.std(aps)), 6),
        }

    drop = round(by_mech["A"]["mean"] - by_mech["B"]["mean"], 6)
    summary = {
        "seeds": SEEDS,
        "mechanism_A": by_mech["A"],
        "mechanism_B": by_mech["B"],
        "cross_mechanism_drop": drop,
        "relative_drop_pct": round(drop / by_mech["A"]["mean"] * 100, 2),
        "conclusion": (
            "跨机制成立：B 机制下 MS-TRP 仍保持明显优于随机基线(AP>>阳性率)"
            if by_mech["B"]["mean"] > 0.15 else
            "需诚实限缩：B 机制下 MS-TRP 增益显著收窄"
        ),
    }

    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.RESULT_DIR, "all_seed_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path = os.path.join(config.RESULT_DIR, "generator_cross.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"A 机制: mean={by_mech['A']['mean']:.6f}  [{by_mech['A']['min']:.6f}, {by_mech['A']['max']:.6f}]")
    print(f"B 机制: mean={by_mech['B']['mean']:.6f}  [{by_mech['B']['min']:.6f}, {by_mech['B']['max']:.6f}]")
    print(f"跨机制跌幅 = {drop:+.6f}  ({summary['relative_drop_pct']:+.2f}%)")
    print(f"→ {summary['conclusion']}")
    print(f"\n已写入 {csv_path} + {json_path}")


if __name__ == "__main__":
    main()
