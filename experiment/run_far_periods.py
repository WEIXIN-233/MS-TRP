"""
B4 —— 冻结阈值分期 FAR（frozen_threshold_far_by_period.csv）
=============================================================
用验证集冻结的阈值 thr_global，在测试集上按「自然季度 / 项目阶段」两套 period 分别报告
n_normal、FP、FAR_per_1000、95% 区间、far_val、far_ratio、delta_abs、denominator_valid。

口径（指导书 §7.2 / B4）：
  · far_val = 验证集正常样本上的 FAR（用同一冻结阈值）。
  · far_ratio = 测试期 FAR / far_val；仅在 n_normal≥50 且 far_val>0 时触发"误报偏离"提示。
  · n_normal<50 只保存原始计数与经验 FAR，不触发倍数警示；far_val=0 时 far_ratio 记 NA。
  · 测试期禁止重新调阈值。

用法：python -m experiment.run_far_periods [--config config/main.yaml] [--data-seed 10]
                            [--out result_raw/frozen_threshold_far_by_period.csv]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config


def _wilson(k, n, z=1.96):
    """Wilson 95% 区间（小样本稳健）。"""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - m) / d), min(1.0, (c + m) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/main.yaml")
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR,
                                                   "frozen_threshold_far_by_period.csv"))
    args = ap.parse_args()

    pred = ec.load_predictions(args.data_seed)
    thr = ec.load_thresholds(args.data_seed)["thr_global"]
    pred["apply_date"] = pd.to_datetime(pred["apply_date"])
    pred["quarter"] = pred["apply_date"].dt.to_period("Q").astype(str)
    pred["is_normal"] = (pred["is_anomaly"] == 0).astype(int)
    pred["flagged"] = (pred["s_final"] > thr).astype(int)

    val = pred[pred["split"] == "val"]
    test = pred[pred["split"] == "test"]
    val_fp = int(((val["flagged"] == 1) & (val["is_normal"] == 1)).sum())
    val_n = int(val["is_normal"].sum())
    far_val = float(val_fp / val_n) if val_n > 0 else 0.0

    rows = []
    for ptype, pcol in [("quarter", "quarter"), ("stage", "stage")]:
        for pid, g in test.groupby(pcol):
            n_normal = int(g["is_normal"].sum())
            fp = int(((g["flagged"] == 1) & (g["is_normal"] == 1)).sum())
            far = fp / n_normal * 1000.0 if n_normal > 0 else np.nan
            ci_lo, ci_hi = _wilson(fp, n_normal) if n_normal > 0 else (np.nan, np.nan)
            ratio = (far / (far_val * 1000.0)) if (n_normal >= 50 and far_val > 0 and np.isfinite(far)) else np.nan
            rows.append({
                "period_type": ptype, "period_id": str(pid),
                "n_normal": n_normal, "fp": fp,
                "far_per_1000": round(far, 4) if np.isfinite(far) else np.nan,
                "far_per_1000_ci_low": round(ci_lo * 1000, 4) if np.isfinite(ci_lo) else np.nan,
                "far_per_1000_ci_high": round(ci_hi * 1000, 4) if np.isfinite(ci_hi) else np.nan,
                "far_val": round(far_val * 1000.0, 4),
                "far_ratio": round(ratio, 4) if pd.notna(ratio) else np.nan,
                "delta_abs": round(far - far_val * 1000.0, 4) if np.isfinite(far) else np.nan,
                "denominator_valid": bool(n_normal >= 50 and far_val > 0),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[B4] 已保存 {args.out}（{len(df)} 行，冻结阈值 τ={thr:.4f}，far_val={far_val*1000:.2f}/1000）")
    # 触发提示
    warn = df[(df["denominator_valid"]) & (df["far_ratio"] > 2)]
    if len(warn):
        print(f"[B4] 提示：{len(warn)} 个 period far_ratio>2，须在论文 6.5 逐项报告经验误报率升高。")


if __name__ == "__main__":
    main()
