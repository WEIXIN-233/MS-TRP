# -*- coding: utf-8 -*-
"""
汇总三个 2025+ 新基线（FoMo-0D / DRL / TCCM）到 metrics_main.csv 同口径表。
============================================================================
- 从 scores_*.npy 读全量分数，用与 build_metrics_main.py 完全相同的阈值协议：
  目标 FPR 校准（alpha_fpr=0.01，验证集正常样本冻结）→ evaluate_point + event_strict_f1。
- 产出 baselines_2025/new_methods_comparison.csv（含原有 18 方法 + 3 新方法，按 AUPRC 排序）。

用法：python -m baselines_2025.consolidate
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src import config, metrics, calibration
from experiment import _export_common as ec
from baselines_2025._prep import build_prep


NEW_METHODS = {
    "FoMo-0D": "scores_fomo.npy",
    "DRL": "scores_drl.npy",
    "TCCM": "scores_tccm.npy",
}


def _recall_at_far(y, s, target_far):
    """Recall@FAR：阈值 = 正常样本分数 (1−FAR) 分位，召回 = 高于阈值正样本占比。"""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    normal = s[y == 0]
    pos = s[y == 1]
    if len(normal) == 0 or len(pos) == 0:
        return np.nan
    thr = np.quantile(normal, 1.0 - target_far)
    return float((pos > thr).mean())


def compute_row(name, s_full, payments, y, train_idx, val_idx, test_idx, events):
    """与 build_metrics_main.py 基线分支同口径（target-FPR alpha=0.01）。"""
    s_full = np.asarray(s_full, dtype=float)
    s_val, s_test = s_full[val_idx], s_full[test_idx]
    y_val, y_test = y[val_idx].astype(int), y[test_idx].astype(int)
    thr = calibration.calibrate_target_fpr(s_val, y_val,
                                           alpha_fpr=calibration.ALPHA_FPR_DEFAULT)
    pe = metrics.evaluate_point(y_test, s_test, thr)
    ev = ec.event_strict_f1(payments, events, test_idx, s_full, thr)
    n_pos_test = int(y_test.sum())
    return {
        "model": name,
        "data_seed": 2026,
        "model_seed": 1,
        "split": "test",
        "n_test": int(len(test_idx)),
        "prevalence": round(n_pos_test / len(test_idx), 6),
        "payment_auprc": round(pe["auprc"], 6),
        "payment_f1": round(pe["f1"], 6),
        "precision": round(pe["precision"], 6),
        "recall": round(pe["recall"], 6),
        "far_1000": round(pe["far_per_1000"], 4),
        "event_f1": round(ev["event_f1"], 6),
        "f1_raw": round(pe["f1"], 6),
        "recall_at_far5": round(_recall_at_far(y_test, s_test, 0.05), 6),
        "recall_at_far10": round(_recall_at_far(y_test, s_test, 0.10), 6),
        "reporting_regime": "single_run",
        "n_reported": 1,
        "status": "completed",
    }


def main():
    X, y, train_idx, val_idx, test_idx, payments, events = build_prep()
    y = y.astype(int)

    old = pd.read_csv(os.path.join(ROOT, "result_raw", "metrics_main.csv"))

    rows = []
    for name, fname in NEW_METHODS.items():
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"[skip] {name}: {fname} 不存在（尚未跑完）")
            continue
        s_full = np.load(path)
        row = compute_row(name, s_full, payments, y, train_idx, val_idx, test_idx, events)
        rows.append(row)
        print(f"[{name}] payment_auprc={row['payment_auprc']}  precision={row['precision']}  "
              f"recall={row['recall']}  f1={row['payment_f1']}  far_1000={row['far_1000']}  "
              f"event_f1={row['event_f1']}")

    if not rows:
        print("没有新方法分数可汇总。")
        return

    new = pd.DataFrame(rows)
    # 只保留与旧表一致的核心列用于对比
    core = ["model", "payment_auprc", "precision", "recall", "payment_f1",
            "far_1000", "event_f1"]
    merged = pd.concat([old[core], new[core]], ignore_index=True)
    merged = merged.sort_values("payment_auprc", ascending=False).reset_index(drop=True)

    out_path = os.path.join(HERE, "new_methods_comparison.csv")
    new.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print("  payment_auprc 排名（原有 18 方法 + 3 个 2025+ 新方法）")
    print("=" * 100)
    print(merged.round(4).to_string(index=False))
    print(f"\n新方法明细已保存: {out_path}")


if __name__ == "__main__":
    main()
