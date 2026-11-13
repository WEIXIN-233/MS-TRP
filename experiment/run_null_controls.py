"""
B5 —— 无技能评分器与事件匹配抗游戏化（random_control_metrics.csv + event_null_distribution.csv）
================================================================================================
指导书 §8.3.2：新增 RandomUniform（固定模型种子 U[0,1]）、ConstantScore（全 0.5）与
随机置换现有分数三种检查，与其他方法遵守相同验证选择/测试冻结规则；常数分数并列值的
阈值与排序处理显式固定（tie_policy：score>=threshold 报警）。

random_control_metrics.csv   model,data_seed,auprc,event_f1,alert_rate,positive_rate
event_null_distribution.csv  置换/随机空模型下 event_f1 的分布（含 MS-TRP 实际值作对照）

不设"Event-F1>0.1 必错"一刀切；如实展示空模型分布，供人工核查，不人工改数。
用法：python -m experiment.run_null_controls [--data-seed 10] [--n-perm 100]
      [--out result_raw/random_control_metrics.csv --out-null result_raw/event_null_distribution.csv]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, data, split, metrics, calibration

NULL_SEED = 2026


def _frozen_thr(s_val, y_val):
    return calibration.calibrate_target_fpr(
        np.asarray(s_val, dtype=float), np.asarray(y_val, dtype=int),
        alpha_fpr=calibration.ALPHA_FPR_DEFAULT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR, "random_control_metrics.csv"))
    ap.add_argument("--out-null", default=os.path.join(config.RESULT_DIR, "event_null_distribution.csv"))
    args = ap.parse_args()

    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    y_all = point_labels["is_anomaly"].astype(int).values
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]
    positive_rate = float(y_test.mean())
    n_test = len(test_idx)

    # MS-TRP 实际分数（用于置换空模型与对照）
    pred = ec.load_predictions(args.data_seed)
    pred = pred[["apply_id", "s_final"]].copy()
    pred["apply_id"] = pred["apply_id"].astype(str)
    s_full_ms = pd.Series(pred.set_index("apply_id")["s_final"].reindex(
        payments["apply_id"].astype(str)).values, index=payments.index)
    thr_ms = ec.load_thresholds(args.data_seed)["thr_global"]

    rng = np.random.RandomState(NULL_SEED)
    rng_perm = np.random.RandomState(NULL_SEED + 1)

    rows = []
    null_rows = []

    # ── 1) RandomUniform ──
    s_ru = rng.uniform(0.0, 1.0, len(payments))
    thr_ru = _frozen_thr(s_ru[val_idx], y_val)
    ev_ru = ec.event_strict_f1(payments, events, test_idx, s_ru, thr_ru)
    alert_ru = float((s_ru[test_idx] > thr_ru).mean())
    rows.append({"model": "random_uniform", "data_seed": args.data_seed if args.data_seed is not None else "main",
                 "auprc": metrics.auprc(y_test, s_ru[test_idx]),
                 "event_f1": ev_ru["event_f1"], "alert_rate": alert_ru,
                 "positive_rate": positive_rate})

    # ── 2) ConstantScore（tie_policy：score>=threshold）──
    s_cs = np.full(len(payments), 0.5)
    thr_cs = _frozen_thr(s_cs[val_idx], y_val)
    ev_cs = ec.event_strict_f1(payments, events, test_idx, s_cs, thr_cs)
    alert_cs = float((s_cs[test_idx] >= thr_cs).mean())
    rows.append({"model": "constant_score", "data_seed": args.data_seed if args.data_seed is not None else "main",
                 "auprc": metrics.auprc(y_test, s_cs[test_idx]),
                 "event_f1": ev_cs["event_f1"], "alert_rate": alert_cs,
                 "positive_rate": positive_rate})

    # ── 3) 置换 MS-TRP 现有分数（冻结阈值不变）──
    ev_ms = ec.event_strict_f1(payments, events, test_idx, s_full_ms.values, thr_ms)
    alert_ms = float((s_full_ms.values[test_idx] > thr_ms).mean())
    rows.append({"model": "MS-TRP", "data_seed": args.data_seed if args.data_seed is not None else "main",
                 "auprc": metrics.auprc(y_test, s_full_ms.values[test_idx]),
                 "event_f1": ev_ms["event_f1"], "alert_rate": alert_ms,
                 "positive_rate": positive_rate})

    base = s_full_ms.values.copy()
    perm_once = base[rng_perm.permutation(len(base))]
    ev_perm = ec.event_strict_f1(payments, events, test_idx, perm_once, thr_ms)
    alert_perm = float((perm_once[test_idx] > thr_ms).mean())
    rows.append({"model": "permutation", "data_seed": args.data_seed if args.data_seed is not None else "main",
                 "auprc": metrics.auprc(y_test, perm_once[test_idx]),
                 "event_f1": ev_perm["event_f1"], "alert_rate": alert_perm,
                 "positive_rate": positive_rate})

    # ── 事件空模型分布：置换 s_final 的 event_f1 分布 ──
    base = s_full_ms.values.copy()
    for rep in range(args.n_perm):
        perm = base[rng_perm.permutation(len(base))]
        ev = ec.event_strict_f1(payments, events, test_idx, perm, thr_ms)
        null_rows.append({"null_type": "permutation", "rep": rep,
                          "event_f1": ev["event_f1"], "event_recall": ev["event_recall"],
                          "event_precision": ev["event_precision"],
                          "tp": ev["tp"], "fp": ev["fp"], "fn": ev["fn"]})

    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(args.out_null, index=False, encoding="utf-8-sig")

    print("[B5] random_control_metrics.csv：")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[B5] event_null_distribution.csv：{len(null_df)} 次置换，"
          f"event_f1 均值={null_df['event_f1'].mean():.4f} "
          f"95分位={null_df['event_f1'].quantile(0.95):.4f} "
          f"（MS-TRP 实际={ev_ms['event_f1']:.4f}）")


if __name__ == "__main__":
    main()
