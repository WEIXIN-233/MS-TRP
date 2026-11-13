"""
B5 —— VUS-PR 最小空模型核查（vus_pr_null_check.csv）
====================================================
指导书 §8.3.3：每个 data_seed 用固定 random_uniform 模型种子，在完全相同网格/投影/实现下
计算随机 VUS-PR，与 MS-TRP 一同写入 vus_pr_null_check.csv：
  data_seed,grid_id,model_vus,null_vus,ratio,warning_threshold,null_flag,n_valid_points
ratio = null_vus / model_vus（分母 0 记 NA）；0.7 为预注册警示值；null_flag 只提示人工核查，
不允许事后删主表或挑指标。网格=自然月分箱（与 metrics.vus_pr 一致），测试期不重选网格。

用法：python -m experiment.run_vus_null [--data-seed 10] [--out result_summary/vus_pr_null_check.csv]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, data, split, metrics

WARNING_THRESHOLD = 0.7
NULL_SEED = 2026
GRID_ID = "natural_month"


def _n_valid_points(y, s, ts):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ts = pd.to_datetime(ts)
    n_valid = 0
    for _, idx in pd.Series(np.arange(len(y))).groupby(ts.dt.to_period("M").values).groups.items():
        idx = np.asarray(idx, dtype=int)
        yk, sk = y[idx], s[idx]
        if yk.sum() == 0 or (1 - yk).sum() == 0 or len(np.unique(sk)) < 2:
            continue
        n_valid += 1
    return n_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR, "vus_pr_null_check.csv"))
    args = ap.parse_args()

    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    y_test = point_labels["is_anomaly"].astype(int).values[test_idx]
    ts_test = payments["apply_date"].iloc[test_idx]

    pred = ec.load_predictions(args.data_seed)
    pred = pred[["apply_id", "s_final"]].copy()
    pred["apply_id"] = pred["apply_id"].astype(str)
    s_model = pred.set_index("apply_id")["s_final"].reindex(
        payments["apply_id"].astype(str)).values[test_idx]

    rng = np.random.RandomState(NULL_SEED)
    s_null = rng.uniform(0.0, 1.0, len(test_idx))

    model_vus = metrics.vus_pr(y_test, s_model, ts_test)
    null_vus = metrics.vus_pr(y_test, s_null, ts_test)
    ratio = null_vus / model_vus if model_vus > 0 else np.nan
    null_flag = bool(ratio > WARNING_THRESHOLD) if pd.notna(ratio) else False

    rows = [{
        "data_seed": args.data_seed if args.data_seed is not None else "main",
        "grid_id": GRID_ID,
        "model_vus": round(model_vus, 6),
        "null_vus": round(null_vus, 6),
        "ratio": round(ratio, 6) if pd.notna(ratio) else np.nan,
        "warning_threshold": WARNING_THRESHOLD,
        "null_flag": null_flag,
        "n_valid_points": _n_valid_points(y_test, s_model, ts_test),
    }]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
