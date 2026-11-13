"""
B6 —— 冷启动分组 × 基线消融（cold_start.csv）
============================================
指导书 §9.2：按评分时点可得历史长度 ≤30 / 31—180 / >180 天分组，分别比较
Self-only、Peer-only、Self/Peer(固定 0.5/0.5 融合) 与完整 MS-TRP（可靠性门控）。
报告组样本量 n_test 与正类数 n_pos，避免小组仅少数正类导致 AP 不稳定。

· Self-only = s_ts（时序基线）
· Peer-only = s_g（图/邻域）
· Self/Peer = 0.5*s_ts + 0.5*s_g（fixed fusion）
· MS-TRP   = s_final（门控）
历史长度取 feat.hist_span_days（评分时点前可见历史跨度，天）。

schema：bucket,model,n_test,n_pos,payment_auprc,far_1000

用法：python -m experiment.run_cold_start [--data-seed 10] [--out result_raw/cold_start.csv]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, data, split, metrics, calibration

BUCKETS = [("<=30", 0, 30), ("31-180", 31, 180), (">180", 181, int(1e9))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR, "cold_start.csv"))
    args = ap.parse_args()

    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    y_all = point_labels["is_anomaly"].astype(int).values
    apply_all = payments["apply_id"].astype(str).tolist()

    # 历史跨度（天）按 apply_id 对齐
    _, _, feat = ec.load_feat(args.data_seed)
    span = dict(zip(payments["apply_id"].astype(str), feat["hist_span_days"].fillna(0).values))

    pred = ec.load_predictions(args.data_seed)
    pred["apply_id"] = pred["apply_id"].astype(str)
    pred = pred.set_index("apply_id")
    pred = pred.reindex(apply_all)

    s_ts = pred["s_ts"].values
    s_g = pred["s_g"].values
    s_final = pred["s_final"].values

    models = {
        "Self-only": s_ts,
        "Peer-only": s_g,
        "Self/Peer": 0.5 * s_ts + 0.5 * s_g,
        "MS-TRP": s_final,
    }

    test_idx_arr = np.asarray(test_idx, dtype=int)
    test_apply = np.asarray(apply_all)[test_idx_arr]
    test_span = np.asarray([span.get(a, 0.0) for a in test_apply], dtype=float)
    y_test = y_all[test_idx_arr]

    rows = []
    for bname, lo, hi in BUCKETS:
        mask = (test_span >= lo) & (test_span <= hi)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        n_test = int(mask.sum())
        n_pos = int(y_test[mask].sum())
        for mname, s in models.items():
            s_t = np.asarray(s, dtype=float)[test_idx_arr]
            s_group = s_t[mask]
            a = metrics.auprc(y_test[mask], s_group)
            if mname == "MS-TRP":
                thr = ec.load_thresholds(args.data_seed)["thr_global"]
            else:
                thr = calibration.calibrate_target_fpr(
                    np.asarray(s, dtype=float)[val_idx], y_all[val_idx],
                    alpha_fpr=calibration.ALPHA_FPR_DEFAULT)
            fp = int((s_group > thr).sum())
            far = fp / n_test * 1000.0 if n_test > 0 else np.nan
            rows.append({
                "bucket": bname,
                "model": mname,
                "n_test": n_test,
                "n_pos": n_pos,
                "payment_auprc": round(a, 6),
                "far_1000": round(far, 4) if np.isfinite(far) else np.nan,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
