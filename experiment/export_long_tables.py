"""
B2 —— 长表导出：feature_table / baseline_table / temporal_score
================================================================
指导书 §5.3 中间文件：
  feature_table.csv   sample_id, score_time, feature_name, window, value, source_max_available_at
  baseline_table.csv  sample_id, mu_self, mu_peer, sigma_self, sigma_peer, lambda, fallback_level
  temporal_score.csv  sample_id, s_ts, top_feature, sample_type

冷启动 λ=0；所有值有历史依据；无不可解释 NaN（数值列填 0 并备注回退来源）。
用法：python -m experiment.export_long_tables [--data-seed 10]
     [--out-dir result_raw/]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, split, temporal_baseline as tb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out-dir", default=config.RESULT_DIR)
    args = ap.parse_args()

    payments, enriched, feat = ec.load_feat(args.data_seed)
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    split_col = pd.Series("train", index=payments.index)
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"

    # 重新计算 peer 均值/尺度（μ_peer/σ_peer 未单独存于 feat）
    peer_stats = tb.compute_peer_stats(enriched.loc[train_idx])
    peer_mean, peer_scale, _ = tb._peer_lookup(feat, peer_stats)

    aid = payments["apply_id"].astype(str).values
    score_time = payments["apply_date"].astype(str).values

    # ── baseline_table.csv ─────────────────────────────────────
    base = pd.DataFrame({
        "sample_id": aid,
        "mu_self": np.round(np.nan_to_num(feat["hist_mean"].values, nan=0.0), 6),
        "mu_peer": np.round(np.nan_to_num(peer_mean, nan=0.0), 6),
        "sigma_self": np.round(np.nan_to_num(feat["hist_std"].values, nan=0.0), 6),
        "sigma_peer": np.round(np.nan_to_num(peer_scale, nan=0.0), 6),
        "lambda": np.round(feat["lambda_self"].values, 6),
        "fallback_level": feat["peer_level"].values.astype(str),
    })
    base.to_csv(os.path.join(args.out_dir, "baseline_table.csv"), index=False, encoding="utf-8-sig")

    # ── temporal_score.csv ─────────────────────────────────────
    s_amt = np.asarray(feat["s_amt"], dtype=float)
    s_freq = np.asarray(feat["s_freq"], dtype=float)
    s_trend = np.asarray(feat["s_trend"], dtype=float)
    top = np.argmax(np.vstack([s_amt, s_freq, s_trend]), axis=0)
    top_name = np.where(top == 0, "amount", np.where(top == 1, "freq", "trend"))
    ts = pd.DataFrame({
        "sample_id": aid,
        "s_ts": np.round(np.asarray(feat["s_ts"], dtype=float), 6),
        "top_feature": top_name,
        "sample_type": "payment",
    })
    ts.to_csv(os.path.join(args.out_dir, "temporal_score.csv"), index=False, encoding="utf-8-sig")

    # ── feature_table.csv（长表）───────────────────────────────
    # 当前值 + 历史窗口聚合 + 派生节奏/基线字段，全部严格时间前向（day_end）。
    feats = {
        "amount": (payments["amount_wan"].values, "none"),
        "log_amount": (np.log1p(payments["amount_wan"].values), "none"),
        "w7_count": (feat["w7_count"].fillna(0.0).values, "7d"),
        "w7_mean": (feat["w7_mean"].fillna(0.0).values, "7d"),
        "w30_count": (feat["w30_count"].fillna(0.0).values, "30d"),
        "w30_mean": (feat["w30_mean"].fillna(0.0).values, "30d"),
        "hist_count": (feat["hist_count"].fillna(0.0).values, "self"),
        "hist_mean": (feat["hist_mean"].fillna(0.0).values, "self"),
        "hist_std": (feat["hist_std"].fillna(0.0).values, "self"),
        "prior_rate": (feat["prior_rate"].fillna(0.0).values, "self"),
        "z_amount": (feat["z_amount"].fillna(0.0).values, "derived"),
        "z_freq": (feat["z_freq"].fillna(0.0).values, "derived"),
        "z_trend": (feat["z_trend"].fillna(0.0).values, "derived"),
        "z_rhythm": (feat["z_rhythm"].fillna(0.0).values, "derived"),
        "z_rhythm_ind": (feat["z_rhythm_ind"].fillna(0.0).values, "derived"),
        "z_progress_snap": (feat["z_progress_snap"].fillna(0.0).values, "derived"),
        "month": (payments["month"].values, "none"),
        "is_december": (payments["is_december"].values, "none"),
    }
    long_rows = []
    for fname, (vals, window) in feats.items():
        for i in range(len(aid)):
            long_rows.append({
                "sample_id": aid[i], "score_time": score_time[i],
                "feature_name": fname, "window": window,
                "value": round(float(vals[i]), 6),
                "source_max_available_at": score_time[i],
            })
    feat_df = pd.DataFrame(long_rows)
    feat_df.to_csv(os.path.join(args.out_dir, "feature_table.csv"), index=False, encoding="utf-8-sig")

    print(f"[B2] baseline_table.csv（{len(base)} 行）")
    print(f"[B2] temporal_score.csv（{len(ts)} 行）")
    print(f"[B2] feature_table.csv（{len(feat_df)} 行 × {len(feats)} 特征）")


if __name__ == "__main__":
    main()
