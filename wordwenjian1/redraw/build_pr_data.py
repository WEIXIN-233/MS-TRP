# -*- coding: utf-8 -*-
"""复现测试集逐支付分数，用于绘制 PR 曲线 / 低误报召回曲线。

做法与 experiment/build_metrics_main.py 完全同源：
  · 基线分数直接调用 baselines.*_scores（同一函数、同一参考统计）；
  · MS-TRP 取 run_main/predictions.csv 的 s_final；
  · Temporal-only 取 max(s_ts, s_rhythm_ch)。
导出后逐方法核对 AUPRC 是否与 metrics_main.csv 一致（不一致即报错）。

用法：python build_pr_data.py
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src import config, data, split, metrics          # noqa: E402
from src import feature_engineering as features        # noqa: E402
from experiment import _export_common as ec            # noqa: E402
import baselines as blm                                # noqa: E402

WANTED = ["fixed_rules", "isolation_forest", "simple_ppr", "concat_if"]


def main():
    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    ref_stats = features.compute_reference_stats(payments, train_idx)
    y_all = point_labels["is_anomaly"].astype(int).values

    funcs = {
        "fixed_rules": lambda: blm.fixed_rules_scores(payments, ref_stats),
        "isolation_forest": lambda: blm.isolation_forest_scores(payments, ref_stats, train_idx),
        "simple_ppr": lambda: blm.simple_ppr_scores(payments),
        "concat_if": lambda: blm.concat_if_scores(payments, ref_stats, train_idx),
    }
    scores = {}
    for name in WANTED:
        scores[name] = np.asarray(funcs[name](), dtype=float)

    pred = pd.read_csv(os.path.join(config.RESULT_DIR, "run_main", "predictions.csv"))
    pred["apply_id"] = pred["apply_id"].astype(str)
    pred = pred.set_index("apply_id")
    pay_ids = payments["apply_id"].astype(str)
    scores["MS-TRP"] = pred["s_final"].reindex(pay_ids).values.astype(float)
    scores["temporal_only"] = np.maximum(
        pred["s_ts"].reindex(pay_ids).values,
        pred["s_rhythm_ch"].reindex(pay_ids).values,
    ).astype(float)

    # 与 metrics_main.csv 核对 AUPRC
    ref = pd.read_csv(os.path.join(config.RESULT_DIR, "metrics_main.csv"), encoding="utf-8-sig")
    report = []
    ok = True
    for name, s in scores.items():
        ap = metrics.auprc(y_all[test_idx], s[test_idx])
        exp = float(ref[ref["model"] == name]["payment_auprc"].iloc[0])
        diff = abs(ap - exp)
        report.append(f"{name:18s} recomputed={ap:.6f}  metrics_main={exp:.6f}  diff={diff:.2e}")
        if diff > 1e-4:
            ok = False
    report.append("MATCH" if ok else "MISMATCH")
    with open(os.path.join(HERE, "pr_verify.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    if not ok:
        raise SystemExit("AUPRC 复现与 metrics_main.csv 不一致，停止导出")

    split_col = np.array(["test"] * len(payments), dtype=object)
    split_col[train_idx] = "train"
    split_col[val_idx] = "val"
    out = pd.DataFrame({
        "apply_id": pay_ids.values,
        "apply_date": payments["apply_date"].values,
        "is_anomaly": y_all,
        "split": split_col,
    })
    for name, s in scores.items():
        out[name] = s
    out.to_csv(os.path.join(HERE, "pr_scores.csv"), index=False)
    print("已导出 pr_scores.csv:", out.shape)


if __name__ == "__main__":
    main()
