"""
B0/B5 —— 困难正常负例误报（hard_negative_far.csv）
==================================================
指导书 §B0/Q2-05：困难正常负例（合法公共收款人、高连接度供应商、年末合法批次、批准延期）
上的误报率。生成器未单独打 hard_negative 标签，故按业务口径在测试正常样本上定义困难子集，
用冻结阈值报告 FP / FAR_per_1000；仅报告原始计数与经验误报，不据此回调阈值。

子集定义（测试集正常样本内，对应指导书四类困难正常负例）：
  legal_public_payee       公共收款人且跨 ≥3 个项目（合法公共收款人，Shared Payee 的合法对照）
  high_connectivity_payee  收款人历史支付数 ≥ 50（高连接度供应商）
  yearend_batch            12 月正常支付（年末合法批次）
  approved_extension       多级审批 approval_levels=8 的正常支付（批准延期/复杂审批近似）

schema：subgroup,definition,n_normal,fp,far_per_1000

用法：python -m experiment.run_hard_negative [--data-seed 10] [--out result_raw/hard_negative_far.csv]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, data, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR, "hard_negative_far.csv"))
    args = ap.parse_args()

    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    thr = ec.load_thresholds(args.data_seed)["thr_global"]

    pred = ec.load_predictions(args.data_seed)
    pred["apply_id"] = pred["apply_id"].astype(str)
    s_map = dict(zip(pred["apply_id"], pred["s_final"]))

    test = payments.iloc[test_idx].copy()
    test["apply_id"] = test["apply_id"].astype(str)
    test["score"] = [s_map.get(a, 0.0) for a in test["apply_id"]]
    test["is_anomaly"] = point_labels["is_anomaly"].iloc[test_idx].values
    normal = test[test["is_anomaly"] == 0].copy()
    normal["flagged"] = (normal["score"] > thr).astype(int)

    # 收款人连接度
    payee_nproj = payments.groupby("payee_id")["project_id"].nunique()
    payee_npay = payments.groupby("payee_id")["apply_id"].count()

    def _far(sub):
        n = int(len(sub))
        fp = int(sub["flagged"].sum())
        return n, fp, (fp / n * 1000.0) if n > 0 else np.nan

    subs = [
        ("legal_public_payee", "公共收款人且跨>=3项目（合法）",
         normal[(normal["public_private"] == "PUBLIC")
                & (normal["payee_id"].map(payee_nproj).fillna(0) >= 3)]),
        ("high_connectivity_payee", "收款人历史支付数>=50",
         normal[normal["payee_id"].map(payee_npay).fillna(0) >= 50]),
        ("yearend_batch", "12月正常支付", normal[normal["month"] == 12]),
        ("approved_extension", "多级审批approval_levels=8",
         normal[normal["approval_levels"] == 8]),
    ]

    rows = []
    for name, desc, sub in subs:
        n, fp, far = _far(sub)
        rows.append({"subgroup": name, "definition": desc, "n_normal": n,
                     "fp": fp, "far_per_1000": round(far, 4) if np.isfinite(far) else np.nan})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
