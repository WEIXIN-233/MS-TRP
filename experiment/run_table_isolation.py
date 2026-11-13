"""
B5 —— 表格四基线 raw/with_derived 输入隔离（table_feature_isolation.csv + feature_column_manifest.json）
====================================================================================================
指导书 §8.3.4：ECOD/COPOD/MCD/CBLOF 各跑两组严格同样本输入：
  · table_baseline_raw        仅评分时点可观测值与预定义历史窗口聚合（白名单，封闭清单），
                              排除 Self/Peer 分层收缩偏离、MAD/IQR 归一化偏离、稳健 Z、tanh 打分。
  · table_baseline_with_derived 在 raw 上追加 MS-TRP 派生字段（z_amount/z_freq/z_trend/z_rhythm/s_*）。
缺失填补与尺度拟合只在训练集 fit，验证/测试仅 transform；主表固定四行一律来自 raw，
with_derived 作为 diagnostic 另列。白名单英文字段名、窗口定义及 hash 写入 feature_column_manifest.json。

schema：model,data_seed,feature_set_id,fit_split,payment_auprc,feature_hash

用法：python -m experiment.run_table_isolation [--data-seed 10]
      [--out result_summary/table_feature_isolation.csv --manifest result_summary/feature_column_manifest.json]
"""
import os
import sys
import json
import hashlib
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, data, split, metrics

# 封闭白名单（raw 允许字段；时间累计均以 score_time 之前可见事件为界）
RAW_WHITELIST = {
    "amount": "amount_wan",
    "sum_amount_7d": "w7_count*w7_mean",
    "sum_amount_30d": "w30_count*w30_mean",
    "payment_count_7d": "w7_count",
    "payment_count_30d": "w30_count",
    "mean_payment_gap_days_to_t": "hist_span_days/max(hist_count,1)",
    "cumulative_paid_to_t": "hist_sum+amount",
    "approved_budget_at_t": "indicator.ind_amount_wan",
    "execution_ratio_to_t": "cumulative_paid/approved_budget",
    "payment_month": "month",
    "days_until_month_end": "days_in_month - day",
    "project_type_train_encoding": "LabelEncoder(project_category)",
    "expense_category_train_encoding": "LabelEncoder(econ_class_code)",
}
DERIVED_COLS = ["z_amount", "z_freq", "z_trend", "z_rhythm", "s_amt", "s_freq", "s_trend", "s_ts"]

# 表格四基线（模型名 → (pyod 类, kwargs)）；build_metrics_main 复用同一套以保证主表 raw 行一致。
def _table_model_classes():
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD
    from pyod.models.cblof import CBLOF
    from pyod.models.mcd import MCD
    return {
        "ecod": (ECOD, {}),
        "copod": (COPOD, {}),
        "mcd": (MCD, {"contamination": 0.1, "random_state": 2026}),
        "cblof": (CBLOF, {"n_clusters": 8, "contamination": 0.1, "random_state": 2026}),
    }


def _build_raw(payments, feat, entities, train_idx):
    """构建 raw 白名单特征矩阵（不含任何 Self/Peer 偏离、稳健 Z、tanh 打分）。"""
    raw = pd.DataFrame(index=payments.index)
    amt = payments["amount_wan"].astype(float)
    raw["amount"] = amt.values
    raw["sum_amount_7d"] = (feat["w7_count"].fillna(0) * feat["w7_mean"].fillna(0)).values
    raw["sum_amount_30d"] = (feat["w30_count"].fillna(0) * feat["w30_mean"].fillna(0)).values
    raw["payment_count_7d"] = feat["w7_count"].fillna(0).values
    raw["payment_count_30d"] = feat["w30_count"].fillna(0).values
    hc = feat["hist_count"].fillna(0).values
    hs = feat["hist_sum"].fillna(0).values
    span = feat["hist_span_days"].fillna(0).values
    raw["mean_payment_gap_days_to_t"] = np.where(hc > 0, span / np.maximum(hc, 1), 0.0)
    raw["cumulative_paid_to_t"] = hs + amt.values

    ind_budget = dict(zip(entities["indicator"]["indicator_id"].astype(str),
                          entities["indicator"]["ind_amount_wan"].astype(float)))
    budget = payments["indicator_id"].astype(str).map(ind_budget).fillna(0.0).values
    raw["approved_budget_at_t"] = budget
    raw["execution_ratio_to_t"] = np.where(budget > 0, raw["cumulative_paid_to_t"].values / np.maximum(budget, 1e-6), 0.0)
    raw["payment_month"] = payments["month"].values

    dt = payments["apply_date"]
    dim = dt.dt.days_in_month.values
    day = dt.dt.day.values
    raw["days_until_month_end"] = dim - day

    # train-fitted 标签编码（未见类别回退 0）
    raw["project_type_train_encoding"] = _fit_label_encode(
        feat["project_category"], train_idx)
    raw["expense_category_train_encoding"] = _fit_label_encode(
        payments["econ_class_code"].astype(str), train_idx)
    return raw


def _fit_label_encode(s, train_idx):
    le = LabelEncoder()
    le.fit(s.iloc[train_idx].astype(str))
    out = np.zeros(len(s), dtype=int)
    known = set(le.classes_)
    for i, v in enumerate(s.astype(str)):
        out[i] = le.transform([v])[0] if v in known else 0
    return out


def _impute_fit_transform(F, train_idx):
    """均值填补：仅训练集 fit。"""
    F = F.copy().apply(pd.to_numeric, errors="coerce")
    mean = F.iloc[train_idx].mean()
    F = F.fillna(mean)
    return F, mean


def _auprc(model_cls, X_raw, X_der, y_all, train_idx, test_idx, kwargs):
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD
    from pyod.models.cblof import CBLOF
    from pyod.models.mcd import MCD
    clf = model_cls(**kwargs)
    clf.fit(X_raw[train_idx])
    s_raw = clf.decision_function(X_raw)
    clf2 = model_cls(**kwargs)
    clf2.fit(X_der[train_idx])
    s_der = clf2.decision_function(X_der)
    y_test = y_all[test_idx]
    return metrics.auprc(y_test, s_raw[test_idx]), metrics.auprc(y_test, s_der[test_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR, "table_feature_isolation.csv"))
    ap.add_argument("--manifest", default=os.path.join(config.RESULT_DIR, "feature_column_manifest.json"))
    args = ap.parse_args()

    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    y_all = point_labels["is_anomaly"].astype(int).values

    _, enriched, feat = ec.load_feat(args.data_seed)
    entities = data.load_entity_tables()

    raw = _build_raw(payments, feat, entities, train_idx)
    X_raw, _ = _impute_fit_transform(raw, train_idx)

    # derived：raw + MS-TRP 派生字段
    der = raw.copy()
    for c in DERIVED_COLS:
        der[c] = feat[c].fillna(0.0).values
    X_der, _ = _impute_fit_transform(der, train_idx)

    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD
    from pyod.models.cblof import CBLOF
    from pyod.models.mcd import MCD

    models = [
        ("ecod", ECOD, {}),
        ("copod", COPOD, {}),
        ("mcd", MCD, {"contamination": 0.1}),
        ("cblof", CBLOF, {"n_clusters": 8, "contamination": 0.1, "random_state": 2026}),
    ]

    raw_hash = hashlib.sha256(
        json.dumps(sorted(RAW_WHITELIST.keys()), ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    der_hash = hashlib.sha256(
        json.dumps(sorted(RAW_WHITELIST.keys()) + sorted(DERIVED_COLS), ensure_ascii=False).encode("utf-8")).hexdigest()[:16]

    rows = []
    for name, cls, kw in models:
        a_raw, a_der = _auprc(cls, X_raw.values, X_der.values, y_all, train_idx, test_idx, kw)
        rows.append({"model": name, "data_seed": args.data_seed if args.data_seed is not None else "main",
                     "feature_set_id": "raw", "fit_split": "train",
                     "payment_auprc": round(a_raw, 6), "feature_hash": raw_hash})
        rows.append({"model": name, "data_seed": args.data_seed if args.data_seed is not None else "main",
                     "feature_set_id": "derived", "fit_split": "train",
                     "payment_auprc": round(a_der, 6), "feature_hash": der_hash})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    manifest = {
        "raw_whitelist": RAW_WHITELIST,
        "raw_excluded": ["Self/Peer位置差", "MAD/IQR归一化偏离", "稳健Z", "tanh打分",
                         "execution_ratio−planned_ratio", "图邻域同质性权重"],
        "derived_appended": DERIVED_COLS,
        "fit_split": "train",
        "impute": "mean（仅训练集 fit）",
        "raw_feature_hash": raw_hash,
        "derived_feature_hash": der_hash,
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(df.to_string(index=False))
    print(f"[B5] feature_column_manifest.json 已写（raw_hash={raw_hash}）")


if __name__ == "__main__":
    main()
