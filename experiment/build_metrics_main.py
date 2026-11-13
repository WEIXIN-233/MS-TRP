"""
构建规范化指标产物：metrics_main.csv + metrics_range.csv + metrics_ranking.csv
================================================================================
指导书 §8.3 唯一规范 schema。三族指标分离到三个文件，不再混入单一宽表：

  metrics_main.csv    点级主表（九行强制验收 + 常规基线 + MS-TRP）
       model,data_seed,model_seed,split,n_test,prevalence,payment_auprc,payment_f1,
       precision,recall,far_1000,event_f1,f1_raw,recall_at_far5,recall_at_far10,
       reporting_regime,n_reported,status
  metrics_range.csv   区间级 VUS-PR（自然月网格）
       model,data_seed,observation_grid,interval_label_source,aggregation_rule,
       threshold_dependent,n_valid,vus_pr,metric_status
  metrics_ranking.csv 排序级 NDCG@K（K=复核预算=测试正样本数，冻结）
       model,data_seed,k,n_alerts,n_pos,ndcg_at_k,metric_status

关键修正：
  · ecod/copod/mcd/cblof 主表四行一律取 table_baseline_raw（白名单）AUPRC/指标，
    与 table_feature_isolation.csv 同源（复用 run_table_isolation 的 raw 构建与模型类）。
  · constant_score 的 VUS-PR/NDCG@K 记 NA 且 metric_status=not_applicable；
    其支付级 AP/F1/FAR/事件F1 等有定义指标按固定 tie_policy(score>thr) 真实计算。
  · status∈{completed,not_run,not_applicable}；未执行保留行并标 not_run，不删行。

用法：python -m experiment.build_metrics_main
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, split, metrics, calibration
from src import feature_engineering as features
from experiment import _export_common as ec
from experiment.run_table_isolation import _build_raw, _impute_fit_transform, _table_model_classes
import baselines as blm


BASELINE_FUNCS = {
    "fixed_rules":      lambda pay, ref, tr: blm.fixed_rules_scores(pay, ref),
    "robust_stats":     lambda pay, ref, tr: blm.robust_stats_scores(pay, ref),
    "isolation_forest": lambda pay, ref, tr: blm.isolation_forest_scores(pay, ref, tr),
    "lof":              lambda pay, ref, tr: blm.lof_scores(pay, ref, tr),
    "residual":         lambda pay, ref, tr: blm.residual_scores(pay, ref),
    "graph":            lambda pay, ref, tr: blm.graph_scores(pay),
    # 指导书九行强制基线（random_uniform / constant_score / 两种结构启发式 /
    # 四种表格基线 / simple_ppr）
    "random_uniform":   lambda pay, ref, tr: blm.random_uniform_scores(pay),
    "constant_score":   lambda pay, ref, tr: blm.constant_score_scores(pay),
    "shared_payee_count": lambda pay, ref, tr: blm.shared_payee_count_scores(pay),
    "shared_account_concentration": lambda pay, ref, tr: blm.shared_account_concentration_scores(pay),
    "ecod":             lambda pay, ref, tr: blm.ecod_scores(pay, ref, tr),
    "copod":            lambda pay, ref, tr: blm.copod_scores(pay, ref, tr),
    "mcd":              lambda pay, ref, tr: blm.mcd_scores(pay, ref, tr),
    "cblof":            lambda pay, ref, tr: blm.cblof_scores(pay, ref, tr),
    "simple_ppr":       lambda pay, ref, tr: blm.simple_ppr_scores(pay),
    # B7 消融
    "concat_if":        lambda pay, ref, tr: blm.concat_if_scores(pay, ref, tr),
}

# 主表固定四行一律来自 raw 白名单的表格基线（不取 derived 派生特征）
TABLE_MODELS = ("ecod", "copod", "mcd", "cblof")


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


def _vus_pr_with_n(y, s, ts):
    """VUS-PR + 有效自然月区间数（与 metrics.vus_pr 同口径，额外返回 n_valid）。"""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    periods = pd.to_datetime(ts).dt.to_period("M")
    vol, w_total, n_valid = 0.0, 0.0, 0
    groups = pd.Series(np.arange(len(y))).groupby(periods.values).groups
    for idx in groups.values():
        idx = np.asarray(idx, dtype=int)
        yk, sk = y[idx], s[idx]
        n_pos, n_neg = int(yk.sum()), int((1 - yk).sum())
        if n_pos == 0 or n_neg == 0 or len(np.unique(sk)) < 2:
            continue
        w = len(idx)
        vol += w * metrics.auprc(yk, sk)
        w_total += w
        n_valid += 1
    vus = vol / w_total if w_total > 0 else np.nan
    return vus, n_valid


def _load_ms_trp_signals():
    """从主流程预测文件取 s_final / s_ts / s_rhythm_ch，按 payments 行序对齐。"""
    pred = pd.read_csv(os.path.join(config.RESULT_DIR, "run_main", "predictions.csv"))
    pred["apply_id"] = pred["apply_id"].astype(str)
    pred = pred.set_index("apply_id")
    return pred[["s_final", "s_ts", "s_rhythm_ch"]]


def _table_raw_scores(payments, feat, entities, train_idx):
    """表格四基线的 raw 白名单分数（训练 fit，全量 decision_function）。"""
    raw = _build_raw(payments, feat, entities, train_idx)
    X_raw, _ = _impute_fit_transform(raw, train_idx)
    X = X_raw.values
    out = {}
    for name, (cls, kw) in _table_model_classes().items():
        clf = cls(**kw)
        clf.fit(X[train_idx])
        out[name] = clf.decision_function(X)
    return out


def main():
    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    ref_stats = features.compute_reference_stats(payments, train_idx)

    y_all = point_labels["is_anomaly"].astype(int).values
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]
    ts_test = payments.loc[test_idx, "apply_date"]
    n_test = int(len(test_idx))
    n_pos_test = int(y_test.sum())
    prevalence = n_pos_test / n_test

    # raw 白名单特征（表格四基线主表行同源）
    _, enriched, feat = ec.load_feat(None)
    entities = data.load_entity_tables()
    table_raw = _table_raw_scores(payments, feat, entities, train_idx)

    # 各方法全量分数（与 payments 等长）
    model_scores = {}
    for name, func in BASELINE_FUNCS.items():
        if name in TABLE_MODELS:
            model_scores[name] = table_raw[name]
        else:
            model_scores[name] = np.asarray(func(payments, ref_stats, train_idx), dtype=float)

    ms = _load_ms_trp_signals()
    pay_ids = payments["apply_id"].astype(str)
    model_scores["MS-TRP"] = ms["s_final"].reindex(pay_ids).values
    # Temporal-only 基线：与消融「Temporal-only(去图)」同口径 = 关掉图分支但保留节奏/季节通道
    # s = max(s_ts, s_rhythm_ch)（pipeline.run_ms_trp(use_graph=False) 的 s_final，V8 软 max）。
    temporal_only = np.maximum(ms["s_ts"], ms["s_rhythm_ch"])
    model_scores["temporal_only"] = temporal_only.reindex(pay_ids).values

    # MS-TRP 主表冻结阈值 = 部署阈值 thr_global（分层收缩校准），与 event_params /
    # diagnostics_pa / frozen_threshold_far_by_period 同源；基线无分层校准，取各自
    # target-FPR 冻结阈值（alpha_fpr=0.01），同一评测器 evaluate_point/event_strict_f1。
    thr_global_ms = ec.load_thresholds(None)["thr_global"]

    main_rows, range_rows, rank_rows = [], [], []
    for name, s_full in model_scores.items():
        s_full = np.asarray(s_full, dtype=float)
        s_val = s_full[val_idx]
        s_test = s_full[test_idx]
        if name == "MS-TRP":
            thr = thr_global_ms
        else:
            thr = calibration.calibrate_target_fpr(s_val, y_val,
                                                   alpha_fpr=calibration.ALPHA_FPR_DEFAULT)
        pe = metrics.evaluate_point(y_test, s_test, thr)
        ev = ec.event_strict_f1(payments, events, test_idx, s_full, thr)
        is_constant = (name == "constant_score")

        main_rows.append({
            "model": name,
            "data_seed": 2026,
            "model_seed": 1,
            "split": "test",
            "n_test": n_test,
            "prevalence": round(prevalence, 6),
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
        })

        vus, n_valid = _vus_pr_with_n(y_test, s_test, ts_test)
        range_rows.append({
            "model": name,
            "data_seed": 2026,
            "observation_grid": "natural_month",
            "interval_label_source": "点级异常标签按月投影",
            "aggregation_rule": "样本量加权 AUPRC 均值",
            "threshold_dependent": False,
            "n_valid": n_valid,
            "vus_pr": round(vus, 6) if not is_constant and np.isfinite(vus) else np.nan,
            "metric_status": "not_applicable" if is_constant else "completed",
        })

        ndcg = metrics.ndcg_at_k(y_test, s_test, k=None)
        rank_rows.append({
            "model": name,
            "data_seed": 2026,
            "k": n_pos_test,
            "n_alerts": n_pos_test,
            "n_pos": n_pos_test,
            "ndcg_at_k": round(ndcg, 6) if not is_constant else np.nan,
            "metric_status": "not_applicable" if is_constant else "completed",
        })

    df_main = pd.DataFrame(main_rows)
    df_range = pd.DataFrame(range_rows)
    df_rank = pd.DataFrame(rank_rows)

    os.makedirs(config.RESULT_DIR, exist_ok=True)
    df_main.to_csv(os.path.join(config.RESULT_DIR, "metrics_main.csv"),
                   index=False, encoding="utf-8-sig")
    df_range.to_csv(os.path.join(config.RESULT_DIR, "metrics_range.csv"),
                    index=False, encoding="utf-8-sig")
    df_rank.to_csv(os.path.join(config.RESULT_DIR, "metrics_ranking.csv"),
                   index=False, encoding="utf-8-sig")

    print("=" * 120)
    print(df_main.sort_values("payment_auprc", ascending=False)
          [["model", "payment_auprc", "payment_f1", "far_1000", "event_f1",
            "recall_at_far5", "recall_at_far10"]].round(4).to_string(index=False))
    print("=" * 120)
    print("[range] metrics_range.csv 行数:", len(df_range))
    print("[rank]  metrics_ranking.csv 行数:", len(df_rank))


if __name__ == "__main__":
    main()
