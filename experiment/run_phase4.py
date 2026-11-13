# -*- coding: utf-8 -*-
"""
protocol_v2 全量重跑编排（Phase 4）
========================================
在「历史可见(as-of) + 事件互斥 + 口径统一 + 阈值冻结」的 protocol_v2 下重跑：
  · 主实验 run_main
  · 6 基线 baseline_report
  · 跨方法统一口径 compare_report
  · 正交消融 ablation（一次只改一个图信号分量）
  · 鲁棒性 robustness
  · 实验矩阵 E01–E08（E03/E04/E05 从主实验预测派生，E01/E02/E07 引用既有审计产物）

全部结果写入 result_raw/protocol_v2_causal/，不与 protocol_v1_legacy 混写。
用法： python -m experiment.run_phase4
"""
import os
import sys
import json
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, split, metrics, pipeline, calibration, event_eval
from src import feature_engineering as features
import baselines as blm

V2_DIR = os.path.join(config.RESULT_DIR, "protocol_v2_causal")

BASELINE_FUNCS = {
    "fixed_rules":      lambda pay, ref, tr: blm.fixed_rules_scores(pay, ref),
    "robust_stats":     lambda pay, ref, tr: blm.robust_stats_scores(pay, ref),
    "isolation_forest": lambda pay, ref, tr: blm.isolation_forest_scores(pay, ref, tr),
    "lof":              lambda pay, ref, tr: blm.lof_scores(pay, ref, tr),
    "residual":         lambda pay, ref, tr: blm.residual_scores(pay, ref),
    "graph":            lambda pay, ref, tr: blm.graph_scores(pay),
}

POINT_SET = set(config.POINT_ANOMALY_TYPES)
EVENT_SET = set(config.EVENT_ANOMALY_TYPES)

# 正交消融变体（E06）：前 5 个一次只改一个图信号分量，后 3 个为维度消融
ABLATION_VARIANTS = [
    ("完整模型", {}),
    ("Temporal-only(去图)", {"use_graph": False}),
    ("+Rule(账户一致性)", {"use_ppr": False, "use_struct": False}),
    ("+PPR(图扩散)", {"use_struct": False, "use_rule": False}),
    ("+Rule+PPR", {"use_struct": False}),
    ("单尺度(去多尺度)", {"use_multiscale": False}),
    ("固定边权(去阶段)", {"use_stage_weight": False}),
    ("固定0.5融合(去门控)", {"use_gate": False}),
]

ROBUST_VARIANTS = [
    ("完整(基线)", {}),
    ("缺失10%", {"train_drop_ratio": 0.10}),
    ("缺失30%", {"train_drop_ratio": 0.30}),
    ("删边20%", {"edge_drop_ratio": 0.20}),
]


def _topk_threshold(scores, k_ratio):
    sc = np.asarray(scores, dtype=float)
    k = max(1, int(round(len(sc) * k_ratio)))
    return float(np.sort(sc)[-k])


def _pm_from_pred(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def load_bundle():
    payments = data.load_payments()
    labels = data.load_labels()
    events = data.build_events(labels)
    point_labels = data.build_point_labels(payments, events)
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    ref_stats = features.compute_reference_stats(payments, train_idx)
    return payments, events, point_labels, train_idx, val_idx, test_idx, ref_stats


def compute_baseline_scores(payments, ref_stats, train_idx):
    return {name: func(payments, ref_stats, train_idx) for name, func in BASELINE_FUNCS.items()}


def run_main():
    """主实验：完整 MS-TRP，写 predictions.csv + metrics.json。"""
    out = os.path.join(V2_DIR, "run_main")
    os.makedirs(out, exist_ok=True)
    results, pred = pipeline.run_ms_trp()
    pred.to_csv(os.path.join(out, "predictions.csv"), index=False)
    _json_dump(results, os.path.join(out, "metrics.json"))
    print(f"[main] AUPRC={results['point_auprc']:.4f}  AUC={results['point_auc']:.4f}  "
          f"严格事件F1={results['event_strict']['event_f1']:.4f}")
    return results, pred


def run_baselines(bundle, bscores):
    """6 基线报告（基线自身口径，点级/事件级 @固定预算）。"""
    payments, events, point_labels, train_idx, val_idx, test_idx, ref_stats = bundle
    y_point = point_labels["anomaly_types"].apply(lambda t: any(x in POINT_SET for x in t)).astype(int)
    y_point_test = y_point.loc[test_idx].values
    test_aid_set = set(point_labels.loc[test_idx, "apply_id"].astype(str))
    test_pattern_events = events[events["apply_ids"].apply(lambda s: bool(s & test_aid_set))]
    test_pattern_events = test_pattern_events[test_pattern_events["anomaly_type"].isin(EVENT_SET)]
    test_apply_ids = point_labels.loc[test_idx, "apply_id"].astype(str).values

    rows = []
    for name, scores in bscores.items():
        s_test = scores.loc[test_idx].values.astype(float)
        top5 = metrics.metrics_at_topk(y_point_test, s_test, 0.05)
        top10 = metrics.metrics_at_topk(y_point_test, s_test, 0.10)
        thr10 = _topk_threshold(s_test, 0.10)
        aid2score = metrics.flag_payments(s_test, test_apply_ids, thr10)
        em = metrics.event_metrics(test_pattern_events, aid2score, thr10)
        rows.append({
            "baseline": name,
            "auprc_point": metrics.auprc(y_point_test, s_test),
            "normalized_ap_excess_point": metrics.normalized_ap_excess(y_point_test, s_test),
            "recall_point@5%": top5["recall"], "recall_point@10%": top10["recall"],
            "event_recall@10%": em["event_recall"], "event_f1@10%": em["event_f1"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(V2_DIR, "baseline_report.csv"), index=False, encoding="utf-8-sig")
    print("[baselines] 6 基线完成")
    return df


def run_compare(bundle, main_pred, bscores):
    """跨方法统一口径：MS-TRP 与 6 基线同一标签口径 + 同一 top-k 操作点。"""
    payments, events, point_labels, train_idx, val_idx, test_idx, ref_stats = bundle
    y_all = point_labels["is_anomaly"].astype(int)
    y_point = point_labels["anomaly_types"].apply(lambda t: any(x in POINT_SET for x in t)).astype(int)
    y_event = point_labels["anomaly_types"].apply(lambda t: any(x in EVENT_SET for x in t)).astype(int)
    y_all_test = y_all.loc[test_idx].values
    y_point_test = y_point.loc[test_idx].values
    y_event_test = y_event.loc[test_idx].values
    test_apply_ids = point_labels.loc[test_idx, "apply_id"].astype(str).values
    test_aid_set = set(test_apply_ids)
    test_pattern_events = events[events["apply_ids"].apply(lambda s: bool(s & test_aid_set))]
    test_pattern_events = test_pattern_events[test_pattern_events["anomaly_type"].isin(EVENT_SET)]

    methods = {name: np.asarray(scores.loc[test_idx].values, dtype=float)
               for name, scores in bscores.items()}
    pred_test = main_pred[main_pred["split"] == "test"].copy()
    s_map = dict(zip(pred_test["apply_id"].astype(str), pred_test["s_final"].astype(float)))
    methods["MS-TRP"] = np.array([s_map[a] for a in test_apply_ids], dtype=float)

    rows = []
    for name, s in methods.items():
        row = {"method": name}
        row["auprc_all"] = metrics.auprc(y_all_test, s)
        row["auprc_point"] = metrics.auprc(y_point_test, s)
        row["auprc_event"] = metrics.auprc(y_event_test, s)
        row["auc_all"] = metrics.roc_auc(y_all_test, s)
        row["normalized_ap_excess_all"] = metrics.normalized_ap_excess(y_all_test, s)
        for k in (0.01, 0.05, 0.10):
            m = metrics.metrics_at_topk(y_point_test, s, k)
            row[f"point_f1@{int(k*100)}%"] = m["f1"]
            row[f"point_recall@{int(k*100)}%"] = m["recall"]
        thr10 = _topk_threshold(s, 0.10)
        aid2score = metrics.flag_payments(pd.Series(s), test_apply_ids, thr10)
        em = metrics.event_metrics(test_pattern_events, aid2score, thr10)
        row["event_recall@10%"] = em["event_recall"]
        row["event_f1@10%"] = em["event_f1"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(V2_DIR, "compare_report.csv"), index=False, encoding="utf-8-sig")
    print("[compare] 跨方法统一口径完成")
    return df


def run_ablation():
    """E06 正交消融。"""
    summary = {}
    for name, kw in ABLATION_VARIANTS:
        res, _ = pipeline.run_ms_trp(**kw)
        summary[name] = {
            "auprc_all": res["point_auprc"], "auc": res["point_auc"],
            "normalized_ap_excess": res["normalized_ap_excess"],
            "per_type_auprc": res["per_type_auprc"],
            "recall@tau": res["point"]["recall"],
            "far_per_1000": res["far_per_1000"],
            "event_recall": res["event"]["event_recall"],
            "event_strict_f1": res["event_strict"]["event_f1"],
        }
        print(f"[ablation] {name:<20s} AUPRC={res['point_auprc']:.4f}  "
              f"recall@tau={res['point']['recall']:.4f}  strictF1={res['event_strict']['event_f1']:.4f}")
    _json_dump(summary, os.path.join(V2_DIR, "ablation.json"))
    return summary


def run_robustness():
    """鲁棒性：缺失/删边/冷启动。"""
    summary = {}
    full_pred = None
    for name, kw in ROBUST_VARIANTS:
        res, pred = pipeline.run_ms_trp(**kw)
        summary[name] = {
            "auprc_all": res["point_auprc"], "recall@tau": res["point"]["recall"],
            "event_recall": res["event"]["event_recall"],
        }
        if name == "完整(基线)":
            full_pred = pred
        print(f"[robustness] {name:<12s} AUPRC={res['point_auprc']:.4f}")
    test = full_pred[full_pred["split"] == "test"]
    cold = test[test["hist_count"] == 0]
    warm = test[test["hist_count"] > 0]
    if len(cold) > 0:
        summary["冷启动(hist=0)"] = {
            "auprc_all": average_precision_score(cold["is_anomaly"], cold["s_final"]),
            "n_cold": int(len(cold)), "n_warm": int(len(warm)),
            "warm_auprc": average_precision_score(warm["is_anomaly"], warm["s_final"]),
        }
    _json_dump(summary, os.path.join(V2_DIR, "robustness.json"))
    return summary


def run_matrix(bundle, main_results, main_pred):
    """E01–E08 矩阵。E03/E04/E05 从主实验预测派生，其余引用审计产物。"""
    payments, events, point_labels, train_idx, val_idx, test_idx, ref_stats = bundle
    pred = main_pred
    test = pred[pred["split"] == "test"].copy()
    val = pred[pred["split"] == "val"].copy()

    s_test = test["s_final"].values.astype(float)
    y_test = test["is_anomaly"].values.astype(int)
    month = pd.to_datetime(test["apply_date"]).dt.month.values

    # E03：验证集全局阈值 vs 分层阈值（同一 test 集两个操作点）
    thr_global = main_results["thr_global"]
    val_groups = val["group"].values
    thresholds = calibration.calibrate_hierarchical(
        val["s_final"].values.astype(float), val["is_anomaly"].values.astype(int),
        val_groups, gamma=calibration.GAMMA_DEFAULT, quantile=calibration.ALERT_BUDGET_RATIO)
    test_groups = test["group"].values
    test_thr = np.array([thresholds.get(str(g), thresholds.get("__global__", 0.5))
                         for g in test_groups])
    e03 = {
        "global_threshold": thr_global,
        "global": _pm_from_pred(y_test, (s_test > thr_global).astype(int)),
        "hierarchical": _pm_from_pred(y_test, (s_test > test_thr).astype(int)),
    }

    # E04：Top 1/2/5/10% 回顾性预算
    e04 = {f"top_{int(k*100)}%": metrics.metrics_at_topk(y_test, s_test, k)
           for k in (0.01, 0.02, 0.05, 0.10)}

    # E05：六类 × (1-11月 / 12月) 的 AP / Recall@τ / 事件命中率 / 正样本率
    thr = main_results["thr_global"]
    test_apply_ids = test["apply_id"].astype(str).values
    flagged = set(test_apply_ids[s_test > thr])
    is_dec = month == 12
    e05 = {}
    for at in config.ANOMALY_TYPES:
        y_at = test["anomaly_types"].str.contains(at, na=False, regex=False).astype(int).values
        e05[at] = {}
        for grp, mask in (("1-11月", ~is_dec), ("12月", is_dec)):
            yy = y_at[mask]
            ss = s_test[mask]
            recall = None
            ap = None
            if yy.sum() > 0:
                ap = average_precision_score(yy, ss)
                tp = int(((ss > thr) & (yy == 1)).sum())
                fn = int(((ss <= thr) & (yy == 1)).sum())
                recall = tp / (tp + fn) if tp + fn else 0.0
            e05[at][grp] = {"ap": ap, "recall@tau": recall, "event_recall": None,
                            "pos_rate": float(yy.mean())}
    # 事件命中率按事件 start_time 所属月份组补算（每类）
    test_aid_set = set(test_apply_ids)
    tev = events[events["apply_ids"].apply(lambda s: bool(s & test_aid_set))]
    for at in config.ANOMALY_TYPES:
        sub = tev[tev["anomaly_type"] == at]
        if sub.empty or "start_time" not in sub.columns:
            continue
        st = pd.to_datetime(sub["start_time"], errors="coerce")
        for grp, is_d in (("1-11月", False), ("12月", True)):
            if is_d:
                in_grp = sub[st.dt.month == 12]
            else:
                in_grp = sub[st.dt.month != 12]
            if len(in_grp) == 0:
                continue
            caught = sum(1 for _, ev in in_grp.iterrows() if ev["apply_ids"] & flagged)
            e05[at][grp]["event_recall"] = caught / len(in_grp)

    # E01/E02/E07/E08：引用既有审计产物
    e01 = {"note": "as-of v2 泄漏审计 max|Δ|=0（audit/leakage_audit.json）；"
                   "未来节点在快照中孤立+零种子，全图与 as-of 历史评分一致。"}
    e02 = {"cross_split_events": 0, "source": "data_fixed/DATA_VERSION.json"}
    e07 = {"max_abs_diff": 0.0, "pass": True, "source": "audit/leakage_audit.json"}
    e08 = {"status": "pending", "note": "需 ≥3 种子重生成数据(仿真V3.py)后各自重跑，本阶段未执行。"}

    matrix = {"E01": e01, "E02": e02, "E03": e03, "E04": e04,
              "E05": e05, "E06_ablation": None, "E07": e07, "E08": e08}
    _json_dump(matrix, os.path.join(V2_DIR, "matrix_E01_E08.json"))
    print("[matrix] E01-E08 已生成（E06 见 ablation.json）")
    return matrix


def main():
    os.makedirs(V2_DIR, exist_ok=True)
    print("=" * 100)
    print("protocol_v2 全量重跑（Phase 4）→ " + V2_DIR)
    print("=" * 100)

    manifest = {
        "protocol_version": "v2_causal_asof",
        "generated_at": "2026-10-16",
        "data": {"db": config.DB_PATH, "sha256": _sha256(config.DB_PATH)},
        "config": {"file": "src/config.py",
                   "sha256": _sha256(os.path.join(config.PROJECT_DIR, "src", "config.py")),
                   "train": [config.TRAIN_START, config.TRAIN_END],
                   "val": [config.VAL_START, config.VAL_END],
                   "test": [config.TEST_START, config.TEST_END]},
        "params": {"rho": 0.6, "tau": 1.0, "gamma": 30.0,
                   "alert_budget_ratio": calibration.ALERT_BUDGET_RATIO,
                   "alpha_fpr": calibration.ALPHA_FPR_DEFAULT},
        "score_time_mode": config.SCORE_TIME_MODE,
    }
    _json_dump(manifest, os.path.join(V2_DIR, "MANIFEST.json"))

    bundle = load_bundle()
    bscores = compute_baseline_scores(bundle[0], bundle[6], bundle[3])
    results, pred = run_main()
    run_baselines(bundle, bscores)
    run_compare(bundle, pred, bscores)
    ablation = run_ablation()
    robustness = run_robustness()
    matrix = run_matrix(bundle, results, pred)
    matrix["E06_ablation"] = ablation
    _json_dump(matrix, os.path.join(V2_DIR, "matrix_E01_E08.json"))

    summary = {
        "main": {k: results[k] for k in ("point_auprc", "point_auc", "normalized_ap_excess",
                                          "far_per_1000", "fp_per_1000_all", "thr_global", "tau_fpr")},
        "point": results["point"], "event_strict": results["event_strict"],
        "n_ablation_variants": len(ablation),
    }
    _json_dump(summary, os.path.join(V2_DIR, "protocol_v2_summary.json"))
    print("\n[phase4] 全部完成 → " + V2_DIR)


if __name__ == "__main__":
    main()
