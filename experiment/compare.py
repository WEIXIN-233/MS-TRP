"""
统一口径对比：MS-TRP 与 6 种基线在同一操作点上对比
====================================================
修复两个口径问题：
  1) 操作点不一致：原 MS-TRP 用分层校准阈值(thr_global≈1.3%误报)报 P/R/F1，
     基线用 top-5%/10% 报，二者不在同一操作点，无法对比；
  2) 标签口径不一致：原 MS-TRP 的 point_auprc 用「全量6类」标签，
     基线的 auprc_point 用「仅点级2类(AMOUNT_SPIKE+FREQ_BURST)」标签，
     直接对比是错配（正确应比 MS-TRP vs 基线的 auprc_all）。

本脚本统一：
  · 标签口径  all  = 全部 6 类；point = AMOUNT_SPIKE+FREQ_BURST；
             event = 其余 4 类(模式级)
  · 阈值无关  AUPRC / ROC-AUC / 归一化AP超额（同一标签口径，可直接比）
  · 固定操作点 top-1% / 5% / 10% 的点级 P/R/F1；top-10% 的事件级 recall/F1
  · 所有方法（含 MS-TRP）用完全相同的函数与操作点计算

输出 result_raw/compare_report.csv
用法： python -m experiment.compare
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, split, metrics
from src import feature_engineering as features
import baselines as blm


BASELINE_FUNCS = {
    "fixed_rules":      lambda pay, ref, tr: blm.fixed_rules_scores(pay, ref),
    "robust_stats":     lambda pay, ref, tr: blm.robust_stats_scores(pay, ref),
    "isolation_forest": lambda pay, ref, tr: blm.isolation_forest_scores(pay, ref, tr),
    "lof":              lambda pay, ref, tr: blm.lof_scores(pay, ref, tr),
    "residual":         lambda pay, ref, tr: blm.residual_scores(pay, ref),
    "graph":            lambda pay, ref, tr: blm.graph_scores(pay),
    # ── 指导书 B5 九行强制基线 ──
    "random_uniform":   lambda pay, ref, tr: blm.random_uniform_scores(pay),
    "constant_score":   lambda pay, ref, tr: blm.constant_score_scores(pay),
    "shared_payee_count": lambda pay, ref, tr: blm.shared_payee_count_scores(pay),
    "shared_account_concentration": lambda pay, ref, tr: blm.shared_account_concentration_scores(pay),
    "ecod":             lambda pay, ref, tr: blm.ecod_scores(pay, ref, tr),
    "copod":            lambda pay, ref, tr: blm.copod_scores(pay, ref, tr),
    "mcd":              lambda pay, ref, tr: blm.mcd_scores(pay, ref, tr),
    "cblof":            lambda pay, ref, tr: blm.cblof_scores(pay, ref, tr),
    "simple_ppr":       lambda pay, ref, tr: blm.simple_ppr_scores(pay),
    # ── B7 消融对照 ──
    "concat_if":        lambda pay, ref, tr: blm.concat_if_scores(pay, ref, tr),
}

POINT_SET = set(config.POINT_ANOMALY_TYPES)   # AMOUNT_SPIKE + FREQ_BURST
EVENT_SET = set(config.EVENT_ANOMALY_TYPES)   # 其余 4 类


def _topk_threshold(scores, k_ratio):
    sc = np.asarray(scores, dtype=float)
    k = max(1, int(round(len(sc) * k_ratio)))
    return float(np.sort(sc)[-k])


def run():
    # ── 数据与标签 ───────────────────────────────
    payments = data.load_payments()
    labels = data.load_labels()
    events = data.build_events(labels)
    point_labels = data.build_point_labels(payments, events)
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    ref_stats = features.compute_reference_stats(payments, train_idx)

    y_all = point_labels["is_anomaly"].astype(int)
    y_point = point_labels["anomaly_types"].apply(
        lambda t: any(x in POINT_SET for x in t)).astype(int)
    y_event = point_labels["anomaly_types"].apply(
        lambda t: any(x in EVENT_SET for x in t)).astype(int)

    y_all_test = y_all.loc[test_idx].values
    y_point_test = y_point.loc[test_idx].values
    y_event_test = y_event.loc[test_idx].values

    test_apply_ids = point_labels.loc[test_idx, "apply_id"].astype(str).values
    test_aid_set = set(test_apply_ids)
    test_all_events = events[events["apply_ids"].apply(lambda s: bool(s & test_aid_set))].copy()
    test_pattern_events = test_all_events[test_all_events["anomaly_type"].isin(EVENT_SET)].copy()

    # ── 各方法在 test 上的分数 ─────────────────────
    methods = {}
    for name, func in BASELINE_FUNCS.items():
        scores = func(payments, ref_stats, train_idx)
        methods[name] = np.asarray(scores.loc[test_idx].values, dtype=float)

    pred = pd.read_csv(os.path.join(config.RESULT_DIR, "run_main", "predictions.csv"))
    pred_test = pred[pred["split"] == "test"].copy()
    assert len(pred_test) == len(test_idx), "MS-TRP test 样本数不匹配"
    s_map = dict(zip(pred_test["apply_id"].astype(str), pred_test["s_final"].astype(float)))
    methods["MS-TRP"] = np.array([s_map[a] for a in test_apply_ids], dtype=float)

    # ── 统一口径指标 ─────────────────────────────
    rows = []
    for name, s in methods.items():
        row = {"method": name}
        # 阈值无关（可跨方法直接比）
        row["auprc_all"] = metrics.auprc(y_all_test, s)
        row["auprc_point"] = metrics.auprc(y_point_test, s)
        row["auprc_event"] = metrics.auprc(y_event_test, s)
        row["auc_all"] = metrics.roc_auc(y_all_test, s)
        row["normalized_ap_excess_all"] = metrics.normalized_ap_excess(y_all_test, s)
        # 固定操作点：点级 P/R/F1（同一 top-k，所有方法一致）
        for k in (0.01, 0.05, 0.10):
            m = metrics.metrics_at_topk(y_point_test, s, k)
            row[f"point_f1@{int(k*100)}%"] = m["f1"]
            row[f"point_recall@{int(k*100)}%"] = m["recall"]
        # 固定操作点：事件级 recall/F1 @ top-10%（仅模式级事件）
        thr10 = _topk_threshold(s, 0.10)
        aid2score = metrics.flag_payments(pd.Series(s), test_apply_ids, thr10)
        em = metrics.event_metrics(test_pattern_events, aid2score, thr10)
        row["event_recall@10%"] = em["event_recall"]
        row["event_f1@10%"] = em["event_f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    res = run().round(4)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)

    print("\n" + "=" * 120)
    print("  统一口径对比（train=2023~2024 / val=2025 / test=2026）")
    print("  标签口径: all=全部6类 | point=AMOUNT_SPIKE+FREQ_BURST | event=其余4类")
    print("=" * 120)

    print("\n【表1】阈值无关指标（可直接跨方法对比）")
    t1 = ["method", "auprc_all", "auprc_point", "auprc_event", "auc_all", "normalized_ap_excess_all"]
    print(res[t1].sort_values("auprc_all", ascending=False).to_string(index=False))

    print("\n【表2】固定操作点 · 点级 F1/Recall（top-1%/5%/10%，所有方法同一操作点）")
    t2 = ["method", "point_f1@1%", "point_recall@1%", "point_f1@5%", "point_recall@5%",
          "point_f1@10%", "point_recall@10%"]
    print(res[t2].sort_values("point_f1@10%", ascending=False).to_string(index=False))

    print("\n【表3】固定操作点 · 事件级（top-10%，仅模式级事件）")
    t3 = ["method", "event_recall@10%", "event_f1@10%"]
    print(res[t3].sort_values("event_recall@10%", ascending=False).to_string(index=False))
    print("=" * 120)

    out = os.path.join(config.RESULT_DIR, "compare_report.csv")
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
