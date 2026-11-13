"""
MS-TRP 检测框架 —— 基线实验主入口
========================================
流程：加载数据 → 构建标签 → 时间前向划分 → 参考期统计 → 6 种基线打分
      → 阈值无关指标(AUPRC/VUS-PR/ROC) + 固定工作点(5%/10%) + 事件级召回

注：本报告为基线自身口径。与 MS-TRP 的跨方法对比（统一标签口径 + 统一 top-k 操作点）
    请运行 python -m experiment.compare（输出 result_raw/compare_report.csv）。

评估口径（对应任务书"点级F1 / 事件级F1 / 每类Recall / AUPRC / VUS-PR"）：
  · 点级异常  = AMOUNT_SPIKE + FREQ_BURST（逐笔金额/频次错误）
  · 事件级异常 = 其余4类模式异常（一个事件=一组受影响支付）
用法：在 MS-TRP/ 目录下执行  python -m experiment.run_baselines
"""
import os
import sys

import numpy as np
import pandas as pd

# 让脚本在 PyCharm(脚本模式)或命令行(python -m)下都能从项目根 MS-TRP/ 导入 src、baselines
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


def _topk_threshold(scores, k_ratio):
    """返回恰好标记 top-k_ratio 比例的分数阈值。"""
    sc = np.asarray(scores, dtype=float)
    k = max(1, int(round(len(sc) * k_ratio)))
    return float(np.sort(sc)[-k])


def _is_point(types_list, point_set):
    return any(t in point_set for t in types_list)


def run(seed: int = 2026):
    # ── 1. 数据与标签 ───────────────────────────────
    payments = data.load_payments()
    labels = data.load_labels()
    events = data.build_events(labels)
    point_labels = data.build_point_labels(payments, events)

    # ── 2. 时间前向划分 ─────────────────────────────
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    print(f"样本划分: train(2023~2024)={len(train_idx)}  val(2025)={len(val_idx)}  "
          f"test(2026)={len(test_idx)}")

    # ── 3. 参考期统计（只在 train 上计算）─────────────
    ref_stats = features.compute_reference_stats(payments, train_idx)

    # ── 4. 标签掩码：全部 / 点级 / 事件级 ─────────────
    POINT_SET = set(config.POINT_ANOMALY_TYPES)
    EVENT_SET = set(config.EVENT_ANOMALY_TYPES)

    y_all = point_labels["is_anomaly"].astype(int)
    y_point = point_labels["anomaly_types"].apply(lambda t: _is_point(t, POINT_SET)).astype(int)

    y_all_test = y_all.loc[test_idx].values
    y_point_test = y_point.loc[test_idx].values
    y_point_val = y_point.loc[val_idx].values

    test_apply_ids = point_labels.loc[test_idx, "apply_id"].astype(str).values
    test_aid_set = set(test_apply_ids)
    test_all_events = events[events["apply_ids"].apply(lambda s: bool(s & test_aid_set))].copy()
    test_pattern_events = test_all_events[test_all_events["anomaly_type"].isin(EVENT_SET)].copy()

    rows = []
    for name, func in BASELINE_FUNCS.items():
        scores = func(payments, ref_stats, train_idx)
        s_test = scores.loc[test_idx].values.astype(float)
        s_val = scores.loc[val_idx].values.astype(float)

        # 阈值无关指标（test）
        auprc_all = metrics.auprc(y_all_test, s_test)
        nap_all = metrics.normalized_ap_excess(y_all_test, s_test)
        roc_all = metrics.roc_auc(y_all_test, s_test)
        auprc_point = metrics.auprc(y_point_test, s_test)
        nap_point = metrics.normalized_ap_excess(y_point_test, s_test)

        # 报警预算工作点（点级，test）
        top5 = metrics.metrics_at_topk(y_point_test, s_test, 0.05)
        top10 = metrics.metrics_at_topk(y_point_test, s_test, 0.10)

        # 事件级（@10% 固定工作点阈值，test）
        thr10 = _topk_threshold(s_test, 0.10)
        aid2score = metrics.flag_payments(s_test, test_apply_ids, thr10)
        em_pattern = metrics.event_metrics(test_pattern_events, aid2score, thr10)
        pc = metrics.per_type_event_recall(test_all_events, aid2score, thr10)

        rows.append({
            "baseline": name,
            "auprc_all": auprc_all, "normalized_ap_excess_all": nap_all, "roc_all": roc_all,
            "auprc_point": auprc_point, "normalized_ap_excess_point": nap_point,
            "recall_point@5%": top5["recall"], "recall_point@10%": top10["recall"],
            "event_recall@10%": em_pattern["event_recall"],
            "event_f1@10%": em_pattern["event_f1"],
            **{f"event_recall_{at}": pc[at] for at in config.ANOMALY_TYPES},
        })

    return pd.DataFrame(rows)


def main():
    res = run().round(4)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)

    print("\n" + "=" * 120)
    print("  基线实验报告（时间前向 train=2023~2024 / val=2025 / test=2026）")
    print("=" * 120)

    print("\n【表1】阈值无关指标（test）")
    t1 = ["baseline", "auprc_all", "normalized_ap_excess_all", "roc_all",
          "auprc_point", "normalized_ap_excess_point"]
    print(res[t1].to_string(index=False))

    print("\n【表2】报警预算工作点（test）")
    t2 = ["baseline", "recall_point@5%", "recall_point@10%", "event_recall@10%", "event_f1@10%"]
    print(res[t2].to_string(index=False))

    print("\n【表3】每类事件召回率 @10% 工作点（test）")
    t3 = ["baseline"] + [f"event_recall_{at}" for at in config.ANOMALY_TYPES]
    print(res[t3].to_string(index=False))
    print("=" * 120)

    out_path = config.RESULT_DIR + "/baseline_report.csv"
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n报告已保存: {out_path}")
    print("与 MS-TRP 的跨方法对比（统一口径 + 统一操作点）：python -m experiment.compare")


if __name__ == "__main__":
    main()
