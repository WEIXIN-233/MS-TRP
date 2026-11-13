"""
B5/B7 —— 六类 category.csv + 事件指标 event_metrics.csv + 三族 metrics_family.csv + paired_effects.csv
========================================================================================================
  category.csv      六类异常从真实事件分组统计 Recall 与样本量（method,type,n_event,n_positive,recall）
  event_metrics.csv MS-TRP 事件级：event_precision/event_recall/event_f1/首警延迟 delay(天)
  metrics_family.csv 三族指标（点/区间/排序）→ 结果文件 + 是否阈值相关 + 是否用于主结论
  paired_effects.csv MS-TRP 与结构启发式/Simple PPR 在 ENTITY_AGGREGATION 上的成对效应（主库）

category / event_metrics 均来自 predictions.csv（MS-TRP）+ 冻结阈值 thr_global，不回溯测试标签选阈值。
用法：python -m experiment.run_summary_metrics
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, data, split


def main():
    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    thr = ec.load_thresholds(None)["thr_global"]

    pred = ec.load_predictions(None)
    pred["apply_id"] = pred["apply_id"].astype(str)
    s_map = pred.set_index("apply_id")["s_final"].to_dict()
    date_map = dict(zip(payments["apply_id"].astype(str), pd.to_datetime(payments["apply_date"])))

    # 测试集事件（真实）
    apply_test = payments["apply_id"].astype(str).iloc[test_idx].values
    test_set = set(apply_test)
    test_events = events[events["apply_ids"].apply(
        lambda ids: any(str(a) in test_set for a in ids))].reset_index(drop=True)

    # ── category.csv：六类事件召回 ──
    cat_rows = []
    for at in config.ANOMALY_TYPES:
        sub = test_events[test_events["anomaly_type"] == at]
        n_event = len(sub)
        n_pos = sum(len([a for a in ev["apply_ids"] if str(a) in test_set])
                    for _, ev in sub.iterrows())
        caught = 0
        for _, ev in sub.iterrows():
            if any(s_map.get(str(a), 0.0) > thr for a in ev["apply_ids"] if str(a) in test_set):
                caught += 1
        cat_rows.append({
            "method": "MS-TRP",
            "type": at,
            "n_event": n_event,
            "n_positive": n_pos,
            "recall": round(caught / n_event, 6) if n_event else np.nan,
        })
    pd.DataFrame(cat_rows).to_csv(os.path.join(config.RESULT_DIR, "category.csv"),
                                  index=False, encoding="utf-8-sig")

    # ── event_metrics.csv：MS-TRP 事件级 + 首警延迟 ──
    ev_match = ec.event_strict_f1(payments, events, test_idx,
                                  pred.set_index("apply_id")["s_final"].reindex(
                                      payments["apply_id"].astype(str)).values, thr)
    delays = []
    for _, ev in test_events.iterrows():
        st = pd.to_datetime(ev["start_time"], errors="coerce")
        if pd.isna(st):
            continue
        member_dates = [date_map[str(a)] for a in ev["apply_ids"]
                        if str(a) in test_set and str(a) in date_map
                        and s_map.get(str(a), 0.0) > thr]
        if member_dates:
            delays.append(float((min(member_dates) - st).total_seconds() / 86400.0))
    mean_delay = float(np.mean(delays)) if delays else np.nan

    ev_rows = [{
        "method": "MS-TRP",
        "event_precision": round(ev_match["event_precision"], 6),
        "event_recall": round(ev_match["event_recall"], 6),
        "event_f1": round(ev_match["event_f1"], 6),
        "delay": round(mean_delay, 4),
    }]
    pd.DataFrame(ev_rows).to_csv(os.path.join(config.RESULT_DIR, "event_metrics.csv"),
                                 index=False, encoding="utf-8-sig")

    # ── metrics_family.csv：三族 → 文件 + 阈值相关 + 主结论 ──
    fam_rows = [
        {"family": "point", "metric": "payment_auprc", "output_file": "metrics_main.csv",
         "threshold_dependent": False, "used_for_main_conclusion": True},
        {"family": "point", "metric": "payment_f1", "output_file": "metrics_main.csv",
         "threshold_dependent": True, "used_for_main_conclusion": True},
        {"family": "point", "metric": "precision", "output_file": "metrics_main.csv",
         "threshold_dependent": True, "used_for_main_conclusion": True},
        {"family": "point", "metric": "recall", "output_file": "metrics_main.csv",
         "threshold_dependent": True, "used_for_main_conclusion": True},
        {"family": "point", "metric": "far_1000", "output_file": "metrics_main.csv",
         "threshold_dependent": True, "used_for_main_conclusion": True},
        {"family": "point", "metric": "event_f1", "output_file": "metrics_main.csv",
         "threshold_dependent": True, "used_for_main_conclusion": True},
        {"family": "point", "metric": "f1_raw", "output_file": "metrics_main.csv",
         "threshold_dependent": True, "used_for_main_conclusion": True},
        {"family": "point", "metric": "recall_at_far5", "output_file": "metrics_main.csv",
         "threshold_dependent": True, "used_for_main_conclusion": True},
        {"family": "point", "metric": "recall_at_far10", "output_file": "metrics_main.csv",
         "threshold_dependent": True, "used_for_main_conclusion": True},
        {"family": "interval", "metric": "vus_pr", "output_file": "metrics_range.csv",
         "threshold_dependent": False, "used_for_main_conclusion": False},
        {"family": "ranking", "metric": "ndcg_at_k", "output_file": "metrics_ranking.csv",
         "threshold_dependent": False, "used_for_main_conclusion": False},
    ]
    pd.DataFrame(fam_rows).to_csv(os.path.join(config.RESULT_DIR, "metrics_family.csv"),
                                  index=False, encoding="utf-8-sig")

    # ── paired_effects.csv：MS-TRP vs 结构启发式/Simple PPR（主库）──
    spp = pd.read_csv(os.path.join(config.RESULT_DIR, "struct_vs_ppr_paired.csv"))
    ms = spp[spp["model"] == "MS-TRP"].iloc[0]
    pair_rows = []
    for _, r in spp[spp["model"] != "MS-TRP"].iterrows():
        pair_rows.append({
            "baseline_model": r["model"],
            "data_seed": r["data_seed"],
            "ms_trp_entity_event_recall": ms["entity_event_recall"],
            "baseline_entity_event_recall": r["entity_event_recall"],
            "paired_effect_entity_recall": round(ms["entity_event_recall"] - r["entity_event_recall"], 6),
            "ms_trp_recall_at_far5": ms["recall_at_far5"],
            "baseline_recall_at_far5": r["recall_at_far5"],
            "paired_effect_recall_at_far5": round(ms["recall_at_far5"] - r["recall_at_far5"], 6),
        })
    pd.DataFrame(pair_rows).to_csv(os.path.join(config.RESULT_DIR, "paired_effects.csv"),
                                   index=False, encoding="utf-8-sig")

    print("[category] category.csv：六类事件召回")
    print(pd.read_csv(os.path.join(config.RESULT_DIR, "category.csv")).to_string(index=False))
    print("\n[event] event_metrics.csv")
    print(pd.read_csv(os.path.join(config.RESULT_DIR, "event_metrics.csv")).to_string(index=False))
    print("\n[paired] paired_effects.csv")
    print(pd.read_csv(os.path.join(config.RESULT_DIR, "paired_effects.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
