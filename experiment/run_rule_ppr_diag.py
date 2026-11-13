# -*- coding: utf-8 -*-
"""审稿意见第4条（优先级1）：规则→PPR 召回回落的逐层中间统计。

对 +Rule(账户一致性规则) 与 +Rule+PPR 两个变体，按同一测试集逐层输出：
  规则命中支付数 → 命中支付最终分数 → 冻结全局阈值 → 支付告警数
  → 聚合后的预测事件数 → 宽松命中事件数（任一成员告警）
  → 严格一对一命中事件数（实体一致 + 时间交叠）
  → 实体聚集类事件的召回（宽松口径）

结论写入 result_raw/rule_ppr_diag.csv。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, event_eval, pipeline  # noqa: E402

VARIANTS = [
    ("+Rule", {"use_ppr": False, "use_struct": False}),
    ("+Rule+PPR", {"use_struct": False}),
    ("完整模型", {}),
]


def main():
    meta = pd.read_csv(os.path.join(config.PROJECT_DIR, "data_fixed", "ground_truth_event.csv"),
                       encoding="utf-8-sig", usecols=["event_id", "anomaly_subtype"])
    subtype = dict(zip(meta["event_id"].astype(str), meta["anomaly_subtype"].astype(str)))
    payments = data.load_payments()
    payments["_aid"] = payments["apply_id"].astype(str)
    labels = data.load_labels()
    events = data.build_events(labels)

    rows = []
    kept = {}
    for name, kw in VARIANTS:
        res, pred = pipeline.run_ms_trp(**kw)
        thr = float(res["thr_global"])
        te = pred[pred["split"] == "test"].copy()
        te["s_final"] = te["s_final"].astype(float)
        te["_aid"] = te["apply_id"].astype(str)

        rule_hit = te["s_mismatch"].astype(float) >= 0.999
        flagged_ids = set(te.loc[te["s_final"] > thr, "_aid"])
        flagged = te["s_final"] > thr

        # 事件侧：与 pipeline 同口径（只保留测试集内有成员的 gt 事件）
        test_ids = set(te["_aid"])
        test_events = events[events["apply_ids"].apply(
            lambda ids: any(str(a) in test_ids for a in ids))].reset_index(drop=True)
        n_events = len(test_events)

        loose_ids = flagged_ids
        loose = sum(1 for _, ev in test_events.iterrows()
                    if ev["apply_ids"] & loose_ids)
        ent_events = test_events[test_events["anomaly_type"] == "ENTITY_AGGREGATION"]
        per_type = {}
        for at in config.ANOMALY_TYPES:
            sub = test_events[test_events["anomaly_type"] == at]
            caught = sum(1 for _, ev in sub.iterrows() if ev["apply_ids"] & loose_ids)
            per_type[at] = (caught, len(sub))
        ent_caught, ent_n = per_type["ENTITY_AGGREGATION"]

        # 严格一对一（实体一致 + 时间交叠）
        true_ev = event_eval.build_true_events(test_events)
        flagged_df = payments[payments["_aid"].isin(flagged_ids)]
        if len(true_ev) and len(flagged_df):
            alerts = event_eval.build_payment_alerts(flagged_df)
            pred_ev = event_eval.cluster_alerts(alerts, delta_event=event_eval.DELTA_EVENT_DEFAULT)
            strict = event_eval.match_events(true_ev, pred_ev)
        else:
            pred_ev, strict = pd.DataFrame(), {"tp": 0, "fn": len(true_ev), "fp": 0,
                                               "event_recall": 0.0, "event_precision": 0.0}

        ent_sub = ent_events.assign(
            subtype=ent_events["event_id"].astype(str).map(subtype).fillna("NA"))
        sub_split = "; ".join(
            f"{st}={sum(1 for _, ev in g.iterrows() if ev['apply_ids'] & loose_ids)}/{len(g)}"
            for st, g in ent_sub.groupby("subtype"))

        # 规则通道单独的覆盖（阈值<1 时与变体无关，用作不变性核对）
        rule_ids = set(te.loc[rule_hit, "_aid"])
        ent_rule = sum(1 for _, ev in ent_events.iterrows()
                       if ev["apply_ids"] & rule_ids)
        kept[name] = (te, test_events, ent_events, rule_ids, ent_rule)

        # 10% 误报预算下的实体聚集子类覆盖（仅作诊断，不作论文主口径）
        val = pred[pred["split"] == "val"]
        thr10 = float(np.quantile(val.loc[val["is_anomaly"] == 0, "s_final"], 0.90))
        ids10 = set(te.loc[te["s_final"] > thr10, "_aid"])
        sub10 = "; ".join(
            f"{st}={sum(1 for _, ev in g.iterrows() if ev['apply_ids'] & ids10)}/{len(g)}"
            for st, g in ent_sub.groupby("subtype"))

        rows.append({
            "variant": name,
            "thr_far10": round(thr10, 4),
            "entity_by_subtype_far10": sub10,
            "thr_global": round(thr, 4),
            "n_test": int(len(te)),
            "n_rule_hit": int(rule_hit.sum()),
            "rule_hit_min_score": round(float(te.loc[rule_hit, "s_final"].min()), 4)
            if rule_hit.any() else np.nan,
            "rule_hit_flagged": int((rule_hit & flagged).sum()),
            "n_alerts": int(len(flagged_ids)),
            "n_pred_events": int(len(pred_ev)),
            "n_events_loose": int(loose),
            "n_events": int(n_events),
            "loose_recall": round(loose / n_events, 4) if n_events else np.nan,
            "strict_tp": int(strict.get("tp", 0)),
            "strict_recall": round(float(strict.get("event_recall", 0.0)), 4),
            "entity_events": int(ent_n),
            "entity_caught_loose": int(ent_caught),
            "entity_recall_loose": round(ent_caught / ent_n, 4) if ent_n else np.nan,
            "point_auprc": round(float(res["point_auprc"]), 4),
            "entity_by_subtype": sub_split,
            "per_type_recall": "; ".join(
                f"{k}={v[0]}/{v[1]}({v[0] / v[1]:.4f})" if v[1] else f"{k}=0/0"
                for k, v in per_type.items()),
        })
        print(rows[-1], flush=True)

    # ── 交叉核对：固定分数、只换阈值（审查意见第3条的阈值分离实验）──
    thr = {r["variant"]: r["thr_global"] for r in rows}
    print("\n固定分数、只换阈值后的实体聚集命中（分母 43）：", flush=True)
    cross = []
    for score_name, (te_s, _, ent_s, _, _) in kept.items():
        for thr_name, t in thr.items():
            ids = set(te_s.loc[te_s["s_final"] > t, "_aid"])
            n = sum(1 for _, ev in ent_s.iterrows() if ev["apply_ids"] & ids)
            cross.append({"scores_from": score_name, "threshold_from": thr_name,
                          "thr": t, "entity_caught": n, "entity_recall": round(n / 43, 4)})
            print(f"  分数={score_name:<10s} 阈值={thr_name:<10s}({t:.4f}) -> {n}/43", flush=True)
    print("规则命中支付对实体聚集事件的覆盖：")
    for name, (_, _, _, rule_ids, ent_rule) in kept.items():
        print(f"  {name:<10s} 规则命中 {len(rule_ids)} 笔，覆盖实体聚集事件 {ent_rule}/43", flush=True)

    pd.DataFrame(cross).to_csv(
        os.path.join(config.RESULT_DIR, "rule_ppr_threshold_cross.csv"),
        index=False, encoding="utf-8-sig")
    out = os.path.join(config.RESULT_DIR, "rule_ppr_diag.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("written:", out)


if __name__ == "__main__":
    main()
