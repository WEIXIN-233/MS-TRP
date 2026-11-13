"""
实验补充 —— 显著性检验 + 超参数敏感性
========================================
在「静态边前向泄漏修复」后的代码上跑，含三块：
  1) 主实验（新代码）→ 得到修复后的 AUPRC/AUC/事件严格F1，与旧值(0.6209)对照；
  2) 显著性：配对 bootstrap（B=1000，test 重抽样）对「MS-TRP − 各基线」的 AUPRC 差值
     报告均值 / 95% 分位 CI / P(差≤0)（单侧：MS-TRP 不优于基线的概率）；
  3) 敏感性：单变量扫描 rho / tau / gamma / delta_event，报告 point AUPRC / AUC /
     归一化AP超额 / 事件严格 F1 / FPR / FAR(每千)。

输出 result_raw/protocol_v2_causal/sensitivity_significance/
  main_metrics.json / significance.json / sensitivity.json
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, split, pipeline, metrics
from src import feature_engineering as features
import baselines as blm

OUT_DIR = os.path.join(config.RESULT_DIR, "protocol_v2_causal", "sensitivity_significance")

BASELINE_FUNCS = {
    "fixed_rules":      lambda pay, ref, tr: blm.fixed_rules_scores(pay, ref),
    "robust_stats":     lambda pay, ref, tr: blm.robust_stats_scores(pay, ref),
    "isolation_forest": lambda pay, ref, tr: blm.isolation_forest_scores(pay, ref, tr),
    "lof":              lambda pay, ref, tr: blm.lof_scores(pay, ref, tr),
    "residual":         lambda pay, ref, tr: blm.residual_scores(pay, ref),
    "graph":            lambda pay, ref, tr: blm.graph_scores(pay),
}

SWEEPS = {
    "rho":         [0.2, 0.4, 0.6, 0.8],
    "tau":         [0.5, 1.0, 2.0, 4.0],
    "gamma":       [10.0, 30.0, 100.0],
    "delta_event": [3.0, 5.0, 7.0, 10.0, 14.0],
}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if f != f else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, ensure_ascii=False, indent=2)


def _grab(res):
    """从 pipeline 结果里抽敏感性要看的指标。"""
    return {
        "point_auprc": res["point_auprc"],
        "point_auc": res["point_auc"],
        "normalized_ap_excess": res["normalized_ap_excess"],
        "event_strict_f1": res["event_strict"]["event_f1"],
        "event_strict_recall": res["event_strict"]["event_recall"],
        "event_strict_precision": res["event_strict"]["event_precision"],
        "fpr": res["fpr"],
        "far_per_1000": res["far_per_1000"],
        "thr_global": res["thr_global"],
    }


def run_significance(payments, point_labels, events, train_idx, val_idx, test_idx,
                     ref_stats, ms_scores):
    y_all = point_labels["is_anomaly"].astype(int)
    y_all_test = y_all.loc[test_idx].values

    # 事件级标签（其余 4 类模式级）
    EVENT_SET = set(config.EVENT_ANOMALY_TYPES)
    y_event = point_labels["anomaly_types"].apply(
        lambda t: any(x in EVENT_SET for x in t)).astype(int)
    y_event_test = y_event.loc[test_idx].values

    base = {}
    for name, func in BASELINE_FUNCS.items():
        s = func(payments, ref_stats, train_idx)
        base[name] = np.asarray(s.loc[test_idx].values, dtype=float)

    rng = np.random.RandomState(2026)
    n = len(test_idx)
    B = 1000
    idx = np.arange(n)

    def _boot(y, s_ms, s_bs):
        diffs = np.empty(B)
        for b in range(B):
            r = rng.choice(idx, n, replace=True)
            diffs[b] = (average_precision_score(y[r], s_ms[r])
                        - average_precision_score(y[r], s_bs[r]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return {"diff_mean": float(diffs.mean()), "ci95_lo": float(lo),
                "ci95_hi": float(hi), "p_le_0": float((diffs <= 0).mean())}

    out = {"B": B, "baselines": {}}
    for name, bs in base.items():
        entry = {
            "auprc": float(average_precision_score(y_all_test, bs)),
            "auprc_event": float(average_precision_score(y_event_test, bs)),
            "signif_all": _boot(y_all_test, ms_scores, bs),
            "signif_event": _boot(y_event_test, ms_scores, bs),
        }
        out["baselines"][name] = entry
    out["ms_trp"] = {
        "auprc": float(average_precision_score(y_all_test, ms_scores)),
        "auprc_event": float(average_precision_score(y_event_test, ms_scores)),
    }
    return out


def run_sensitivity():
    rows = {}
    for param, vals in SWEEPS.items():
        rows[param] = []
        for v in vals:
            kw = {param: v}
            res, _ = pipeline.run_ms_trp(**kw)
            rec = {"value": v, **_grab(res)}
            rows[param].append(rec)
            print(f"  [sensitivity] {param}={v:<5}  AUPRC={res['point_auprc']:.4f}  "
                  f"strictF1={res['event_strict']['event_f1']:.4f}  "
                  f"FAR/1000={res['far_per_1000']:.2f}", flush=True)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 100)
    print("显著性 + 敏感性（静态边泄漏修复后代码）")
    print("=" * 100, flush=True)

    # ── 数据与标签 ──────────────────────────────
    payments = data.load_payments()
    labels = data.load_labels()
    events = data.build_events(labels)
    point_labels = data.build_point_labels(payments, events)
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    ref_stats = features.compute_reference_stats(payments, train_idx)

    # ── 1. 主实验（修复后）──────────────────────
    print("[1/3] 主实验（修复后）...", flush=True)
    res, pred = pipeline.run_ms_trp()
    pred_test = pred[pred["split"] == "test"].reset_index(drop=True)
    ms_scores = pred_test["s_final"].astype(float).values
    main_metrics = _grab(res)
    main_metrics["old_auprc_before_fix"] = 0.6209   # 阶段4 记录的旧值，供对照
    _dump(main_metrics, os.path.join(OUT_DIR, "main_metrics.json"))
    print(f"  → AUPRC={res['point_auprc']:.4f}（旧=0.6209）  AUC={res['point_auc']:.4f}  "
          f"strictF1={res['event_strict']['event_f1']:.4f}  "
          f"FAR/1000={res['far_per_1000']:.2f}", flush=True)

    # ── 2. 显著性 ──────────────────────────────
    print("[2/3] 显著性（配对 bootstrap B=1000）...", flush=True)
    signif = run_significance(payments, point_labels, events, train_idx, val_idx,
                              test_idx, ref_stats, ms_scores)
    _dump(signif, os.path.join(OUT_DIR, "significance.json"))
    for name, e in signif["baselines"].items():
        s = e["signif_all"]
        print(f"  MS-TRP - {name:<16s}  AUPRC差={s['diff_mean']:+.4f}  "
              f"95%CI=[{s['ci95_lo']:+.4f},{s['ci95_hi']:+.4f}]  "
              f"P(≤0)={s['p_le_0']:.3f}", flush=True)

    # ── 3. 敏感性 ──────────────────────────────
    print("[3/3] 敏感性扫描...", flush=True)
    sens = run_sensitivity()
    _dump(sens, os.path.join(OUT_DIR, "sensitivity.json"))

    print("\n" + "=" * 100)
    print(f"完成，产物见 {OUT_DIR}/")
    print("=" * 100)


if __name__ == "__main__":
    main()
