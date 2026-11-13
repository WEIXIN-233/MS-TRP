"""
实验 E12 —— 多尺度信息量匹配消融（窗口多样性 vs 特征数增加）
============================================================
验收判据（导师）：明确多尺度增益是"窗口多样性"带来的，而非仅"特征数量增加"。

设计（s_ts 由「金额长尺度 + 若干频次窗口」取 max 组成，每项都算一个尺度信号）：

  单因素去留（回答"哪个尺度在起作用"）：
    完整(amount+7+30+Q)   4 尺度，参照
    单尺度(amount)        1 尺度
    去7日(amount+30+Q)    3 尺度
    去30日(amount+7+Q)    3 尺度
    去金额(7+30+Q)        3 尺度
    去当季(amount+7+30)   3 尺度
    仅当季(amount+Q)      2 尺度   ← 导师"7d/30d/Q/Y 各自实验"
    仅年度(amount+365)    2 尺度

  等维对照（固定 4 个尺度信号，只换窗口值 → 隔离"特征数"与"窗口选择"）：
    完整(amount+7+30+Q)     4 尺度
    等维Q/Y(amount+90+365+Q)4 尺度
    等维5/60(amount+5+60+Q) 4 尺度

结论判据：
  · 多尺度增益 = 完整 − 单尺度（>0 说明多尺度优于单尺度）
  · 窗口选择增益 = 完整 − max(等维Q/Y, 等维5/60)（>0 说明 7/30 窗口选择本身优于
    其他同维窗口，即增益来自窗口多样性而非仅特征数增加）

输出 result_raw/window_ablation.csv + window_ablation.json。
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Windows GBK 控制台会因 U+2212 等字符崩溃，统一重配 stdout 为 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import config, pipeline

OUT_DIR = os.path.join(config.RESULT_DIR, "window_ablation")

# (变体名, n_尺度信号, 说明, kwargs)。当季截至当前（Q）默认常开，算作一个尺度信号。
VARIANTS = [
    ("完整(amount+7+30+Q)", 4, "金额+7日+30日+当季", {}),
    ("单尺度(amount)", 1, "仅金额长尺度", {"use_multiscale": False}),
    ("去7日(amount+30+Q)", 3, "金额+30日+当季", {"freq_windows": (30,)}),
    ("去30日(amount+7+Q)", 3, "金额+7日+当季", {"freq_windows": (7,)}),
    ("去金额(7+30+Q)", 3, "7日+30日+当季（无金额）", {"use_amount_scale": False}),
    ("去当季(amount+7+30)", 3, "金额+7日+30日（无当季）", {"use_quarter_scale": False}),
    ("仅当季(amount+Q)", 2, "金额+当季", {"freq_windows": ()}),
    ("仅年度(amount+365)", 2, "金额+365日年度频次", {"freq_windows": (365,), "use_quarter_scale": False}),
    ("等维Q/Y(amount+90+365+Q)", 4, "金额+90日+365日+当季（同维对照）", {"freq_windows": (90, 365)}),
    ("等维5/60(amount+5+60+Q)", 4, "金额+5日+60日+当季（同维对照）", {"freq_windows": (5, 60)}),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 100)
    print("E12 多尺度信息量匹配消融：窗口多样性 vs 特征数增加")
    print("=" * 100)

    rows = []
    summary = {}
    csv_rows = []
    for name, n_scale, desc, kw in VARIANTS:
        res, pred = pipeline.run_ms_trp(**kw)
        ap = res["point_auprc"]
        row = [name, str(n_scale), desc, f"{ap:.4f}"]
        for at in config.ANOMALY_TYPES:
            row.append(f"{res['per_type_auprc'][at]:.4f}")
        rows.append(row)
        summary[name] = {
            "n_scales": n_scale,
            "freq_windows": list(kw.get("freq_windows", (7, 30))),
            "use_amount": kw.get("use_amount_scale", True),
            "use_quarter": kw.get("use_quarter_scale", True),
            "overall_auprc": ap,
            "per_type_auprc": res["per_type_auprc"],
            "event_recall": res["event"]["event_recall"],
            "event_f1": res["event"]["event_f1"],
        }
        csv_rows.append({
            "variant": name,
            "n_scales": n_scale,
            "freq_windows": "+".join(map(str, kw.get("freq_windows", (7, 30)))),
            "use_amount": kw.get("use_amount_scale", True),
            "use_quarter": kw.get("use_quarter_scale", True),
            "AP": round(ap, 6),
            "F1": round(res["point"]["f1"], 6),
            "FAR_1000": round(res["far_per_1000"], 4),
        })
        print(f"  {name:<26s} ({n_scale}尺度, {desc:<18s}) 整体 AUPRC = {ap:.4f}")

    # 结论判据
    full = summary["完整(amount+7+30+Q)"]["overall_auprc"]
    single = summary["单尺度(amount)"]["overall_auprc"]
    equal_max = max(summary["等维Q/Y(amount+90+365+Q)"]["overall_auprc"],
                    summary["等维5/60(amount+5+60+Q)"]["overall_auprc"])
    verdict = {
        "multi_scale_gain": round(full - single, 4),
        "window_choice_gain": round(full - equal_max, 4),
        "conclusion": (
            "增益来自窗口多样性（7/30 选择本身有效），非仅特征数增加"
            if (full - equal_max) > 1e-4 else
            "增益可能主要来自特征数增加，窗口多样性贡献未得到证实"
        ),
    }
    summary["verdict"] = verdict
    print("\n结论判据：")
    print(f"  多尺度增益 = 完整 - 单尺度          = {verdict['multi_scale_gain']:+.4f}")
    print(f"  窗口选择增益 = 完整 - max(等维对照)  = {verdict['window_choice_gain']:+.4f}")
    print(f"  => {verdict['conclusion']}")

    headers = ["变体", "尺度数", "说明", "整体 AUPRC"] + config.ANOMALY_TYPES
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(headers))
    sep = "  " + "  ".join("-" * w[i] for i in range(len(headers)))
    body = "\n".join("  " + "  ".join(r[i].ljust(w[i]) for i in range(len(headers))) for r in rows)
    print("\n" + line + "\n" + sep + "\n" + body)

    with open(os.path.join(OUT_DIR, "window_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, "window_ablation.csv"),
                                  index=False, encoding="utf-8-sig")
    # 同时落一份到 result_raw 根目录（导师验收点名 window_ablation.csv）
    pd.DataFrame(csv_rows).to_csv(os.path.join(config.RESULT_DIR, "window_ablation.csv"),
                                  index=False, encoding="utf-8-sig")
    print(f"\n结果已写入 {OUT_DIR}/window_ablation.csv(+.json) 及 {config.RESULT_DIR}/window_ablation.csv")


if __name__ == "__main__":
    main()
