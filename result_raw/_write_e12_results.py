# -*- coding: utf-8 -*-
"""E12 结果落盘（数字来自已完成的 run_window_ablation.py 9 变体整体 AUPRC）。
脚本原版因 GBK 编码在"结论判据"打印处崩溃、未写 CSV/JSON；此处按同一批数字补写。"""
import os, sys, json
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT = os.path.dirname(os.path.abspath(__file__))  # result_raw/
OUT_DIR = os.path.join(RESULT, "window_ablation")

# (variant, n_scales, freq_windows, use_amount, AP)
data = [
    ("完整(amount+7+30)", 3, "7+30", True, 0.4649),
    ("单尺度(amount)", 1, "", True, 0.3831),
    ("去7日(amount+30)", 2, "30", True, 0.4525),
    ("去30日(amount+7)", 2, "7", True, 0.4601),
    ("去金额(7+30)", 2, "7+30", False, 0.3974),
    ("amount+90(Q)", 2, "90", True, 0.4041),
    ("amount+365(Y)", 2, "365", True, 0.3727),
    ("等维Q/Y(amount+90+365)", 3, "90+365", True, 0.4012),
    ("等维5/60(amount+5+60)", 3, "5+60", True, 0.4420),
]

rows = []
summary = {}
for name, ns, fw, ua, ap in data:
    rows.append({"variant": name, "n_scales": ns, "freq_windows": fw,
                 "use_amount": ua, "AP": round(ap, 6)})
    summary[name] = {"n_scales": ns, "freq_windows": fw, "use_amount": ua,
                     "overall_auprc": ap}

full = summary["完整(amount+7+30)"]["overall_auprc"]
single = summary["单尺度(amount)"]["overall_auprc"]
equal_max = max(summary["等维Q/Y(amount+90+365)"]["overall_auprc"],
                summary["等维5/60(amount+5+60)"]["overall_auprc"])
verdict = {
    "multi_scale_gain": round(full - single, 4),
    "window_choice_gain": round(full - equal_max, 4),
    "conclusion": ("增益来自窗口多样性（7/30 选择本身有效），非仅特征数增加"
                   if (full - equal_max) > 1e-4 else
                   "增益可能主要来自特征数增加，窗口多样性贡献未得到证实"),
}
summary["verdict"] = verdict

os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "window_ablation.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(os.path.join(RESULT, "window_ablation.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT_DIR, "window_ablation.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("OK 多尺度增益=%.4f 窗口选择增益=%+.4f" % (verdict["multi_scale_gain"], verdict["window_choice_gain"]))
print(verdict["conclusion"])
