# -*- coding: utf-8 -*-
"""生成论文图2：主要方法 Payment-AUPRC 比较（含 3 个 2025 新方法）。

数据源：result_raw/metrics_main.csv + baselines_2025/new_methods_comparison.csv
输出：baselines_2025/_paper/fig2_auprc.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

old = pd.read_csv(os.path.join(ROOT, "result_raw", "metrics_main.csv"), encoding="utf-8-sig")
new = pd.read_csv(os.path.join(ROOT, "baselines_2025", "new_methods_comparison.csv"), encoding="utf-8-sig")

# 主要对照方法：AUPRC >= 0.34 的非无技能参照（原有 9 个 + 3 个 2025 方法）
MAIN = [
    ("concat_if", "Concat-IF"),
    ("isolation_forest", "Isolation Forest"),
    ("temporal_only", "Temporal-only"),
    ("MS-TRP", "MS-TRP"),
    ("fixed_rules", "Fixed Rule"),
    ("DRL", "DRL"),
    ("simple_ppr", "Simple PPR"),
    ("residual", "Forecast Residual"),
    ("graph", "Graph-only"),
    ("robust_stats", "Robust MAD"),
    ("FoMo-0D", "FoMo-0D"),
    ("TCCM", "TCCM"),
]

vals = []
for m, label in MAIN:
    if m in set(old["model"]):
        v = float(old[old["model"] == m]["payment_auprc"].iloc[0])
    else:
        v = float(new[new["model"] == m]["payment_auprc"].iloc[0])
    vals.append(v)

order = np.argsort(vals)  # 升序，横向柱状图自下而上
y = np.arange(len(MAIN))
colors = ["#d62728" if MAIN[i][1] == "MS-TRP" else "#4c72b0" for i in range(len(MAIN))]

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh(y, [vals[i] for i in order], color=[colors[i] for i in order], height=0.62)
ax.set_yticks(y)
ax.set_yticklabels([MAIN[i][1] for i in order], fontsize=8.5)
ax.set_xlabel("Payment-AUPRC")
ax.set_title("主要方法的测试集 Payment-AUPRC 比较")
ax.set_xlim(0, max(vals) * 1.12)
for i in range(len(MAIN)):
    v = vals[order[i]]
    ax.text(v + 0.004, y[i], f"{v:.4f}", va="center", fontsize=7.5)
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
p = os.path.join(HERE, "fig2_auprc.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print("[图2] 已生成", p)
print("方法顺序:", [MAIN[i][1] for i in order])
