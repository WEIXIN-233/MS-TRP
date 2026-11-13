# -*- coding: utf-8 -*-
"""重绘论文图2/3/4（数据图）：AUPRC比较 / 冷启动 / 季度FAR。

数据源：../result_raw/ 下的 CSV（与论文表格完全一致，已核对）。
输出：redraw/fig2_*.png 等，供替换论文 docx 内图片。
用法：python redraw_data_figs.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "result_raw")
OUT = HERE

# ── 图2：主要方法 Payment-AUPRC 比较 ──────────────────────────────
def fig2():
    df = pd.read_csv(os.path.join(RES, "metrics_main.csv"), encoding="utf-8-sig")
    # 主要对照方法：AUPRC >= 0.40 的非无技能参照，按 AUPRC 降序
    main = [
        "concat_if", "isolation_forest", "temporal_only", "MS-TRP",
        "fixed_rules", "simple_ppr", "residual", "graph", "robust_stats",
    ]
    labels = [
        "Concat-IF", "Isolation Forest", "Temporal-only", "MS-TRP",
        "Fixed Rule", "Simple PPR", "Forecast Residual", "Graph-only", "Robust MAD",
    ]
    auprc = [float(df[df["model"] == m]["payment_auprc"].iloc[0]) for m in main]
    order = np.argsort(auprc)  # 升序，横向柱状图自下而上
    y = np.arange(len(main))
    colors = ["#d62728" if labels[i] == "MS-TRP" else "#4c72b0" for i in range(len(main))]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.barh(y, [auprc[i] for i in order], color=[colors[i] for i in order], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel("Payment-AUPRC")
    ax.set_title("Payment-AUPRC of Main Methods on the Test Set")
    ax.set_xlim(0, max(auprc) * 1.12)
    for i in range(len(main)):
        v = auprc[order[i]]
        ax.text(v + 0.004, y[i], f"{v:.4f}", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_auprc.png")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[图2] 已生成", p)

# ── 图3：冷启动分组 AUPRC ────────────────────────────────────────
def fig3():
    df = pd.read_csv(os.path.join(RES, "cold_start.csv"), encoding="utf-8-sig")
    buckets = ["<=30", "31-180", ">180"]
    bucket_labels = ["≤30 days", "31–180 days", ">180 days"]
    models = ["Self-only", "Peer-only", "Self/Peer", "MS-TRP"]
    model_labels = ["Self-only", "Peer-only", "Self/Peer", "MS-TRP"]
    x = np.arange(len(buckets))
    width = 0.2
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#d62728"]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for j, m in enumerate(models):
        vals = [float(df[(df["bucket"] == b) & (df["model"] == m)]["payment_auprc"].iloc[0])
                for b in buckets]
        ax.bar(x + (j - 1.5) * width, vals, width, label=model_labels[j], color=colors[j])
        for xi, v in zip(x + (j - 1.5) * width, vals):
            ax.text(xi, v + 0.012, f"{v:.4f}", ha="center", fontsize=8, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("Indicator History Span")
    ax.set_ylabel("Payment-AUPRC")
    ax.set_title("Cold Start: AUPRC by Reference Source")
    ax.set_ylim(0, 0.72)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_cold_start.png")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[图3] 已生成", p)

# ── 图4：冻结阈值下季度误报 + 验证集经验参照 ───────────────────────
def fig4():
    df = pd.read_csv(os.path.join(RES, "frozen_threshold_far_by_period.csv"), encoding="utf-8-sig")
    q = df[df["period_type"] == "quarter"].copy()
    quarters = ["2026Q1", "2026Q2", "2026Q3", "2026Q4"]
    far = [float(q[q["period_id"] == p]["far_per_1000"].iloc[0]) for p in quarters]
    val_ref = float(q["far_val"].iloc[0])  # 1.4153

    x = np.arange(len(quarters))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(x, far, color="#4c72b0", width=0.55)
    ax.axhline(val_ref, color="#d62728", linestyle="--", linewidth=1.4,
               label=f"Validation empirical {val_ref:.2f}")
    for xi, v in zip(x, far):
        ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters)
    ax.set_xlabel("Test Quarter")
    ax.set_ylabel("FAR / 1000")
    ax.set_title("Quarterly False Alarm Rate under the Frozen Threshold")
    ax.set_ylim(0, max(val_ref, max(far)) * 1.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_quarter_far.png")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[图4] 已生成", p)


if __name__ == "__main__":
    fig2()
    fig3()
    fig4()
    print("完成。")
