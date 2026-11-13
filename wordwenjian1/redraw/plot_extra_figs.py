# -*- coding: utf-8 -*-
"""按审阅意见补图：
  fig2_combined.png   图2升级：AUPRC概览 + PR曲线 + 低误报召回曲线
  fig3_cold_start2.png 图3：补样本量与阳性比例
  fig_ablation.png    新增：消融双面板（AUPRC + 实体聚集召回）
  fig_generator.png   新增：跨生成机制配对散点

数据源：
  pr_scores.csv（build_pr_data.py 导出，AUPRC 已与 metrics_main.csv 核对）
  metrics_main.csv / cold_start.csv / run_ablation/ablation.csv / all_seed_results.csv
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
RES = os.path.abspath(os.path.join(HERE, "..", "..", "result_raw"))
OUT = HERE

MAIN_METHODS = [
    ("MS-TRP", "MS-TRP", "#d62728"),
    ("temporal_only", "Temporal-only", "#ff7f0e"),
    ("fixed_rules", "Fixed Rule", "#55a868"),
    ("isolation_forest", "Isolation Forest", "#4c72b0"),
    ("concat_if", "Concat-IF", "#8172b3"),
    ("simple_ppr", "Simple PPR", "#8c564b"),
]


def _pr_curve(y, s):
    """返回 recall, precision 曲线（按分数从高到低）。"""
    order = np.argsort(-s, kind="mergesort")
    y = y[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(y.sum(), 1e-12)
    return recall, precision


def _recall_at_far_curve(y, s, fars):
    normal = np.sort(s[y == 0])
    pos = s[y == 1]
    out = []
    for f in fars:
        thr = np.quantile(normal, 1.0 - f)
        out.append(float((pos > thr).mean()))
    return np.asarray(out)


def fig2_combined():
    df = pd.read_csv(os.path.join(HERE, "pr_scores.csv"), encoding="utf-8-sig")
    test = df[df["split"] == "test"]
    y = test["is_anomaly"].astype(int).values
    prev = y.mean()
    mm = pd.read_csv(os.path.join(RES, "metrics_main.csv"), encoding="utf-8-sig")

    fig = plt.figure(figsize=(15.0, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.28)

    # (a) AUPRC 概览
    ax = fig.add_subplot(gs[0, 0])
    names = [m[1] for m in MAIN_METHODS]
    auprc = [float(mm[mm["model"] == m[0]]["payment_auprc"].iloc[0]) for m in MAIN_METHODS]
    order = np.argsort(auprc)
    yy = np.arange(len(names))
    colors = [MAIN_METHODS[i][2] for i in order]
    ax.barh(yy, [auprc[i] for i in order], color=colors, height=0.62)
    ax.set_yticks(yy)
    ax.set_yticklabels([names[i] for i in order], fontsize=10)
    for i in range(len(order)):
        ax.text(auprc[order[i]] + 0.006, yy[i], f"{auprc[order[i]]:.4f}",
                va="center", fontsize=9)
    ax.set_xlim(0, max(auprc) * 1.16)
    ax.set_xlabel("Payment-AUPRC")
    ax.set_title("(a) AUPRC of main methods", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) PR 曲线
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(prev, color="0.45", linestyle=":", linewidth=1.2,
               label=f"Prevalence = {prev:.4f}")
    for key, label, color in MAIN_METHODS:
        r, p = _pr_curve(y, test[key].values)
        ax.plot(r, p, color=color, linewidth=1.6, label=label)
    # MS-TRP 冻结阈值工作点（thr_global，验证集冻结）
    ms = mm[mm["model"] == "MS-TRP"].iloc[0]
    ax.scatter([ms["recall"]], [ms["precision"]], marker="*", s=180, color="#d62728",
               edgecolor="black", linewidth=0.6, zorder=5,
               label="MS-TRP frozen operating point")
    ax.annotate(f"P={ms['precision']:.4f}\nR={ms['recall']:.4f}",
                (ms["recall"] + 0.03, ms["precision"] - 0.06), fontsize=7.6, color="#d62728")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(b) Precision-Recall on the test set", fontsize=11)
    ax.legend(fontsize=7.4, loc="upper right", frameon=False)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) 低误报区间召回
    ax = fig.add_subplot(gs[0, 2])
    fars = np.linspace(0.002, 0.10, 200)
    for key, label, color in MAIN_METHODS:
        rc = _recall_at_far_curve(y, test[key].values, fars)
        ax.plot(fars * 100, rc, color=color, linewidth=1.7, label=label)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("False alarm rate on normal payments (%)")
    ax.set_ylabel("Recall")
    ax.set_title("(c) Recall under low-FAR budget", fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p = os.path.join(OUT, "fig2_combined.png")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[图2] 已生成", p)


def fig3_cold_start():
    df = pd.read_csv(os.path.join(RES, "cold_start.csv"), encoding="utf-8-sig")
    buckets = ["<=30", "31-180", ">180"]
    models = ["Self-only", "Peer-only", "Self/Peer", "MS-TRP"]
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#d62728"]
    x = np.arange(len(buckets))
    width = 0.2

    labels = []
    for b in buckets:
        sub = df[df["bucket"] == b]
        n = int(sub["n_test"].iloc[0])
        pos = int(sub["n_pos"].iloc[0])
        labels.append(f"{b.replace('<=', '≤')} days\nn={n:,}, pos={pos / n:.1%}")

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    for j, m in enumerate(models):
        vals = [float(df[(df["bucket"] == b) & (df["model"] == m)]["payment_auprc"].iloc[0])
                for b in buckets]
        ax.bar(x + (j - 1.5) * width, vals, width, label=m, color=colors[j])
        for xi, v in zip(x + (j - 1.5) * width, vals):
            ax.text(xi, v + 0.011, f"{v:.3f}", ha="center", fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Indicator history span")
    ax.set_ylabel("Payment-AUPRC")
    ax.set_title("Cold start: AUPRC by reference source", fontsize=11)
    ax.set_ylim(0, 0.74)
    ax.legend(ncol=2, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_cold_start2.png")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[图3] 已生成", p)


def fig_ablation():
    df = pd.read_csv(os.path.join(RES, "run_ablation", "ablation.csv"), encoding="utf-8-sig")
    label_map = {
        "完整模型": "Full model",
        "Temporal-only(去图)": "Temporal-only",
        "+Rule(账户一致性)": "+Account rule",
        "+PPR(图扩散)": "+PPR",
        "+Rule+PPR": "+Rule+PPR",
        "单尺度(去多尺度)": "Single scale",
        "固定边权(去阶段)": "Fixed edge weight",
        "固定0.5融合(去门控)": "Fixed 0.5 fusion",
    }
    order = list(label_map)
    df = df.set_index("variant").loc[order]
    labels = [label_map[v] for v in order]
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    specs = [("AP", "Payment-AUPRC", "(a) Ranking performance", "#4c72b0", 0.62),
             ("entity_recall", "Entity-aggregation recall", "(b) Event recall of the entity type",
              "#55a868", 0.70)]
    for ax, (col, xlabel, title, color, xmax) in zip(axes, specs):
        vals = df[col].astype(float).values
        cols = ["#d62728" if l == "Full model" else color for l in labels]
        ax.barh(y, vals, color=cols, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v + xmax * 0.012, yi, f"{v:.3f}", va="center", fontsize=8.6)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9.2)
    axes[0].invert_yaxis()
    fig.suptitle("Component variants: ranking gain vs. rule-driven entity recall",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_ablation.png")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[消融图] 已生成", p)


def fig_generator():
    df = pd.read_csv(os.path.join(RES, "all_seed_results.csv"), encoding="utf-8-sig")
    seeds = sorted(df["seed"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, col, ylabel, title in [
        (axes[0], "AP", "Payment-AUPRC", "(a) Payment-AUPRC"),
        (axes[1], "event_recall", "Event recall", "(b) Event recall"),
    ]:
        a = [float(df[(df["mechanism"] == "A") & (df["seed"] == s)][col].iloc[0]) for s in seeds]
        b = [float(df[(df["mechanism"] == "B") & (df["seed"] == s)][col].iloc[0]) for s in seeds]
        for i, s in enumerate(seeds):
            ax.plot([0, 1], [a[i], b[i]], color="0.55", linewidth=1.1, marker="o",
                    markersize=6, zorder=2)
            ax.annotate(f"seed {s}", (1.02, b[i]), fontsize=8.4, va="center")
        ax.scatter([0] * len(seeds), a, color="#4c72b0", s=60, zorder=3, label="Mechanism A")
        ax.scatter([1] * len(seeds), b, color="#d62728", s=60, zorder=3, label="Mechanism B")
        ax.set_xlim(-0.35, 1.28)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["A", "B"])
        ax.set_xlabel("Generator mechanism")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("MS-TRP under two generator mechanisms (3 data seeds each)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_generator.png")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[跨机制图] 已生成", p)


if __name__ == "__main__":
    if not os.path.exists(os.path.join(HERE, "pr_scores.csv")):
        raise SystemExit("缺少 pr_scores.csv，请先运行 build_pr_data.py")
    fig2_combined()
    fig3_cold_start()
    fig_ablation()
    fig_generator()
    print("完成。")
