# -*- coding: utf-8 -*-
"""中文版图5：冻结阈值下的测试季度误报（含验证集经验参照）。

数据源：result_raw/frozen_threshold_far_by_period.csv（run_far_periods 输出）
输出：redraw/fig4_quarter_far_cn.png 与 latex/figures/image4.png
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
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "result_raw")


def main():
    df = pd.read_csv(os.path.join(RES, "frozen_threshold_far_by_period.csv"),
                     encoding="utf-8-sig")
    q = df[df["period_type"] == "quarter"].copy()
    quarters = ["2026Q1", "2026Q2", "2026Q3", "2026Q4"]
    rows = [q[q["period_id"] == p].iloc[0] for p in quarters]
    far = np.array([float(r["far_per_1000"]) for r in rows])
    ci_lo = np.array([float(r["far_per_1000_ci_low"]) for r in rows])
    ci_hi = np.array([float(r["far_per_1000_ci_high"]) for r in rows])
    val_ref = float(q["far_val"].iloc[0])

    x = np.arange(len(quarters))
    fig, ax = plt.subplots(figsize=(7.5, 4.35))

    # 验证集经验 FPR 参照（横向虚线）
    ax.axhline(val_ref, color="#c0392b", linestyle="--", linewidth=1.4,
               zorder=2, label=f"验证集经验 FPR {val_ref:.2f}")
    # 带 95% 置信区间（Wilson）的折线
    ax.fill_between(x, ci_lo, ci_hi, color="#4c72b0", alpha=0.16,
                    linewidth=0, zorder=2, label="95% 置信区间")
    ax.errorbar(x, far, yerr=[far - ci_lo, ci_hi - far], fmt="none",
                ecolor="#4c72b0", elinewidth=1.3, capsize=4, zorder=3)
    ax.plot(x, far, color="#4c72b0", linewidth=1.8, marker="o", markersize=7,
            zorder=4)
    for xi, v, hi in zip(x, far, ci_hi):
        ax.text(xi, hi + 0.10, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters)
    ax.set_xlim(-0.35, len(quarters) - 0.65)
    ax.set_xlabel("测试季度")
    ax.set_ylabel("FAR / 1000（每千笔正常支付误报数）")
    ax.set_title("冻结阈值下的测试季度误报（含 95% 置信区间）")
    ax.set_ylim(0, max(val_ref, ci_hi.max()) * 1.28)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out1 = os.path.join(HERE, "fig4_quarter_far_cn.png")
    fig.savefig(out1, dpi=600, bbox_inches="tight")
    out2 = os.path.join(ROOT, "wordwenjian1", "latex", "figures", "image4.png")
    fig.savefig(out2, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[图5中文] saved", out1)
    print("[图5中文] saved", out2)
    print("quarters:", dict(zip(quarters, [round(v, 4) for v in far])), "val_ref:", round(val_ref, 4))


if __name__ == "__main__":
    main()
