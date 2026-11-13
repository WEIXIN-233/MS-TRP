# -*- coding: utf-8 -*-
"""关系证据子图 + 正常共享对照（审阅意见优先级 4）。

同一"共享收款人"拓扑在两个实体上都存在：
  案例 PY000473（长三角咨询服务股份有限公司）：67 笔 / 52 项目 / 3 账户
  对照 PY000390（长三角咨询服务有限公司）  ：60 笔 / 53 项目 / 3 账户
差别不在"共享"本身，而在实体级短窗口密度（模型的关系证据 s_entity_freq）。

所有节点 ID、笔数、账户数、日期与分值均来自
  data_fixed/payment_apply.csv
  result_raw/run_main/predictions.csv
边权为 graph_builder.REL_STRENGTH 的默认值（paid_to=0.9, owns_account=0.7）。
图中只给"风险关联"，不给违规因果判断。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PRED = os.path.join(ROOT, "result_raw", "run_main", "predictions.csv")
PAY = os.path.join(ROOT, "data_fixed", "payment_apply.csv")

CASE, CTRL = "PY000473", "PY000390"
CASE_ACC = ["ACCT000978", "ACCT000979", "ACCT000980"]
CTRL_ACC = ["ACCT000817", "ACCT000818", "ACCT000816"]
C_CASE, C_CTRL = "#c0392b", "#2e86c1"

FS_NODE, FS_TITLE, FS_NOTE = 7.2, 10.0, 7.0


def load():
    pred = pd.read_csv(PRED, encoding="utf-8-sig",
                       usecols=["apply_id", "s_entity_freq", "s_g", "s_final",
                                "is_anomaly", "split"])
    pay = pd.read_csv(PAY, encoding="utf-8-sig",
                      usecols=["apply_id", "apply_date", "amount_wan", "project_id",
                               "detail_id", "unit_name", "payee_id", "account_id"])
    df = pred.merge(pay, on="apply_id", how="left")
    df["day"] = pd.to_datetime(df["apply_date"]).dt.normalize()
    return df


def node_box(ax, x, y, w, h, text, color, fs=FS_NODE, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.005,rounding_size=0.02",
                                linewidth=1.0, edgecolor=color,
                                facecolor="white", zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal", linespacing=1.2)


def payee_circle(ax, x, y, r, text, color):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=3,
                        alpha=0.12))
    ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor=color, zorder=4,
                        linewidth=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.8, zorder=5,
            fontweight="bold", color=color)


def edge(ax, p1, p2, rad=0.0, color="0.35", lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=7,
                                 connectionstyle=f"arc3,rad={rad}", color=color,
                                 linewidth=lw, linestyle=ls, zorder=2))


def draw_panel(ax, title, payee, accounts, projects, accent, note):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=FS_TITLE, loc="left", pad=5)

    px, py = 0.60, 0.48
    payee_circle(ax, px, py, 0.125, payee, accent)

    ys = [0.87, 0.655, 0.44, 0.225]
    for (pid, unit, n_pay, hot), y in zip(projects, ys):
        node_box(ax, 0.155, y, 0.30, 0.135, f"{pid}\n{unit}",
                 accent if hot else "0.45", bold=hot)
        k = min(n_pay, 9)
        for j in range(k):
            yy = y + (j - (k - 1) / 2) * 0.018
            ax.add_patch(Circle((0.385, yy), 0.0105,
                                facecolor=accent if hot else "0.65",
                                edgecolor="none", zorder=3))
        edge(ax, (0.395, y), (px - 0.128, py), rad=-0.16 if y > py else 0.16,
             color=accent if hot else "0.55")

    for j, acc in enumerate(accounts):
        yy = py + (1 - j) * 0.20
        node_box(ax, 0.875, yy, 0.175, 0.105, acc, "0.45", fs=6.2)
        edge(ax, (px + 0.128, py), (0.785, yy), color="0.55", lw=0.8)

    ax.text(0.55, 0.985, "paid_to  a=0.9", ha="center", va="top",
            fontsize=FS_NOTE, color="0.35")
    ax.text(0.70, 0.81, "owns_account a=0.7", fontsize=FS_NOTE, color="0.35")
    ax.text(0.50, 0.02, note, ha="center", va="bottom", fontsize=FS_NOTE,
            color="0.3")


def main():
    df = load()
    case = df[df["payee_id"] == CASE]
    ctrl = df[df["payee_id"] == CTRL]

    fig = plt.figure(figsize=(7.4, 6.9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.75, 0.95, 0.95],
                          hspace=0.45, wspace=0.14)

    axl = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1])

    draw_panel(
        axl, "(a) Case: shared payee, burst", CASE, CASE_ACC,
        [("P000458", "Greening Office", 9, True),
         ("P000003", "Education Bureau", 1, False),
         ("P000242", "Blood Centre", 2, False),
         ("P000547", "Audit Centre", 2, False)],
        C_CASE,
        f"{len(case)} payments / {case['project_id'].nunique()} projects / "
        f"{case['account_id'].nunique()} accounts, 2023-2026\n"
        f"9 of them on 2026-10-01~03")

    draw_panel(
        axr, "(b) Control: shared payee, no burst", CTRL, CTRL_ACC,
        [("P000167", "No.1 Middle School", 1, False),
         ("P000097", "Ecology Bureau", 1, False),
         ("P000262", "Housing Requisition", 1, False),
         ("P000413", "Funeral Services", 1, False)],
        C_CTRL,
        f"{len(ctrl)} payments / {ctrl['project_id'].nunique()} projects / "
        f"{ctrl['account_id'].nunique()} accounts, 2023-2026")

    # (c) 2026 月度笔数
    ax = fig.add_subplot(gs[1, :])
    months = pd.date_range("2026-01-01", "2026-12-01", freq="MS")
    per = [m.to_period("M") for m in months]
    mc = case[case["apply_date"] >= "2026-01-01"].groupby(
        case["day"].dt.to_period("M")).size().reindex(per, fill_value=0)
    mt = ctrl[ctrl["apply_date"] >= "2026-01-01"].groupby(
        ctrl["day"].dt.to_period("M")).size().reindex(per, fill_value=0)
    x = np.arange(len(months))
    w = 0.38
    ax.bar(x - w / 2, mc.values, w, color=C_CASE, label=f"Case {CASE}")
    ax.bar(x + w / 2, mt.values, w, color=C_CTRL, label=f"Control {CTRL}")
    ax.set_xticks(x)
    ax.set_xticklabels([m.strftime("%b") for m in months], fontsize=8.2)
    ax.set_ylabel("Payments per month\n(2026)", fontsize=8.6)
    ax.set_title("(c) Same relation structure, different payment density",
                 fontsize=FS_TITLE, loc="left")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left", ncol=2)
    ax.set_ylim(0, max(mc.max(), mt.max()) * 1.42)
    ax.grid(axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (d) 实体级频次证据
    ax = fig.add_subplot(gs[2, :])
    for sub, lab, col in [(case, f"Case {CASE}", C_CASE),
                          (ctrl, f"Control {CTRL}", C_CTRL)]:
        s = sub[sub["apply_date"] >= "2026-01-01"]
        ax.plot(s["day"], s["s_entity_freq"], linestyle="none", marker="o",
                markersize=3.6, color=col, label=lab, alpha=0.9)
    ax.axhline(0.5, color="0.4", linestyle=":", linewidth=1.0)
    ax.text(pd.Timestamp("2026-01-05"), 0.53, "0.5", fontsize=7.4, color="0.4",
            ha="left", va="bottom")
    ax.annotate(f"max {case[case['apply_date'] >= '2026-01-01']['s_entity_freq'].max():.3f}",
                xy=(pd.Timestamp("2026-10-03"), 0.99),
                xytext=(pd.Timestamp("2026-05-20"), 0.88), fontsize=8.0, color=C_CASE,
                arrowprops=dict(arrowstyle="->", color=C_CASE, lw=0.9))
    ax.annotate(f"max {ctrl[ctrl['apply_date'] >= '2026-01-01']['s_entity_freq'].max():.3f}",
                xy=(pd.Timestamp("2026-12-23"), 0.52),
                xytext=(pd.Timestamp("2026-10-18"), 0.68), fontsize=8.0, color=C_CTRL,
                arrowprops=dict(arrowstyle="->", color=C_CTRL, lw=0.9))
    ax.set_ylim(0, 1.12)
    ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))
    ax.set_xticks(pd.to_datetime(["2026-01-01", "2026-03-01", "2026-05-01",
                                  "2026-07-01", "2026-09-01", "2026-11-01"]))
    ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"], fontsize=8.2)
    ax.set_ylabel("s_entity_freq", fontsize=8.6)
    ax.set_title("(d) The entity-level frequency evidence that separates them",
                 fontsize=FS_TITLE, loc="left")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax.grid(axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p = os.path.join(HERE, "fig_relation_evidence.png")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[relation] saved", p)


if __name__ == "__main__":
    main()
