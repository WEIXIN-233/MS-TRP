# -*- coding: utf-8 -*-
"""典型异常的多尺度时序案例图（审阅意见优先级 2）。

案例：2026-10-01~03，明细 DT00004556 在三天内向同一收款人 PY000473 连续支付 9 笔。
数据全部取自落盘的 result_raw/run_main/predictions.csv（不重算、不改分），
逐日取当日最大值；只画有支付的日期，不做插值。

三面板（共用日期轴）：
  (a) 业务变化：当日金额与笔数，参照为"此前历史单笔中位数"
  (b) 尺度响应：模型实际使用的五个时序通道（金额长尺度 / 7日 / 30日 / 当季截至当前 / 当年节奏-季节）
  (c) 告警过程：s_ts、s_g、s_final 与验证集冻结阈值

注意：数据里只有业务申请日期（apply_date），没有入库/评分/更正时点，
      因此图与图题不宣称任何"评分时延"或"整改时点"。
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
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PRED = os.path.join(ROOT, "result_raw", "run_main", "predictions.csv")
PAY = os.path.join(ROOT, "data_fixed", "payment_apply.csv")

DETAIL = "DT00004556"
T_START, T_END = "2026-09-01", "2026-10-10"
WIN = ("2026-10-01", "2026-10-03")
THR = 0.8414934427356722          # 验证集冻结的 thr_global（见 run_main/metrics.json）

CHANNELS = [
    ("s_amt", "Amount baseline\n(long run)"),
    ("s_freq", "7-day\nfrequency"),
    ("s_trend", "30-day\ntrend"),
    ("s_quarter", "Quarter-to-date\n(QTD count)"),
    ("s_rhythm_ch", "YTD rhythm /\nseason"),
]


def load():
    pred = pd.read_csv(PRED, encoding="utf-8-sig", parse_dates=["apply_date"])
    pay = pd.read_csv(PAY, encoding="utf-8-sig",
                      usecols=["apply_id", "detail_id", "payee_id"])
    df = pred.merge(pay, on="apply_id", how="left")
    return df[df["detail_id"] == DETAIL].sort_values("apply_date")


def build_daily(df):
    d = df[(df["apply_date"] >= T_START) & (df["apply_date"] <= T_END)].copy()
    d["day"] = d["apply_date"].dt.normalize()
    agg = {"n": ("apply_id", "size"), "amt": ("amount_wan", "sum")}
    for c, _ in CHANNELS:
        agg[c] = (c, "max")
    for c in ["s_ts", "s_g", "s_final"]:
        agg[c] = (c, "max")
    return d.groupby("day").agg(**agg).reset_index()


def main():
    df = load()
    g = build_daily(df)

    prior = df[df["apply_date"] < pd.Timestamp(WIN[0])]
    med_amt = float(prior["amount_wan"].median())
    n_prior_12m = int((prior["apply_date"] >= pd.Timestamp("2025-09-01")).sum())
    amt_12m = float(prior.loc[prior["apply_date"] >= pd.Timestamp("2025-09-01"),
                              "amount_wan"].sum())

    out_csv = os.path.join(HERE, "case_series.csv")
    g.to_csv(out_csv, index=False, encoding="utf-8-sig")

    xs = np.arange(len(g))
    ticks = [f"{d:%m-%d}\n{n} pays" for d, n in zip(g["day"], g["n"])]
    n_day = len(g)
    xlim = (-0.72, n_day - 0.28)

    # 突增窗口（10-01~10-03）对应的 x 范围，用于三面板统一高亮
    win_mask = (g["day"] >= pd.Timestamp(WIN[0])) & (g["day"] <= pd.Timestamp(WIN[1]))
    win_x = np.where(win_mask)[0]
    win_lo, win_hi = (win_x.min() - 0.5, win_x.max() + 0.5) if len(win_x) else (None, None)

    fig = plt.figure(figsize=(10.2, 8.9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.25, 1.2], hspace=0.36)

    # ---------------- (a) 业务变化 ----------------
    ax = fig.add_subplot(gs[0, 0])
    if win_lo is not None:
        ax.axvspan(win_lo, win_hi, color="#f1c40f", alpha=0.13, zorder=0)
    ax.bar(xs, g["amt"], width=0.52, color="#c0392b", zorder=3)
    for x, a in zip(xs, g["amt"]):
        ax.text(x, a + 10, f"{a:.0f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(med_amt, color="0.4", linestyle="--", linewidth=1.1, zorder=2)
    if win_lo is not None:
        ax.text((win_lo + win_hi) / 2, 160, "burst window\n10-01 ~ 10-03",
                ha="center", va="top", fontsize=7.6, color="#8a6d00")
    ax.set_ylabel("Daily amount\n(10k CNY)", fontsize=9.5)
    ax.set_title("(a) Business volume: 9 payments to one payee in three days",
                 fontsize=11, loc="left")
    ax.set_ylim(0, 165)
    ax.annotate(f"Previous 12 months: {n_prior_12m} payments, {amt_12m:.0f} in total"
                f"   (dashed line: prior per-payment median = {med_amt:.1f})",
                xy=(0.015, 0.92), xycoords="axes fraction", fontsize=8.6, color="0.25")
    ax.grid(axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(*xlim)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, fontsize=8.4)
    ax.tick_params(axis="x", length=0)

    # ---------------- (b) 尺度响应热图 ----------------
    ax = fig.add_subplot(gs[1, 0])
    M = np.array([[float(g[c].iloc[i]) for i in range(n_day)] for c, _ in CHANNELS])
    # 用 pcolormesh 而非 imshow：imshow 会栅格化，pcolormesh 保持矢量
    xe = np.arange(n_day + 1) - 0.5
    ye = np.arange(len(CHANNELS) + 1) - 0.5
    im = ax.pcolormesh(xe, ye, M, cmap="YlOrRd", vmin=0, vmax=1,
                       edgecolors="white", linewidth=0.6)
    if win_lo is not None:
        ax.axvspan(win_lo, win_hi, color="#f1c40f", alpha=0.20, zorder=2)
        for bx in (win_lo, win_hi):
            ax.axvline(bx, color="#8a6d00", linestyle="--", linewidth=1.0, zorder=4)
    ax.set_ylim(len(CHANNELS) - 0.5, -0.5)
    for j in range(len(CHANNELS)):
        for i in range(n_day):
            v = M[j, i]
            ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if v < 0.62 else "white")
    ax.set_yticks(range(len(CHANNELS)))
    ax.set_yticklabels([lbl for _, lbl in CHANNELS], fontsize=8.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d:%m-%d}" for d in g["day"]], fontsize=8.6)
    ax.set_title("(b) Scale response of the temporal channels the model actually uses",
                 fontsize=11, loc="left")
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.026)
    cb.solids.set_rasterized(False)   # matplotlib 默认把色条栅格化，这里强制矢量
    cb.set_label("channel score", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax.tick_params(axis="x", length=0)

    # ---------------- (c) 告警过程 ----------------
    ax = fig.add_subplot(gs[2, 0])
    # 高亮：突增窗口 + 阈值以上的告警区（面积填充，仿高分期刊标记事件区）
    if win_lo is not None:
        ax.axvspan(win_lo, win_hi, color="#f1c40f", alpha=0.13, zorder=1)
    ax.fill_between(xs, THR, 1.52, color="#c0392b", alpha=0.05,
                    linewidth=0, zorder=1)
    ax.text(xlim[0] + 0.04, 1.46, "alarm region (score ≥ threshold)",
            fontsize=7.6, color="#c0392b", ha="left", va="top")
    spec = [("s_ts", "s_ts (temporal)", "#e67e22", "o"),
            ("s_g", "s_g (graph / structure)", "#7b52ab", "s"),
            ("s_final", "s_final (fused)", "#c0392b", "D")]
    for c, lbl, col, mk in spec:
        ax.plot(xs, g[c], color=col, linewidth=1.8, marker=mk, markersize=8,
                label=lbl, zorder=3)
    ax.axhline(THR, color="0.15", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(n_day - 0.42, THR + 0.035, f"frozen threshold (val) = {THR:.4f}",
            ha="right", va="bottom", fontsize=8.6, color="0.15")
    first = g[g["s_final"] >= THR]
    if len(first):
        f = first.iloc[0]
        xi = int(np.where(g["day"].values == f["day"])[0][0])
        ax.scatter([xi], [f["s_final"]], s=220, marker="*", color="#f1c40f",
                   edgecolor="black", linewidth=0.7, zorder=5)
        ax.annotate(f"first alarm {f['day']:%m-%d}\ns_final = {f['s_final']:.3f}",
                    xy=(xi, f["s_final"]), xytext=(0.02, 0.16), fontsize=8.8,
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9))
    ax.set_ylabel("Score", fontsize=9.5)
    ax.set_ylim(0, 1.52)
    ax.set_xlim(*xlim)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d:%m-%d}" for d in g["day"]], fontsize=8.6)
    ax.set_xlabel("Business application date (the dataset carries no ingest or scoring timestamp)",
                  fontsize=9)
    ax.set_title("(c) Alarm process: daily maximum of each score", fontsize=11, loc="left")
    ax.legend(fontsize=8.6, loc="upper left", frameon=False, ncol=1,
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
    ax.grid(axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    p = os.path.join(HERE, "fig_case_multiscale.png")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[case] saved", p)
    print("[case] series", out_csv)
    print(g.to_string())


if __name__ == "__main__":
    main()
