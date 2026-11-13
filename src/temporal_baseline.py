"""
MS-TRP Step 1~3 —— 多尺度时序分支
========================================
Step 1  多尺度特征：7日/30日窗口 + 指标自身历史（严格时间前向，只用 apply_date 之前的数据）
Step 2  Self/Peer 分层动态基线：μ = λ·μ_self + (1-λ)·μ_peer，
        λ 由有效历史长度与稳定性启发式给出，冷启动自动依赖 Peer。
Step 3  稳健时序风险：z=|x-μ|/(σ+ε) → tanh 有界映射 → 汇总为 0~1 的 s_ts。

输入约定：payments_df 已按 apply_date 升序且 index 为 RangeIndex（data.load_payments 保证）。
"""
import numpy as np
import pandas as pd

# 短期窗口（时间前向，closed='left' 保证不含当前支付）
WINDOWS = {"w7": pd.Timedelta(days=7), "w30": pd.Timedelta(days=30)}

# Self/Peer 收缩系数 λ 的启发式参数（指导书 5.2 默认实现）
LAMBDA_MIN, LAMBDA_MAX = 0.1, 0.9
LAMBDA_SLOPE = 0.8
HIST_LEN_SAT = 20.0          # 历史笔数达到该值视为"充分"
PEER_MIN_SAMPLES = 5         # peer 组最小样本数，不足则逐级放宽

# 多尺度时序风险的汇总：对三个尺度的有界风险取最大值（任一尺度显著偏离即报警），
# 避免加权平均稀释稀疏的强信号（与图分支账户一致性硬报警的 OR 设计一致）。
Z_SAT = 3.0                  # z-score 饱和尺度（tanh 内）
FREQ_ZOOM = 7.0              # 短尺度期望：历史"笔/天"速率 × 7 天
TREND_ZOOM = 30.0            # 中尺度期望：历史"笔/天"速率 × 30 天
RHYTHM_SAT = 0.35            # 历史节奏累计占比偏离的饱和尺度（z_rhythm 的量纲是"份额"，非 z 分数）
CONC_SAT = 0.5               # 年末集中 z_conc 的饱和尺度（z_conc 是"12月占比"的 z 分数，量纲与 z_rhythm 不同）
EPS = 1e-6


def enrich_payments(payments_df, projects_df, units_df):
    """为支付补齐 project_category / unit_type / 执行阶段(stage)，供 peer 分组使用。
    用 map 逐列补齐而非 merge：merge 会重排行序并重置 index，破坏与点级标签的对齐。"""
    proj = projects_df.set_index("project_id")
    unit_type = units_df.set_index("unit_id")["unit_type"]

    df = payments_df.copy()
    df["project_category"] = df["project_id"].map(proj["project_category"])
    df["unit_type"] = df["project_id"].map(proj["unit_id"]).map(unit_type)
    df["start_year"] = df["project_id"].map(proj["start_year"])
    df["end_year"] = df["project_id"].map(proj["end_year"])

    span = (df["end_year"] - df["start_year"]).clip(lower=0)
    elapsed = (df["year"] - df["start_year"]).clip(lower=0)
    # 进度裁剪到 [0,1]：单年度项目在后续年份支付时进度会超过 1，否则 stage 出现 NaN
    progress = np.clip(elapsed / (span + 1.0), 0.0, 1.0)
    df["stage"] = pd.cut(progress, bins=[-0.01, 0.33, 0.66, 1.01],
                         labels=["Early", "Middle", "Late"])
    return df


def compute_multi_scale_features(payments_df, freq_windows=(7, 30)):
    """
    Step 1。返回 feature DataFrame（与 payments_df 等长、同 index）。
    所有历史统计均"含当前再 shift"或 closed='left'，即只用严格早于当前支付的历史。
    """
    # 只重排、不重置 index：feat 的 index 与 payments_df 完全一致，
    # 后续一律按 index 对齐赋值，避免任何行序错位（尤其是 enrich 之后）。
    df = payments_df.sort_values("apply_date")
    feat = pd.DataFrame(index=payments_df.index)

    # 当前值
    feat["amount_wan"] = df["amount_wan"]
    feat["log_amount"] = np.log1p(df["amount_wan"])
    feat["month"] = df["month"]
    feat["is_december"] = df["is_december"]
    feat["year"] = df["year"]
    feat["indicator_id"] = df["indicator_id"]
    feat["project_id"] = df["project_id"]
    feat["unit_name"] = df["unit_name"]
    feat["econ_class_code"] = df["econ_class_code"]

    # 透传 peer 分组所需的 enrich 列（若输入已 enrich）
    for col in ["unit_type", "project_category", "stage"]:
        if col in df.columns:
            feat[col] = df[col]

    def _feat(sub):
        sub = sub.sort_values("apply_date")          # sub.index = df 的 RangeIndex 子集
        s = sub["amount_wan"]
        exp = s.expanding(min_periods=1)
        out = pd.DataFrame(index=sub.index)
        out["hist_count"] = np.arange(len(s))
        out["hist_sum"] = s.cumsum().shift(1)
        out["hist_mean"] = exp.mean().shift(1)
        out["hist_std"] = exp.std().shift(1)
        out["hist_median"] = exp.median().shift(1)
        out["interval_days"] = sub["apply_date"].diff().dt.days
        # 历史速率：严格早于当前的笔数 / 自首笔以来天数（用于频次突增的期望值）
        out["hist_span_days"] = (sub["apply_date"] - sub["apply_date"].iloc[0]).dt.days
        out["prior_rate"] = out["hist_count"] / np.maximum(out["hist_span_days"], 1.0)
        ts = sub.set_index("apply_date")["amount_wan"]
        for d in freq_windows:
            wd = pd.Timedelta(days=d)
            r = ts.rolling(window=wd, closed="left")
            out[f"w{d}_count"] = r.count().values
            out[f"w{d}_mean"] = r.mean().values
        # 当季截至当前（quarter-to-date，长中尺度之间的"季节内累计"口径）：
        #   w_q_count = 本指标在本自然季度内、严格早于当前的笔数；
        #   q_days    = 本自然季度已过天数（季首日为 0）。
        # 用 cumcount 实现"季度内前向累计"：groupby 后按 apply_date 顺序计数，天然不含当前。
        qkey = sub["apply_date"].dt.to_period("Q")
        out["w_q_count"] = sub.groupby(qkey, sort=False).cumcount().loc[sub.index].values
        qstart = sub["apply_date"].dt.to_period("Q").dt.start_time
        out["q_days"] = (sub["apply_date"] - qstart).dt.days.values
        return out

    hist = df.groupby("indicator_id", sort=False, group_keys=False).apply(_feat)
    hist = hist.reindex(df.index)                    # 恢复到全局顺序
    for c in hist.columns:
        feat[c] = hist[c]                            # 按 index 对齐
    return feat


def compute_rhythm_features(payments_df, feat):
    """
    Step 1c。历史节奏 / 执行进度偏离特征（V7.1，救 HISTORICAL_SHIFT / PROGRESS_LAG）。

    产出两个口径的累计进度偏离标量（均为 z = actual − due，天然有界于 [-1,1]）：
      · z_rhythm（明细级）：actual/due 均按明细(detail_id)累计。z>0 → 该明细当年进度快于
        往年同期（抓 HISTORICAL_SHIFT，明细级注入的"早期超花"）。
      · z_rhythm_ind（指标级）：actual/due 均按指标(indicator_id)累计。z<0 → 该指标当年进度
        慢于往年同期（抓 PROGRESS_LAG，指标级注入的"前9月欠执行、差额后置"）。

    二者粒度分别对齐各自异常的注入粒度，避免明细级 ~3 笔/年的量化噪声把指标级压缩信号淹没。
    无泄漏：due 只用"往年(y<Y)"数据；actual 只用"当年 y=Y 且 月≤m"的已发生支付。
    该实体无往年时，due 回退到其单位(unit_name)的往年节奏。
    """
    df = payments_df.sort_values("apply_date")
    # 单位级往年月度金额：unit_name -> {year -> 月金额向量[13]}
    unit_year = {}
    for uname, g in df.groupby("unit_name", sort=False):
        by_year = {}
        for y, sy in g.groupby("year"):
            ma = np.zeros(13)
            for m in range(1, 13):
                ma[m] = float(sy.loc[sy["month"] == m, "amount_wan"].sum())
            by_year[y] = ma
        unit_year[uname] = by_year

    def _group_rhythm(group_col):
        """按 group_col 分组，返回每条支付的 actual−due 累计占比偏离（长度=len(df)）。"""
        z = np.zeros(len(df))
        for _gid, g in df.groupby(group_col, sort=False):
            g = g.sort_values("apply_date")
            uname = g["unit_name"].iloc[0]
            prior_ma = np.zeros(13)
            prior_total = 0.0
            for y in sorted(g["year"].unique()):
                sy = g[g["year"] == y]
                cur_ma = np.zeros(13)
                for m in range(1, 13):
                    cur_ma[m] = float(sy.loc[sy["month"] == m, "amount_wan"].sum())
                cur_ytd = np.cumsum(cur_ma)
                cur_total = cur_ytd[12]

                # due：优先该实体往年累计占比；否则该单位往年累计占比
                if prior_total > 0:
                    due_cum = np.cumsum(prior_ma) / prior_total
                else:
                    uprior_ma = np.zeros(13)
                    for uy, uma in unit_year.get(uname, {}).items():
                        if uy < y:
                            uprior_ma += uma
                    uprior_total = float(uprior_ma.sum())
                    if uprior_total > 0:
                        due_cum = np.cumsum(uprior_ma) / uprior_total
                    else:
                        # 无任何往年基线：中性，仅累计当年再跳过
                        prior_ma += cur_ma
                        prior_total += cur_total
                        continue
                if cur_total <= EPS:
                    prior_ma += cur_ma
                    prior_total += cur_total
                    continue
                actual_cum = cur_ytd / cur_total
                dev = actual_cum - due_cum
                for m in range(1, 13):
                    pos = sy.index[sy["month"].values == m]
                    if len(pos):
                        z[pos] = dev[m]

                prior_ma += cur_ma
                prior_total += cur_total
        return z

    feat = feat.copy()
    feat["z_rhythm"] = _group_rhythm("detail_id")        # 明细级：抓 HISTORICAL_SHIFT
    feat["z_rhythm_ind"] = _group_rhythm("indicator_id")  # 指标级：抓 PROGRESS_LAG
    return feat


def compute_year_end_concentration(payments_df):
    """
    Step 1c-2。年末集中支付偏离特征（V8，抓 YEAREND_CONC）。

    YEAREND_CONC 注入在"单位×年度"粒度：把该单位某一年内非12月支付的 45% 挪到当年12月，
    使该单位当年12月的支付占比（笔数/金额）远超往年同期。累计占比类节奏特征
    （z_rhythm/z_rhythm_ind）在12月恒为 1，对"年末集中"是天然盲区，故单独建
    "12月占比 vs 往年12月占比"的 z 分数（笔数、金额两口径取较大者，只取正偏离）。

    无泄漏：due 只用往年(y<Y)的12月占比；只对12月支付赋值（年末集中仅在年末可判）。
    该单位往年12月占比样本不足 1 年时 z=0（保守不报，避免单点 std=0 造成爆炸）。
    输入 payments_df 需含 unit_name / year / month / amount_wan / apply_date。
    """
    df = payments_df.sort_values("apply_date")
    out = np.zeros(len(df))
    for uname, g in df.groupby("unit_name", sort=False):
        prior_c = []   # 往年 12月"笔数"占比
        prior_a = []   # 往年 12月"金额"占比
        for y in sorted(g["year"].unique()):
            gy = g[g["year"] == y]
            dec = gy[gy["month"] == 12]
            n_dec = float(len(dec))
            n_tot = float(len(gy))
            a_dec = float(dec["amount_wan"].sum())
            a_tot = float(gy["amount_wan"].sum())
            share_c = n_dec / max(n_tot, EPS)
            share_a = a_dec / max(a_tot, EPS)
            zc = za = 0.0
            if len(prior_c) >= 1:
                zc = (share_c - float(np.mean(prior_c))) / (float(np.std(prior_c)) + 0.05)
            if len(prior_a) >= 1:
                za = (share_a - float(np.mean(prior_a))) / (float(np.std(prior_a)) + 0.05)
            z = max(zc, za)
            if z > 0:
                out[dec.index.values] = z
            prior_c.append(share_c)
            prior_a.append(share_a)
    return out


# 进度滞后可检窗口（月，闭区间）：due_progress 已累计到可观水平、actual_progress 尚未追平。
# PROGRESS_LAG 注入口径 = "前9月欠执行、差额后置Q4"，故 lag=due−actual 只在年中的
# 欠执行期（due 已涨、actual 仍低）显著；1~5 月 due 太小、11~12 月 actual 追平，lag 天然趋 0，
# 若全年平均反而稀释信号、并在正常"天然后置"指标上制造等量噪声（实测仅取 6~10 月，
# PROGRESS_LAG AUPRC 0.069→0.131）。
PROGRESS_LAG_MONTHS = (6, 10)


def compute_snapshot_progress_lag(feat):
    """
    任务B 月末快照级 PROGRESS_LAG 信号（指导书 §3.2 / Q2：需 indicator_snapshot 的
    due_progress/actual_progress，而非按支付聚合）。
    读 indicator_snapshot，按（指标×年-月）给每条支付附上快照的
        lag = due_progress − actual_progress（clip≥0）
    计划进度明显领先于实际执行 → lag 大 → 执行进度滞后（PROGRESS_LAG）。
    仅保留 PROGRESS_LAG_MONTHS 窗口内的 lag（见上注），窗口外置 0。
    无快照匹配时回退 0（正常）。输入 feat 需含 indicator_id / year / month 列。
    """
    import sqlite3
    from src import config
    conn = sqlite3.connect(config.DB_PATH)
    snap = pd.read_sql(
        "SELECT indicator_id, month_end, due_progress, actual_progress FROM indicator_snapshot",
        conn)
    conn.close()
    lag_map = {}
    for iid, me, due, act in zip(snap["indicator_id"].astype(str),
                                 snap["month_end"].astype(str),
                                 snap["due_progress"], snap["actual_progress"]):
        lag_map[(iid, me)] = max(float(due) - float(act), 0.0)
    keys = list(zip(feat["indicator_id"].astype(str),
                    feat["year"].astype(str) + "-" +
                    feat["month"].astype(str).str.zfill(2)))
    lag = np.asarray([lag_map.get(k, 0.0) for k in keys], dtype=float)
    lo, hi = PROGRESS_LAG_MONTHS
    month = np.asarray(feat["month"].astype(int).values, dtype=int)
    return lag * ((month >= lo) & (month <= hi))


def compute_peer_stats(train_df):
    """
    Step 2 的 peer 部分。在参考期 train_df（已 enrich：含 unit_type/project_category/stage）上，
    按四级分组统计金额 mean 与 MAD，返回 {level: DataFrame}。
    层级（指导书 5.2）：L0 单位类型+项目类别+支出分类+阶段 → L1 单位类型+项目类别
    → L2 项目类别 → L3 全局。
    """
    train = train_df.copy()
    train["g0"] = list(zip(train["unit_type"].astype(str), train["project_category"].astype(str),
                           train["econ_class_code"].astype(str), train["stage"].astype(str)))
    train["g1"] = list(zip(train["unit_type"].astype(str), train["project_category"].astype(str)))
    train["g2"] = train["project_category"].astype(str)

    def _agg(key):
        g = train.groupby(key, dropna=False)["amount_wan"]
        mean = g.mean()
        med = g.median()
        mad = (train["amount_wan"] - train[key].map(med)).abs().groupby(train[key]).median()
        return pd.DataFrame({"mean": mean, "mad": mad, "count": g.size()})

    return {"L0": _agg("g0"), "L1": _agg("g1"), "L2": _agg("g2"),
            "L3": {"mean": train["amount_wan"].mean(),
                   "mad": (train["amount_wan"] - train["amount_wan"].median()).abs().median(),
                   "count": len(train)}}


def _peer_lookup(feat, peer_stats):
    """对每条支付按 L0→L1→L2→L3 逐级查询 peer 统计，返回 (peer_mean, peer_scale, level)。"""
    keys = {
        "L0": list(zip(feat["unit_type"].astype(str), feat["project_category"].astype(str),
                       feat["econ_class_code"].astype(str), feat["stage"].astype(str))),
        "L1": list(zip(feat["unit_type"].astype(str), feat["project_category"].astype(str))),
        "L2": feat["project_category"].astype(str).tolist(),
    }
    n = len(feat)
    peer_mean = np.full(n, np.nan)
    peer_scale = np.full(n, np.nan)
    level = np.full(n, "L3", dtype=object)

    # 逐级填充：后面级别只覆盖仍为 NaN 的行
    for lv in ["L0", "L1", "L2"]:
        stats = peer_stats[lv]
        mean_map = stats["mean"].to_dict()
        mad_map = stats["mad"].to_dict()
        cnt_map = stats["count"].to_dict()
        k = keys[lv]
        for i in range(n):
            if np.isnan(peer_mean[i]):
                g = k[i]
                if g in mean_map and cnt_map.get(g, 0) >= PEER_MIN_SAMPLES:
                    peer_mean[i] = mean_map[g]
                    peer_scale[i] = mad_map[g]
                    level[i] = lv
    # L3 全局兜底
    g = peer_stats["L3"]
    peer_mean = np.where(np.isnan(peer_mean), g["mean"], peer_mean)
    peer_scale = np.where(np.isnan(peer_scale), g["mad"], peer_scale)
    return peer_mean, peer_scale, level


def build_self_peer_baseline(feat, peer_stats):
    """
    Step 2。返回在 feat 上新增的列：baseline_mean / baseline_scale / lambda_self / peer_level。
    λ=clip(0.1+0.8·q, 0.1, 0.9)；q 由有效历史长度与历史稳定性归一化取平均。
    无历史(hist_count=0)时 λ=0，完全依赖 peer。
    """
    out = feat.copy()
    # 历史充分度 + 稳定性 → q → λ
    hist_len_norm = np.clip(out["hist_count"] / HIST_LEN_SAT, 0.0, 1.0)
    cv = np.abs(out["hist_std"]) / (out["hist_mean"].abs() + EPS)
    stability = 1.0 / (1.0 + np.nan_to_num(cv, nan=1.0))
    q = 0.5 * hist_len_norm + 0.5 * stability
    lam = np.clip(LAMBDA_MIN + LAMBDA_SLOPE * q, LAMBDA_MIN, LAMBDA_MAX)
    lam = np.where(out["hist_count"] <= 0, 0.0, lam)

    peer_mean, peer_scale, level = _peer_lookup(out, peer_stats)

    out["baseline_mean"] = lam * out["hist_mean"].fillna(0.0) + (1.0 - lam) * peer_mean
    # scale：自身用历史 std，peer 用 MAD，按 λ 混合；首笔无 std 时全用 peer
    self_scale = out["hist_std"].fillna(0.0)
    out["baseline_scale"] = np.where(
        out["hist_count"] > 0,
        lam * self_scale + (1.0 - lam) * peer_scale,
        peer_scale,
    )
    out["baseline_scale"] = out["baseline_scale"].clip(lower=EPS)
    out["lambda_self"] = lam
    out["peer_level"] = level
    return out


def compute_temporal_risk(feat, use_amount=True, freq_windows=(7, 30), use_quarter=True):
    """
    Step 3。基于动态基线计算多个尺度的偏离风险，tanh 有界映射后取最大值汇总为 s_ts∈[0,1]。
      · s_amt（长尺度/历史基线）：金额偏离 |x-μ|/(σ+ε) 的有界 z 分数；
      · s_freq（短尺度/7 日）：7 日支付笔数相对该指标历史"笔/天"速率的稳健 z 分数
        （只取突增方向的正偏离，避免高频正常指标被误报）；
      · s_trend（中尺度/30 日）：30 日支付笔数相对历史速率的稳健 z 分数（年末集中）；
      · s_quarter（当季截至当前）：本自然季度内已发生笔数相对"历史日速率 × 本季已过天数"
        的稳健 z 分数（只取突增方向），补足固定日窗看不到的"整季持续偏快"；
      · s_progress（执行进度滞后，V7.1）：指标级累计进度落后往年同期（z_rhythm_ind<0）→ 抓 PROGRESS_LAG；
      · s_rhythm（历史节奏偏离，V7.1）：明细级累计进度快于往年同期（z_rhythm>0）→ 抓 HISTORICAL_SHIFT；
      · s_ts = max(s_amt, s_freq, s_trend, s_quarter)：只聚合通用尺度，节奏信号不并入 s_ts，
        而是由 pipeline 作为独立通道与门控结果取 OR，避免被通用信号的高误报淹没。
    输入 feat 需已含 baseline_mean / baseline_scale（build_self_peer_baseline 输出）、
    w7_count / w30_count / prior_rate（compute_multi_scale_features 输出）及
    z_rhythm / z_rhythm_ind（compute_rhythm_features 输出）。
    """
    z_amt = np.abs(feat["amount_wan"] - feat["baseline_mean"]) / (feat["baseline_scale"] + EPS)
    s_amt = np.tanh(z_amt / Z_SAT)

    # 频次尺度：对每个窗口 d 天，算"d 日笔数 vs 历史速率期望"的稳健 z 分数（只取突增方向）。
    # 默认 freq_windows=(7,30) 即 s_freq(7日)/s_trend(30日)；换任意窗口可做等维对照（E12）。
    prior_rate = np.asarray(feat["prior_rate"].fillna(0.0), dtype=float)
    freq_z_signals = {}
    freq_signals = {}
    for d in freq_windows:
        wd = np.asarray(feat[f"w{d}_count"].fillna(0.0), dtype=float)
        exp = prior_rate * float(d)
        z = (wd - exp) / (np.sqrt(np.maximum(exp, 0.1)) + 0.5)
        freq_z_signals[d] = z
        freq_signals[d] = np.tanh(np.maximum(z, 0.0) / Z_SAT)

    # 输出列保持旧语义：s_freq/z_freq = 第一个频次窗口，s_trend/z_trend = 第二个；不足则置 0
    fz = [freq_z_signals[d] for d in freq_windows]
    fs = [freq_signals[d] for d in freq_windows]
    z0 = np.zeros_like(s_amt)
    freq_z = fz[0] if fz else z0
    trend_z = fz[1] if len(fz) >= 2 else z0
    s_freq = fs[0] if fs else z0
    s_trend = fs[1] if len(fs) >= 2 else z0

    # 当季截至当前（季节内累计尺度 s_quarter）：本季已发生笔数 vs 历史"笔/天"速率 × 本季已过天数。
    # 与 7/30 日固定窗互补：日窗只覆盖近端，季度窗把"整个季度一直偏快/偏慢"的累计偏离也纳入。
    # 季首日 q_days=0、本季前向笔数为 0 → z=0，不会产生季初虚警。
    if use_quarter and "w_q_count" in feat.columns:
        q_count = np.asarray(feat["w_q_count"].fillna(0.0), dtype=float)
        q_days = np.asarray(feat["q_days"].fillna(0.0), dtype=float)
        exp_q = prior_rate * np.maximum(q_days, 1.0)
        z_quarter = (q_count - exp_q) / (np.sqrt(np.maximum(exp_q, 0.1)) + 0.5)
        s_quarter = np.tanh(np.maximum(z_quarter, 0.0) / Z_SAT)
    else:
        z_quarter = np.zeros_like(s_amt)
        s_quarter = np.zeros_like(s_amt)

    # V7.1 历史节奏 / 执行进度偏离（Step 1c，按注入粒度拆成两个硬报警信号）：
    #   s_progress 用月末快照级 lag=due−actual（任务B口径，indicator_snapshot）→ PROGRESS_LAG；
    #   s_rhythm   用明细级 z_rhythm（快于往年 → HISTORICAL_SHIFT，明细级注入）
    if "z_rhythm" in feat.columns:
        z_rhythm = np.asarray(feat["z_rhythm"].fillna(0.0), dtype=float)
    else:
        z_rhythm = np.zeros_like(s_amt)
    if "z_rhythm_ind" in feat.columns:
        z_rhythm_ind = np.asarray(feat["z_rhythm_ind"].fillna(0.0), dtype=float)
    else:
        z_rhythm_ind = np.zeros_like(s_amt)
    z_progress_snap = compute_snapshot_progress_lag(feat)               # 快照级（任务B）
    s_progress = np.tanh(z_progress_snap / RHYTHM_SAT)                  # 计划领先实际 → PROGRESS_LAG
    s_rhythm = np.tanh(np.maximum(z_rhythm, 0.0) / RHYTHM_SAT)         # 明细级超前（HISTORICAL_SHIFT）
    # V8 年末集中支付偏离（Step 1c-2，单位级12月占比偏离 → YEAREND_CONC）
    if "z_conc" in feat.columns:
        z_conc = np.asarray(feat["z_conc"].fillna(0.0), dtype=float)
    else:
        z_conc = np.zeros_like(s_amt)
    s_conc = np.tanh(np.maximum(z_conc, 0.0) / CONC_SAT)

    # V7.2：节奏信号(s_rhythm/s_progress)是"特异且稀疏"的强信号，若与通用信号
    # (s_amt/s_freq/s_trend) 一起 max，会被后者的高误报淹没（实测 s_rhythm 单独 AUPRC 0.24、
    # 与通用信号 max 后塌到 0.04）。故 s_ts 只聚合通用三尺度；节奏通道单独输出，
    # 由 pipeline 与门控结果取 OR（同 s_mismatch 硬报警口径），避免被软平均稀释。
    active = []
    if use_amount:
        active.append(s_amt)
    active.extend(freq_signals[d] for d in freq_windows)
    if use_quarter:
        active.append(s_quarter)
    if active:
        s_ts = active[0]
        for a in active[1:]:
            s_ts = np.maximum(s_ts, a)
    else:
        s_ts = np.zeros_like(s_amt)
    feat = feat.copy()
    feat["z_amount"] = z_amt
    feat["z_freq"] = freq_z
    feat["z_trend"] = trend_z
    feat["z_rhythm"] = z_rhythm
    feat["z_rhythm_ind"] = z_rhythm_ind
    feat["z_progress_snap"] = z_progress_snap
    feat["z_conc"] = z_conc
    feat["s_amt"] = s_amt
    feat["s_freq"] = s_freq
    feat["s_trend"] = s_trend
    feat["z_quarter"] = z_quarter
    feat["s_quarter"] = s_quarter
    feat["s_progress"] = s_progress
    feat["s_rhythm"] = s_rhythm
    feat["s_conc"] = s_conc
    feat["s_ts"] = s_ts
    return feat
