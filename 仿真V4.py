"""
预算执行仿真数据生成器 B（独立生成机制 · E22 跨生成机制验证）
================================================================
与生成器 A（仿真V3.py）**独立**，用于检验 MS-TRP 的投稿论断是否依赖单一生成机制。

独立性边界（只改"检测相关机制"，不改"业务骨架/任务定义"）：
  · 复用（业务骨架，定义"任务"而非"机制"）：
      单位/项目/指标/明细/收款人/账户 层级结构、科目与项目分类词表、
      支付方式/资金渠道口径、凭证/审批/预警/更正/退款 事后日志、月末快照。
  · 替换（检测相关机制，A/B 各自独立）：
      1) 年度节奏曲线  A:(m/12)^θ×12月加成（单调、年底集中）
                      B:log-正弦基波+二次谐波（光滑、可有双峰，相位随单位不同）
      2) 正常支付流     A:月度预算约束联动（月内金额 gamma 抽样后归一，合计恒等于月预算）
                      B:独立 lognormal 金额（逐笔 i.i.d.，不强制合计=预算）
      3) 异常注入       A/B 六类异常机制逐类不同（见 inject_anomalies_B）

关键不变式（与 A 一致，保证公平可比较）：
  · 同一 schema、同一 6 类异常、同一时间划分(2023-2026)、同一主种子协议；
  · 异常"数量/比率目标"与 A 相同（只改机制，不改难度）；
  · P0-2 事件成员锁单一自然年、不跨 split。

用法：
    python 仿真V4.py --seed 10 --out data_fixed_b/seed10
    python 仿真V4.py --seed 20 --out data_fixed_b/seed20
    python 仿真V4.py --seed 30 --out data_fixed_b/seed30
"""
import os
import sys
import copy
import importlib.util

import pandas as pd
import numpy as np
import sqlite3

# ── 以模块方式加载生成器 A（仿真V3.py），复用业务骨架生成器与词表常量 ──
_ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("genA", os.path.join(_ROOT, "仿真V3.py"))
genA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(genA)

CFG = copy.deepcopy(genA.CFG)
CFG["generator"] = "仿真V4.py"          # 独立生成器 B
CFG["db_name"] = "budget_data_v5.db"    # 与 A 同库名，便于 pipeline 直接读
ANOMALY_TYPES = genA.ANOMALY_TYPES
FUND_CHANNEL_PAY_TYPES = genA.FUND_CHANNEL_PAY_TYPES
PAYMENT_TYPE_MAP = genA.PAYMENT_TYPE_MAP
ECON_SALARY_CODES = genA.ECON_SALARY_CODES
GOV_CARD_ECON = genA.GOV_CARD_ECON
_rand_day = genA._rand_day
_days_in_month = genA._days_in_month
MONTHLY_SALARY = {"30101", "30102", "30107", "30113", "30302"}

# 收款人其它类采样（与 A 同口径）
PAYEE_OTHER = ["VENDOR", "PERSONAL", "BANK_INTERIM"]
PAYEE_OTHER_PROBS = np.array([0.65, 0.25, 0.10])

_MIN_AMT = {"办公费":0.05,"印刷费":0.03,"水费":0.50,"电费":0.50,"邮电费":0.10,"差旅费":0.10,
            "维修（护）费":0.50,"培训费":0.20,"劳务费":0.10,"委托业务费":1.00,"工会经费":0.50,
            "公务用车运行维护费":0.30,"办公设备购置":0.50,"专用设备购置":0.50,"房屋建筑物购建":10.00,
            "大型修缮":10.00,"信息网络及软件购置更新":1.00,"物业管理费":0.50,"会议费":0.20,
            "专用材料费":0.30,"其他交通费用":0.10,"其他商品和服务支出":0.10,"其他资本性支出":1.00,
            "基本工资":0.10,"津贴补贴":0.10,"奖金":0.10,"绩效工资":0.10,"住房公积金":0.10,
            "其他工资福利支出":0.10,"退休费":0.50}


# ═══════════════════════════════════════════════════════════
# B1. 年度节奏曲线（独立：log-正弦基波 + 二次谐波）
# ═══════════════════════════════════════════════════════════
def generate_unit_rhythms_B(units_df, seed):
    """单位年度花钱节奏（B 机制）：p_m ∝ exp(A·cos(2π(m−φ)/12) + H·cos(4π(m−φ)/12))。
    与 A 的 (m/12)^θ×12月加成（单调年底集中）结构不同：B 为光滑周期型，
    相位 φ 随单位不同，可有双峰；跨年以相位缓慢漂移延续。返回结构同 A。
    """
    np.random.seed(seed)
    years = list(CFG["years"])
    profiles = {}
    rows = []
    for _, u in units_df.iterrows():
        uid = u["unit_id"]
        amp = float(np.clip(np.random.normal(0.5, 0.2), 0.15, 0.9))   # 基波振幅
        harm = float(np.clip(np.random.normal(0.0, 0.12), -0.25, 0.25))  # 二次谐波
        phase = float(np.random.uniform(0.0, 12.0))                   # 相位(月)
        drift = float(np.clip(np.random.normal(0.0, 0.03), -0.08, 0.08))  # 逐年相位漂移
        prof = {}
        for y in years:
            ph = phase + drift * (y - years[0]) * 6.0
            m = np.arange(1, 13, dtype=float)
            shape = np.exp(amp * np.cos(2*np.pi*(m - ph)/12.0)
                           + harm * np.cos(4*np.pi*(m - ph)/12.0))
            shape = shape * np.exp(np.random.normal(0.0, 0.08, size=12))
            shape = np.clip(shape, 0.0, None)
            shape = shape / shape.sum()
            prof[y] = shape
            rows.append({
                "unit_id": uid, "unit_name": u["unit_name"], "year": y,
                "theta": round(amp, 4), "drift": round(drift, 4),
                "december_bump": round(ph, 4),
                "profile": ",".join(f"{x:.4f}" for x in shape),
            })
        profiles[uid] = prof
    return profiles, pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# B2. 正常支付流（独立：Poisson 月笔数 + 独立 lognormal 金额）
# ═══════════════════════════════════════════════════════════
def generate_payments_B(details_df, projects_df, payees_df, accounts_df, unit_profiles, detail_delta, seed):
    """B 机制的正常支付生成。与 A 的关键差异：
      · 月内金额 = 独立 lognormal（逐笔 i.i.d.，期望 ≈ 明细年度预算/总笔数），
        **不强制**"月内合计=月预算"（A 的预算约束联动被打破）；
      · 月度节奏 = 正弦曲线（B1）而非 A 的幂律年底集中。
    业务骨架（收款人路由/支付方式/生命周期/状态）与 A 一致，schema 完全一致。
    """
    np.random.seed(seed)
    target = CFG["total_payments_target"]
    amounts = details_df["detail_amount_wan"].values
    weights = np.clip(amounts / amounts.sum() * target, 5, 150).astype(int)
    total_alloc = weights.sum()
    if total_alloc < target:
        shortfall = target - total_alloc
        top = np.argsort(amounts)[-min(len(amounts), shortfall//5+1):]
        weights[top] += max(1, shortfall//len(top))
    pay_counts = np.clip(weights, 0, 200)
    cold_mask = np.random.random(len(pay_counts)) < 0.05
    pay_counts[cold_mask] = np.random.randint(1, 4, size=cold_mask.sum())

    life = projects_df.set_index("project_id")[["start_year", "end_year"]].to_dict("index")
    base_year_weight = {y: w for y, w in zip(CFG["years"], [0.22, 0.24, 0.26, 0.28])}

    pool_by_type = payees_df.groupby("payee_category")["payee_id"].apply(list).to_dict()
    payee_accs = accounts_df.groupby("payee_id")["account_id"].apply(list).to_dict()
    payee_name = payees_df.set_index("payee_id")["payee_name"].to_dict()
    payee_type = payees_df.set_index("payee_id")["payee_type"].to_dict()
    acc_info = accounts_df.set_index("account_id")[["bank_name", "bank_account"]].to_dict("index")
    salary_by_unit = payees_df[payees_df["payee_category"] == "SALARY_ACCT"].set_index("unit_id")["payee_id"].to_dict()
    salary_fallback = pool_by_type["SALARY_ACCT"][0]

    threshold = CFG["large_pay_threshold_wan"]
    statuses = ["DRAFT", "SUBMITTED", "APPROVED", "PAID", "SETTLED"]
    s_early = np.array([0.12, 0.18, 0.23, 0.27, 0.20])
    s_late = np.array([0.03, 0.07, 0.15, 0.35, 0.40])

    records, pay_id = [], 0
    for idx, (_, det) in enumerate(details_df.iterrows()):
        n_pay = int(pay_counts[idx])
        if n_pay <= 0:
            continue
        det_amt = det["detail_amount_wan"]
        unit_name = det["unit_name"]
        eco_name = det["econ_class_name"]
        min_a = _MIN_AMT.get(eco_name, 0.05)

        lp = life.get(det["project_id"])
        valid_years = (list(CFG["years"]) if lp is None
                       else [y for y in CFG["years"] if lp["start_year"] <= y <= lp["end_year"]]
                       or [lp["start_year"]])
        vw = np.array([base_year_weight[y] for y in valid_years], dtype=float)
        vw = vw * np.exp(np.random.normal(0.0, 0.12, size=len(valid_years)))
        vw = vw / vw.sum()

        up = unit_profiles.get(det["unit_id"])
        dd = detail_delta.get(det["detail_id"], np.ones(12))
        year_prof = {}
        for y in valid_years:
            prof = (up[y] * dd) if (up is not None and y in up) else (np.full(12, 1.0/12.0) * dd)
            prof = prof / prof.sum()
            year_prof[y] = prof

        # 把 n_pay 分配到 (年,月) 单元格（年度份额 × 正弦节奏），最大余数法
        cells = []
        for jy, y in enumerate(valid_years):
            for m in range(1, 13):
                cells.append((y, m, jy, vw[jy] * year_prof[y][m - 1]))
        probs = np.array([c[3] for c in cells], dtype=float)
        probs = probs / probs.sum()
        raw_counts = probs * n_pay
        n_cell = np.floor(raw_counts).astype(int)
        rem = n_pay - int(n_cell.sum())
        if rem > 0:
            order = np.argsort(-(raw_counts - n_cell))
            for k in range(int(rem)):
                n_cell[order[k]] += 1

        # 工资统发：月度固定发放（与 A 一致；统发工资为业务事实，非检测机制差异）
        salary_monthly = {}
        if det["econ_class_code"] in MONTHLY_SALARY:
            exec_ratio = float(np.clip(np.random.normal(0.92, 0.08), 0.5, 1.0))
            total_to_pay = det_amt * exec_ratio
            n_months = len(valid_years) * 12
            monthly_base = total_to_pay / n_months
            monthly = monthly_base * np.random.normal(1.0, 0.02, n_months)
            monthly = np.clip(monthly, monthly_base * 0.9, monthly_base * 1.1)
            monthly = monthly / monthly.sum() * total_to_pay
            monthly = np.maximum(np.round(monthly, 1), 0.01)
            monthly[np.argmax(monthly)] += round(total_to_pay - monthly.sum(), 1)
            k = 0
            for y in valid_years:
                for m in range(1, 13):
                    salary_monthly[(y, m)] = float(monthly[k])
                    k += 1
            n_cell = np.ones(len(cells), dtype=int)  # 每月恰一笔

        # B 机制：独立 lognormal 金额参数（期望 = 明细年度预算 / 总笔数）
        mu_d = np.log(max(det_amt / max(n_pay, 1), min_a)) - 0.5 * 0.6 * 0.6

        for ci, (y, m, jy, _p) in enumerate(cells):
            nc = int(n_cell[ci])
            if nc <= 0:
                continue
            for _ in range(nc):
                key = (y, m)
                if key in salary_monthly:
                    amt = salary_monthly[key]
                else:
                    amt = float(np.random.lognormal(mu_d, 0.6))
                    amt = min(max(amt, min_a), det_amt * 0.5)  # 单笔不超明细预算一半
                amt = round(amt, 1)
                if amt < 0.01:
                    amt = 0.01

                ch_code = det["fund_channel_code"]
                allowed = FUND_CHANNEL_PAY_TYPES[ch_code]
                codes = [c for c, _w in allowed]
                wts = np.array([w for _c, w in allowed], dtype=float)
                pt_code = codes[np.random.choice(len(codes), p=wts / wts.sum())]

                is_salary = det["econ_class_code"] in ECON_SALARY_CODES
                if is_salary:
                    pt_code = "FINANCE_ZERO_BALANCE"

                is_gov_card = "N"
                if ch_code == "FISCAL" and eco_name in GOV_CARD_ECON and amt <= 5.0 and np.random.random() < 0.15:
                    is_gov_card = "Y"
                    pt_code = "UNIT_ZERO_BALANCE"
                    amt = round(min(amt, np.random.uniform(0.1, 5.0)), 1)

                pt = PAYMENT_TYPE_MAP[pt_code]
                day = _rand_day(y, m)
                probs_s = s_early if m <= 9 else s_late
                probs_s = probs_s / probs_s.sum()
                status = statuses[np.random.choice(len(statuses), p=probs_s)]

                if is_gov_card == "Y":
                    pcat = "CARD_HOLDER"
                elif is_salary:
                    pcat = "SALARY_ACCT"
                else:
                    pcat = PAYEE_OTHER[np.random.choice(len(PAYEE_OTHER), p=PAYEE_OTHER_PROBS)]

                if pcat == "SALARY_ACCT":
                    pid_ = salary_by_unit.get(det["unit_id"], salary_fallback)
                else:
                    pid_ = np.random.choice(pool_by_type[pcat])
                acc_ids = payee_accs[pid_]
                account_id = np.random.choice(acc_ids)
                ai = acc_info[account_id]

                pay_id += 1
                records.append({
                    "apply_id": f"PA{str(pay_id).zfill(10)}",
                    "unit_name": unit_name,
                    "project_id": det["project_id"],
                    "indicator_id": det["indicator_id"],
                    "detail_id": det["detail_id"],
                    "fund_channel_code": ch_code, "fund_channel": det["fund_channel"],
                    "year": y,
                    "pay_type_code": pt["code"], "pay_type_label": pt["label"],
                    "approval_levels": pt["levels"],
                    "is_gov_card": is_gov_card,
                    "payee_id": pid_,
                    "payee_name": payee_name[pid_],
                    "payee_bank": ai["bank_name"],
                    "payee_account": ai["bank_account"],
                    "account_id": account_id,
                    "public_private": payee_type[pid_],
                    "func_class_code": det["func_class_code"],
                    "func_class_name": det["func_class_name"],
                    "econ_class_code": det["econ_class_code"],
                    "econ_class_name": det["econ_class_name"],
                    "procurement_cat": det["procurement_cat"],
                    "amount_wan": amt,
                    "approval_status": status,
                    "biz_status": "SETTLED" if status == "SETTLED" else ("PAID" if status == "PAID" else "IN_FLIGHT"),
                    "is_large": "Y" if amt >= threshold else "N",
                    "apply_date": f"{y}-{m:02d}-{day:02d}",
                    "is_executed": "Y" if status in ["PAID", "SETTLED"] else "N",
                    "is_gov_proc_pay": det["is_procurement"],
                })
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════
# B3. 异常注入（独立：六类异常逐类换机制）
# ═══════════════════════════════════════════════════════════
def inject_anomalies_B(payments_df, details_df, projects_df, indicators_df, unit_profiles, detail_delta, seed):
    np.random.seed(seed)
    events = []
    df = payments_df.copy()
    acfg = CFG["anomaly_config"]
    det_budget = details_df.set_index("detail_id")["detail_amount_wan"].to_dict()
    life_map = projects_df.set_index("project_id")[["start_year", "end_year"]].to_dict("index")

    TEST_YEAR = int(max(CFG["years"]))
    TEST_BIAS = 0.5

    def pick_year(years):
        ys = sorted({int(y) for y in years})
        if TEST_YEAR in ys and np.random.random() < TEST_BIAS:
            return TEST_YEAR
        return int(np.random.choice(ys))

    def cap_amt(det_id, new_amt):
        budget = det_budget.get(det_id, 99999)
        return min(new_amt, max(budget * 2.0, 100))

    aid2idx = {str(a): i for i, a in enumerate(df["apply_id"])}
    locked_year = {}

    # ── AMOUNT_SPIKE：加性偏移（A 为乘性倍数；排除工资统发）──
    spike_pool = df.index[~df["econ_class_code"].isin(ECON_SALARY_CODES)].tolist()
    n1 = max(100, int(len(df) * acfg["AMOUNT_SPIKE"]["ratio"]))
    n1 = min(n1, len(spike_pool))
    det_std = df.groupby("detail_id")["amount_wan"].std().to_dict()
    for i in np.random.choice(spike_pool, n1, replace=False):
        orig = df.at[i, "amount_wan"]
        sd = det_std.get(df.at[i, "detail_id"], 0.0)
        # 单笔明细 std=NaN（ddof=1），NaN 为真值会穿透 or 回退，须显式兜底
        if sd is None or not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        k = np.random.choice([4, 6, 8, 10, 12])
        new_amt = cap_amt(df.at[i, "detail_id"], round(orig + k * sd, 1))
        df.at[i, "amount_wan"] = new_amt
        events.append({
            "anomaly_type": "AMOUNT_SPIKE",
            "anomaly_subtype": np.random.choice(["录入错误", "异常大额", "指标偏差"]),
            "risk_level": "HIGH" if k > 7 else "MEDIUM",
            "target_entity_type": "PAYMENT",
            "target_entity_id": str(df.at[i, "apply_id"]),
            "related_entities": f"{df.at[i,'unit_name']},{df.at[i,'project_id']}",
            "inject_parameters": f"orig={orig:.2f},add={k*sd:.2f}(k={k})",
            "affected": [str(df.at[i, "apply_id"])],
        })

    # ── FREQ_BURST：10 天窗口内的持续高频（A 为 2-3 天短促）──
    freq_pool = details_df.index[~details_df["econ_class_code"].isin(ECON_SALARY_CODES)].tolist()
    n2 = max(15, int(len(details_df) * acfg["FREQ_BURST"]["indicator_ratio"]))
    n2 = min(n2, len(freq_pool))
    for ti in np.random.choice(freq_pool, n2, replace=False):
        dcode = details_df.iloc[ti]["detail_id"]
        iidx = df[df["detail_id"] == dcode].index.tolist()
        if len(iidx) < 8:
            continue
        detail_years = sorted({str(df.at[i, "apply_date"])[:4] for i in iidx})
        by = int(np.random.choice(detail_years))
        bm = np.random.randint(2, 11)
        ns = np.random.randint(8, 16)  # 比 A 的 (5,10) 更多、更分散
        spike_ids = []
        for _ in range(ns):
            new_row = df.iloc[iidx[0]].copy()
            new_row["apply_id"] = f"BURST{np.random.randint(1000,9999)}{np.random.randint(1000,9999)}"
            new_row["amount_wan"] = round(np.random.gamma(shape=2, scale=15), 1)
            new_row["year"] = by
            new_row["apply_date"] = f"{by}-{bm:02d}-{np.random.randint(1,11):02d}"  # 前10天窗口
            df.loc[len(df)] = new_row
            spike_ids.append(str(new_row["apply_id"]))
        for aid in spike_ids:
            locked_year[aid] = by
        events.append({
            "anomaly_type": "FREQ_BURST",
            "anomaly_subtype": "持续高频(10日)",
            "risk_level": "MEDIUM",
            "target_entity_type": "INDICATOR_DETAIL",
            "target_entity_id": str(dcode),
            "related_entities": f"{new_row['unit_name']},{dcode}",
            "inject_parameters": f"detail={dcode},n={ns},days=1-10",
            "affected": spike_ids,
        })

    # ── PROGRESS_LAG：光滑幂律阻尼（A 为"前9月×0.2 + Q4 回填"的阶跃压缩）──
    lag_pool = indicators_df.index[~indicators_df["econ_class_code"].isin(ECON_SALARY_CODES)].tolist()
    n3 = max(10, int(len(indicators_df) * acfg["PROGRESS_LAG"]["indicator_ratio"]))
    n3 = min(n3, len(lag_pool))
    for ti in np.random.choice(lag_pool, n3, replace=False):
        irow = indicators_df.iloc[ti]
        ipay = df[df["indicator_id"] == irow["indicator_id"]]
        if len(ipay) < 10:
            continue
        up = unit_profiles.get(irow["unit_id"])
        if up is None:
            continue
        year_cnt = ipay["apply_date"].astype(str).str[:4].value_counts()
        cand = [int(y) for y, c in year_cnt.items() if c >= 6 and int(y) in up]
        if not cand:
            continue
        ty = pick_year(cand)
        ipay_y = ipay[ipay["apply_date"].astype(str).str.startswith(str(ty))]
        normal = np.asarray(up[ty], dtype=float)
        normal = normal / normal.sum()
        lagged = normal * (np.arange(1, 13, dtype=float) / 12.0) ** 1.8  # 早月光滑阻尼
        lagged = lagged / lagged.sum()
        mids = []
        for pi in ipay_y.index:
            m = int(np.random.choice(range(1, 13), p=lagged))
            df.at[pi, "apply_date"] = f"{ty}-{m:02d}-{_rand_day(ty, m):02d}"
            mids.append(str(df.at[pi, "apply_id"]))
        for aid in mids:
            locked_year[aid] = ty
        events.append({
            "anomaly_type": "PROGRESS_LAG",
            "anomaly_subtype": "早月光滑欠执行(幂律)",
            "risk_level": "HIGH" if irow["ind_amount_wan"] > 500 else "MEDIUM",
            "target_entity_type": "INDICATOR",
            "target_entity_id": str(irow["indicator_id"]),
            "related_entities": f"{irow['unit_name']},{irow['project_id']}",
            "inject_parameters": f"indicator={irow['indicator_id']},year={ty},mode=power1.8",
            "affected": mids,
        })

    # ── YEAREND_CONC：挪到 11-12 月两月窗口（A 为只挪到 12 月）──
    n4 = max(12, int(len(genA.ALL_UNITS) * acfg["YEAREND_CONC"]["unit_ratio"]))
    all_names = df["unit_name"].unique()
    for uname in np.random.choice(all_names, min(n4, len(all_names)), replace=False):
        upay = df[df["unit_name"] == uname]
        if len(upay) < 10:
            continue
        years = sorted({str(a)[:4] for a in upay["apply_date"]})
        if not years:
            continue
        ty = pick_year(years)
        upay_y = upay[upay["apply_date"].astype(str).str.startswith(str(ty))]
        non_late = upay_y[~upay_y["apply_date"].astype(str).str[5:7].isin(["11", "12"])
                         & ~upay_y["econ_class_code"].isin(ECON_SALARY_CODES)].index
        if len(non_late) == 0:
            continue
        mc = max(3, int(len(non_late) * acfg["YEAREND_CONC"]["move_ratio"]))
        mc = min(mc, len(non_late))
        moved_ids = []
        for mi in np.random.choice(non_late, mc, replace=False):
            tm = np.random.choice([11, 12])
            df.at[mi, "apply_date"] = f"{ty}-{tm:02d}-{_rand_day(ty, tm):02d}"
            moved_ids.append(str(df.at[mi, "apply_id"]))
        for aid in moved_ids:
            locked_year[aid] = ty
        events.append({
            "anomaly_type": "YEAREND_CONC",
            "anomaly_subtype": "年末突击(11-12月)",
            "risk_level": "MEDIUM",
            "target_entity_type": "UNIT",
            "target_entity_id": str(uname),
            "related_entities": str(uname),
            "inject_parameters": f"unit={uname},moved_to_novdec={mc}",
            "affected": moved_ids,
        })

    # ── HISTORICAL_SHIFT：相位平移（A 为"前后反转"）──
    shift_pool = details_df.index[~details_df["econ_class_code"].isin(ECON_SALARY_CODES)].tolist()
    n5 = max(10, int(len(details_df) * acfg["HISTORICAL_SHIFT"]["indicator_ratio"]))
    n5 = min(n5, len(shift_pool))
    for ti in np.random.choice(shift_pool, n5, replace=False):
        drow = details_df.iloc[ti]
        dpay = df[df["detail_id"] == drow["detail_id"]]
        if len(dpay) < 5:
            continue
        years = sorted({str(a)[:4] for a in dpay["apply_date"]})
        if not years:
            continue
        ty = int(np.random.choice(years))
        dpay_y = dpay[dpay["apply_date"].astype(str).str.startswith(str(ty))]
        if len(dpay_y) < 3:
            continue
        up = unit_profiles.get(drow["unit_id"])
        dd = detail_delta.get(drow["detail_id"], np.ones(12))
        if up is None or ty not in up:
            continue
        normal = up[ty] * dd
        normal = normal / normal.sum()
        shift = int(np.random.choice([3, 6, 9]))
        shifted_p = np.roll(normal, shift)  # 循环相位平移，非反转
        shifted_p = shifted_p / shifted_p.sum()
        shifted = []
        for pi in dpay_y.index:
            m = int(np.random.choice(range(1, 13), p=shifted_p))
            df.at[pi, "apply_date"] = f"{ty}-{m:02d}-{_rand_day(ty, m):02d}"
            shifted.append(str(df.at[pi, "apply_id"]))
        for aid in shifted:
            locked_year[aid] = int(ty)
        events.append({
            "anomaly_type": "HISTORICAL_SHIFT",
            "anomaly_subtype": f"{ty}年节奏平移(+{shift}月)",
            "risk_level": "HIGH",
            "target_entity_type": "INDICATOR_DETAIL",
            "target_entity_id": str(drow["detail_id"]),
            "related_entities": f"{drow['unit_name']},{drow['project_id']}",
            "inject_parameters": f"detail={drow['detail_id']},year={ty},mode=shift{shift}",
            "affected": shifted,
        })

    # ── ENTITY_AGGREGATION：重连实体边（不改日期；A 为"挪日期进 5 日窗口"）──
    ent = acfg["ENTITY_AGGREGATION"]
    aid2idx = {str(a): i for i, a in enumerate(df["apply_id"])}

    # Shared Payee：把不同项目的支付"重连"到同一收款人（保持各自日期，纯图异常）
    for _ in range(int(ent["shared_payee_count"])):
        ty = pick_year(CFG["years"])
        cand = df[(df["year"] == ty) & (~df["apply_id"].astype(str).isin(set(locked_year)))]
        if len(cand) < 3:
            continue
        projs = cand["project_id"].unique()
        n_proj = min(np.random.randint(3, 5), len(projs))
        chosen = np.random.choice(projs, n_proj, replace=False)
        # 目标收款人：任选一笔（属于某项目）作为共享收款人来源
        src = cand.sample(1).iloc[0]
        py = src["payee_id"]
        target_row = df[df["payee_id"] == py].iloc[0]
        affected = []
        for p in chosen:
            sub = cand[cand["project_id"] == p]
            ap = str(sub.iloc[0]["apply_id"])
            idx = aid2idx[ap]
            df.at[idx, "payee_id"] = py
            df.at[idx, "payee_name"] = target_row["payee_name"]
            df.at[idx, "payee_bank"] = target_row["payee_bank"]
            df.at[idx, "payee_account"] = target_row["payee_account"]
            df.at[idx, "account_id"] = target_row["account_id"]
            affected.append(ap)
        if len(affected) < 2:
            continue
        for aid in affected:
            locked_year[aid] = ty
        events.append({
            "anomaly_type": "ENTITY_AGGREGATION",
            "anomaly_subtype": "Shared Payee(重连)",
            "risk_level": "HIGH",
            "target_entity_type": "PAYEE",
            "target_entity_id": str(py),
            "related_entities": f"payee={py},projects={','.join(chosen)}",
            "inject_parameters": f"shared_payee={py},n_projects={n_proj},mode=rewire",
            "affected": affected,
        })

    # Shared Account：不同收款人的支付重连到同一账户（不改日期，保持各自收款人）
    for _ in range(int(ent["shared_account_count"])):
        per_payee = df.drop_duplicates("payee_id")
        if len(per_payee) < 2:
            break
        n = min(np.random.randint(2, 4), len(per_payee))
        picks = per_payee.sample(n)
        src = picks.iloc[0]
        shared = src["account_id"]
        shared_bank = src["payee_bank"]
        shared_acc = src["payee_account"]
        affected = []
        for py in picks["payee_id"].tolist()[1:]:
            sub = df[(df["payee_id"] == py) & (~df["apply_id"].astype(str).isin(set(locked_year)))]
            if len(sub) == 0:
                continue
            ap = str(sub.iloc[0]["apply_id"])
            idx = aid2idx[ap]
            df.at[idx, "account_id"] = shared
            df.at[idx, "payee_bank"] = shared_bank
            df.at[idx, "payee_account"] = shared_acc
            affected.append(ap)
        if len(affected) < 1:
            continue
        events.append({
            "anomaly_type": "ENTITY_AGGREGATION",
            "anomaly_subtype": "Shared Account(重连)",
            "risk_level": "HIGH",
            "target_entity_type": "ACCOUNT",
            "target_entity_id": str(shared),
            "related_entities": f"account={shared},payees={','.join(picks['payee_id'].tolist())}",
            "inject_parameters": f"shared_account={shared},n_payees={n},mode=rewire",
            "affected": affected,
        })

    # 统一事件标签结构（与 A 相同）
    final_idx = {str(a): i for i, a in enumerate(df["apply_id"])}
    rows = []
    for eid, ev in enumerate(events, start=1):
        affected = [a for a in ev["affected"] if a in final_idx]
        if not affected:
            continue
        dates = sorted(str(df.at[final_idx[a], "apply_date"]) for a in affected)
        rows.append({
            "event_id": f"EV{eid:06d}",
            "anomaly_type": ev["anomaly_type"],
            "anomaly_subtype": ev.get("anomaly_subtype"),
            "risk_level": ev.get("risk_level"),
            "target_entity_type": ev.get("target_entity_type"),
            "target_entity_id": ev.get("target_entity_id"),
            "start_time": dates[0],
            "end_time": dates[-1],
            "anchor_apply_id": affected[0],
            "affected_apply_ids": ",".join(affected),
            "related_entities": ev.get("related_entities"),
            "inject_parameters": ev.get("inject_parameters"),
        })
    labels_df = pd.DataFrame(rows)
    return df, labels_df


# ═══════════════════════════════════════════════════════════
# 主流程（业务骨架与事后日志复用 A，只替换节奏/支付/注入三处）
# ═══════════════════════════════════════════════════════════
def main():
    if "--seed" in sys.argv:
        CFG["seed"] = int(sys.argv[sys.argv.index("--seed") + 1])
    if "--out" in sys.argv:
        CFG["output_dir"] = sys.argv[sys.argv.index("--out") + 1]
    print("=" * 60)
    print(f"  Budget Simulation B (独立机制)  |  seed={CFG['seed']}")
    print("=" * 60)

    np.random.seed(CFG["seed"])
    out = CFG["output_dir"]
    os.makedirs(out, exist_ok=True)
    db_path = os.path.join(out, CFG["db_name"])
    if os.path.exists(db_path):
        os.remove(db_path)

    # 1-3 实体骨架（复用 A）
    units_df = genA.generate_units(CFG["seed"])
    unit_profiles, unit_rhythm_df = generate_unit_rhythms_B(units_df, CFG["seed"] + 15)
    projects_df = genA.generate_projects(units_df, CFG["seed"] + 1)
    indicators_df = genA.generate_indicators(projects_df, CFG["seed"] + 2)
    details_df = genA.generate_indicator_details(indicators_df, CFG["seed"] + 12)
    detail_delta = genA.generate_detail_rhythm_deltas(details_df, CFG["seed"] + 16)
    plans_df = genA.generate_large_fund_plans(projects_df, indicators_df, CFG["seed"] + 11)
    expense_cat_df = genA.generate_expense_categories(CFG["seed"] + 13)
    procurement_item_df = genA.generate_procurement_items(CFG["seed"] + 14)
    payees_df, accounts_df = genA.generate_payee_pool(units_df, CFG["seed"] + 9)
    print(f"  -> {len(units_df)} units, {len(projects_df)} projects, "
          f"{len(indicators_df)} indicators, {len(details_df)} details")

    # 4 支付（B 机制）
    payments_df = generate_payments_B(details_df, projects_df, payees_df, accounts_df,
                                      unit_profiles, detail_delta, CFG["seed"] + 3)
    print(f"  -> {len(payments_df)} payments (normal)")

    # 5 异常注入（B 机制）+ 一致性修复
    payments_df, ground_truth = inject_anomalies_B(payments_df, details_df, projects_df,
                                                   indicators_df, unit_profiles, detail_delta,
                                                   CFG["seed"] + 4)
    payments_df = genA.sanitize_payments(payments_df)
    for at in ANOMALY_TYPES:
        print(f"     - {at}: {(ground_truth['anomaly_type'] == at).sum()}")

    snapshots_df = genA.generate_indicator_snapshots(indicators_df, payments_df, unit_profiles, CFG["seed"] + 17)

    # 回填已支付金额
    paid_det = payments_df.groupby("detail_id")["amount_wan"].sum()
    for i, row in details_df.iterrows():
        details_df.at[i, "paid_amount_wan"] = round(paid_det.get(row["detail_id"], 0.0), 1)
    paid_ind = payments_df.groupby("indicator_id")["amount_wan"].sum()
    for i, row in indicators_df.iterrows():
        indicators_df.at[i, "paid_amount_wan"] = round(paid_ind.get(row["indicator_id"], 0.0), 1)

    # 事后日志（复用 A）
    vouchers_df = genA.generate_vouchers(payments_df, CFG["seed"] + 10)
    approvals_df = genA.generate_approvals(payments_df, CFG["seed"] + 5)
    warnings_df = genA.generate_warnings(payments_df, CFG["seed"] + 6)
    corrections_df = genA.generate_corrections(payments_df, CFG["seed"] + 7)
    refunds_df = genA.generate_refunds(payments_df, CFG["seed"] + 8)

    # 落盘（schema 与 A 完全一致）
    conn = sqlite3.connect(db_path)
    for df, name in [(units_df, "unit"), (projects_df, "project"), (indicators_df, "indicator"),
                     (details_df, "indicator_detail"), (plans_df, "large_fund_plan"),
                     (expense_cat_df, "expense_category"), (procurement_item_df, "procurement_item"),
                     (payments_df, "payment_apply"), (vouchers_df, "payment_voucher"),
                     (payees_df, "payee"), (accounts_df, "bank_account"), (approvals_df, "payment_approval")]:
        df.to_sql(name, conn, if_exists="replace", index=False)
    ground_truth.to_sql("ground_truth_event", conn, if_exists="replace", index=False)
    unit_rhythm_df.to_sql("unit_rhythm", conn, if_exists="replace", index=False)
    snapshots_df.to_sql("indicator_snapshot", conn, if_exists="replace", index=False)
    for df, name in [(warnings_df, "warning"), (corrections_df, "payment_correction"), (refunds_df, "refund")]:
        df.to_sql(name, conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_pay_unit ON payment_apply(unit_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pay_date ON payment_apply(apply_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pay_account ON payment_apply(account_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gt_type ON ground_truth_event(anomaly_type)")
    genA.generate_data_dict(conn, out)
    conn.close()

    print(f"\n  B Complete | {len(payments_df):,} payments | seed={CFG['seed']} | out={out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
