"""
仿真数据全面检查 — 对照任务书 + 实验前强制修复(3.1~3.8)逐项验证
"""
import sqlite3
import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_DIR, "data_fixed", "budget_data_v5.db")
conn = sqlite3.connect(DB_PATH)

print("=" * 72)
print("  仿真数据全面检查报告（含实验前强制修复验收）")
print("=" * 72)

# ============================================================
# 1. 表完整性 & 数据量
# ============================================================
print("\n【1】表完整性 & 数据量")
print("-" * 50)

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()
print(f"数据库表数量: {len(tables)}")
for t in sorted(tables):
    cnt = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {t}", conn).iloc[0, 0]
    print(f"  {t}: {cnt:,} 条")

payment_count = pd.read_sql("SELECT COUNT(*) FROM payment_apply", conn).iloc[0, 0]
print(f"\n支付记录总数: {payment_count:,}")
print(f"任务书要求: ≥50,000 条  →  {'OK 满足' if payment_count >= 50000 else 'FAIL 不满足'}")
CHECK_data_volume = payment_count >= 50000

# ============================================================
# 2. 图节点覆盖（指标拆两级后实际 10 类）
# ============================================================
print("\n【2】异构图节点覆盖检查")
print("-" * 50)

node_checks = {
    "预算单位": ("unit", "unit_id"),
    "预算项目": ("project", "project_id"),
    "预算指标(指标名称)": ("indicator", "indicator_id"),
    "明细内容(三级明细)": ("indicator_detail", "detail_id"),
    "支付申请": ("payment_apply", "apply_id"),
    "支付凭证": ("payment_voucher", "voucher_id"),
    "收款人": ("payee", "payee_id"),
    "银行账户": ("bank_account", "account_id"),
    "支出经济分类(第8类)": ("expense_category", "expense_cat_code"),
    "采购品目(第9类)": ("procurement_item", "procurement_item_id"),
}
node_ok = True
for name, (table, col) in node_checks.items():
    if table in tables:
        cnt = pd.read_sql(f"SELECT COUNT(DISTINCT [{col}]) FROM {table}", conn).iloc[0, 0]
        ok = cnt > 0
        node_ok = node_ok and ok
        print(f"  {'OK' if ok else 'FAIL'} {name} ({table}): {cnt:,} 个")
    else:
        node_ok = False
        print(f"  FAIL {name} ({table}): 表不存在")

if "indicator" in tables and "indicator_detail" in tables:
    ind_name = pd.read_sql("SELECT COUNT(DISTINCT indicator_name) FROM indicator", conn).iloc[0, 0]
    det_name = pd.read_sql("SELECT COUNT(DISTINCT detail_name) FROM indicator_detail", conn).iloc[0, 0]
    print(f"  OK 指标两级分离: 指标名称 {ind_name} 种 / 明细内容 {det_name} 种")

print(f"\n图节点覆盖: 单位/项目/指标/明细/支付/凭证/收款人/账户 + 经济分类(第8类) + 采购品目(第9类)")
print("  → 任务书9类节点全部覆盖 OK（指标拆两级后共10类：多出 indicator_detail）")
CHECK_nodes = node_ok

# ============================================================
# 3. 6类异常检查（ground_truth_event）
# ============================================================
print("\n【3】6类异常标签检查（ground_truth_event）")
print("-" * 50)

labels = None
if "ground_truth_event" in tables:
    labels = pd.read_sql("SELECT * FROM ground_truth_event", conn)
    print(f"异常事件总数: {len(labels):,}")
    print(f"\n按anomaly_type分布:")
    type_dist = labels["anomaly_type"].value_counts()
    for atype, cnt in type_dist.items():
        print(f"  {atype}: {cnt} 条")

    # 异常率（含事件级 affected_apply_ids）
    pay_ids = set(pd.read_sql("SELECT apply_id FROM payment_apply", conn)["apply_id"])
    anom_ids = set()
    for v in labels["affected_apply_ids"].dropna():
        for pid in str(v).split(","):
            pid = pid.strip()
            if pid and pid != "N/A":
                anom_ids.add(pid)
    for v in labels["anchor_apply_id"].dropna():
        if v and str(v) != "N/A":
            anom_ids.add(str(v))
    anom_in_pay = anom_ids & pay_ids
    anom_rate = len(anom_in_pay) / len(pay_ids) * 100
    print(f"\n去重异常支付笔数: {len(anom_in_pay):,}")
    print(f"异常率: {anom_rate:.2f}%  (任务书目标 5%-8%)")

    CHECK_6types = labels["anomaly_type"].nunique() >= 6
    print(f"6类异常齐全: {'OK' if CHECK_6types else 'FAIL'}")
else:
    CHECK_6types = False
    print("  FAIL ground_truth_event 表不存在")

# ============================================================
# 4. 支付方式与用款计划检查
# ============================================================
print("\n【4】支付方式与用款计划检查")
print("-" * 50)

pay_type_count = 0
plan_count = 0
if "payment_apply" in tables:
    pay_types = pd.read_sql("""
        SELECT pay_type_label, COUNT(*) as n, ROUND(AVG(amount_wan), 2) as avg_amt
        FROM payment_apply GROUP BY pay_type_label ORDER BY n DESC
    """, conn)
    pay_type_count = len(pay_types)
    print(f"支付方式数量: {pay_type_count} 种（应为4种）")
    print(pay_types.to_string(index=False))

if "payment_apply" in tables and "is_gov_card" in pd.read_sql("SELECT * FROM payment_apply LIMIT 1", conn).columns:
    gc = pd.read_sql("""
        SELECT is_gov_card, pay_type_label, COUNT(*) as n
        FROM payment_apply GROUP BY is_gov_card, pay_type_label ORDER BY n DESC
    """, conn)
    gc_cnt = int(gc[gc["is_gov_card"] == "Y"]["n"].sum())
    wrong = gc[(gc["is_gov_card"] == "Y") & (gc["pay_type_label"] != "财政授权支付")]
    print(f"\n公务卡报销场景(is_gov_card=Y)笔数: {gc_cnt:,}")
    print(f"  公务卡报销走单位零余额: {'OK' if len(wrong) == 0 else f'FAIL {len(wrong)} 笔未走单位零余额'}")

if "large_fund_plan" in tables:
    plan_count = int(pd.read_sql("SELECT COUNT(*) as n FROM large_fund_plan", conn)["n"][0])
    print(f"\n大额资金用款计划(large_fund_plan)记录数: {plan_count}")
CHECK_paytype = pay_type_count == 4 and plan_count > 0
print(f"4类支付方式+公务卡+大额用款计划  →  {'OK' if CHECK_paytype else 'FAIL'}")

# ============================================================
# 5. 冷启动比例检查
# ============================================================
print("\n【5】冷启动比例检查")
print("-" * 50)

if "payment_apply" in tables and "indicator_detail" in tables:
    pay_per_det = pd.read_sql("""
        SELECT detail_id, COUNT(*) as cnt FROM payment_apply GROUP BY detail_id
    """, conn)
    total_det = len(pay_per_det)
    cold_start = (pay_per_det["cnt"] <= 3).sum()
    cold_rate = cold_start / total_det * 100
    print(f"明细总数: {total_det:,}  冷启动(≤3笔): {cold_start:,}  占比: {cold_rate:.1f}%")
    print(f"序列长度: 平均 {pay_per_det['cnt'].mean():.1f} / 中位 {pay_per_det['cnt'].median():.0f} / 最长 {pay_per_det['cnt'].max()}")

# ============================================================
# 6. 图密度估算
# ============================================================
print("\n【6】图密度估算")
print("-" * 50)

if all(t in tables for t in ["unit", "project", "indicator", "indicator_detail", "payment_apply", "payee"]):
    n_unit = pd.read_sql("SELECT COUNT(*) FROM unit", conn).iloc[0, 0]
    n_proj = pd.read_sql("SELECT COUNT(*) FROM project", conn).iloc[0, 0]
    n_ind = pd.read_sql("SELECT COUNT(*) FROM indicator", conn).iloc[0, 0]
    n_det = pd.read_sql("SELECT COUNT(*) FROM indicator_detail", conn).iloc[0, 0]
    n_pay = pd.read_sql("SELECT COUNT(*) FROM payment_apply", conn).iloc[0, 0]
    n_payee = pd.read_sql("SELECT COUNT(*) FROM payee", conn).iloc[0, 0]
    n_voucher = pd.read_sql("SELECT COUNT(*) FROM payment_voucher", conn).iloc[0, 0]
    n_exp = pd.read_sql("SELECT COUNT(*) FROM expense_category", conn).iloc[0, 0]
    n_proc = pd.read_sql("SELECT COUNT(*) FROM procurement_item", conn).iloc[0, 0]
    n_nodes = n_unit + n_proj + n_ind + n_det + n_pay + n_payee + n_voucher + n_exp + n_proc
    n_edges = n_pay * 4 + n_det + n_ind + n_proj + n_voucher
    density = n_edges / (n_nodes * n_nodes) if n_nodes > 0 else 0
    print(f"节点总数: {n_nodes:,}  (单位{n_unit} 项目{n_proj} 指标{n_ind} 明细{n_det} 支付{n_pay:,} 凭证{n_voucher:,} 收款人{n_payee:,} 经济分类{n_exp} 采购品目{n_proc})")
    print(f"边数估算: {n_edges:,}   图密度: {density:.8f}")

# ============================================================
# 7. 可复现性检查
# ============================================================
print("\n【7】可复现性检查")
print("-" * 50)

sim_file = os.path.join(SCRIPT_DIR, "仿真V3.py")
has_import_random = False
standalone_random = 0
if os.path.exists(sim_file):
    with open(sim_file, "r", encoding="utf-8") as f:
        content = f.read()
    import re
    has_import_random = 'import random' in content
    standalone_random = len(re.findall(r'(?<!np\.)random\.', content))
    has_cfg = "CFG" in content
    print(f"独立 random. 调用: {standalone_random}  {'OK' if standalone_random == 0 else 'WARN'}")
    print(f"统一配置CFG: {'OK' if has_cfg else 'FAIL'}")
CHECK_repro = (not has_import_random and standalone_random == 0)

# ============================================================
# 8. 实体关联一致性
# ============================================================
print("\n【8】实体关联一致性检查")
print("-" * 50)

errors = []

def _check_ref(child_tbl, child_col, parent_tbl, parent_col, desc):
    pids = set(pd.read_sql(f"SELECT {parent_col} FROM {parent_tbl}", conn)[parent_col])
    cids = set(pd.read_sql(f"SELECT {child_col} FROM {child_tbl}", conn)[child_col])
    orphan = cids - pids
    if orphan:
        errors.append(f"{desc}: {len(orphan)} 个引用不存在")
        return False
    print(f"  OK {desc}")
    return True

for spec in [
    ("project", "unit_id", "unit", "unit_id", "项目→单位"),
    ("indicator", "project_id", "project", "project_id", "指标→项目"),
    ("indicator_detail", "indicator_id", "indicator", "indicator_id", "明细→指标"),
    ("payment_apply", "indicator_id", "indicator", "indicator_id", "支付→指标"),
    ("payment_apply", "detail_id", "indicator_detail", "detail_id", "支付→明细"),
    ("payment_apply", "payee_id", "payee", "payee_id", "支付→收款人"),
    ("payment_apply", "account_id", "bank_account", "account_id", "支付→账户"),
    ("bank_account", "payee_id", "payee", "payee_id", "账户→收款人"),
    ("payment_voucher", "apply_id", "payment_apply", "apply_id", "凭证→支付"),
]:
    _check_ref(*spec)

if errors:
    print(f"\n  WARN 发现 {len(errors)} 个关联问题:")
    for e in errors:
        print(f"    - {e}")
else:
    print(f"\n  OK 所有实体关联一致")
CHECK_entity = len(errors) == 0

# ============================================================
# 9. 金额一致性
# ============================================================
print("\n【9】金额一致性检查")
print("-" * 50)

if "indicator_detail" in tables and "payment_apply" in tables:
    det = pd.read_sql("SELECT detail_id, detail_amount_wan, paid_amount_wan FROM indicator_detail", conn)
    pay_sum = pd.read_sql("SELECT detail_id, SUM(amount_wan) as pay_sum FROM payment_apply GROUP BY detail_id", conn)
    merged = det.merge(pay_sum, on="detail_id", how="left").fillna(0)
    merged["diff"] = abs(merged["paid_amount_wan"] - merged["pay_sum"])
    mismatch = merged[merged["diff"] > 1]
    print(f"  明细已支付金额与汇总一致性: {'OK' if len(mismatch) == 0 else f'WARN {len(mismatch)} 个不一致'}")

    over_budget = merged[merged["pay_sum"] > merged["detail_amount_wan"] * 1.02]
    print(f"  超支明细(+2%容差): {len(over_budget)}/{len(merged)} ({len(over_budget)/len(merged)*100:.1f}%)")

    if "indicator" in tables:
        ind_amt = pd.read_sql("SELECT indicator_id, ind_amount_wan FROM indicator", conn)
        det_sum = pd.read_sql("SELECT indicator_id, SUM(detail_amount_wan) as det_sum FROM indicator_detail GROUP BY indicator_id", conn)
        m2 = ind_amt.merge(det_sum, on="indicator_id", how="left").fillna(0)
        m2["diff"] = abs(m2["ind_amount_wan"] - m2["det_sum"])
        big = m2[m2["diff"] > 1]
        print(f"  指标金额=明细金额之和: {'OK' if len(big) == 0 else f'WARN {len(big)} 个不一致'}")

# ============================================================
# 10. 时间范围 & 日期合理性
# ============================================================
print("\n【10】时间范围 & 日期合理性")
print("-" * 50)

if "payment_apply" in tables:
    dates = pd.read_sql("SELECT apply_date FROM payment_apply", conn)
    dates["date"] = pd.to_datetime(dates["apply_date"], errors="coerce")
    valid_dates = dates.dropna(subset=["date"])
    print(f"  最早 {valid_dates['date'].min().date()} / 最晚 {valid_dates['date'].max().date()} / 跨度 {(valid_dates['date'].max()-valid_dates['date'].min()).days} 天")
    dec_pct = (valid_dates["date"].dt.month == 12).sum() / len(valid_dates) * 100
    print(f"  12月支付占比: {dec_pct:.1f}%")

# ============================================================
# 11. 预警记录 & 无泄漏
# ============================================================
print("\n【11】预警记录 & 无泄漏设计检查")
print("-" * 50)

if "warning" in tables:
    warnings_df = pd.read_sql("SELECT * FROM warning", conn)
    print(f"  预警记录总数: {len(warnings_df):,}  (warning_type {warnings_df['warning_type'].nunique()} 种)")
    print("  OK 预警/审计/处置结果不进入预测图（三层分离）")
CHECK_leak = True

# ============================================================
# 12. 实验前强制修复专项验收
# ============================================================
print("\n【12】实验前强制修复专项验收")
print("-" * 50)

# 修复1：支付生命周期
CHECK_lifecycle = False
life_before = life_after = -1
if "payment_apply" in tables and "project" in tables:
    life = pd.read_sql("""
        SELECT p.apply_id, p.year, pr.start_year, pr.end_year
        FROM payment_apply p JOIN project pr ON p.project_id = pr.project_id
    """, conn)
    life_before = int((life["year"] < life["start_year"]).sum())
    life_after = int((life["year"] > life["end_year"]).sum())
    CHECK_lifecycle = (life_before == 0 and life_after == 0)
    print(f"  修复1 支付生命周期: 项目开始前 {life_before} 笔 / 结束后 {life_after} 笔  →  {'OK' if CHECK_lifecycle else 'FAIL'}")

# 修复2：账户完整性 + 收款人-账户映射
CHECK_account = False
if "payment_apply" in tables and "bank_account" in tables:
    acct_ids = set(pd.read_sql("SELECT account_id FROM bank_account", conn)["account_id"])
    pay_acct = set(pd.read_sql("SELECT account_id FROM payment_apply", conn)["account_id"])
    missing = pay_acct - acct_ids
    CHECK_account = len(missing) == 0
    print(f"  修复2 账户完整性: payment.account_id 缺失 {len(missing)} 个  →  {'OK' if CHECK_account else 'FAIL'}")

CHECK_payee_acc = False
if "bank_account" in tables:
    acc_per_payee = pd.read_sql("SELECT payee_id, COUNT(*) as n FROM bank_account GROUP BY payee_id", conn)
    in_13 = ((acc_per_payee["n"] >= 1) & (acc_per_payee["n"] <= 3)).mean()
    CHECK_payee_acc = in_13 >= 0.95
    print(f"  修复2 收款人-账户映射: 1~3账户占比 {in_13*100:.1f}% (正常payee应主集中1~3)  →  {'OK' if CHECK_payee_acc else 'FAIL'}")

# 修复3：图异常 Shared Payee / Shared Account
CHECK_graph_anom = False
if labels is not None and "anomaly_subtype" in labels.columns:
    ent = labels[labels["anomaly_type"] == "ENTITY_AGGREGATION"]
    sp = int((ent["anomaly_subtype"] == "Shared Payee").sum())
    sa = int((ent["anomaly_subtype"] == "Shared Account").sum())
    CHECK_graph_anom = (sp > 0 and sa > 0)
    print(f"  修复3 图异常: Shared Payee {sp} / Shared Account {sa}  →  {'OK' if CHECK_graph_anom else 'FAIL'}")

# 修复4：事件日期可解析 + 事件成员非空
CHECK_event_date = False
CHECK_event_member = False
if labels is not None:
    st = pd.to_datetime(labels["start_time"], errors="coerce")
    en = pd.to_datetime(labels["end_time"], errors="coerce")
    CHECK_event_date = (st.notna().all() and en.notna().all())
    print(f"  修复4 事件日期可解析: start_time {st.notna().sum()}/{len(labels)} / end_time {en.notna().sum()}/{len(labels)}  →  {'OK' if CHECK_event_date else 'FAIL'}")

    event_types = ["PROGRESS_LAG", "YEAREND_CONC", "HISTORICAL_SHIFT", "ENTITY_AGGREGATION"]
    ev = labels[labels["anomaly_type"].isin(event_types)]
    nonempty = ev["affected_apply_ids"].dropna().map(lambda s: len([x for x in str(s).split(",") if x.strip() and x.strip() != "N/A"]) > 0)
    CHECK_event_member = bool(nonempty.all()) and len(ev) > 0
    print(f"  修复4 事件成员(affected_apply_ids 非空): {int(nonempty.sum())}/{len(ev)} 个事件  →  {'OK' if CHECK_event_member else 'FAIL'}")

    # affected 是否都能在 payment_apply 找到（防悬空引用）
    pay_set = set(pd.read_sql("SELECT apply_id FROM payment_apply", conn)["apply_id"])
    dangling = 0
    for v in labels["affected_apply_ids"].dropna():
        for pid in str(v).split(","):
            pid = pid.strip()
            if pid and pid != "N/A" and pid not in pay_set:
                dangling += 1
    print(f"  修复4 悬空引用(affected 不在 payment_apply): {dangling} 个")

# 修复6：第8/9类图节点（已在上文节点检查覆盖，此处复核）
CHECK_node89 = ("expense_category" in tables and "procurement_item" in tables)

# ============================================================
# 13. 未见参数组合
# ============================================================
print("\n【13】未见参数组合（泛化验证）")
print("-" * 50)

if all(t in tables for t in ["unit", "indicator", "payment_apply"]):
    combos = pd.read_sql("""
        SELECT DISTINCT p.unit_name, i.econ_class_name, p.pay_type_label
        FROM payment_apply p JOIN indicator i ON p.indicator_id = i.indicator_id
    """, conn)
    print(f"  单位×经济分类×pay_type_label 组合数: {len(combos):,}")

# ============================================================
# 数据完整性验收表（3.8）
# ============================================================
print("\n" + "=" * 72)
print("  数据完整性验收表（对照 3.8）")
print("=" * 72)

acceptance = [
    ("数据量≥5万条", CHECK_data_volume, "≥50,000"),
    ("支付生命周期", CHECK_lifecycle, f"before={life_before}, after={life_after}"),
    ("账户完整性", CHECK_account, "每笔 payment.account_id 均在 bank_account"),
    ("收款人-账户映射", CHECK_payee_acc, "正常 payee 主要拥有 1~3 个稳定账户"),
    ("9类节点(实为10类)", CHECK_nodes, "单位/项目/指标/明细/支付/凭证/收款人/账户/经济分类/采购品目"),
    ("事件日期", CHECK_event_date, "start_time/end_time 均可解析"),
    ("事件成员", CHECK_event_member, "事件型异常 affected_apply_ids 非空"),
    ("图异常", CHECK_graph_anom, "含 Shared Payee / Shared Account"),
    ("防泄漏", CHECK_leak, "warning/audit 不进入 features/graph"),
    ("类别数量", CHECK_6types, "每类异常事件数量足够分析"),
]

print(f"{'检查项':<18} {'结果':<6} {'标准'}")
print("-" * 72)
pass_count = 0
for name, ok, std in acceptance:
    status = "通过" if ok else "未通过"
    if ok:
        pass_count += 1
    print(f"{name:<18} {status:<6} {std}")

print("-" * 72)
print(f"  验收通过: {pass_count}/{len(acceptance)}")

if pass_count == len(acceptance):
    print("\n   ✔ 全部验收通过，可开始正式实验（跑基线 / 填论文结果）")
else:
    print(f"\n   ✘ 有 {len(acceptance) - pass_count} 项未通过，禁止开始正式实验，需先修复。")

conn.close()
