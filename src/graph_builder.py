"""
MS-TRP Step 4 —— 构建 9 类预算执行异构图
========================================
9 类节点（指导书 4.4.3）：
  Unit(预算单位) / Project(预算项目) / Indicator(预算指标) / PaymentApplication(支付申请)
  Voucher(支付凭证) / Payee(收款人) / Account(银行账户) / ExpenseCategory(支出分类)
  ProcurementItem(采购品目)

关系（指导书表 10 + 直接业务键）：
  Unit→Project→Indicator→PaymentApplication→Voucher→{Payee, Account}
  PaymentApplication→{ExpenseCategory, ProcurementItem}
  Payee→Account（共享账户归属，用于 Shared Account 检测）

构图只用评分时点之前的业务事实；warning/correction/refund 等审计日志不进入图。
同时计算"关系结构风险"（图分支自身的证据，非时序）：
  · 账户一致性：支付 payee 与账户归属 payee 不一致 → Shared Account 信号；
  · 实体级频次突增：收款人/账户在短窗口内的支付密度显著高于其历史速率 → Shared Payee 信号。
"""
import numpy as np
import pandas as pd
import sqlite3

from src import config

NODE_TYPES = ["Unit", "Project", "Indicator", "PaymentApplication",
              "Voucher", "Payee", "Account", "ExpenseCategory", "ProcurementItem"]

# 关系类型 → 强度 a_r（默认，验证集可调）
REL_STRENGTH = {
    "owns": 1.0,
    "allocates": 1.0,
    "used_by": 1.0,
    "executed_as": 0.8,
    "paid_to": 0.9,
    "through": 0.9,
    "owns_account": 0.7,
    "classified_as": 0.5,
    "belongs_to": 0.5,
}

# 实体级频次突增窗口（天）：跨项目/跨收款人短时聚集的检测尺度
ENTITY_WINDOW_DAYS = 7.0
ENTITY_ZOOM = 7.0           # 期望频次 = 历史速率 × 该天数
Z_SAT = 3.0
EPS = 1e-6


def _nid(t, i):
    return f"{t}::{i}"


def load_graph_tables():
    """读取构图所需的全部业务表（不含 warning/correction/refund 等审计表）。"""
    conn = sqlite3.connect(config.DB_PATH)
    tables = {t: pd.read_sql(f"SELECT * FROM {t}", conn) for t in [
        "unit", "project", "indicator", "payment_apply",
        "payment_voucher", "payee", "bank_account",
        "expense_category", "procurement_item"]}
    conn.close()
    tables["payment_apply"]["apply_date"] = pd.to_datetime(tables["payment_apply"]["apply_date"])
    tables["payment_voucher"]["gen_date"] = pd.to_datetime(
        tables["payment_voucher"]["gen_date"], errors="coerce")
    return tables


def build_nodes(tables):
    """
    收集 9 类节点，返回 (node_ids, node_type, id2idx)。
    node_ids: list[str]，node_type: dict[node_id -> type]。
    """
    node_ids, node_type = [], {}
    sets = [
        ("Unit", tables["unit"]["unit_id"]),
        ("Project", tables["project"]["project_id"]),
        ("Indicator", tables["indicator"]["indicator_id"]),
        ("PaymentApplication", tables["payment_apply"]["apply_id"]),
        ("Voucher", tables["payment_voucher"]["voucher_id"]),
        ("Payee", tables["payee"]["payee_id"]),
        ("Account", tables["bank_account"]["account_id"]),
        ("ExpenseCategory", tables["expense_category"]["expense_cat_code"]),
        ("ProcurementItem", tables["procurement_item"]["procurement_cat"]),
    ]
    for typ, ids in sets:
        for i in ids.dropna().astype(str).unique():
            nid = _nid(typ, i)
            node_ids.append(nid)
            node_type[nid] = typ
    id2idx = {nid: k for k, nid in enumerate(node_ids)}
    return node_ids, node_type, id2idx


def build_edges(tables, id2idx):
    """
    构建边列表：list[dict(src, dst, rel, date)]。
    date 为 pd.Timestamp（时变边 / 带 valid_from 的项目结构边）或 None（无生命周期的静态关系，如 Payee→Account）。
    无向传播：每条关系只登记一次，传播时视为对称。
    """
    project = tables["project"]
    indicator = tables["indicator"]
    pay = tables["payment_apply"]
    voucher = tables["payment_voucher"]
    account = tables["bank_account"]

    edges = []

    def add(src_type, src_id, dst_type, dst_id, rel, date=None):
        sid = id2idx.get(_nid(src_type, src_id))
        did = id2idx.get(_nid(dst_type, dst_id))
        if sid is not None and did is not None:
            edges.append({"src": sid, "dst": did, "rel": rel, "date": date})

    # 项目生命周期 → 结构边 valid_from（修复：防止未来项目骨架前向泄漏）。
    # project.start_year 决定 Unit→Project→Indicator 结构何时起可见；已关闭项目的历史结构
    # 保留在图中（不构成未来信息泄漏，end_year 不用于剔除）。
    proj_start = {str(pid): sy for pid, sy in zip(project["project_id"], project["start_year"])}

    def _start_date(year):
        try:
            return pd.Timestamp(year=int(year), month=1, day=1)
        except (TypeError, ValueError):
            return None

    # 静态链：Unit→Project→Indicator（valid_from = project.start_year-01-01）
    for _, row in project.iterrows():
        add("Unit", row["unit_id"], "Project", row["project_id"], "owns",
            _start_date(row["start_year"]))
    for _, row in indicator.iterrows():
        add("Project", row["project_id"], "Indicator", row["indicator_id"], "allocates",
            _start_date(proj_start.get(str(row["project_id"]))))

    # Payee→Account 归属（Shared Account 检测的对照关系）
    for _, row in account.iterrows():
        add("Payee", row["payee_id"], "Account", row["account_id"], "owns_account")

    # 支付相关边（时间随 apply_date 传递）
    for _, row in pay.iterrows():
        d = row["apply_date"]
        aid = row["apply_id"]
        add("Indicator", row["indicator_id"], "PaymentApplication", aid, "used_by", d)
        if row["payee_id"] is not None and str(row["payee_id"]) != "nan":
            add("PaymentApplication", aid, "Payee", row["payee_id"], "paid_to", d)
        if row["account_id"] is not None and str(row["account_id"]) != "nan":
            add("PaymentApplication", aid, "Account", row["account_id"], "through", d)
        if row["econ_class_code"] is not None:
            add("PaymentApplication", aid, "ExpenseCategory", row["econ_class_code"], "classified_as", d)
        if row["procurement_cat"] is not None and str(row["procurement_cat"]) != "nan":
            add("PaymentApplication", aid, "ProcurementItem", row["procurement_cat"], "belongs_to", d)

    # Voucher 边
    for _, row in voucher.iterrows():
        d = row["gen_date"] if pd.notna(row["gen_date"]) else None
        add("PaymentApplication", row["apply_id"], "Voucher", row["voucher_id"], "executed_as", d)

    # Indicator → 支出分类/采购品目（属性边，valid_from 随项目 start_year）
    for _, row in indicator.iterrows():
        sd = _start_date(proj_start.get(str(row["project_id"])))
        if row["econ_class_code"] is not None:
            add("Indicator", row["indicator_id"], "ExpenseCategory", row["econ_class_code"], "classified_as", sd)
        if row["procurement_cat"] is not None and str(row["procurement_cat"]) != "nan":
            add("Indicator", row["indicator_id"], "ProcurementItem", row["procurement_cat"], "belongs_to", sd)

    return edges


def split_edges_by_time(edges):
    """把 build_edges 输出拆成 (static_edges, timed_edges)，timed 按 date 升序。
    静态边：date 为 None/NaT 的无生命周期持久关系（如 Payee→Account）；
    时变边：date 为支付 apply_date / 凭证 gen_date / 项目 start_year 的边，随评分时点逐步可见。
    （项目结构边 valid_from=start_year 进入时变路径，但 beta=0 不随时间衰减。）"""
    static, timed = [], []
    for e in edges:
        d = e.get("date")
        if d is None or pd.isna(d):
            static.append(e)
        else:
            timed.append(e)
    timed.sort(key=lambda e: pd.Timestamp(e["date"]))
    return static, timed


def make_asof_snapshot(edges, cutoff):
    """评分时点 cutoff 的可见边集合（P0-3 快照语义）。
    静态关系恒可见；时变边仅当其业务日期 <= cutoff 时可见（日终批量，含当日）。
    对预构建的边列表做单次 O(|edges|) 过滤，供测试/审计复用；
    增量 as-of 扫描在 src/asof.py 中用「排序 + 前缀」实现，避免反复全量过滤。"""
    cutoff = pd.Timestamp(cutoff)
    out = []
    for e in edges:
        d = e.get("date")
        if d is None or pd.isna(d):
            out.append(e)
        elif pd.Timestamp(d) <= cutoff:
            out.append(e)
    return out


def compute_structural_risk(tables, payments_df):
    """
    图分支的结构关系风险（点级，与 payments_df 等长、同 index）：
      · account_mismatch：支付 payee 与账户归属 payee 不一致 → [0,1]（Shared Account 信号）；
      · entity_freq_z：收款人/账户级短窗口频次突增的稳健 z 分数（Shared Payee 信号）；
    返回 DataFrame，列：s_account_mismatch / s_entity_freq。
    """
    pay = tables["payment_apply"]
    account = tables["bank_account"]
    owner = dict(zip(account["account_id"].astype(str), account["payee_id"].astype(str)))

    out = pd.DataFrame(index=payments_df.index)
    pay = pay.set_index("apply_id").reindex(payments_df["apply_id"].astype(str))
    payee_id = pay["payee_id"].astype(str)
    account_id = pay["account_id"].astype(str)
    owner_payee = account_id.map(lambda a: owner.get(a, None))
    owner_payee = owner_payee.fillna(payee_id)   # 账户缺失归属时视为一致
    out["s_account_mismatch"] = (payee_id != owner_payee).astype(float).values

    # 实体级频次突增：对 payee 与 account 分别计算前向 7 日笔数 vs 历史速率
    df = payments_df[["apply_id", "payee_id", "account_id", "apply_date"]].copy()
    df["apply_date"] = pd.to_datetime(df["apply_date"])
    payee_freq = _entity_freq_burst(df, "payee_id")
    account_freq = _entity_freq_burst(df, "account_id")
    out["s_entity_freq"] = np.maximum(
        np.nan_to_num(payee_freq.values, nan=0.0),
        np.nan_to_num(account_freq.values, nan=0.0),
    )
    return out


def _entity_freq_burst(df, key):
    """按实体 key 计算短窗口频次突增风险（tanh 有界，严格前向）。"""
    res = pd.Series(0.0, index=df.index)
    for _, grp in df.groupby(key, dropna=False):
        if grp[key].isna().all():
            continue
        grp = grp.sort_values("apply_date")
        ts = grp.set_index("apply_date")["apply_id"]
        w7 = ts.rolling(window=pd.Timedelta(days=ENTITY_WINDOW_DAYS), closed="left").count().values
        d = grp["apply_date"]
        span = (d - d.iloc[0]).dt.days
        prior_cnt = np.arange(len(grp))
        rate = prior_cnt / np.maximum(span, 1.0)
        exp = rate * ENTITY_ZOOM
        z = (w7 - exp) / (np.sqrt(np.maximum(exp, 0.1)) + 0.5)
        res.loc[grp.index] = np.tanh(np.maximum(z, 0.0) / Z_SAT)
    return res
