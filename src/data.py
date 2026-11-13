"""
MS-TRP 检测框架 —— 数据加载与标签构建
========================================
从 budget_data_v5.db 读取支付申请与真实标签，构建：
  1) 支付特征帧(带解析后的日期/年/月字段)
  2) 点级标签  apply_id -> 是否异常(可多类型)
  3) 事件级标签 每行一条事件 -> 受影响支付集合(主单号+linked)
"""
import pandas as pd
import sqlite3

from src import config


def load_payments() -> pd.DataFrame:
    """读取 payment_apply，解析日期，按时间排序，派生 year/month/is_december 字段。"""
    conn = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql("SELECT * FROM payment_apply", conn)
    conn.close()

    df["apply_date"] = pd.to_datetime(df["apply_date"], errors="coerce")
    df = df.dropna(subset=["apply_date"]).sort_values("apply_date").reset_index(drop=True)
    df["year"] = df["apply_date"].dt.year.astype(int)
    df["month"] = df["apply_date"].dt.month.astype(int)
    df["is_december"] = (df["month"] == 12).astype(int)
    return df


def load_labels() -> pd.DataFrame:
    """读取 ground_truth_event 原始事件标签表。"""
    conn = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql("SELECT * FROM ground_truth_event", conn)
    conn.close()
    return df


def _parse_ids(value) -> list:
    """把 affected_apply_ids 这类逗号分隔字符串解析成列表，空/None 返回 []。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


def build_events(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建事件级标签：一行 = 一条事件。
    每事件 = {anchor_apply_id} ∪ affected_apply_ids（受影响的支付集合）。
    返回列：event_id, anomaly_type, target_entity_type, target_entity_id,
            start_time, end_time, apply_ids(set)。
    target_entity_id/start_time/end_time 直接来自 ground_truth_event，供事件级
    严格一对一匹配使用（P0-4）。
    """
    events = []
    for i, row in labels_df.iterrows():
        ids = set()
        anchor = row.get("anchor_apply_id")
        if anchor is not None and not (isinstance(anchor, float) and pd.isna(anchor)):
            ids.add(str(anchor))
        for x in _parse_ids(row.get("affected_apply_ids")):
            ids.add(x)
        events.append({
            "event_id": row.get("event_id", i),
            "anomaly_type": row["anomaly_type"],
            "target_entity_type": row.get("target_entity_type"),
            "target_entity_id": row.get("target_entity_id"),
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
            "apply_ids": ids,
        })
    return pd.DataFrame(events)


def build_point_labels(payments_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    构建点级标签：每个支付一条，标记是否异常 + 命中的异常类型(可多个)。
    """
    valid_ids = set(payments_df["apply_id"].astype(str))
    type_of = {}   # apply_id -> set(anomaly_type)
    for _, ev in events.iterrows():
        at = ev["anomaly_type"]
        for aid in ev["apply_ids"]:
            if aid in valid_ids:
                type_of.setdefault(aid, set()).add(at)

    out = payments_df[["apply_id"]].copy()
    out["is_anomaly"] = out["apply_id"].astype(str).map(lambda x: int(x in type_of))
    out["anomaly_types"] = out["apply_id"].astype(str).map(
        lambda x: sorted(type_of.get(x, [])))
    return out


def load_entity_tables() -> dict:
    """读取 indicator / indicator_detail / project / unit 四张实体表，供特征工程使用。"""
    conn = sqlite3.connect(config.DB_PATH)
    tables = {}
    for name in ["indicator", "indicator_detail", "project", "unit"]:
        tables[name] = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return tables
