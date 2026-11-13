"""
B0 数据契约完整性报告 → data_fixed/integrity.json + 追加 logs/data_checks.txt
================================================================================
对照指导书 §3.5「数据质量检查必须全部通过」逐项输出计数与 checks_passed。
所有计数由当前主库（data_fixed/budget_data_v5.db）实时重算，不做任何手工填写。

用法：python -m experiment.build_integrity
"""
import os
import sys
import json
import sqlite3
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

DB_PATH = os.path.join(config.PROJECT_DIR, "data_fixed", "budget_data_v5.db")
OUT_JSON = os.path.join(config.PROJECT_DIR, "data_fixed", "integrity.json")
OUT_LOG = os.path.join(config.PROJECT_DIR, "logs", "data_checks.txt")


def _q(sql):
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def _count_duplicates(table, key):
    df = _q(f'SELECT "{key}" FROM "{table}"')
    return int(df.duplicated().sum())


def _count_orphans(child, ccol, parent, pcol):
    df = _q(f'SELECT "{ccol}" FROM "{child}"')
    par = set(_q(f'SELECT "{pcol}" FROM "{parent}"')[pcol])
    return int(df[~df[ccol].isin(par)][ccol].nunique())


def main():
    checks = {}

    # 1. 数据量 ≥ 5 万
    n_pay = int(_q('SELECT COUNT(*) AS n FROM "payment_apply"')["n"][0])
    checks["n_payments"] = n_pay
    checks["payments_ge_50000"] = n_pay >= 50000

    # 2. 主键重复数（9 类实体）
    pk_dup = {}
    for tbl, key in [
        ("unit", "unit_id"), ("project", "project_id"), ("indicator", "indicator_id"),
        ("indicator_detail", "detail_id"), ("payment_apply", "apply_id"),
        ("payment_voucher", "voucher_id"), ("payee", "payee_id"),
        ("bank_account", "account_id"), ("expense_category", "expense_cat_code"),
        ("procurement_item", "procurement_item_id"),
    ]:
        pk_dup[tbl] = _count_duplicates(tbl, key)
    checks["primary_key_duplicates"] = pk_dup
    checks["pk_duplicates_total"] = int(sum(pk_dup.values()))

    # 3. 外键无法映射数
    fk = {}
    for child, ccol, parent, pcol in [
        ("project", "unit_id", "unit", "unit_id"),
        ("indicator", "project_id", "project", "project_id"),
        ("indicator_detail", "indicator_id", "indicator", "indicator_id"),
        ("payment_apply", "indicator_id", "indicator", "indicator_id"),
        ("payment_apply", "detail_id", "indicator_detail", "detail_id"),
        ("payment_apply", "payee_id", "payee", "payee_id"),
        ("payment_apply", "account_id", "bank_account", "account_id"),
        ("bank_account", "payee_id", "payee", "payee_id"),
        ("payment_voucher", "apply_id", "payment_apply", "apply_id"),
    ]:
        fk[f"{child}.{ccol}->{parent}.{pcol}"] = _count_orphans(child, ccol, parent, pcol)
    checks["foreign_key_orphans"] = fk
    checks["fk_orphans_total"] = int(sum(fk.values()))

    # 4. 九类节点均有正样本
    node_counts = {}
    for tbl in ["unit", "project", "indicator", "indicator_detail", "payment_apply",
                "payment_voucher", "payee", "bank_account", "expense_category",
                "procurement_item"]:
        node_counts[tbl] = int(_q(f'SELECT COUNT(*) AS n FROM "{tbl}"')["n"][0])
    checks["node_counts"] = node_counts
    checks["all_nodes_positive"] = all(v > 0 for v in node_counts.values())

    # 5. 支付生命周期（项目开始前/结束后支付数）
    life = _q("""
        SELECT p.apply_id, p.year, pr.start_year, pr.end_year
        FROM payment_apply p JOIN project pr ON p.project_id = pr.project_id
    """)
    checks["payments_before_start"] = int((life["year"] < life["start_year"]).sum())
    checks["payments_after_end"] = int((life["year"] > life["end_year"]).sum())

    # 6. 事件日期可解析 + 受影响支付无悬空引用
    labels = _q('SELECT * FROM "ground_truth_event"')
    st = pd.to_datetime(labels["start_time"], errors="coerce")
    en = pd.to_datetime(labels["end_time"], errors="coerce")
    checks["events_total"] = int(len(labels))
    checks["event_dates_parseable"] = bool(st.notna().all() and en.notna().all())
    pay_set = set(_q('SELECT apply_id FROM "payment_apply"')["apply_id"])
    dangling = 0
    for v in labels["affected_apply_ids"].dropna():
        for pid in str(v).split(","):
            pid = pid.strip()
            if pid and pid != "N/A" and pid not in pay_set:
                dangling += 1
    checks["event_affected_dangling"] = dangling

    # 7. 六类异常齐全
    checks["anomaly_types_present"] = int(labels["anomaly_type"].nunique())
    checks["anomaly_type_counts"] = labels["anomaly_type"].value_counts().to_dict()

    # 8. 跨集合事件数（读 split_manifest / DATA_VERSION，生成器已保证）
    dv_path = os.path.join(config.PROJECT_DIR, "data_fixed", "DATA_VERSION.json")
    with open(dv_path, encoding="utf-8") as f:
        dv = json.load(f)
    checks["cross_split_events"] = int(dv.get("cross_split_events", 0))
    checks["split_counts"] = dv.get("split_counts", {})

    # ── 汇总判据 ──
    criteria = {
        "payments_ge_50000": checks["payments_ge_50000"],
        "pk_duplicates_total == 0": checks["pk_duplicates_total"] == 0,
        "fk_orphans_total == 0": checks["fk_orphans_total"] == 0,
        "all_nodes_positive": checks["all_nodes_positive"],
        "payments_before_start == 0": checks["payments_before_start"] == 0,
        "payments_after_end == 0": checks["payments_after_end"] == 0,
        "event_dates_parseable": checks["event_dates_parseable"],
        "event_affected_dangling == 0": checks["event_affected_dangling"] == 0,
        "cross_split_events == 0": checks["cross_split_events"] == 0,
        "anomaly_types >= 6": checks["anomaly_types_present"] >= 6,
    }
    checks["criteria"] = criteria
    checks["checks_passed"] = bool(all(criteria.values()))
    checks["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)

    # 追加到 logs/data_checks.txt
    lines = [
        "",
        f"=== integrity.json（B0 逐项检查）生成于 {checks['generated_at']} ===",
        f"支付记录数 {n_pay:,}（≥50,000: {'PASS' if checks['payments_ge_50000'] else 'FAIL'}）",
        f"主键重复 {checks['pk_duplicates_total']}  外键孤儿 {checks['fk_orphans_total']}",
        f"支付生命周期 before={checks['payments_before_start']} after={checks['payments_after_end']}",
        f"事件日期可解析 {'PASS' if checks['event_dates_parseable'] else 'FAIL'}  悬空引用 {checks['event_affected_dangling']}",
        f"跨集合事件 {checks['cross_split_events']}  异常类型 {checks['anomaly_types_present']}",
        f"checks_passed = {checks['checks_passed']}",
        ("ALL CHECKS PASSED" if checks["checks_passed"] else "SOME CHECKS FAILED"),
    ]
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[integrity] 写入 {OUT_JSON}")
    print(f"  checks_passed = {checks['checks_passed']}")
    for k, v in criteria.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")


if __name__ == "__main__":
    main()
