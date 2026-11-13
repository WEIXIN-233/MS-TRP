"""
B3/B6 —— 种子落点诊断（seed_diagnostics.csv）
=============================================
对每个 ENTITY_AGGREGATION 真事件离线计算 PPR 重启向量（时序种子 s_ts）的落点诊断，
输出指导书 §6.5 强制字段。仅用于测试后离线分层，绝不回流到 PPR 评分或阈值。

冻结规则（§6.5 / B3-B6）：
  · event_source_id = 注入时选定的共享 Payee/Account 聚集核心（target_entity_id）。
  · affected_apply_ids = 注入实际修改了金额/频次/时间的支付（不含未改的合法关联支付）。
  · source_bridge_apply_ids = 其中在评分时点合法传播图上直接邻接核心的支付
    （Shared Payee：payee_id==core；Shared Account：account_id==core）。
  · 对桥接与非桥接两组都非空、且总质量 M=Σs_ts>ε 的事件：
        r_bridge=Σ桥接 s_ts/M，>0.5 source_dominant、<0.5 victim_dominant、=0.5 diffuse。
    M≤ε 记 zero；映射缺失 / 非 EA / 两集合无法区分 / 时间权限不成立 → unknown。
  · ε=1e-12、0.5 容差、pattern_rule_version 冻结；不得测试后改阈值。

关键事实（生成器口径）：仿真V3 的 ENTITY_AGGREGATION 只修改金额/频次/时间并让受影响的
支付全部直接邻接核心（Shared Payee 全部 payee_id==core；Shared Account 全部 account_id==core），
没有独立的"非桥接受害者"。因此按冻结规则"两集合都非空"不成立 → seed_dominance_pattern 如实记 unknown；
本表仍如实报告 mass / concentration / localization_proxy / 时序与图分支检出，不预设"越集中增益越大"。

用法：python -m experiment.run_seed_diagnostics [--data-seed 10] [--out result_raw/seed_diagnostics.csv]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, data

EPS_MASS = 1e-12
R_BRIDGE_TOL = 1e-9
PATTERN_RULE_VERSION = "v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR, "seed_diagnostics.csv"))
    args = ap.parse_args()

    pred = ec.load_predictions(args.data_seed)
    thr = ec.load_thresholds(args.data_seed)["thr_global"]
    pred["apply_id"] = pred["apply_id"].astype(str)
    s_ts = dict(zip(pred["apply_id"], pred["s_ts"]))
    s_g = dict(zip(pred["apply_id"], pred["s_g"]))

    payments, labels, events, point_labels = ec.load_standard()
    pay_id = payments["apply_id"].astype(str).tolist()
    payee_of = dict(zip(pay_id, payments["payee_id"].astype(str)))
    account_of = dict(zip(pay_id, payments["account_id"].astype(str)))

    # 候选高风险节点集合边界：验证集 s_ts 的 99 分位（冻结，仅作离线落点比例 H）
    val = pred[pred["split"] == "val"]
    q99 = float(val["s_ts"].quantile(0.99)) if len(val) else np.nan

    ea = events[events["anomaly_type"] == "ENTITY_AGGREGATION"]
    rows = []
    for _, ev in ea.iterrows():
        eid = str(ev.get("event_id"))
        stype = str(ev.get("target_entity_type") or "")
        sid = str(ev.get("target_entity_id") or "")
        affected = sorted(ev.get("apply_ids") or set())
        affected = [a for a in affected if a in s_ts]

        # 桥接 = 直接邻接核心（paid_to / through 边）
        if stype == "PAYEE":
            bridge = [a for a in affected if payee_of.get(a, "") == sid]
        elif stype == "ACCOUNT":
            bridge = [a for a in affected if account_of.get(a, "") == sid]
        else:
            bridge = []
        non_bridge = [a for a in affected if a not in set(bridge)]

        m = float(sum(s_ts[a] for a in affected))
        vals = np.asarray([s_ts[a] for a in affected], dtype=float)
        conc = float(vals.max() / vals.mean()) if (len(vals) and vals.mean() > EPS_MASS) else np.nan
        # 落点比例 H = 事件种子质量落在候选高风险集合内的比例
        if m > EPS_MASS:
            loc = float(sum(s_ts[a] for a in affected if s_ts[a] > q99) / m)
            r_bridge = float(sum(s_ts[a] for a in bridge) / m)
        else:
            loc = 0.0
            r_bridge = np.nan

        # 冻结 pattern 规则
        if m <= EPS_MASS:
            pattern = "zero"
        elif stype not in ("PAYEE", "ACCOUNT") or not bridge or not non_bridge:
            pattern = "unknown"
        elif abs(r_bridge - 0.5) <= R_BRIDGE_TOL:
            pattern = "diffuse"
        elif r_bridge > 0.5:
            pattern = "source_dominant"
        else:
            pattern = "victim_dominant"

        temporal_detected = int(any(s_ts[a] > thr for a in affected))
        graph_detected = int(any(s_g[a] > thr for a in affected))

        rows.append({
            "data_seed": args.data_seed if args.data_seed is not None else "main",
            "event_id": eid,
            "event_source_id": sid,
            "event_source_type": stype,
            "affected_apply_ids": ",".join(affected),
            "source_bridge_apply_ids": ",".join(bridge),
            "n_members": len(affected),
            "mass": round(m, 8),
            "concentration": round(conc, 6) if np.isfinite(conc) else np.nan,
            "localization_proxy": round(loc, 6),
            "source_bridge_seed_share": round(r_bridge, 6) if np.isfinite(r_bridge) else np.nan,
            "seed_dominance_pattern": pattern,
            "source_definition": str(ev.get("inject_parameters") or ""),
            "pattern_rule_version": PATTERN_RULE_VERSION,
            "temporal_detected": temporal_detected,
            "graph_detected": graph_detected,
            "delta_recall": graph_detected - temporal_detected,
            "group_id": str(ev.get("anomaly_subtype") or stype),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[B6] seed_diagnostics.csv：{len(df)} 个 ENTITY_AGGREGATION 事件")
    print("[B6] seed_dominance_pattern 分布：", df["seed_dominance_pattern"].value_counts().to_dict())
    g = df.groupby(["group_id", "seed_dominance_pattern"]).agg(
        n=("event_id", "size"),
        temporal_recall=("temporal_detected", "mean"),
        graph_recall=("graph_detected", "mean"),
    ).reset_index()
    g["temporal_recall"] = g["temporal_recall"].round(4)
    g["graph_recall"] = g["graph_recall"].round(4)
    print(g.to_string(index=False))
    print(f"[B6] 注意：生成器未制造独立非桥接受害者，故 pattern 依冻结规则记 unknown（见文件头注释）。")


if __name__ == "__main__":
    main()
