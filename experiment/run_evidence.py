"""
B4 —— 四类证据 JSONL（evidence.jsonl）
========================================
对每条 HighRisk（s_final>θ_high）与 Warning（θ_low<s_final≤θ_high）报警，输出最低 JSON
（指导书 §7.3）：feature / history / rule / relation 四类证据 + replay 回放版本 + q2_optional。

口径：
  · θ_high = thr_global（验证集冻结主阈值）；θ_low = tau_fpr（更低一档 Warning 阈值），
    二者均来自 metrics.json，保证 θ_low ≤ θ_high。
  · 每条证据的 evidence_id 可回查原始业务字段；window 只取评分时点之前的可见信息。
  · 未命中规则时 rule.result 写 "not_triggered"，绝不伪造理由。
  · q2_optional 两字段在 B 阶段均为 null。

用法：python -m experiment.run_evidence [--data-seed 10] [--out result_raw/evidence.jsonl]
"""
import os
import sys
import json
import hashlib
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, propagation, reliability_gate, calibration, graph_builder, split, temporal_baseline as tb

PARAM_VERSION = ("v1"
                 f"-rho{propagation.RHO_DEFAULT}"
                 f"-tau{reliability_gate.TAU_DEFAULT}"
                 f"-gamma{calibration.GAMMA_DEFAULT}"
                 f"-q{calibration.ALPHA_HIGH}"
                 f"-afpr{calibration.ALPHA_FPR_DEFAULT}")


def _neighbor_map(edges, node_ids):
    """node_id -> list[(neighbor_node_id, rel, base_weight)]（业务边一次登记，方向无关）。"""
    m = {}
    for e in edges:
        s = node_ids[e["src"]]
        d = node_ids[e["dst"]]
        rel = e["rel"]
        w = graph_builder.REL_STRENGTH.get(rel, 1.0)
        m.setdefault(s, []).append((d, rel, w))
        m.setdefault(d, []).append((s, rel, w))
    return m


def _top_feature(r):
    """从时序分量里取贡献最大的特征（feature 证据块）。"""
    cands = [
        ("amount", float(r.get("s_amt", 0.0) or 0.0), 0.5, "7d"),
        ("freq", float(r.get("s_freq", 0.0) or 0.0), 0.5, "7d"),
        ("trend", float(r.get("s_trend", 0.0) or 0.0), 0.5, "30d"),
        ("rhythm", float(r.get("s_rhythm_ch", 0.0) or 0.0), 0.8, "rhythm/seasonal"),
    ]
    return max(cands, key=lambda c: c[1])


def _peer_count_lookup(payments, enriched, feat):
    """返回 apply_id -> peer 组样本数（与 _peer_lookup 逐级查询一致，反映实际用到的 peer_n）。"""
    train_idx, _, _ = split.time_forward_split(payments)
    peer_stats = tb.compute_peer_stats(enriched.loc[train_idx])
    keys = {
        "L0": list(zip(feat["unit_type"].astype(str), feat["project_category"].astype(str),
                       feat["econ_class_code"].astype(str), feat["stage"].astype(str))),
        "L1": list(zip(feat["unit_type"].astype(str), feat["project_category"].astype(str))),
        "L2": feat["project_category"].astype(str).tolist(),
    }
    levels = ["L0", "L1", "L2"]
    out = {}
    n = len(feat)
    for i in range(n):
        cnt = int(peer_stats["L3"]["count"])
        for lv in levels:
            cnt_map = peer_stats[lv]["count"].to_dict()
            g = keys[lv][i]
            if g in cnt_map and cnt_map.get(g, 0) >= tb.PEER_MIN_SAMPLES:
                cnt = int(cnt_map[g])
                break
        out[str(enriched["apply_id"].iloc[i])] = cnt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR, "evidence.jsonl"))
    args = ap.parse_args()

    pred = ec.load_predictions(args.data_seed)
    thr = ec.load_thresholds(args.data_seed)
    theta_high = float(thr["thr_global"])
    theta_low = float(thr["tau_fpr"])
    if theta_low > theta_high:
        theta_low, theta_high = theta_high, theta_low

    # 特征长表（lambda_self / peer_level / hist 字段来自 feat，与 pred 按 apply_id 对齐）
    payments, enriched, feat = ec.load_feat(args.data_seed)
    feat_map = feat.copy()
    feat_map["apply_id"] = enriched["apply_id"].astype(str)
    aid2feat = feat_map.set_index("apply_id").to_dict("index")
    peer_n_map = _peer_count_lookup(payments, enriched, feat)

    # 图（relation 证据的 1 跳邻居与边权）
    tables, node_ids, node_type, id2idx, edges = ec.load_graph(args.data_seed)
    nb = _neighbor_map(edges, node_ids)
    graph_hash = "g-" + hashlib.sha256(
        f"{len(node_ids)}:{len(edges)}".encode("utf-8")).hexdigest()[:12]
    threshold_version = "thr-" + hashlib.sha256(
        f"{theta_low:.12f}:{theta_high:.12f}".encode("utf-8")).hexdigest()[:12]

    pred["apply_date"] = pd.to_datetime(pred["apply_date"])
    flagged = pred[pred["s_final"] > theta_low]

    n_high = n_warn = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _, r in flagged.iterrows():
            aid = str(r["apply_id"])
            score = float(r["s_final"])
            alert = "HighRisk" if score > theta_high else "Warning"
            if alert == "HighRisk":
                n_high += 1
            else:
                n_warn += 1

            ft = aid2feat.get(aid, {})
            s_time = r["apply_date"]
            name, val, base, win = _top_feature(r)

            # relation：1 跳邻居（PaymentApplication 节点）
            node = f"PaymentApplication::{aid}"
            nbrs = nb.get(node, [])
            # 只保留前 6 条，按边权降序，避免证据体过大
            nbrs_sorted = sorted(nbrs, key=lambda x: -x[2])[:6]

            obj = {
                "sample_id": aid,
                "evidence_id": f"ev-{aid}",
                "entity_id": f"Indicator::{ft.get('indicator_id', '') or 'NA'}",
                "window_start": str((s_time - pd.Timedelta(days=30)).date()),
                "window_end": str(s_time.date()),
                "score_time": str(s_time.date()),
                "score": round(score, 8),
                "alert": alert,
                "split": r.get("split", ""),
                "feature": {
                    "name": name, "value": round(val, 6),
                    "baseline": base, "window": win,
                },
                "history": {
                    "self_n": int(r.get("hist_count", 0) or 0),
                    "peer_n": int(peer_n_map.get(aid, 0)),
                    "fallback_level": str(ft.get("peer_level", "L0")),
                    "lambda": float(ft.get("lambda_self", 0.0) or 0.0),
                },
                "rule": {
                    "rule_id": ("large_pay" if int(r.get("is_large", 0) or 0) == 1
                                else "december" if int(r.get("is_december", 0) or 0) == 1
                                else "not_triggered"),
                    "version": "v1",
                    "exception": "" if (int(r.get("is_large", 0) or 0) or
                                        int(r.get("is_december", 0) or 0)) else "not_triggered",
                },
                "relation": {
                    "entities": [n for n, _, _ in nbrs_sorted],
                    "paths": [rel for _, rel, _ in nbrs_sorted],
                    "edge_weights": [round(w, 4) for _, _, w in nbrs_sorted],
                },
                "replay": {
                    "graph_hash": graph_hash,
                    "param_version": PARAM_VERSION,
                    "threshold_version": threshold_version,
                },
                "q2_optional": {
                    "temporal_link_id": None,
                    "relation_type_calibration": None,
                },
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[B4] evidence.jsonl：{len(flagged)} 条报警（HighRisk {n_high} / Warning {n_warn}）")
    print(f"[B4] 阈值 θ_low={theta_low:.6f} θ_high={theta_high:.6f}；graph_hash={graph_hash}")


if __name__ == "__main__":
    main()
