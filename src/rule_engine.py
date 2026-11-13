"""
MS-TRP Step 8 —— 四类可解释证据
========================================
对每条报警支付给出"为何报警"的四类证据，供审计解释与案例研究：

  1) 特征证据 FEATURE  ：当前笔金额/频次特征偏离（s_amt / s_freq，Step 3）
  2) 历史证据 HISTORY  ：相对 Self/Peer 动态历史基线的稳健 z 偏离（z_amount，Step 2~3）
  3) 规则证据 RULE     ：固定业务规则命中（区级大额支付 / 年末集中支付）
  4) 关系证据 RELATION ：账户一致性违反 / 实体频次聚集 / 图风险传播（Step 4~6）

证据按分量独立触发，可同时命中多类，最终风险 s_final 是各分量的门控融合。
输入为 pipeline 产出的 predictions DataFrame 行（Series）。
"""
import pandas as pd

# 各证据的触发阈值（与各分量有界分数同尺度，均可解释）
FEATURE_AMT_TH = 0.5      # s_amt 金额偏离
FEATURE_FREQ_TH = 0.5     # s_freq 频次突增
HISTORY_Z_TH = 2.0        # 历史基线稳健 z 偏离
RELATION_MISMATCH_TH = 0.5
RELATION_ENTITY_TH = 0.5
RELATION_DIFFUSE_TH = 0.5


def explain(row):
    """
    对单条支付（predictions 的一行）生成证据列表。
    返回 list[dict(type, label, detail, score)]，按分数降序。
    """
    ev = []

    # 1) 特征证据
    s_amt = float(row.get("s_amt", 0.0) or 0.0)
    s_freq = float(row.get("s_freq", 0.0) or 0.0)
    if s_amt >= FEATURE_AMT_TH:
        ev.append({"type": "FEATURE", "label": "金额特征偏离",
                   "detail": f"金额偏离有界分数 s_amt={s_amt:.2f}",
                   "score": s_amt})
    if s_freq >= FEATURE_FREQ_TH:
        ev.append({"type": "FEATURE", "label": "频次特征突增",
                   "detail": f"7 日支付频次突增 s_freq={s_freq:.2f}",
                   "score": s_freq})

    # 2) 历史证据
    z = float(row.get("z_amount", 0.0) or 0.0)
    if abs(z) >= HISTORY_Z_TH:
        ev.append({"type": "HISTORY", "label": "偏离历史基线",
                   "detail": f"相对 Self/Peer 动态基线 z={z:.2f}",
                   "score": min(abs(z) / 5.0, 1.0)})

    # 3) 规则证据（is_large 可为 0/1 或 Y/N，统一解析）
    def _flag(v):
        if isinstance(v, str):
            return v.strip().upper() in ("Y", "YES", "1", "TRUE", "是")
        return int(v or 0) == 1

    if _flag(row.get("is_large", 0)):
        ev.append({"type": "RULE", "label": "区级大额支付",
                   "detail": f"金额 {float(row.get('amount_wan', 0.0)):.1f} 万元 ≥ 大额阈值",
                   "score": 0.7})
    if _flag(row.get("is_december", 0)):
        ev.append({"type": "RULE", "label": "年末集中支付",
                   "detail": "支付发生在 12 月（年末集中支付窗口）",
                   "score": 0.5})

    # 4) 关系证据
    s_mm = float(row.get("s_mismatch", 0.0) or 0.0)
    s_ef = float(row.get("s_entity_freq", 0.0) or 0.0)
    s_df = float(row.get("s_diffuse", 0.0) or 0.0)
    if s_mm >= RELATION_MISMATCH_TH:
        ev.append({"type": "RELATION", "label": "账户一致性违反",
                   "detail": "收款人与账户归属收款人不一致（共享账户）",
                   "score": 1.0})
    if s_ef >= RELATION_ENTITY_TH:
        ev.append({"type": "RELATION", "label": "实体频次聚集",
                   "detail": f"收款人/账户短窗口频次突增 s={s_ef:.2f}",
                   "score": s_ef})
    if s_df >= RELATION_DIFFUSE_TH:
        ev.append({"type": "RELATION", "label": "图风险传播",
                   "detail": f"关联节点传播风险 s_diffuse={s_df:.2f}",
                   "score": s_df})

    ev.sort(key=lambda d: -d["score"])
    return ev


def explain_flagged(pred, threshold):
    """
    对 s_final > threshold 的报警支付生成证据表。
    返回 DataFrame：apply_id, s_final, anomaly_types, evidence_types, evidence_detail。
    """
    rows = []
    for _, r in pred.iterrows():
        if float(r["s_final"]) <= threshold:
            continue
        ev = explain(r)
        types = ";".join(sorted({e["type"] for e in ev}))
        detail = " | ".join(f"[{e['type']}]{e['label']}" for e in ev)
        rows.append({"apply_id": r["apply_id"], "s_final": float(r["s_final"]),
                     "is_anomaly": int(r["is_anomaly"]),
                     "anomaly_types": r.get("anomaly_types", ""),
                     "evidence_types": types, "evidence_detail": detail})
    return pd.DataFrame(rows)
