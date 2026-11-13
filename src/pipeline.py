"""
MS-TRP 主流程编排（Step 1~8 串联）
========================================
  s_ts      时序风险（temporal_baseline，Step 1~3）= max(金额长尺度, 7日, 30日, 当季截至当前)
  s_struct  结构关系风险（graph_builder，账户一致性 + 实体频次突增，Step 4）
  s_diffuse = PPR(s_ts)（stage_weight + propagation，Step 5~6）
  s_g       = max(s_struct, s_diffuse)   —— 保留稀疏结构信号，图扩散仅作平滑
  s_gate    = gate_fusion(s_ts, s_g, r_ts, r_g)（reliability_gate，Step 7）
  s_rhythm_ch = max(W_RHYTHM·s_rhythm, W_CONC·s_conc)  —— 软加权节奏/季节通道（V8）
  s_final   = max(1 − (1−s_gate)(1−s_mismatch), s_rhythm_ch)  —— 账户一致性违反
              是确定性硬报警，与门控结果取 OR；节奏/季节通道用软权重 max，避免硬阈值假阳性（Step 7）
  阈值      = 分层收缩校准（calibration，Step 8）

泄漏控制：
  · 时序风险 s_ts 严格时间前向（历史统计只用 apply_date 之前的数据）；
  · peer 基线统计只在 train 段估计；
  · 结构风险由静态账户归属关系与实体频次前向窗口计算；
  · 图节点/边一次性由全部业务表构建（转导式评分），评分时所有节点已知，
    结构本身不含未来标签信息；warning/correction/refund 审计日志不进入图。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config, data, split, temporal_baseline
from src import graph_builder, propagation, reliability_gate, calibration, asof, metrics, event_eval

# V8 节奏/季节通道软权重（验证集冻结；见 hyperparams.json）：
#   W_CONC   → s_conc（年末集中，单位级12月占比偏离 → YEAREND_CONC）
#   W_RHYTHM → s_rhythm（历史节奏，明细级累计进度超前 → HISTORICAL_SHIFT）
W_CONC = 0.65
W_RHYTHM = 0.70


def run_ms_trp(rho=propagation.RHO_DEFAULT, tau=reliability_gate.TAU_DEFAULT,
               gamma=calibration.GAMMA_DEFAULT, quantile=calibration.ALPHA_HIGH,
               use_multiscale=True, freq_windows=(7, 30), use_amount_scale=True,
               use_quarter_scale=True,
               use_stage_weight=True,
               use_graph=True, use_gate=True,
               use_struct=True, use_ppr=True, use_rule=True,
               train_drop_ratio=0.0, edge_drop_ratio=0.0,
               delta_event=event_eval.DELTA_EVENT_DEFAULT,
               data_seed=None, model_seed=1):
    """
    运行完整 MS-TRP 流程，返回 (results dict, predictions DataFrame)。
    消融开关（正交：一次只改一个图信号分量，use_graph 为主开关）：
      use_multiscale=False  → 单尺度（s_ts 只用金额长尺度，去掉 7日/30日 频次与趋势）
      freq_windows=(7,30)   → 频次尺度窗口集合（E12 等维对照可改 (5,60)/(7,)/(30,)）
      use_amount_scale=False→ 去掉金额长尺度，s_ts 只由频次/季度窗口组成（E12 去金额）
      use_quarter_scale=False→去掉当季截至当前尺度（E12 去当季）
      use_stage_weight=False→ 固定边权（阶段因子恒 1.0，去阶段感知）
      use_graph=False       → 无图分支（s = s_ts）
      use_gate=False        → 固定 0.5/0.5 融合（去可靠度门控）
      use_struct=False      → 去掉实体频次突增（s_entity）信号
      use_ppr=False         → 去掉 PPR 图扩散（s_diffuse）信号
      use_rule=False        → 去掉账户一致性规则（s_mismatch）硬报警
    鲁棒性开关（种子固定 2026）：
      train_drop_ratio      → 随机丢弃该比例的训练支付（模拟历史数据缺失）
      edge_drop_ratio       → 随机删除该比例的图边（模拟关系数据缺失）
    """
    # ── 0. 数据种子切换（多种子实验；默认 None = 沿用 data_fixed/ 主库）──
    if data_seed is not None:
        config.set_data_seed(data_seed)

    # ── 1. 数据与标签 ──────────────────────────────────────────
    payments = data.load_payments()
    labels = data.load_labels()
    events = data.build_events(labels)
    point_labels = data.build_point_labels(payments, events)
    entities = data.load_entity_tables()
    projects_df, units_df = entities["project"], entities["unit"]

    # ── 2. 时间前向划分 ────────────────────────────────────────
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    # 鲁棒性：随机丢弃训练支付（模拟历史数据缺失），种子固定
    if train_drop_ratio > 0:
        rng = np.random.RandomState(model_seed)
        keep = rng.rand(len(train_idx)) > train_drop_ratio
        train_idx = [train_idx[i] for i in range(len(train_idx)) if keep[i]]

    # ── 3~5. 时序分支：多尺度特征 → Self/Peer 基线 → s_ts ──────
    enriched = temporal_baseline.enrich_payments(payments, projects_df, units_df)
    feat = temporal_baseline.compute_multi_scale_features(enriched, freq_windows=freq_windows)
    feat = temporal_baseline.compute_rhythm_features(enriched, feat)
    feat["z_conc"] = temporal_baseline.compute_year_end_concentration(enriched)
    peer_stats = temporal_baseline.compute_peer_stats(enriched.loc[train_idx])
    feat = temporal_baseline.build_self_peer_baseline(feat, peer_stats)
    feat = temporal_baseline.compute_temporal_risk(
        feat, use_amount=use_amount_scale, freq_windows=freq_windows,
        use_quarter=use_quarter_scale)
    if use_multiscale:
        s_ts = np.asarray(feat["s_ts"], dtype=float)
    else:  # 单尺度：仅金额长尺度（去频次尺度）
        s_ts = np.asarray(feat["s_amt"], dtype=float)

    # ── 6. 图构建（按需；正交消融开关 use_struct/use_ppr/use_rule）──
    apply_ids = payments["apply_id"].astype(str)
    n_pay = len(payments)
    if use_graph:
        tables = graph_builder.load_graph_tables()
        # 结构关系风险（账户一致性规则 + 实体频次突增；点级，本身时间前向）
        struct = graph_builder.compute_structural_risk(tables, payments)
        raw_mismatch = np.asarray(struct["s_account_mismatch"], dtype=float)
        raw_entity = np.asarray(struct["s_entity_freq"], dtype=float)
    else:
        tables = None
        raw_mismatch = np.zeros(n_pay)
        raw_entity = np.zeros(n_pay)

    # 正交消融：use_rule/use_struct 分别关掉账户一致性 / 实体频次信号
    s_mismatch = raw_mismatch if use_rule else np.zeros(n_pay)
    s_entity = raw_entity if use_struct else np.zeros(n_pay)
    s_struct = np.maximum(s_mismatch, s_entity)

    # ── 7. as-of PPR 扩散（day_end）+ 节点度（r_g 用）─────────
    if use_graph and use_ppr:
        node_ids, node_type, id2idx = graph_builder.build_nodes(tables)
        edges = graph_builder.build_edges(tables, id2idx)
        # 鲁棒性：随机删除图边（模拟关系数据缺失），种子固定
        if edge_drop_ratio > 0:
            rng = np.random.RandomState(model_seed)
            keep = rng.rand(len(edges)) > edge_drop_ratio
            edges = [edges[i] for i in range(len(edges)) if keep[i]]
        pay_node = apply_ids.map(lambda a: id2idx.get(f"PaymentApplication::{a}", -1))
        pay_idx = np.asarray(pay_node.tolist(), dtype=np.int64)
        node_stage = {}
        for aid, st in zip(apply_ids, enriched["stage"].astype(str)):
            node_stage[f"PaymentApplication::{aid}"] = st
        s_diffuse, deg_pay, n_iters = asof.score_asof_diffuse(
            edges, node_ids, node_stage, payments, s_ts, pay_idx,
            rho=rho, use_stage=use_stage_weight)
    else:
        s_diffuse = np.zeros(n_pay)
        deg_pay = np.zeros(n_pay)
        n_iters = 0.0

    s_g = np.maximum(s_struct, s_diffuse)

    # V8 节奏/季节通道：软最大融合（替代 V7.4 的硬阈值 noisy-OR）。
    # 现实数据修复（执行率 46%→92%）后，硬阈值(0.95)让 ~1.4% 正常支付灌入高节奏噪声，
    # noisy-OR 把它们抬到接近 1，拖垮整体 AUPRC（val 0.4364→0.3444）。改为软权重 max，
    # 有界且不灌入硬假阳性：
    #   s_conc  （年末集中，单位级12月占比偏离）→ YEAREND_CONC（W_CONC）
    #   s_rhythm（历史节奏，明细级累计进度超前）→ HISTORICAL_SHIFT（W_RHYTHM）
    # 权重在验证集冻结（见 hyperparams.json）。s_progress 仍计算并导出供分型分析，
    # 但快照滞后在现实数据上与 s_trend 的 Q4 突增冗余（验证集搜索权重≈0），不并入最终分。
    s_rhythm_ch = np.maximum(W_RHYTHM * np.asarray(feat["s_rhythm"], dtype=float),
                             W_CONC * np.asarray(feat["s_conc"], dtype=float))

    # ── 8. 可靠度门控融合 + 硬关系违规 OR ─────────────────────
    r_ts = reliability_gate.compute_temporal_reliability(feat)
    if use_graph:
        r_g = reliability_gate.compute_graph_reliability(deg_pay, s_struct)
        if use_gate:
            s_gate, g_ts, g_g = reliability_gate.gate_fusion(s_ts, s_g, r_ts, r_g, tau=tau)
        else:  # 固定 0.5/0.5 融合（去门控）
            s_gate = 0.5 * s_ts + 0.5 * s_g
            g_ts = np.full_like(s_ts, 0.5)
            g_g = np.full_like(s_ts, 0.5)
        # 账户一致性违反（payee≠账户归属 payee）是确定性关系违规，作为硬报警与门控结果取 OR；
        # 节奏/季节通道用软权重 max（有界），避免硬阈值假阳性灌入最终分。
        s_core = 1.0 - (1.0 - s_gate) * (1.0 - s_mismatch)
        s_final = np.maximum(s_core, s_rhythm_ch)
    else:  # 无图分支：s = s_ts，节奏/季节通道同样软 max
        s_gate = s_ts.copy()
        g_ts = np.ones_like(s_ts)
        g_g = np.zeros_like(s_ts)
        r_g = np.zeros_like(s_ts)
        s_final = np.maximum(s_ts, s_rhythm_ch)

    # ── 10. 分层阈值校准（val 上冻结）──────────────────────────
    groups = (enriched["unit_type"].astype(str) + "|" +
              enriched["project_category"].astype(str) + "|" +
              enriched["stage"].astype(str))
    y_all = point_labels["is_anomaly"].values.astype(int)
    thresholds = calibration.calibrate_hierarchical(
        s_final[val_idx], y_all[val_idx], groups.values[val_idx],
        gamma=gamma, quantile=quantile)
    thr_global = thresholds["__global__"]
    # 目标 FPR 阈值（P0-1 协议 A：验证集正常样本 quantile(1−alpha_fpr) 冻结，与预算阈值并列报告）
    tau_fpr = calibration.calibrate_target_fpr(
        s_final[val_idx], y_all[val_idx], alpha_fpr=calibration.ALPHA_FPR_DEFAULT)

    # ── 11. 评估（test，阈值冻结）───────────────────────────────
    test_mask = np.zeros(len(payments), dtype=bool)
    test_mask[test_idx] = True
    y_test = y_all[test_idx]
    s_test = s_final[test_idx]
    apply_test = apply_ids.values[test_idx]

    results = {
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "ppr_iters": n_iters, "rho": rho, "tau": tau, "gamma": gamma,
        "thr_global": thr_global, "tau_fpr": tau_fpr,
        "data_seed": data_seed, "model_seed": model_seed,
    }

    # 点级评估（阈值无关 + 阈值相关，统一口径 evaluate_point）
    point_eval = metrics.evaluate_point(y_test, s_test, thr_global)
    results["point_auprc"] = point_eval["auprc"]
    results["point_auc"] = point_eval["auc"]
    results["normalized_ap_excess"] = point_eval["normalized_ap_excess"]
    results["point"] = {
        "precision": point_eval["precision"], "recall": point_eval["recall"],
        "f1": point_eval["f1"], "tp": point_eval["tp"], "fp": point_eval["fp"], "fn": point_eval["fn"],
    }
    results["fpr"] = point_eval["fpr"]
    results["far_per_1000"] = point_eval["far_per_1000"]          # 指导书口径：FP/N_normal×1000
    results["fp_per_1000_all"] = point_eval["fp_per_1000_all"]    # 旁证口径：FP/N_total×1000
    results["topk"] = metrics.metrics_at_topk(y_test, s_test, quantile)
    # 目标 FPR 操作点（协议 A：冻结阈值，与报警预算协议 B 并列报告）
    results["point_at_target_fpr"] = metrics.evaluate_point(y_test, s_test, tau_fpr)

    # 每类点级 AUPRC（正样本 = 该类事件受影响支付）
    type_of = {}
    for _, ev in events.iterrows():
        for aid in ev["apply_ids"]:
            type_of.setdefault(str(aid), set()).add(ev["anomaly_type"])
    per_type_auprc = {}
    for at in config.ANOMALY_TYPES:
        y_at = np.asarray([1 if at in type_of.get(a, ()) else 0 for a in apply_test])
        if y_at.sum() > 0:
            per_type_auprc[at] = average_precision_score(y_at, s_test)
        else:
            per_type_auprc[at] = float("nan")
    results["per_type_auprc"] = per_type_auprc

    # 每类点级召回率（P0-4 修正：只传 test 子集，避免非 test 成员计入召回分母）
    score_map = {str(a): float(s) for a, s in zip(apply_test, s_test)}
    test_point_labels = point_labels.iloc[test_idx].reset_index(drop=True)
    results["per_class_recall"] = metrics.per_class_recall(test_point_labels, score_map, thr_global)

    # 事件级指标（全局阈值；松散口径 = 至少命中一个成员）
    test_events = events[events["apply_ids"].apply(
        lambda ids: any(str(a) in set(apply_test) for a in ids))]
    results["event"] = metrics.event_metrics(test_events, score_map, thr_global)
    results["per_type_event_recall"] = metrics.per_type_event_recall(test_events, score_map, thr_global)

    # 事件级严格一对一匹配（P0-4 新增：实体一致 + 时间交叠，杜绝重复命中虚高）
    true_ev = event_eval.build_true_events(test_events)
    flagged = payments[test_mask & (s_final > thr_global)]
    if len(true_ev) and len(flagged):
        alerts = event_eval.build_payment_alerts(flagged)
        pred_ev = event_eval.cluster_alerts(alerts, delta_event=delta_event)
        results["event_strict"] = event_eval.match_events(true_ev, pred_ev)
    else:
        results["event_strict"] = {"tp": 0, "fn": int(len(true_ev)), "fp": 0,
                                   "event_recall": 0.0, "event_precision": 0.0,
                                   "event_f1": 0.0, "pairs": []}

    # ── 12. 预测输出（供案例研究 / 后续实验复用）────────────────
    split_col = np.full(len(payments), "train", dtype=object)
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"
    pred = pd.DataFrame({
        "apply_id": apply_ids.values,
        "apply_date": payments["apply_date"].values,
        "amount_wan": payments["amount_wan"].values,
        "is_large": (payments["is_large"].astype(str).str.upper() == "Y").astype(int).values
        if "is_large" in payments.columns else 0,
        "is_december": payments["is_december"].values,
        "unit_type": enriched["unit_type"].values,
        "project_category": enriched["project_category"].values,
        "stage": enriched["stage"].values,
        "group": groups.values,
        "is_anomaly": y_all,
        "anomaly_types": [",".join(x) for x in point_labels["anomaly_types"].values],
        "z_amount": np.asarray(feat["z_amount"]),
        "hist_count": np.asarray(feat["hist_count"]),
        "s_amt": np.asarray(feat["s_amt"]),
        "s_freq": np.asarray(feat["s_freq"]),
        "s_trend": np.asarray(feat["s_trend"]),
        "z_quarter": np.asarray(feat["z_quarter"]),
        "s_quarter": np.asarray(feat["s_quarter"]),
        "z_rhythm": np.asarray(feat["z_rhythm"]),
        "z_rhythm_ind": np.asarray(feat["z_rhythm_ind"]),
        "z_conc": np.asarray(feat["z_conc"]),
        "s_progress": np.asarray(feat["s_progress"]),
        "s_rhythm": np.asarray(feat["s_rhythm"]),
        "s_conc": np.asarray(feat["s_conc"]),
        "s_rhythm_ch": s_rhythm_ch,
        "s_ts": s_ts,
        "s_struct": s_struct,
        "s_mismatch": s_mismatch,
        "s_entity_freq": s_entity,
        "s_diffuse": s_diffuse,
        "s_g": s_g,
        "r_ts": np.asarray(r_ts),
        "r_g": np.asarray(r_g),
        "g_ts": np.asarray(g_ts),
        "g_g": np.asarray(g_g),
        "s_gate": s_gate,
        "s_final": s_final,
        "split": split_col,
    })
    return results, pred
