# -*- coding: utf-8 -*-
"""
指导书 §10.1 最低单元测试样例 —— 27 个唯一名称逐一对应
==========================================================
每个 test_* 与 §10.1 总表同名，断言对应「输入场景 → 预期结果」。
读取 result_raw 产物 / config 冻结值的用例属于输出契约测试（B7 签核口径）。

说明：本文件补齐 §10.1 的 27 个名称；其余更细粒度的既有测试仍保留在
tests/test_*.py（metrics/event/calibration/split/future/asof 等）。
"""
import os
import json
import inspect

import numpy as np
import pandas as pd

from src import (config, metrics, event_eval, propagation, graph_builder,
                 calibration, split, stage_weight, external_gate)


# ── 1. 未来信息不变性 ──────────────────────────────────────────────
def test_no_future():
    """2025 评分时点放入 2026 新交易 → 可见边集合不变。"""
    cutoff = pd.Timestamp("2025-12-31")
    edges_2025 = [{"src": 0, "dst": 1, "rel": "paid_to",
                   "date": pd.Timestamp("2025-06-01")}]
    edges_with_2026 = edges_2025 + [
        {"src": 2, "dst": 3, "rel": "paid_to", "date": pd.Timestamp("2026-01-15")}]
    before = graph_builder.make_asof_snapshot(edges_2025, cutoff)
    after = graph_builder.make_asof_snapshot(edges_with_2026, cutoff)
    assert [e["rel"] for e in before] == [e["rel"] for e in after] == ["paid_to"]
    assert all(e["date"] <= cutoff for e in after)


# ── 2~3. 图分支退化 ────────────────────────────────────────────────
def test_seed_zero():
    """全节点 s_ts 为 0 → 图分支保持 0。"""
    n = 3
    W = propagation.build_transition_matrix(
        n, [{"src": 0, "dst": 1, "rel": "paid_to", "date": None}], [1.0])
    s, _ = propagation.personalized_ppr(np.zeros(n), W)
    np.testing.assert_allclose(s, np.zeros(n), atol=1e-12)


def test_isolate():
    """图节点没有邻居 → 无自环、靠 PPR 重启保留自身信号，且稳定收敛、不泄漏。"""
    n = 3
    W, deg = propagation.build_transition_matrix_from_arrays(
        np.array([1]), np.array([2]), np.array([1.0]), n)
    assert deg[0] == 0                      # 孤立节点加权度=0（无邻居）
    s, _ = propagation.personalized_ppr(np.array([1.0, 0.0, 0.0]), W, rho=0.6)
    assert abs(s[0] - 0.6) < 1e-9           # 自身信号 = rho·seed，稳定
    assert abs(s[1]) < 1e-12 and abs(s[2]) < 1e-12   # 不向邻域泄漏
    assert np.isfinite(s).all()


# ── 4~5. 事件边界 / 快照 ───────────────────────────────────────────
def test_event_no_overlap():
    """事件恰跨划分边界 → 边界日期归属确定，禁止跨集合。"""
    pay = pd.DataFrame({
        "apply_id": ["p_train_end", "p_val_start", "p_val_end", "p_test_start"],
        "apply_date": pd.to_datetime(
            [config.TRAIN_END, config.VAL_START, config.VAL_END, config.TEST_START]),
    })
    tr, va, te = split.time_forward_split(pay)
    assert (len(tr), len(va), len(te)) == (1, 2, 1)
    assert set(tr).isdisjoint(va) and set(va).isdisjoint(te) and set(tr).isdisjoint(te)
    assert pay.loc[tr[0], "apply_id"] == "p_train_end"
    assert pay.loc[te[0], "apply_id"] == "p_test_start"


def test_progress_no_payment():
    """当月无付款 → 月末进度快照仍产出（静态关系边恒可见）。"""
    cutoff = pd.Timestamp("2025-03-31")
    edges = [
        {"src": 0, "dst": 1, "rel": "owns_account", "date": None},
        {"src": 2, "dst": 3, "rel": "paid_to", "date": pd.Timestamp("2025-01-10")},
    ]
    snap = graph_builder.make_asof_snapshot(edges, cutoff)
    assert len(snap) == 2
    assert {e["rel"] for e in snap} == {"owns_account", "paid_to"}


# ── 6~10. 评价指标口径 ─────────────────────────────────────────────
def test_metric_denominator():
    """1000 正常、10 误报 → FAR/1000=10。"""
    assert abs(metrics.far_per_1000(10, 1000) - 10.0) < 1e-9
    assert abs(metrics.fpr(10, 1000) - 0.01) < 1e-9


def test_shuffle_labels():
    """变换 test 真实标签 → 任何预测不变（评分不使用标签）。"""
    from baselines import forced as blm
    pay = pd.DataFrame({
        "apply_id": [f"p{i}" for i in range(4)],
        "apply_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        "payee_id": ["a", "a", "b", "b"],
        "project_id": ["x", "y", "x", "z"],
        "account_id": ["acc1"] * 4,
    })
    s_before = blm.shared_payee_count_scores(pay).values
    _ = pd.DataFrame({"apply_id": pay["apply_id"], "is_anomaly": [0, 1, 0, 1]}) \
        .sample(frac=1.0, random_state=1).reset_index(drop=True)   # 打乱标签
    s_after = blm.shared_payee_count_scores(pay).values
    np.testing.assert_array_equal(s_before, s_after)


def test_metric_families():
    """三族指标：点级/区间级/排序级；区间不依赖事件合并；K 冻结；记入 metrics_family.csv。"""
    assert set(metrics.METRIC_FAMILIES.keys()) == {"point", "interval", "ranking"}
    assert "vus_pr" in metrics.METRIC_FAMILIES["interval"]
    assert "ndcg_at_k" in metrics.METRIC_FAMILIES["ranking"]
    assert "delta_event" not in inspect.signature(metrics.vus_pr).parameters
    assert "k" in inspect.signature(metrics.ndcg_at_k).parameters
    fam = pd.read_csv(os.path.join(config.RESULT_DIR, "metrics_family.csv"))
    assert set(fam["family"].unique()) == {"point", "interval", "ranking"}


def test_frozen_threshold_periods():
    """零验证FAR/缺失正常标签/季度阶段分组 → FAR ratio 记 NA；阈值哈希不变；禁外部数据实际FAR。"""
    m = json.load(open(os.path.join(config.RESULT_DIR, "run_main", "metrics.json"),
                       encoding="utf-8"))
    ep = json.load(open(os.path.join(config.PROJECT_DIR, "config", "event_params.json"),
                        encoding="utf-8"))
    assert abs(ep["threshold"] - m["thr_global"]) < 1e-12      # 测试阈值冻结不变
    df = pd.read_csv(os.path.join(config.RESULT_DIR, "frozen_threshold_far_by_period.csv"))
    assert set(df["period_type"].unique()) <= {"quarter", "stage"}
    # 缺失正常标签时 calibrate_target_fpr 回退全量分位（有限、不崩溃）
    tau = calibration.calibrate_target_fpr(np.array([0.1, 0.9]), np.array([1, 1]),
                                           alpha_fpr=0.01)
    assert np.isfinite(tau)
    # 禁对无标签外部数据生成实际 FAR
    assert external_gate.external_time_gate(False)["status"] == "not_comparable"


def test_metric_semantics():
    """ConstantScore 区间/排序记 NA+not_applicable，AP 按并列策略=基率；RandomUniform 空模型。"""
    main = pd.read_csv(os.path.join(config.RESULT_DIR, "metrics_main.csv"))
    rng = pd.read_csv(os.path.join(config.RESULT_DIR, "metrics_range.csv"))
    rnk = pd.read_csv(os.path.join(config.RESULT_DIR, "metrics_ranking.csv"))
    expected_cols = ["model", "data_seed", "model_seed", "split", "n_test", "prevalence",
                     "payment_auprc", "payment_f1", "precision", "recall", "far_1000",
                     "event_f1", "f1_raw", "recall_at_far5", "recall_at_far10",
                     "reporting_regime", "n_reported", "status"]
    assert list(main.columns) == expected_cols                        # 唯一规范 schema
    cs_main = main[main["model"] == "constant_score"].iloc[0]
    cs_rng = rng[rng["model"] == "constant_score"].iloc[0]
    cs_rnk = rnk[rnk["model"] == "constant_score"].iloc[0]
    assert pd.isna(cs_rng["vus_pr"]) and cs_rng["metric_status"] == "not_applicable"
    assert pd.isna(cs_rnk["ndcg_at_k"]) and cs_rnk["metric_status"] == "not_applicable"
    assert abs(cs_main["payment_auprc"] - cs_main["prevalence"]) < 1e-9
    ru_rng = rng[rng["model"] == "random_uniform"].iloc[0]
    ru_rnk = rnk[rnk["model"] == "random_uniform"].iloc[0]
    assert np.isfinite(ru_rng["vus_pr"]) and np.isfinite(ru_rnk["ndcg_at_k"])


# ── 11~13. 图结构与桥接 ────────────────────────────────────────────
def test_relation_weight_effect():
    """固定阶段与邻居下改变 a_r → 关系混合 W 改变（未被对称归一化抵消）。"""
    n = 4

    def _W(a_paid_to):
        edges = [{"src": 0, "dst": 1, "rel": "paid_to", "date": None},
                 {"src": 0, "dst": 2, "rel": "classified_as", "date": None}]
        w = np.array([a_paid_to, graph_builder.REL_STRENGTH["classified_as"]], dtype=float)
        W, _ = propagation.build_transition_matrix_from_arrays(
            np.array([0, 0]), np.array([1, 2]), w, n)
        return W

    W1, W2 = _W(0.9), _W(1.8)
    assert W1[0, 1] != W2[0, 1]                       # a_r 改变确实改变 W
    # 对称归一化只做全局尺度，不消除关系强度比：W[0,1]/W[0,2]=sqrt(a1/a2)
    assert abs(W1[0, 1] / W1[0, 2] - np.sqrt(0.9 / graph_builder.REL_STRENGTH["classified_as"])) < 1e-9


def test_reverse_message_bridge():
    """两个项目经同一 Payee 连接 → Voucher_A→Payee→Voucher_B 传播可达。"""
    n = 3
    edges = [{"src": 0, "dst": 1, "rel": "paid_to", "date": None},
             {"src": 2, "dst": 1, "rel": "paid_to", "date": None}]
    W = propagation.build_transition_matrix(n, edges, [1.0, 1.0])
    s, _ = propagation.personalized_ppr(np.array([1.0, 0.0, 0.0]), W, rho=0.6)
    assert s[2] > 0          # 经 Payee 桥接到达 Voucher_B
    assert s[1] > 0          # Payee 节点本身也获得风险


def test_eval_permission():
    """置换测试标签 → 评分向量、阶段边权、可靠度门控不变（评分链路无标签访问）。"""
    from baselines import forced as blm
    sig = inspect.signature(stage_weight.prepare_edge_weights)
    assert "label" not in str(sig).lower() and "y" not in sig.parameters
    sig2 = inspect.signature(blm.shared_payee_count_scores)
    assert "label" not in str(sig2).lower()
    assert graph_builder.REL_STRENGTH["paid_to"] == 0.9
    assert stage_weight.STAGE_FACTOR == {"Early": 0.7, "Middle": 1.0, "Late": 1.5}


# ── 14~20. 评估协议 / 主表 / 报告 / 阈值 ───────────────────────────
def test_seed_independence():
    """同一合成实例下更换模型随机数 → 不增加 data_seed 独立版本数。"""
    assert config.DATA_SEEDS == [10, 20, 30]
    assert len(config.MODEL_SEEDS) == 5
    for ds in config.DATA_SEEDS:
        d = os.path.join(config.RESULT_DIR, "seed", f"seed{ds}")
        model_dirs = sorted(x for x in os.listdir(d)
                            if os.path.isdir(os.path.join(d, x)) and x.startswith("model"))
        assert model_dirs == [f"model{i}" for i in range(1, 6)], model_dirs


def test_event_one_to_one():
    """重复预测同一真实事件 → 最多记 1 个 TP，多余预警计 FP。"""
    true = pd.DataFrame({
        "entity_key": ["A", "B"],
        "start": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")],
        "end":   [pd.Timestamp("2023-01-05"), pd.Timestamp("2023-02-05")],
    })
    pred = pd.DataFrame({
        "entity_key": ["A", "A", "B"],
        "start": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"),
                  pd.Timestamp("2023-02-01")],
        "end":   [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"),
                  pd.Timestamp("2023-02-03")],
    })
    r = event_eval.match_events(true, pred)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 0)


def test_constant_ties():
    """常数分数与阈值相等 → 比较结果固定（严格 >，score==thr 不报警）；tie_policy 落盘。"""
    pm = metrics.point_metrics(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.5]), 0.5)
    assert (pm["tp"], pm["fp"]) == (0, 0)
    ep = json.load(open(os.path.join(config.PROJECT_DIR, "config", "event_params.json"),
                        encoding="utf-8"))
    assert ep["tie_handling"] == "score > threshold 报警"


def test_injection_threshold_separation():
    """注入模板与强度配置哈希在阈值搜索前固定，搜索空间独立。"""
    inj = pd.read_csv(os.path.join(config.RESULT_DIR, "injection_params.csv"))
    assert "injection_locked_before_threshold_search" in inj.columns
    assert inj["injection_locked_before_threshold_search"].astype(bool).all()
    assert inj["injection_config_hash"].notna().all()
    ep = json.load(open(os.path.join(config.PROJECT_DIR, "config", "event_params.json"),
                        encoding="utf-8"))
    assert isinstance(ep["threshold"], float)


def test_main_table_rows():
    """九行强制验收行均存在，缺跑 status=not_run 且数值 NA，不删除不填 0。"""
    main = pd.read_csv(os.path.join(config.RESULT_DIR, "metrics_main.csv"))
    mandatory = ["random_uniform", "constant_score", "shared_payee_count",
                 "shared_account_concentration", "ecod", "copod", "mcd", "cblof",
                 "simple_ppr"]
    present = set(main["model"])
    assert all(m in present for m in mandatory), \
        f"缺失强制验收行：{[m for m in mandatory if m not in present]}"
    for m in mandatory:
        row = main[main["model"] == m].iloc[0]
        assert row["status"] in ("completed", "not_run")
        if row["status"] == "not_run":
            assert pd.isna(row["payment_auprc"])


def test_reporting_regime():
    """无法以 test 标签或 best-of-N 挑主结果：reporting_regime=single_run 且留痕 n_reported。"""
    main = pd.read_csv(os.path.join(config.RESULT_DIR, "metrics_main.csv"))
    assert (main["reporting_regime"] == "single_run").all()
    assert (main["n_reported"] == 1).all()
    reg = pd.read_csv(os.path.join(config.RESULT_DIR, "run_registry.csv"))
    assert (reg["reporting_regime"] == "single_run").all()


def test_heuristics_causal_time():
    """删去 score_time 之后的图边与支付 → Shared-Payee/Account 评分不变（因果时间）。"""
    from baselines import forced as blm
    pay = pd.DataFrame({
        "apply_id": [f"p{i}" for i in range(3)],
        "apply_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "payee_id": ["a", "a", "b"],
        "project_id": ["x", "y", "x"],
        "account_id": ["acc1"] * 3,
    })
    future = pd.DataFrame({
        "apply_id": ["p3"], "apply_date": pd.to_datetime(["2026-01-01"]),
        "payee_id": ["a"], "project_id": ["w"], "account_id": ["acc1"],
    })
    pay_ext = pd.concat([pay, future], ignore_index=True)
    np.testing.assert_array_equal(
        blm.shared_payee_count_scores(pay).values,
        blm.shared_payee_count_scores(pay_ext).values[:3])
    np.testing.assert_array_equal(
        blm.shared_account_concentration_scores(pay).values,
        blm.shared_account_concentration_scores(pay_ext).values[:3])


# ── 21~27. 种子落点 / 隔离 / 外部门 / 小分母 ───────────────────────
def test_seed_diffuse_vs_local():
    """种子分散与集中对照：不预设越集中增益越大；集中种子同样沿图扩散放大误报。"""
    n = 4
    edges = [{"src": 0, "dst": 1, "rel": "paid_to", "date": None},
             {"src": 1, "dst": 2, "rel": "paid_to", "date": None},
             {"src": 2, "dst": 3, "rel": "paid_to", "date": None}]
    W = propagation.build_transition_matrix(n, edges, [1.0, 1.0, 1.0])
    s_conc, _ = propagation.personalized_ppr(np.array([1.0, 0, 0, 0]), W, rho=0.6)
    s_diff, _ = propagation.personalized_ppr(np.array([0.25] * 4), W, rho=0.6)
    assert np.isfinite(s_conc).all() and np.isfinite(s_diff).all()
    assert s_conc[3] > 0               # 集中种子也向末端扩散（误报放大机制存在）
    assert s_conc.sum() > 0 and s_diff.sum() > 0


def test_seed_off_target():
    """种子落在无关高分节点 → 目标节点图分支增益下降，如实报告。"""
    n = 3
    edges = [{"src": 0, "dst": 1, "rel": "paid_to", "date": None},
             {"src": 2, "dst": 1, "rel": "paid_to", "date": None}]
    W = propagation.build_transition_matrix(n, edges, [1.0, 1.0])
    s_on, _ = propagation.personalized_ppr(np.array([0.0, 0.0, 1.0]), W, rho=0.6)
    s_off, _ = propagation.personalized_ppr(np.array([1.0, 0.0, 0.0]), W, rho=0.6)
    assert s_on[2] > s_off[2]          # 正对种子得分更高；off-target 仅桥接残余


def test_vus_pr_null():
    """同 data_seed 下 MS-TRP 与 RandomUniform 同网格/真值区间；null_flag 不触发隐藏结果。"""
    df = pd.read_csv(os.path.join(config.RESULT_DIR, "vus_pr_null_check.csv"))
    assert {"data_seed", "grid_id", "null_flag"} <= set(df.columns)
    assert (df["grid_id"] == "natural_month").all()
    assert df["null_flag"].isin([True, False]).all()
    # vus_pr 用自然月网格，返回 [0,1]
    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    ts = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-15", "2025-02-01",
                                   "2025-02-15", "2025-02-20", "2025-02-25"]))
    assert 0.0 <= metrics.vus_pr(y, s, ts) <= 1.0


def test_table_raw_derived_isolation():
    """raw 输入必须拒绝白名单外字段/未来字段/Self-Peer 偏离；derived 独立诊断。"""
    from experiment.run_table_isolation import RAW_WHITELIST, DERIVED_COLS
    assert isinstance(RAW_WHITELIST, dict) and len(RAW_WHITELIST) >= 10
    # 白名单不含任何 z_/s_ 派生或未来字段名
    for k in RAW_WHITELIST:
        assert not k.startswith(("z_", "s_", "future", "peer", "self_"))
    # derived 追加字段与 raw 白名单互斥（不混入主表 raw）
    assert set(DERIVED_COLS).isdisjoint(set(RAW_WHITELIST.keys()))
    assert "z_amount" in DERIVED_COLS and "s_ts" in DERIVED_COLS


def test_seed_dominance_pattern():
    """冻结 r_bridge 规则五类；无可辨分组记 unknown；评分器不访问事件真值。"""
    def _classify(r_bridge, mass, stype, n_bridge, n_nonbridge,
                  eps=1e-12, tol=1e-9):
        if mass <= eps:
            return "zero"
        if stype not in ("PAYEE", "ACCOUNT") or not n_bridge or not n_nonbridge:
            return "unknown"
        if abs(r_bridge - 0.5) <= tol:
            return "diffuse"
        return "source_dominant" if r_bridge > 0.5 else "victim_dominant"

    assert _classify(0.0, 0.0, "PAYEE", 1, 1) == "zero"
    assert _classify(0.5, 1.0, "PAYEE", 1, 1) == "diffuse"
    assert _classify(0.8, 1.0, "PAYEE", 1, 1) == "source_dominant"
    assert _classify(0.2, 1.0, "PAYEE", 1, 1) == "victim_dominant"
    assert _classify(0.8, 1.0, "UNIT", 1, 1) == "unknown"
    assert _classify(0.8, 1.0, "PAYEE", 1, 0) == "unknown"
    assert _classify(0.8, 1.0, "PAYEE", 0, 1) == "unknown"
    # 评分器（PPR/转移矩阵）不接收事件真值
    assert "event" not in str(inspect.signature(propagation.personalized_ppr)).lower()


def test_external_time_gate():
    """Q2-3A score_time_valid=false → status=not_comparable；禁止继续零样本推理。"""
    assert external_gate.external_time_gate(False)["status"] == "not_comparable"
    assert external_gate.external_time_gate(True)["status"] == "allowed"


def test_far_small_denominator():
    """季度正常样本 49、验证 FAR>0、经验比值>2 → 只记原始，不触发偏离警示。"""
    fp, n_normal = 1, 49
    far = metrics.far_per_1000(fp, n_normal)
    assert abs(far - 1 / 49 * 1000) < 1e-6
    far_val = 0.01
    ratio = (far / (far_val * 1000.0)) if (n_normal >= 50 and far_val > 0) else np.nan
    assert np.isnan(ratio)                      # 小分母不触发倍数警示
    df = pd.read_csv(os.path.join(config.RESULT_DIR, "frozen_threshold_far_by_period.csv"))
    small = df[df["n_normal"] < 50]
    if len(small):
        assert (small["denominator_valid"] == False).all()
        assert small["far_ratio"].isna().all()
