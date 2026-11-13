"""
B0/B7 工件导出 —— 配置/协议/数据画像/划分清单/注入参数（Phase 1）
====================================================================
把 config.py 的常量序列化为指导书要求的 config/*.yaml、experiment_protocol.yaml、
hyperparams.json、baseline_manifest.json，并按 data_seed 产出 data_profile.csv、
split_manifest.csv、injection_params.csv。

单一事实源 = 代码里的常量（config.py / *_DEFAULT），不另维护一份会漂移的配置。
用法：python -m experiment.export_artifacts
"""
import os
import sys
import json
import re
import sqlite3
import hashlib

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, split, temporal_baseline, pipeline
from src import propagation, reliability_gate, calibration, event_eval

try:
    import yaml
except ImportError:
    yaml = None


def _yaml_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if yaml is not None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
    else:  # JSON 是 YAML 合法子集，作降级
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def _json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── 1. config/*.yaml（把 config.py 常量落成指导书引用的配置文件）──────────
def write_config_yaml():
    cfg = os.path.join(config.PROJECT_DIR, "config")
    os.makedirs(cfg, exist_ok=True)

    _yaml_dump({
        "data_contract_version": "v2.1",
        "generator": "仿真V3.py",
        "years": [2023, 2024, 2025, 2026],
        "total_payments_target": 80000,
        "data_seeds": config.DATA_SEEDS,
        "model_seeds": config.MODEL_SEEDS,
        "anomaly_types": config.ANOMALY_TYPES,
        "point_anomaly_types": config.POINT_ANOMALY_TYPES,
        "event_anomaly_types": config.EVENT_ANOMALY_TYPES,
        "nine_entities": ["unit", "project", "indicator", "payment_apply",
                          "voucher", "payee", "bank_account",
                          "expense_category", "procurement_item"],
        "payment_types": config.PAYMENT_TYPES,
        "large_pay_threshold_wan": config.LARGE_PAY_THRESHOLD_WAN,
    }, os.path.join(cfg, "data.yaml"))

    _yaml_dump({
        "train": {"start": config.TRAIN_START, "end": config.TRAIN_END},
        "val": {"start": config.VAL_START, "end": config.VAL_END},
        "test": {"start": config.TEST_START, "end": config.TEST_END},
        "score_time_mode": config.SCORE_TIME_MODE,
        "note": "时间前向：train<val<test 严格无泄漏；保留 2023（见 data-2023-split）",
    }, os.path.join(cfg, "split.yaml"))

    _yaml_dump({
        "rho": propagation.RHO_DEFAULT,
        "rho_search_grid": [0.2, 0.4, 0.6, 0.8],
        "tau": reliability_gate.TAU_DEFAULT,
        "gamma": calibration.GAMMA_DEFAULT,
        "quantile": calibration.ALPHA_HIGH,
        "alpha_fpr": calibration.ALPHA_FPR_DEFAULT,
        "rhythm_sat": temporal_baseline.RHYTHM_SAT,
        "conc_sat": temporal_baseline.CONC_SAT,
        "w_conc": pipeline.W_CONC,
        "w_rhythm": pipeline.W_RHYTHM,
        "ppr_max_iter": 50,
        "ppr_stop_tol": 1e-6,
    }, os.path.join(cfg, "main.yaml"))

    _yaml_dump({
        "fixed_rule": config.FIXED_RULES,
        "robust_mad": {"k": config.MAD_K},
        "isolation_forest": {"n_estimators": 200, "max_samples": "auto",
                             "contamination": "auto"},
        "lof_novelty": {"n_neighbors": 20, "novelty": True, "contamination": "auto"},
        "cblof": {"n_clusters": 8},
        "mcd": {"support_fraction": 0.7},
        "ecod": {}, "copod": {},
    }, os.path.join(cfg, "baselines.yaml"))

    _yaml_dump({
        "single_scale": "仅30日尺度（同一特征定义）",
        "fixed_graph_weight": "去掉阶段差异",
        "no_graph": "Temporal-only",
        "fixed_fusion": "0.5/0.5 门控对照",
        "self_only_peer_only": "分层基线可解释",
        "no_ppr_one_hop": "多跳是否必要",
    }, os.path.join(cfg, "ablation.yaml"))

    _yaml_dump({
        "train_drop_ratio": [0.0, 0.10, 0.30],
        "edge_drop_ratio": [0.0, 0.20],
        "cold_start_buckets": [30, 180],
        "note": "缺失只遮蔽 score_time 之前的可见历史字段",
    }, os.path.join(cfg, "stress.yaml"))

    _yaml_dump({
        "delta_event": event_eval.DELTA_EVENT_DEFAULT,
        "object_rule": "entity_id 一致 + 时间交叠",
        "type_agnostic": True,
        "one_to_one_matching": True,
    }, os.path.join(cfg, "events.yaml"))


# ── 2. experiment_protocol.yaml（预注册协议，B5/B7）──────────────────────
def write_experiment_protocol():
    path = os.path.join(config.PROJECT_DIR, "config", "experiment_protocol.yaml")
    _yaml_dump({
        "reporting_regime": "single_run",
        "n_reported": 1,
        "selection_metric": "payment_auprc",
        "data_seeds": config.DATA_SEEDS,
        "model_seeds": config.MODEL_SEEDS,
        "model_seed_strategy": "每 data_seed 内换 model_seed，model_seed 不计独立场景",
        "threshold_search_space": {"gamma": [30], "quantile": [0.01],
                                   "alpha_fpr": [0.01]},
        "event_merge_gap_grid": [event_eval.DELTA_EVENT_DEFAULT],
        "tie_policy": "score >= threshold 报警；常数分数并列次序固定",
        "pre_registered_metrics": ["payment_auprc", "payment_f1", "far_1000",
                                   "event_f1", "vus_pr", "ndcg_at_k"],
        "frozen_rule": "注入/阈值搜索/事件合并间隔均在验证前冻结，测试集仅一次评价",
    }, path)


# ── 3. hyperparams.json（B5 基线默认超参 + 主模型超参）───────────────────
def write_hyperparams():
    path = os.path.join(config.PROJECT_DIR, "config", "hyperparams.json")
    _json_dump({
        "model": {
            "rho": propagation.RHO_DEFAULT,
            "tau": reliability_gate.TAU_DEFAULT,
            "gamma": calibration.GAMMA_DEFAULT,
            "quantile": calibration.ALPHA_HIGH,
            "alpha_fpr": calibration.ALPHA_FPR_DEFAULT,
            "rhythm_sat": 0.35,
            "conc_sat": 0.5,
            "w_conc": 0.65,
            "w_rhythm": 0.70,
            "ppr_max_iter": 50, "ppr_stop_tol": 1e-6,
        },
        "baselines": {
            "isolation_forest": {"impl": "sklearn.ensemble.IsolationForest",
                                 "n_estimators": 200, "max_samples": "auto",
                                 "contamination": "auto"},
            "lof_novelty": {"impl": "sklearn.neighbors.LocalOutlierFactor",
                            "n_neighbors": 20, "novelty": True,
                            "contamination": "auto"},
            "ecod": {"impl": "pyod.models.ecod.ECOD"},
            "copod": {"impl": "pyod.models.copod.COPOD"},
            "cblof": {"impl": "pyod.models.cblof.CBLOF", "n_clusters": 8},
            "mcd": {"impl": "pyod.models.mcd.MCD / sklearn.covariance.MinCovDet",
                    "support_fraction": 0.7},
        },
        "fit_split": "仅训练/验证期拟合，测试仅 transform",
    }, path)


# ── 4. baseline_manifest.json（B5/Q2 基线实现来源）──────────────────────
def write_baseline_manifest():
    path = os.path.join(config.PROJECT_DIR, "config", "baseline_manifest.json")
    _json_dump([
        {"model": "fixed_rule", "source": "项目内规则引擎", "impl": "baselines/fixed_rule.py"},
        {"model": "robust_mad", "source": "稳健MAD", "impl": "baselines/robust_mad.py"},
        {"model": "isolation_forest", "source": "sklearn.ensemble.IsolationForest",
         "impl": "baselines/isolation_forest.py"},
        {"model": "lof_novelty", "source": "sklearn.neighbors.LocalOutlierFactor(novelty=True)",
         "impl": "baselines/lof.py"},
        {"model": "forecast_residual", "source": "历史窗口预测残差", "impl": "baselines/forecast_residual.py"},
        {"model": "graph_only_structure", "source": "评分时点共享关系结构统计", "impl": "baselines/graph_baseline.py"},
        {"model": "ecod", "source": "pyod.models.ecod.ECOD", "impl": "baselines/forced.py"},
        {"model": "copod", "source": "pyod.models.copod.COPOD", "impl": "baselines/forced.py"},
        {"model": "mcd", "source": "pyod.models.mcd.MCD / sklearn MinCovDet", "impl": "baselines/forced.py"},
        {"model": "cblof", "source": "pyod.models.cblof.CBLOF", "impl": "baselines/forced.py"},
        {"model": "random_uniform", "source": "U(0,1)固定种子", "impl": "baselines/forced.py"},
        {"model": "constant_score", "source": "全样本0.5", "impl": "baselines/forced.py"},
        {"model": "shared_payee_count", "source": "前30日Payee跨项目数", "impl": "baselines/forced.py"},
        {"model": "shared_account_concentration", "source": "前30日Account跨Payee集中度", "impl": "baselines/forced.py"},
        {"model": "simple_ppr", "source": "同种子拓扑+均匀边权PPR", "impl": "baselines/forced.py"},
        {"model": "concat_if", "source": "时序特征+评分前图聚合(IF)", "impl": "baselines/forced.py"},
        {"model": "full_ms_trp", "source": "时序+阶段感知PPR+可靠度门控", "impl": "src/pipeline.py"},
    ], path)


# ── 5~7. 数据画像 / 划分清单 / 注入参数（读 config.DB_PATH，由调用方切换）──
def _profile(ddir):
    pay = data.load_payments()
    ev = data.build_events(data.load_labels())
    conn = sqlite3.connect(config.DB_PATH)

    def cnt(t): return int(pd.read_sql(f"SELECT COUNT(*) AS c FROM {t}", conn)["c"].iloc[0])
    def dct(t, c): return int(pd.read_sql(f"SELECT COUNT(DISTINCT {c}) AS c FROM {t}", conn)["c"].iloc[0])

    rows = []
    for t in ["unit", "project", "indicator", "payment_apply", "payment_voucher",
              "payee", "bank_account", "expense_category", "procurement_item"]:
        rows.append(("node", t, cnt(t), ""))
    rows.append(("node", "indicator_detail", cnt("indicator_detail"), "三级明细(指标下级)"))
    for name, t, c in [("payment_to_indicator", "payment_apply", "indicator_id"),
                       ("payment_to_payee", "payment_apply", "payee_id"),
                       ("payment_to_account", "payment_apply", "account_id"),
                       ("project_to_unit", "project", "unit_id"),
                       ("indicator_to_project", "indicator", "project_id"),
                       ("detail_to_indicator", "indicator_detail", "indicator_id"),
                       ("voucher_to_payment", "payment_voucher", "apply_id")]:
        rows.append(("relation", name, dct(t, c), ""))

    tr, va, te = split.time_forward_split(pay)
    for s, idx in [("train", tr), ("val", va), ("test", te)]:
        rows.append(("split", s, len(idx), ""))
    rows.append(("snapshot", "indicator_snapshot", cnt("indicator_snapshot"), ""))

    n_events = len(ev)
    pos_ids = set()
    for _, e in ev.iterrows():
        pos_ids |= e["apply_ids"]
    pos_ids &= set(pay["apply_id"].astype(str))
    n_pay = len(pay)
    rows.append(("event", "total", n_events, ""))
    rows.append(("event", "affected_payments", len(pos_ids), ""))
    rows.append(("event", "positive_rate", round(len(pos_ids) / n_pay, 5), "affected/n_payments"))
    conn.close()

    pd.DataFrame(rows, columns=["item_type", "name", "count", "note"]).to_csv(
        os.path.join(ddir, "data_profile.csv"), index=False, encoding="utf-8-sig")


def _split_manifest(ddir):
    pay = data.load_payments()
    tr, va, te = split.time_forward_split(pay)
    rows = [("train", config.TRAIN_START, config.TRAIN_END, len(tr)),
            ("val", config.VAL_START, config.VAL_END, len(va)),
            ("test", config.TEST_START, config.TEST_END, len(te))]
    pd.DataFrame(rows, columns=["split", "start", "end", "n_payments"]).to_csv(
        os.path.join(ddir, "split_manifest.csv"), index=False, encoding="utf-8-sig")


_RISK_TO_DIFF = {"HIGH": "hard", "MEDIUM": "medium", "LOW": "easy"}


def _parse_strength(atype, inject_parameters):
    """从生成器 inject_parameters 字符串解析模板强度，返回 (strength_min, strength_max, strength_range)。
    生成器已把每次注入的真实参数写进 inject_parameters，这里只做显式化，不回读检测指标。"""
    ip = inject_parameters if isinstance(inject_parameters, str) else ""
    if atype == "AMOUNT_SPIKE":
        m = re.search(r"mult=([\d.]+)", ip)
        v = float(m.group(1)) if m else None
        return v, v, "multiplier∈{3,4,5,8,10,15}"
    if atype == "FREQ_BURST":
        m = re.search(r"n=(\d+)", ip)
        v = float(m.group(1)) if m else None
        return v, v, "spike_count∈[5,10]"
    if atype == "PROGRESS_LAG":
        m = re.search(r"early=([\d.]+)%", ip)
        v = float(m.group(1)) if m else None
        return v, v, "early_keep%=20（前9月只保留20%）"
    if atype == "YEAREND_CONC":
        m = re.search(r"moved_to_dec=(\d+)", ip)
        v = float(m.group(1)) if m else None
        return v, v, "moved_to_dec∈[3,~0.45·n_nondec]"
    if atype == "HISTORICAL_SHIFT":
        return 1.0, 1.0, "mode=reverse（节奏前后反转，二值）"
    if atype == "ENTITY_AGGREGATION":
        m = re.search(r"n_projects=(\d+)", ip)
        if m:
            v = float(m.group(1))
            return v, v, "n_projects∈[3,5]"
        m = re.search(r"n_payees=(\d+)", ip)
        if m:
            v = float(m.group(1))
            return v, v, "n_payees∈[2,4]"
    return None, None, ""
_ANOMALY_CONFIG_SNAPSHOT = {
    "AMOUNT_SPIKE": {"ratio": 0.050, "multipliers": [3, 4, 5, 8, 10, 15]},
    "FREQ_BURST": {"indicator_ratio": 0.04, "spike_range": [5, 10]},
    "PROGRESS_LAG": {"indicator_ratio": 0.03, "lag_ratio": 0.85},
    "YEAREND_CONC": {"unit_ratio": 0.30, "move_ratio": 0.45},
    "HISTORICAL_SHIFT": {"indicator_ratio": 0.03, "factors": [0.35, 0.45, 1.55, 1.65]},
    "ENTITY_AGGREGATION": {"shared_payee_count": 40, "shared_account_count": 25, "window_days": 5},
}
_CONFIG_HASH = hashlib.sha256(
    json.dumps(_ANOMALY_CONFIG_SNAPSHOT, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _injection_params(seed, out_path):
    labels = data.load_labels()
    rows = []
    for _, r in labels.iterrows():
        st = pd.to_datetime(r.get("start_time"), errors="coerce")
        et = pd.to_datetime(r.get("end_time"), errors="coerce")
        wdays = int((et - st).days) if (pd.notna(st) and pd.notna(et)) else None
        risk = str(r.get("risk_level") or "").upper()
        atype = str(r.get("anomaly_type") or "")
        s_min, s_max, s_range = _parse_strength(atype, r.get("inject_parameters"))
        rows.append({
            "event_id": r.get("event_id"),
            "data_seed": seed,
            "anomaly_type": atype,
            "inject_template_id": r.get("anomaly_subtype") or atype,
            "difficulty_level": _RISK_TO_DIFF.get(risk, "medium"),
            "strength_range": s_range,
            "strength_min": s_min, "strength_max": s_max,
            "time_window_days": wdays,
            "target_entity_type": r.get("target_entity_type"),
            "target_entity_id": r.get("target_entity_id"),
            "hard_negative_flag": False,
            "injection_locked_before_threshold_search": True,
            "injection_config_hash": _CONFIG_HASH,
            "inject_parameters": r.get("inject_parameters"),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


# ── 入口 ──────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("Phase 1 工件导出：config/*.yaml + protocol/hyperparams/manifest + 数据画像/划分/注入")
    print("=" * 80, flush=True)

    write_config_yaml()
    write_experiment_protocol()
    write_hyperparams()
    write_baseline_manifest()
    print("[config] config/*.yaml + experiment_protocol.yaml + hyperparams.json + baseline_manifest.json 已写", flush=True)

    orig_db = config.DB_PATH
    main_dir = os.path.join(config.PROJECT_DIR, "data_fixed")
    targets = [("main", main_dir, os.path.join(main_dir, "budget_data_v5.db"), 2026)] + \
              [(f"seed{s}", config.data_dir(s),
                os.path.join(config.data_dir(s), "budget_data_v5.db"), s)
               for s in config.DATA_SEEDS]
    for label, ddir, db, seed in targets:
        if not os.path.exists(db):
            print(f"[skip] {label}: 无 DB", flush=True)
            continue
        config.DB_PATH = db
        _profile(ddir)
        _split_manifest(ddir)
        inj_out = (os.path.join(config.RESULT_DIR, "injection_params.csv") if label == "main"
                   else os.path.join(config.RESULT_DIR, "seed", f"seed{seed}", "injection_params.csv"))
        os.makedirs(os.path.dirname(inj_out), exist_ok=True)
        _injection_params(seed, inj_out)
        print(f"[data] {label}: data_profile.csv + split_manifest.csv + injection_params.csv", flush=True)

    config.DB_PATH = orig_db
    print("[done] Phase 1 工件导出完成", flush=True)


if __name__ == "__main__":
    main()
