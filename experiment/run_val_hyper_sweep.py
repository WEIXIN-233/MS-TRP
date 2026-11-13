"""
实验补充 —— 验证集上 rho / tau 选参（把"rho=0.6 验证集选出"落实为有依据的选择）
========================================
只用验证集(2025)选择 rho/tau，报告 val 与 test 的 AUPRC/AUC，并标出 val 上的最优值。
  rho 固定 tau=1.0 扫描 {0.2,0.4,0.6,0.8}；tau 固定 rho=0.6 扫描 {0.5,1.0,2.0,4.0}。
阈值无关的 AUPRC 作为选参指标（避免测试集泄漏：选参只看 val，test 仅作报告）。

输出 result_raw/protocol_v2_causal/sensitivity_significance/val_sweep.json
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, pipeline

OUT_DIR = os.path.join(config.RESULT_DIR, "protocol_v2_causal", "sensitivity_significance")

RHO_GRID = [0.2, 0.4, 0.6, 0.8]
TAU_GRID = [0.5, 1.0, 2.0, 4.0]


def _eval(pred):
    """从 pred 里抽 val/test 的 AUPRC/AUC（阈值无关）。"""
    out = {}
    for split in ("val", "test"):
        sub = pred[pred["split"] == split]
        y = sub["is_anomaly"].astype(int).values
        s = sub["s_final"].astype(float).values
        out[split] = {
            "auprc": float(average_precision_score(y, s)),
            "auc": float(roc_auc_score(y, s)),
        }
    return out


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if f != f else f
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    result = {"rho": [], "tau": []}

    print("=== rho 扫描（tau=1.0）===", flush=True)
    for rho in RHO_GRID:
        res, pred = pipeline.run_ms_trp(rho=rho)
        m = _eval(pred)
        result["rho"].append({"rho": rho, **m,
                              "test_strict_f1": res["event_strict"]["event_f1"]})
        print(f"  rho={rho}  valAUPRC={m['val']['auprc']:.4f}  "
              f"testAUPRC={m['test']['auprc']:.4f}  "
              f"strictF1={res['event_strict']['event_f1']:.4f}", flush=True)

    print("=== tau 扫描（rho=0.6）===", flush=True)
    for tau in TAU_GRID:
        res, pred = pipeline.run_ms_trp(tau=tau)
        m = _eval(pred)
        result["tau"].append({"tau": tau, **m,
                              "test_strict_f1": res["event_strict"]["event_f1"]})
        print(f"  tau={tau}  valAUPRC={m['val']['auprc']:.4f}  "
              f"testAUPRC={m['test']['auprc']:.4f}  "
              f"strictF1={res['event_strict']['event_f1']:.4f}", flush=True)

    # 选参结论：val 上 AUPRC 最优
    best_rho = max(result["rho"], key=lambda r: r["val"]["auprc"])
    best_tau = max(result["tau"], key=lambda r: r["val"]["auprc"])
    result["selected"] = {
        "best_rho": best_rho["rho"], "best_rho_val_auprc": best_rho["val"]["auprc"],
        "best_tau": best_tau["tau"], "best_tau_val_auprc": best_tau["val"]["auprc"],
    }
    print(f"\nval 最优：rho={best_rho['rho']} (valAUPRC={best_rho['val']['auprc']:.4f})  "
          f"tau={best_tau['tau']} (valAUPRC={best_tau['val']['auprc']:.4f})", flush=True)

    with open(os.path.join(OUT_DIR, "val_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(_clean(result), f, ensure_ascii=False, indent=2)
    print(f"已写入 {OUT_DIR}/val_sweep.json")


if __name__ == "__main__":
    main()
