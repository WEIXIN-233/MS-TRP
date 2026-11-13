"""
实验 E08 —— 多种子独立仿真（稳健性 / 95% CI）
========================================
目的：验证 MS-TRP 在"同一生成器、不同随机种子"下指标是否稳定，报告
      point AUPRC / AUC / AP 超额 / 事件严格 F1 等的均值±std 与 95% CI。

安全流程（避免污染 seed-2026 主结果库）：
  1. 备份当前 data_fixed/ → data_fixed_seed2026/
  2. 逐种子 仿真V3.py --seed N → pipeline.run_ms_trp() → 记录指标 + 存档该种子 DB
  3. 全部跑完后 恢复 data_fixed/（seed-2026）
  4. 汇总多种子 均值±std / 95% CI

输出：result_raw/protocol_v2_causal/seeds/seed{N}/metrics.json 与
      result_raw/protocol_v2_causal/seeds/summary.json
"""
import os
import sys
import json
import shutil
import hashlib
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, pipeline, data

DATA_DIR = os.path.join(config.PROJECT_DIR, "data_fixed")
BACKUP_DIR = os.path.join(config.PROJECT_DIR, "data_fixed_seed2026")
SEED_ROOT = os.path.join(config.PROJECT_DIR, "result_raw", "protocol_v2_causal", "seeds")
SEEDS = list(config.SEEDS)          # [10, 20, 30, 40, 50]（指导书推荐 5 组）


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, ensure_ascii=False, indent=2)


def cross_split_count():
    pay = data.load_payments()
    yr_of = dict(zip(pay["apply_id"].astype(str),
                     pd.to_datetime(pay["apply_date"]).dt.year))
    events = data.build_events(data.load_labels())
    return sum(1 for _, ev in events.iterrows()
               if len({yr_of.get(str(a)) for a in ev["apply_ids"]}) > 1)


def run_seed(seed: int) -> dict:
    seed_out = os.path.join(SEED_ROOT, f"seed{seed}")
    os.makedirs(seed_out, exist_ok=True)

    # 1. 重新生成该种子数据（覆盖 data_fixed/budget_data_v5.db + CSVs）
    log_path = os.path.join(seed_out, "generator.log")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run([sys.executable, "仿真V3.py", "--seed", str(seed)],
                       cwd=config.PROJECT_DIR, stdout=lf, stderr=subprocess.STDOUT,
                       check=True)

    # 2. 跑主实验（读刚生成的数据）
    res, _pred = pipeline.run_ms_trp()

    # 3. 跨集合审计 + 存档
    rec = {
        "seed": seed,
        "db_sha256": sha256(config.DB_PATH),
        "cross_split_events": cross_split_count(),
        "n_train": res["n_train"], "n_val": res["n_val"], "n_test": res["n_test"],
        "point_auprc": res["point_auprc"],
        "point_auc": res["point_auc"],
        "normalized_ap_excess": res["normalized_ap_excess"],
        "fpr": res["fpr"],
        "far_per_1000": res["far_per_1000"],
        "per_type_auprc": res["per_type_auprc"],
        "event_loose_recall": res["event"]["event_recall"],
        "event_loose_f1": res["event"]["event_f1"],
        "event_strict_recall": res["event_strict"]["event_recall"],
        "event_strict_precision": res["event_strict"]["event_precision"],
        "event_strict_f1": res["event_strict"]["event_f1"],
    }
    _dump(rec, os.path.join(seed_out, "metrics.json"))
    shutil.copy(config.DB_PATH, os.path.join(seed_out, "budget_data_v5.db"))
    shutil.copy(os.path.join(DATA_DIR, "ground_truth_event.csv"),
                os.path.join(seed_out, "ground_truth_event.csv"))

    print(f"[E08] seed={seed}  AUPRC={res['point_auprc']:.4f}  "
          f"AUC={res['point_auc']:.4f}  strictF1={res['event_strict']['event_f1']:.4f}  "
          f"cross_split={rec['cross_split_events']}", flush=True)
    return rec


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


def summarize(records):
    from scipy import stats
    flats = [_flat(r) for r in records]
    all_keys = sorted(set().union(*(set(f) for f in flats)))
    numeric = [k for k in all_keys
               if all(isinstance(f.get(k), (int, float)) and not isinstance(f.get(k), bool)
                      for f in flats)]
    out = {"n_seeds": len(records), "seeds": [r["seed"] for r in records], "metrics": {}}
    for k in numeric:
        vals = [float(f[k]) for f in flats]
        arr = np.asarray(vals)
        mean, std = float(arr.mean()), float(arr.std(ddof=1))
        n = len(vals)
        if n >= 2:
            tcrit = stats.t.ppf(0.975, n - 1)
            lo = mean - tcrit * std / np.sqrt(n)
            hi = mean + tcrit * std / np.sqrt(n)
        else:
            lo = hi = mean
        out["metrics"][k] = {
            "mean": mean, "std": std, "ci95_lo": lo, "ci95_hi": hi, "values": vals,
        }
    return out


def main():
    os.makedirs(SEED_ROOT, exist_ok=True)
    print("=" * 100)
    print(f"E08 多种子仿真：seeds={SEEDS}（安全流程，跑完恢复 seed-2026 库）")
    print("=" * 100, flush=True)

    # 1. 备份当前 seed-2026 数据
    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(DATA_DIR, BACKUP_DIR)
    print(f"[backup] seed-2026 已备份 → {BACKUP_DIR}  (db_sha256={sha256(config.DB_PATH)})", flush=True)

    records = []
    try:
        for seed in SEEDS:
            records.append(run_seed(seed))
    finally:
        # 3. 恢复 seed-2026 数据（无论成功失败都要恢复）
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        shutil.copytree(BACKUP_DIR, DATA_DIR)
        print(f"[restore] 已恢复 data_fixed/ → seed-2026  (db_sha256={sha256(config.DB_PATH)})", flush=True)

    # 4. 汇总
    summary = summarize(records)
    _dump(summary, os.path.join(SEED_ROOT, "summary.json"))
    print("\n" + "=" * 100)
    for k, m in summary["metrics"].items():
        if k in ("point_auprc", "point_auc", "normalized_ap_excess",
                 "event_strict_recall", "event_strict_precision", "event_strict_f1",
                 "fpr", "far_per_1000", "cross_split_events"):
            print(f"  {k:<24s} {m['mean']:.4f} ± {m['std']:.4f}  "
                  f"95%CI=[{m['ci95_lo']:.4f}, {m['ci95_hi']:.4f}]")
    print("=" * 100)
    print(f"[E08] 完成，结果在 {SEED_ROOT}/  （summary.json + seed{{N}}/metrics.json）")


if __name__ == "__main__":
    main()
