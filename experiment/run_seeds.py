"""
实验 —— 多种子 × 多模型种子 正式跑批（指导书 B0/B5/B7 口径）
================================================================
每个独立 data_seed（默认 10/20/30）下，换 5 个 model_seed 各跑一次完整
MS-TRP 主流程，逐次落盘 predictions.csv + metrics.json，并写 run_registry.csv
留痕（不 best-of-N 挑结果，全部披露）。最后按 data_seed 汇总 5 个 model_seed 的
均值±std。

数据按 data_fixed/seed{N}/ 独立归档（与主库 data_fixed/ 隔离，不覆盖）。
用法：
    python -m experiment.run_seeds
    python -m experiment.run_seeds --data-seeds 10 --model-seeds 1,2,3,4,5
"""
import os
import sys
import csv
import json
import time
import shutil
import hashlib
import subprocess
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, pipeline

RUN_ROOT = os.path.join(config.RESULT_DIR, "seed")
REGISTRY_PATH = os.path.join(config.RESULT_DIR, "run_registry.csv")

REGISTRY_COLS = [
    "run_id", "data_seed", "model_seed", "data_version", "data_sha256",
    "start_time", "end_time", "duration_s", "status",
    "point_auprc", "point_auc", "normalized_ap_excess", "event_strict_f1",
    "event_strict_recall", "far_per_1000", "n_train", "n_val", "n_test",
    "selection_metric", "reporting_regime", "notes",
]


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


def ensure_data(seed: int) -> str:
    """确保 data_fixed/seed{N}/budget_data_v5.db 存在；缺则调用生成器生成。"""
    ddir = config.data_dir(seed)
    db = os.path.join(ddir, "budget_data_v5.db")
    if os.path.exists(db):
        return db
    os.makedirs(ddir, exist_ok=True)
    log_path = os.path.join(ddir, "generator.log")
    print(f"[gen] 生成 data_seed={seed} → {ddir} ...", flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run([sys.executable, "仿真V3.py", "--seed", str(seed), "--out", ddir],
                       cwd=config.PROJECT_DIR, stdout=lf, stderr=subprocess.STDOUT,
                       check=True)
    return db


def data_version(seed: int) -> str:
    p = os.path.join(config.data_dir(seed), "DATA_VERSION.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            try:
                d = json.load(f)
                return str(d.get("data_version") or d.get("version") or "unknown")
            except Exception:
                pass
    # 生成器暂不逐种子写 DATA_VERSION.json；用 seed 作标签，唯一性靠 data_sha256 保证
    return f"seed{seed}"


def run_one(seed: int, model_seed: int, db_sha: str, writer) -> dict:
    out_dir = os.path.join(RUN_ROOT, f"seed{seed}", f"model{model_seed}")
    os.makedirs(out_dir, exist_ok=True)
    run_id = f"seed{seed}_model{model_seed}"
    start = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    try:
        res, pred = pipeline.run_ms_trp(data_seed=seed, model_seed=model_seed)
        status = "completed"
        pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
        _dump(res, os.path.join(out_dir, "metrics.json"))
        row = {
            "run_id": run_id, "data_seed": seed, "model_seed": model_seed,
            "data_version": data_version(seed), "data_sha256": db_sha,
            "start_time": start, "end_time": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - t0, 2), "status": status,
            "point_auprc": res["point_auprc"], "point_auc": res["point_auc"],
            "normalized_ap_excess": res["normalized_ap_excess"],
            "event_strict_f1": res["event_strict"]["event_f1"],
            "event_strict_recall": res["event_strict"]["event_recall"],
            "far_per_1000": res["far_per_1000"],
            "n_train": res["n_train"], "n_val": res["n_val"], "n_test": res["n_test"],
            "selection_metric": "payment_auprc", "reporting_regime": "single_run",
            "notes": "",
        }
    except Exception as e:  # 单次失败不中断整批
        status = "failed"
        row = {
            "run_id": run_id, "data_seed": seed, "model_seed": model_seed,
            "data_version": data_version(seed), "data_sha256": db_sha,
            "start_time": start, "end_time": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - t0, 2), "status": status,
            "point_auprc": None, "point_auc": None, "normalized_ap_excess": None,
            "event_strict_f1": None, "event_strict_recall": None,
            "far_per_1000": None, "n_train": None, "n_val": None, "n_test": None,
            "selection_metric": "payment_auprc", "reporting_regime": "single_run",
            "notes": repr(e)[:200],
        }
    writer.writerow(row)
    print(f"[run] seed={seed} model={model_seed} status={status}  "
          f"AUPRC={row['point_auprc']}  strictF1={row['event_strict_f1']}", flush=True)
    return row


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


def summarize(all_rows):
    """按 data_seed 汇总 5 个 model_seed 的均值±std（及逐值）。"""
    from scipy import stats
    key_metrics = ["point_auprc", "point_auc", "normalized_ap_excess",
                   "event_strict_f1", "event_strict_recall", "far_per_1000"]
    by_seed = {}
    for r in all_rows:
        if r["status"] != "completed":
            continue
        by_seed.setdefault(r["data_seed"], []).append(r)
    out = {"n_data_seeds": len(by_seed), "n_model_seeds_per_seed": config.MODEL_SEEDS,
           "seeds": {}, "overall": {}}
    for seed, rows in sorted(by_seed.items(), key=lambda kv: kv[0]):
        m = {}
        for k in key_metrics:
            vals = [float(r[k]) for r in rows if r.get(k) is not None]
            if not vals:
                continue
            arr = np.asarray(vals)
            mean, std = float(arr.mean()), float(arr.std(ddof=1)) if len(vals) > 1 else 0.0
            n = len(vals)
            if n >= 2:
                tcrit = stats.t.ppf(0.975, n - 1)
                lo = mean - tcrit * std / np.sqrt(n)
                hi = mean + tcrit * std / np.sqrt(n)
            else:
                lo = hi = mean
            m[k] = {"mean": mean, "std": std, "ci95_lo": lo, "ci95_hi": hi,
                    "values": vals}
        out["seeds"][str(seed)] = m
    # 跨 data_seed 汇总（报告逐种子值 + 均值/范围，指导书 B0 口径）
    overall = {}
    for k in key_metrics:
        per_seed_means = [m[k]["mean"] for m in out["seeds"].values() if k in m]
        if not per_seed_means:
            continue
        arr = np.asarray(per_seed_means)
        overall[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max()),
            "per_seed_means": per_seed_means,
        }
    out["overall"] = overall
    return out


def main():
    ds_list = [int(x) for x in os.environ.get("DATA_SEEDS", "").split(",") if x.strip()] or config.DATA_SEEDS
    ms_list = [int(x) for x in os.environ.get("MODEL_SEEDS", "").split(",") if x.strip()] or config.MODEL_SEEDS
    # 兼容命令行 --data-seeds / --model-seeds
    for flag, key in (("--data-seeds", "DATA_SEEDS"), ("--model-seeds", "MODEL_SEEDS")):
        if flag in sys.argv:
            os.environ[key] = sys.argv[sys.argv.index(flag) + 1]
            if key == "DATA_SEEDS":
                ds_list = [int(x) for x in os.environ[key].split(",") if x.strip()]
            else:
                ms_list = [int(x) for x in os.environ[key].split(",") if x.strip()]

    os.makedirs(RUN_ROOT, exist_ok=True)
    print("=" * 96)
    print(f"多种子 × 多模型种子跑批：data_seeds={ds_list}  model_seeds={ms_list}  "
          f"(共 {len(ds_list) * len(ms_list)} 次)")
    print("=" * 96, flush=True)

    all_rows = []
    new_file = not os.path.exists(REGISTRY_PATH)
    with open(REGISTRY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLS)
        if new_file:
            writer.writeheader()
        for seed in ds_list:
            db = ensure_data(seed)
            db_sha = sha256(db)
            print(f"[data] seed={seed} db_sha256={db_sha[:16]}...", flush=True)
            for ms in ms_list:
                all_rows.append(run_one(seed, ms, db_sha, writer))

    summary = summarize(all_rows)
    _dump(summary, os.path.join(RUN_ROOT, "summary.json"))

    print("\n" + "=" * 96)
    for seed, m in sorted(summary["seeds"].items(), key=lambda kv: int(kv[0])):
        print(f"seed {seed}:  AUPRC={m['point_auprc']['mean']:.4f}±{m['point_auprc']['std']:.4f}  "
              f"strictF1={m['event_strict_f1']['mean']:.4f}±{m['event_strict_f1']['std']:.4f}  "
              f"FAR/1000={m['far_per_1000']['mean']:.2f}")
    print("=" * 96)
    print(f"完成：{sum(1 for r in all_rows if r['status']=='completed')}/{len(all_rows)} 次成功")
    print(f"结果：{RUN_ROOT}/seed{{N}}/model{{M}}/  +  run_registry.csv  +  summary.json")


if __name__ == "__main__":
    main()
