"""
B0 —— 把主库 run（data_fixed/budget_data_v5.db，data_seed=2026）追加进 run_registry.csv。
============================================================================================
run_seeds.py 只登记 3 个 data_seed（10/20/30）× 5 model_seed；主库 run_main 此前未登记。
本脚本补登记一行，口径与 run_seeds.py 完全一致（同列、同 selection_metric=payment_auprc、
single_run 披露）。指标从 result_raw/run_main/metrics.json 读取，data_sha256 现算主库哈希。
start/end 时间取 metrics.json 落盘时间（run_main 原始起止时间未单独记录，notes 注明）。
用法：python -m experiment.register_main_run
"""
import os
import sys
import csv
import json
import hashlib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

REGISTRY_PATH = os.path.join(config.RESULT_DIR, "run_registry.csv")
REGISTRY_COLS = [
    "run_id", "data_seed", "model_seed", "data_version", "data_sha256",
    "start_time", "end_time", "duration_s", "status",
    "point_auprc", "point_auc", "normalized_ap_excess", "event_strict_f1",
    "event_strict_recall", "far_per_1000", "n_train", "n_val", "n_test",
    "selection_metric", "reporting_regime", "notes",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    db = config.DB_PATH
    db_sha = sha256(db)
    m_path = os.path.join(config.RESULT_DIR, "run_main", "metrics.json")
    with open(m_path, encoding="utf-8") as f:
        m = json.load(f)

    mtime = datetime.fromtimestamp(os.path.getmtime(m_path)).isoformat(timespec="seconds")

    # 若已登记（幂等），跳过
    with open(REGISTRY_PATH, newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))
    if any(r.get("run_id") == "main_model1" for r in existing):
        print("main_model1 已在 run_registry.csv 中，跳过。")
        return

    row = {
        "run_id": "main_model1",
        "data_seed": 2026,
        "model_seed": 1,
        "data_version": "main",
        "data_sha256": db_sha,
        "start_time": mtime, "end_time": mtime, "duration_s": "",
        "status": "completed",
        "point_auprc": m["point_auprc"], "point_auc": m["point_auc"],
        "normalized_ap_excess": m["normalized_ap_excess"],
        "event_strict_f1": m["event_strict"]["event_f1"],
        "event_strict_recall": m["event_strict"]["event_recall"],
        "far_per_1000": m["far_per_1000"],
        "n_train": m["n_train"], "n_val": m["n_val"], "n_test": m["n_test"],
        "selection_metric": "payment_auprc", "reporting_regime": "single_run",
        "notes": "主库 run_main；start/end 取 metrics.json 落盘时间（原始起止未单独记录）",
    }
    with open(REGISTRY_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLS)
        w.writerow(row)
    print(f"已登记 main_model1  data_seed=2026  sha256={db_sha[:16]}  AUPRC={row['point_auprc']:.6f}")


if __name__ == "__main__":
    main()
