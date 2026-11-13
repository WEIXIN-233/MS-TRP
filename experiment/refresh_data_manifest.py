"""
B0 —— 刷新 data_fixed/DATA_VERSION.json 使其与当前主库（budget_data_v5.db）一致。
====================================================================================
DATA_VERSION.json 由生成器写于旧库版本（split 39446/22302/23582），主库后经 P0-2 修复
与 2023 加入后重新生成（当前 36253/23094/25983）。本脚本按当前库重算：
  db_sha256 / db_size_mb / payments_total / split_counts / events_* / affected / cross_split
保留 generator/seed/note 等生成器元信息，更新 generated_at 与数据摘要。
用法：python -m experiment.refresh_data_manifest
"""
import os
import sys
import json
import hashlib
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, split


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    db = config.DB_PATH
    payments, labels, events, point_labels = None, None, None, None
    from experiment import _export_common as ec
    payments, labels, events, point_labels = ec.load_standard()
    tr, va, te = split.time_forward_split(payments)

    # 事件跨集合计数
    split_map = {}
    for name, ids in [("train", tr), ("val", va), ("test", te)]:
        for i in ids:
            split_map[str(payments.loc[i, "apply_id"])] = name
    cross = 0
    for _, ev in events.iterrows():
        g = {split_map[a] for a in ev["apply_ids"] if a in split_map}
        if len(g) > 1:
            cross += 1

    test_set = {str(payments.loc[i, "apply_id"]) for i in te}
    by_type = events["anomaly_type"].value_counts().to_dict()
    by_type_test = events[events["apply_ids"].apply(
        lambda ids: any(str(a) in test_set for a in ids))
    ]["anomaly_type"].value_counts().to_dict()

    affected_total = int(sum(len(ev["apply_ids"]) for _, ev in events.iterrows()))
    point_total = int(sum(len(ev["apply_ids"]) for _, ev in events.iterrows()
                          if ev["anomaly_type"] in ("AMOUNT_SPIKE", "FREQ_BURST")))

    out = {
        "data_version": "v2.1",
        "generator": "仿真V3.py",
        "seed": 2026,
        "generated_at": date.today().isoformat(),
        "db": os.path.basename(db),
        "db_sha256": sha256(db),
        "db_size_mb": round(os.path.getsize(db) / (1024 * 1024), 1),
        "payments_total": int(len(payments)),
        "split_counts": {"train": len(tr), "val": len(va), "test": len(te)},
        "events_total": int(len(events)),
        "cross_split_events": cross,
        "events_by_type": {k: int(v) for k, v in sorted(by_type.items())},
        "events_by_type_in_test": {k: int(v) for k, v in sorted(by_type_test.items())},
        "affected_payments_total": affected_total,
        "point_anomalies_total": point_total,
        "note": "P0-2 修复 + 2023 合法数据加入后重生成：事件成员锁定单一自然年（跨集合 0）；"
                "YEAREND_CONC/ENTITY_AGGREGATION 目标年份向 test(2026) 偏移。本文件由 "
                "experiment/refresh_data_manifest.py 按当前主库重算。",
    }
    p = os.path.join(config.PROJECT_DIR, "data_fixed", "DATA_VERSION.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[B0] 已刷新 {p}")
    print(f"  db_sha256={out['db_sha256'][:16]}…  split={out['split_counts']}  "
          f"events={out['events_total']}  cross={cross}")


if __name__ == "__main__":
    main()
