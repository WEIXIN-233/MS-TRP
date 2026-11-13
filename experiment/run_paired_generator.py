# -*- coding: utf-8 -*-
"""审稿意见第6条（优先级3）：跨生成机制的数据级配对重复。

对 3 个数据种子 × 2 种独立生成机制（A=仿真V3，B=仿真V4）共 6 张库，
在同一批库上计算 MS-TRP 与 4 个基线的 Payment-AUPRC，按 (机制, 种子) 配对求差：
  Δ = AP(MS-TRP) − AP(基线)
报告每对的结果、均值与极差、Δ>0 的配对数。重复单位是数据，不是算法随机性。

MS-TRP 的 AP 取 result_raw/all_seed_results.csv（同一评估代码、阈值各自验证集冻结）。
输出：
  result_raw/paired_generator.csv   (mechanism, seed, method, AP, delta_vs_mstrp)
  result_raw/paired_generator.json  (每个基线的均值/极差/符号一致性)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, metrics, split  # noqa: E402
from src import feature_engineering as features  # noqa: E402
from experiment.run_baselines import BASELINE_FUNCS  # noqa: E402

SEEDS = [10, 20, 30]
MECH_DIR = {
    "A": "data_fixed/seed{seed}/budget_data_v5.db",
    "B": "data_fixed_b/seed{seed}/budget_data_v5.db",
}
PAIRED_BASELINES = ["isolation_forest", "concat_if", "fixed_rules", "simple_ppr"]


def baseline_aps():
    payments = data.load_payments()
    labels = data.load_labels()
    events = data.build_events(labels)
    point_labels = data.build_point_labels(payments, events)
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    y_test = point_labels["is_anomaly"].astype(int).values[test_idx]
    ref_stats = features.compute_reference_stats(payments, train_idx)
    out = {}
    for name in PAIRED_BASELINES:
        scores = BASELINE_FUNCS[name](payments, ref_stats, train_idx).values.astype(float)
        out[name] = metrics.auprc(y_test, scores[test_idx])
    return out


def main():
    ms = pd.read_csv(os.path.join(config.RESULT_DIR, "all_seed_results.csv"),
                     encoding="utf-8-sig")
    ms_map = {(r["mechanism"], int(r["seed"])): float(r["AP"]) for _, r in ms.iterrows()}

    rows = []
    for mech in ("A", "B"):
        for seed in SEEDS:
            db = os.path.join(config.PROJECT_DIR, MECH_DIR[mech].format(seed=seed))
            config.DB_PATH = db
            aps = baseline_aps()
            for name, ap in aps.items():
                rows.append({
                    "mechanism": mech, "seed": seed, "method": name,
                    "AP": round(ap, 6),
                    "delta_vs_mstrp": round(ms_map[(mech, seed)] - ap, 6),
                })
            print(mech, seed, {k: round(v, 4) for k, v in aps.items()}, flush=True)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(config.RESULT_DIR, "paired_generator.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = {}
    for name in PAIRED_BASELINES:
        d = df[df["method"] == name]["delta_vs_mstrp"].values
        summary[name] = {
            "mean_delta": round(float(d.mean()), 6),
            "min_delta": round(float(d.min()), 6),
            "max_delta": round(float(d.max()), 6),
            "n_pairs": int(len(d)),
            "n_mstrp_higher": int((d > 0).sum()),
        }
    out_json = os.path.join(config.RESULT_DIR, "paired_generator.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("written:", out_csv, out_json)


if __name__ == "__main__":
    main()
