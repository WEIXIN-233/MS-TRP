"""
重建 result_raw/seed/summary.json
=================================
run_seeds 跑批被分批执行时（例如先跑 seed10/20，再单独补 seed30），
其内置汇总只覆盖本次入参。本脚本直接读取三次 data_seed × 五个 model_seed
落盘的 metrics.json，按 run_seeds.summarize 的同一口径重建 summary.json，
不重算任何模型、不改动任何指标。

用法：python -m experiment.rebuild_seed_summary
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from experiment import run_seeds

RUN_ROOT = os.path.join(config.RESULT_DIR, "seed")


def main():
    data_seeds = list(config.DATA_SEEDS)
    model_seeds = list(config.MODEL_SEEDS)
    rows = []
    missing = []
    for s in data_seeds:
        for ms in model_seeds:
            p = os.path.join(RUN_ROOT, f"seed{s}", f"model{ms}", "metrics.json")
            if not os.path.exists(p):
                missing.append(p)
                continue
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
            rows.append({
                "run_id": f"seed{s}_model{ms}",
                "data_seed": s,
                "model_seed": ms,
                "status": "completed",
                "point_auprc": m["point_auprc"],
                "point_auc": m["point_auc"],
                "normalized_ap_excess": m["normalized_ap_excess"],
                "event_strict_f1": m["event_strict"]["event_f1"],
                "event_strict_recall": m["event_strict"]["event_recall"],
                "far_per_1000": m["far_per_1000"],
                "thr_global": m.get("thr_global"),
            })

    if missing:
        print("[warn] 缺少 metrics.json：")
        for p in missing:
            print("   ", p)

    summary = run_seeds.summarize(rows)
    summary["source"] = "由 experiment/rebuild_seed_summary.py 从 seed{}/model{}/metrics.json 重建"
    out = os.path.join(RUN_ROOT, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for s in data_seeds:
        m = summary["seeds"].get(str(s))
        if m:
            print(f"seed {s}: AUPRC={m['point_auprc']['mean']:.6f}±{m['point_auprc']['std']:.2e} "
                  f"strictF1={m['event_strict_f1']['mean']:.4f} FAR/1000={m['far_per_1000']['mean']:.2f} "
                  f"(n={len(m['point_auprc']['values'])})")
    ov = summary["overall"].get("point_auprc", {})
    if ov:
        print(f"跨种子 AUPRC: mean={ov['mean']:.6f} [{ov['min']:.6f}, {ov['max']:.6f}]")
    print("已写入", out)


if __name__ == "__main__":
    main()
