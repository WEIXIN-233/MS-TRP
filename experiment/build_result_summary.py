"""
Phase 5 —— 汇总 result_summary/（指导书 §10.2 / line 318 交付目录）
====================================================================
把分散在 result_raw/ 的规范产物复制到 result_summary/ 单目录交付，
并写 result_summary/artifact_command_map.csv（产物 → 复现命令/来源脚本）。
缺失文件跳过并告警（如 ablation/robustness 仍在后台重跑）。

产物清单：
  metrics_main.csv / metrics_range.csv / metrics_ranking.csv / metrics_family.csv
  heuristic_structure_metrics.csv / diagnostics_pa.csv / table_feature_isolation.csv
  vus_pr_null_check.csv / paired_effects.csv / cold_start.csv / category.csv
  event_metrics.csv / ablation.csv / robustness.csv
  feature_column_manifest.json

用法：python -m experiment.build_result_summary
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

SUMMARY_DIR = os.path.join(config.PROJECT_DIR, "result_summary")

# (目标文件名, 源路径, 复现命令)
ARTIFACTS = [
    ("metrics_main.csv", os.path.join(config.RESULT_DIR, "metrics_main.csv"),
     "python -m experiment.build_metrics_main"),
    ("metrics_range.csv", os.path.join(config.RESULT_DIR, "metrics_range.csv"),
     "python -m experiment.build_metrics_main"),
    ("metrics_ranking.csv", os.path.join(config.RESULT_DIR, "metrics_ranking.csv"),
     "python -m experiment.build_metrics_main"),
    ("metrics_family.csv", os.path.join(config.RESULT_DIR, "metrics_family.csv"),
     "python -m experiment.run_summary_metrics"),
    ("heuristic_structure_metrics.csv",
     os.path.join(config.RESULT_DIR, "heuristic_structure_metrics.csv"),
     "python -m experiment.run_struct_heuristics"),
    ("diagnostics_pa.csv", os.path.join(config.RESULT_DIR, "diagnostics_pa.csv"),
     "python -m experiment.run_pa_diagnostics"),
    ("table_feature_isolation.csv",
     os.path.join(config.RESULT_DIR, "table_feature_isolation.csv"),
     "python -m experiment.run_table_isolation"),
    ("feature_column_manifest.json",
     os.path.join(config.RESULT_DIR, "feature_column_manifest.json"),
     "python -m experiment.run_table_isolation"),
    ("vus_pr_null_check.csv", os.path.join(config.RESULT_DIR, "vus_pr_null_check.csv"),
     "python -m experiment.run_vus_null"),
    ("paired_effects.csv", os.path.join(config.RESULT_DIR, "paired_effects.csv"),
     "python -m experiment.run_summary_metrics"),
    ("cold_start.csv", os.path.join(config.RESULT_DIR, "cold_start.csv"),
     "python -m experiment.run_cold_start"),
    ("category.csv", os.path.join(config.RESULT_DIR, "category.csv"),
     "python -m experiment.run_summary_metrics"),
    ("event_metrics.csv", os.path.join(config.RESULT_DIR, "event_metrics.csv"),
     "python -m experiment.run_summary_metrics"),
    ("ablation.csv", os.path.join(config.RESULT_DIR, "run_ablation", "ablation.csv"),
     "python -m experiment.run_ablation"),
    ("robustness.csv", os.path.join(config.RESULT_DIR, "run_robustness", "robustness.csv"),
     "python -m experiment.run_robustness"),
    ("gate_response.csv", os.path.join(config.RESULT_DIR, "gate_response.csv"),
     "python -m experiment.run_gate_response"),
]


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    map_rows, missing = [], []
    for name, src, cmd in ARTIFACTS:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SUMMARY_DIR, name))
            map_rows.append({"artifact": name, "source": src, "command": cmd,
                             "status": "copied"})
        else:
            missing.append(name)
            map_rows.append({"artifact": name, "source": src, "command": cmd,
                             "status": "missing"})

    import pandas as pd
    pd.DataFrame(map_rows).to_csv(os.path.join(SUMMARY_DIR, "artifact_command_map.csv"),
                                  index=False, encoding="utf-8-sig")

    print(f"[summary] 已写入 {SUMMARY_DIR}/")
    for r in map_rows:
        print(f"  {r['status']:<8s} {r['artifact']}")
    if missing:
        print(f"\n[warning] 缺失 {len(missing)} 个产物：{missing}")


if __name__ == "__main__":
    main()
