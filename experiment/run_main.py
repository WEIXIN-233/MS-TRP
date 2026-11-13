"""
实验 1 —— MS-TRP 主模型端到端评估
========================================
运行完整主流程（Step 1~8），打印点级/事件级/各类别指标，
并把预测结果与指标写入 result_raw/run_main/。
"""
import os
import sys
import json

import numpy as np
import pandas as pd

# 让脚本在 PyCharm(脚本模式)或命令行(python -m)下都能从项目根 MS-TRP/ 导入 src、baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, pipeline

OUT_DIR = os.path.join(config.RESULT_DIR, "run_main")


def _fmt_table(rows, headers):
    w = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(headers))
    sep = "  " + "  ".join("-" * w[i] for i in range(len(headers)))
    body = "\n".join("  " + "  ".join(str(r[i]).ljust(w[i]) for i in range(len(headers))) for r in rows)
    return line + "\n" + sep + "\n" + body


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("MS-TRP 主模型评估  (rho=0.6, tau=1.0, gamma=30, quantile=0.01)")
    print("=" * 78)

    results, pred = pipeline.run_ms_trp()

    print(f"\n数据规模  train={results['n_train']}  val={results['n_val']}  test={results['n_test']}")
    print(f"PPR 收敛  迭代 {results['ppr_iters']} 轮   校准阈值 τ={results['thr_global']:.4f}\n")

    # 阈值无关（label=全量 6 类逐笔；point/event 分开的口径见 experiment/compare.py）
    print("【阈值无关 · 全量口径(all=6类)】")
    print(_fmt_table([
        ["AUPRC(all=6类)", f"{results['point_auprc']:.4f}"],
        ["ROC-AUC", f"{results['point_auc']:.4f}"],
        ["归一化AP超额", f"{results['normalized_ap_excess']:.4f}"],
    ], ["指标", "值"]))

    # 校准阈值 —— MS-TRP 自身部署操作点，非跨方法对比口径
    pm = results["point"]
    print("\n【MS-TRP 部署操作点 @分层校准阈值（仅自身，非跨方法对比口径）】")
    print(_fmt_table([
        ["precision", f"{pm['precision']:.4f}"],
        ["recall", f"{pm['recall']:.4f}"],
        ["f1", f"{pm['f1']:.4f}"],
        ["tp / fp / fn", f"{pm['tp']} / {pm['fp']} / {pm['fn']}"],
        ["FAR/1000（FP/正常支付×1000）", f"{results['far_per_1000']:.2f}"],
        ["误报/千笔全部（旁证）", f"{results['fp_per_1000_all']:.2f}"],
    ], ["指标", "值"]))

    # 目标 FPR 操作点（P0-1 协议 A：验证集正常样本 quantile(1−alpha_fpr) 冻结阈值）
    pf = results["point_at_target_fpr"]
    print("\n【协议A · 目标FPR操作点 @冻结阈值（α_fpr=1%，与报警预算协议B并列）】")
    print(_fmt_table([
        ["τ_fpr（冻结）", f"{results['tau_fpr']:.4f}"],
        ["precision", f"{pf['precision']:.4f}"],
        ["recall", f"{pf['recall']:.4f}"],
        ["f1", f"{pf['f1']:.4f}"],
        ["tp / fp / fn", f"{pf['tp']} / {pf['fp']} / {pf['fn']}"],
        ["FAR/1000", f"{pf['far_per_1000']:.2f}"],
    ], ["指标", "值"]))

    # 每类 AUPRC 与召回（召回@校准阈值，仅自身口径）
    print("\n【每类异常 · AUPRC / 召回@校准阈值】")
    rows = []
    for at in config.ANOMALY_TYPES:
        rows.append([at, f"{results['per_type_auprc'][at]:.4f}",
                     f"{results['per_class_recall'][at]:.4f}"])
    print(_fmt_table(rows, ["异常类型", "AUPRC", "召回@τ"]))

    # 事件级（@校准阈值，仅自身口径）
    ev = results["event"]
    print("\n【事件级 @校准阈值（仅自身）】")
    print(_fmt_table([
        ["事件召回", f"{ev['event_recall']:.4f}  ({ev['caught']}/{ev['n_events']})"],
        ["事件精确", f"{ev['event_precision']:.4f}"],
        ["事件 F1", f"{ev['event_f1']:.4f}"],
    ], ["指标", "值"]))
    rows = [[at, f"{results['per_type_event_recall'][at]:.4f}"]
            for at in config.ANOMALY_TYPES]
    print(_fmt_table(rows, ["异常类型", "事件召回"]))

    # 事件级严格一对一匹配（P0-4：实体一致 + 时间交叠，杜绝重复命中虚高）
    es = results["event_strict"]
    print("\n【事件级严格一对一匹配（P0-4）】")
    print(_fmt_table([
        ["TP / FN / FP", f"{es['tp']} / {es['fn']} / {es['fp']}"],
        ["事件召回（严格）", f"{es['event_recall']:.4f}"],
        ["事件精确（严格）", f"{es['event_precision']:.4f}"],
        ["事件 F1（严格）", f"{es['event_f1']:.4f}"],
    ], ["指标", "值"]))

    # 写盘
    pred.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已写入 {OUT_DIR}/predictions.csv 与 metrics.json")
    print("跨方法对比（统一标签口径 + 统一 top-k 操作点）请运行：python -m experiment.compare")


if __name__ == "__main__":
    main()
