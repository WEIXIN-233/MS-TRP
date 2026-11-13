"""
实验 4 —— 案例研究（四类证据解释）
========================================
选取 2 个典型案例，用四类证据（特征/历史/规则/关系）解释报警原因：

  案例 1：真阳性 · 共享账户（ENTITY_AGGREGATION）—— 展示图分支与硬关系证据；
  案例 2：漏报 · 无账户不一致的实体聚集 —— 展示模型对"软"关系异常的局限。

输出 result_raw/run_case_study/case_study.txt。
"""
import os
import sys

import numpy as np
import pandas as pd

# 让脚本在 PyCharm(脚本模式)或命令行(python -m)下都能从项目根 MS-TRP/ 导入 src、baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, pipeline, rule_engine

OUT_DIR = os.path.join(config.RESULT_DIR, "run_case_study")


def _dump(name, row, buf):
    buf.append("=" * 78)
    buf.append(name)
    buf.append("=" * 78)
    buf.append(f"  apply_id     : {row['apply_id']}   金额 {float(row['amount_wan']):.2f} 万元")
    buf.append(f"  真实异常类型 : {row['anomaly_types'] or '(无)'}")
    buf.append(f"  最终风险 s   : {float(row['s_final']):.4f}")
    buf.append(f"  时序分量     : s_ts={float(row['s_ts']):.4f}  "
               f"(金额 s_amt={float(row['s_amt']):.4f} / 频次 s_freq={float(row['s_freq']):.4f} "
               f"/ 趋势 s_trend={float(row['s_trend']):.4f})")
    buf.append(f"  关系分量     : 账户不一致 s_mismatch={float(row['s_mismatch']):.2f}  "
               f"实体聚集 s_entity={float(row['s_entity_freq']):.4f}  "
               f"图扩散 s_diffuse={float(row['s_diffuse']):.4f}")
    buf.append(f"  门控权重     : g_ts={float(row['g_ts']):.3f}  g_g={float(row['g_g']):.3f}  "
               f"(可靠度 r_ts={float(row['r_ts']):.3f} / r_g={float(row['r_g']):.3f})")
    buf.append(f"  历史笔数     : {int(row['hist_count'])}   单位类型 {row['unit_type']}   "
               f"项目类别 {row['project_category']}  阶段 {row['stage']}")
    ev = rule_engine.explain(row)
    buf.append("  四类证据     :")
    if ev:
        for e in ev:
            buf.append(f"    [{e['type']:<8s}] {e['label']}  ——  {e['detail']}  (score={e['score']:.2f})")
    else:
        buf.append("    （无显著证据触发）")
    buf.append("")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("运行完整流程以生成案例所需预测…")
    res, pred = pipeline.run_ms_trp()
    test = pred[pred["split"] == "test"].copy()

    # 案例 1：真阳性 —— 共享账户（账户一致性违反 + 确实异常）
    tp = test[(test["s_mismatch"] > 0.5) & (test["is_anomaly"] == 1)]
    if len(tp) == 0:
        tp = test[test["is_anomaly"] == 1].nlargest(5, "s_final")
    case1 = tp.nlargest(1, "s_final").iloc[0]

    # 案例 2：漏报 —— 实体聚集但无账户不一致（低分异常）
    fn = test[(test["is_anomaly"] == 1)
              & (test["anomaly_types"].str.contains("ENTITY_AGGREGATION", na=False))
              & (test["s_mismatch"] <= 0.5)]
    if len(fn) == 0:
        fn = test[test["is_anomaly"] == 1].nsmallest(5, "s_final")
    case2 = fn.nsmallest(1, "s_final").iloc[0]

    buf = ["MS-TRP 案例研究（四类证据解释）", ""]
    _dump("案例 1 · 真阳性：共享账户关系异常（图分支命中）", case1, buf)
    _dump("案例 2 · 漏报：无账户不一致的实体聚集（模型局限）", case2, buf)

    text = "\n".join(buf)
    print(text)
    with open(os.path.join(OUT_DIR, "case_study.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"结果已写入 {OUT_DIR}/case_study.txt")


if __name__ == "__main__":
    main()
