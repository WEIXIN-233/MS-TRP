# -*- coding: utf-8 -*-
"""第二轮：更新 LaTeX 正文叙述中的实验数字（当季尺度加入后的重跑结果）。"""
import io

P = "latex/main.tex"

REPS = [
    # 摘要
    ("测试集上 MS-TRP 的 Payment-AUPRC 为 0.6001，高于 Isolation Forest 的 0.4143 与 Concat-IF 的 0.4008；"
     "冻结阈值下精确率 0.9252、召回率 0.1667、FAR/1000 为 2.18，属于低召回、高精度的工作点。"
     "把误报预算放宽到 10\\% 后，召回升到 0.6139。",
     "测试集上 MS-TRP 的 Payment-AUPRC 为 0.6006，高于 Isolation Forest 的 0.4143 与 Concat-IF 的 0.4008；"
     "冻结阈值下精确率 0.9447、召回率 0.1624、FAR/1000 为 1.53，属于低召回、高精度的工作点。"
     "把误报预算放宽到 10\\% 后，召回升到 0.6169。"),
    # 6.1 主结果
    ("MS-TRP 的 Payment-AUPRC 为 0.6001，高于 Temporal-only 的 0.5703、Isolation Forest 的 0.4143、"
     "Concat-IF 的 0.4008，在全部 21 种方法或参照里排第 1。相对 Fixed Rule 提高 0.1959。",
     "MS-TRP 的 Payment-AUPRC 为 0.6006，高于 Temporal-only 的 0.5734、Isolation Forest 的 0.4143、"
     "Concat-IF 的 0.4008，在全部 21 种方法或参照里排第 1。相对 Fixed Rule 提高 0.1964。"),
    # 冻结阈值工作点
    ("冻结阈值下，MS-TRP 精确率 0.9252、召回率 0.1667、FAR/1000 2.18。",
     "冻结阈值下，MS-TRP 精确率 0.9447、召回率 0.1624、FAR/1000 1.53。"),
    # 低误报
    ("误报率放到 10\\% 时，MS-TRP 召回 0.6139，高于 Temporal-only 的 0.5443、COPOD 的 0.4542、"
     "ECOD 的 0.4113；收紧到 5\\% 时召回 0.4738，仍高于各对照。",
     "误报率放到 10\\% 时，MS-TRP 召回 0.6169，高于 Temporal-only 的 0.5440、COPOD 的 0.4542、"
     "ECOD 的 0.4113；收紧到 5\\% 时召回 0.4680，仍高于各对照。"),
    # 表7 表注
    ("故Temporal-only的FAR/1000在表4为8.46（目标误报率口径）、本表为1.58（分层收缩口径）；"
     "MS-TRP两表均为2.18（同用分层收缩）。",
     "故Temporal-only的FAR/1000在表4为8.42（目标误报率口径）、本表为1.58（分层收缩口径）；"
     "MS-TRP两表均为1.53（同用分层收缩）。"),
    # 消融叙述
    ("实体聚集事件上，Temporal-only 召回 0.0930。只加账户一致性规则升到 0.6279，只加 PPR 扩散仍是 0.0930，"
     "规则与 PPR 联合落到 0.4186。",
     "实体聚集事件上，Temporal-only 召回 0.0930。只加账户一致性规则升到 0.6279，只加 PPR 扩散是 0.0465，"
     "规则与 PPR 联合落到 0.3953。"),
    ("时序加规则组合的 Payment-AUPRC 为 0.4970，比 Temporal-only 的 0.5703 低 0.0733，"
     "等于拿支付级排序换实体聚集覆盖。规则单独作用聚集召回 0.6279，与 PPR 联合后回到 0.4186，"
     "这个回落来自二者在分层收缩阈值口径下分数范围与阈值接口的相互作用，不是 PPR 抵消了规则的检出信号。",
     "时序加规则组合的 Payment-AUPRC 为 0.5078，比 Temporal-only 的 0.5734 低 0.0656，"
     "等于拿支付级排序换实体聚集覆盖。规则单独作用聚集召回 0.6279，与 PPR 联合后回到 0.3953，"
     "这个回落来自二者在分层收缩阈值口径下分数范围与阈值接口的相互作用，不是 PPR 抵消了规则的检出信号。"),
    ("阶段边权固定后 AUPRC 0.6006，与完整模型差 0.0005；门控换成固定等权融合后 0.6008，略高于完整模型。",
     "阶段边权固定后 AUPRC 0.6011，与完整模型差 0.0005；门控换成固定等权融合后 0.6029，略高于完整模型。"),
    # 7.x Temporal-only FAR
    ("Temporal-only 关掉关系分支、保留节奏与季节通道，FAR/1000 为 8.46，告警与事件指标都能正常算。",
     "Temporal-only 关掉关系分支、保留节奏与季节通道，FAR/1000 为 8.42，告警与事件指标都能正常算。"),
]


def main():
    with io.open(P, encoding="utf-8", newline="") as f:
        s = f.read().replace("\r\n", "\n")
    n = 0
    for old, new in REPS:
        if old not in s:
            print("MISS:", old[:60])
            continue
        s = s.replace(old, new, 1)
        n += 1
    with io.open(P, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(s)
    print(f"patched {n}/{len(REPS)}")


if __name__ == "__main__":
    main()
