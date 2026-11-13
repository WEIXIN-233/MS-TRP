# -*- coding: utf-8 -*-
"""更新 Word 论文正文叙述中的实验数字（加入当季尺度后的重跑结果）。

只改段落文字，保留段落首个 run 的字体格式；每处替换都做存在性校验，避免误改。
"""
import os
import shutil
import time

import docx

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "MS-TRP_中文双栏论文 .docx")

# 段落号 → [(old, new, count)]，count 为该段内期望替换次数（None = 全部）
EDITS = {
    2: [("Payment-AUPRC 为 0.6001", "Payment-AUPRC 为 0.6006", 1),
        ("精确率 0.9252、召回率 0.1667、FAR/1000 为 2.18", "精确率 0.9447、召回率 0.1624、FAR/1000 为 1.53", 1),
        ("召回升到 0.6139", "召回升到 0.6169", 1)],
    98: [("Payment-AUPRC 为 0.6001", "Payment-AUPRC 为 0.6006", 1),
         ("Temporal-only 的 0.5703", "Temporal-only 的 0.5734", 1),
         ("提高 0.1959", "提高 0.1964", 1)],
    104: [("精确率 0.9252、召回率 0.1667、FAR/1000 2.18",
           "精确率 0.9447、召回率 0.1624、FAR/1000 1.53", 1)],
    106: [("MS-TRP 召回 0.6139", "MS-TRP 召回 0.6169", 1),
          ("Temporal-only 的 0.5443", "Temporal-only 的 0.5440", 1),
          ("召回 0.4738", "召回 0.4680", 1)],
    109: [("召回 0.1505", "召回 0.1299", 1),
          ("各为 0.7000、0.4444", "各为 0.7500、0.3889", 1),
          ("各为 0.3333、0.2500", "各为 0.3333、0.1875", 1)],
    110: [("六类分别命中约 182、14、7、8、4、18 个", "六类分别命中约 157、15、7、7、3、18 个", 1),
          ("合计 233 个命中", "合计 207 个命中", 1),
          ("微平均召回约 0.1756", "微平均召回约 0.1560", 1)],
    113: [("在表4为8.46", "在表4为8.42", 1),
          ("MS-TRP两表均为2.18（同用分层收缩）", "MS-TRP两表均为1.53（同用分层收缩）", 1)],
    114: [("AUPRC 从 0.6001 掉到 0.4268，绝对下降 0.1733",
           "AUPRC 从 0.6006 掉到 0.4268，绝对下降 0.1738", 1),
          ("完整模型 0.6001 高于 Temporal-only 的 0.5703",
           "完整模型 0.6006 高于 Temporal-only 的 0.5734", 1),
          ("金额尺度配 7日/30日频次窗口得 0.6001，配 5日/60日得 0.5808，配 90日/365日得 0.5299，"
           "窗口选择本身带来 0.0194 增益（按未取整 AUPRC 算）。7日/30日窗口和注入的异常模式更对齐，"
           "增益至少有部分来自窗口多样性，而非单纯的特征数量。",
           "金额尺度配 7日/30日频次窗口再加当季尺度得 0.6006，配 5日/60日加当季得 0.5932，"
           "配 90日/365日加当季得 0.5781，窗口选择本身带来 0.0074 增益（按未取整 AUPRC 算）。"
           "7日/30日窗口和注入的异常模式更对齐，增益有一部分来自窗口多样性，但幅度有限，"
           "不宜把等维对照当作强证据。当季尺度单因素的边际效应更小：去掉它，AUPRC 从 0.6006 变到 0.6001，"
           "排序几乎不变；它的作用主要在告警侧，完整模型冻结阈值下 FAR/1000 为 1.53，去掉当季后回到 2.18。", 1)],
    115: [("Payment-AUPRC 为 0.4970", "Payment-AUPRC 为 0.5078", 1),
          ("Temporal-only 的 0.5703 低 0.0733", "Temporal-only 的 0.5734 低 0.0656", 1),
          ("只加 PPR 扩散仍是 0.0930", "只加 PPR 扩散是 0.0465", 1),
          ("规则与 PPR 联合落到 0.4186", "规则与 PPR 联合落到 0.3953", 1),
          ("与 PPR 联合后回到 0.4186", "与 PPR 联合后回到 0.3953", 1)],
    116: [("阶段边权固定后 AUPRC 0.6006", "阶段边权固定后 AUPRC 0.6011", 1),
          ("固定等权融合后 0.6008", "固定等权融合后 0.6029", 1)],
    156: [("FAR/1000 为 8.46", "FAR/1000 为 8.42", 1)],
}


def set_text_keep_fmt(p, text):
    runs = p.runs
    if not runs:
        p.add_run(text)
        return
    first = runs[0]
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    first.text = text


def main():
    d = docx.Document(SRC)
    ps = d.paragraphs
    applied = 0
    for idx, edits in EDITS.items():
        p = ps[idx]
        t = p.text
        new = t
        for old, rep, cnt in edits:
            if old not in new:
                print(f"MISS [{idx}] {old[:40]}")
                continue
            new = new.replace(old, rep) if cnt is None else new.replace(old, rep, cnt)
            applied += 1
        if new != t:
            set_text_keep_fmt(p, new)
    bak = os.path.join(HERE, f"MS-TRP_中文双栏论文_正文数字前_{time.strftime('%Y%m%d-%H%M%S')}.docx")
    shutil.copy2(SRC, bak)
    d.save(SRC)
    print(f"applied {applied}; 备份 {bak}")


if __name__ == "__main__":
    main()
