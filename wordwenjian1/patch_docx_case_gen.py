# -*- coding: utf-8 -*-
"""更新 Word 段落 143（图6 五通道 + 新阈值）与 161（图8 跨机制数字）。"""
import os
import shutil
import time

import docx

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "MS-TRP_中文双栏论文 .docx")

E143 = [
    ("业务量与四个时序通道", "业务量与五个时序通道"),
    ("当年累计节奏通道三天内稳定在 0.58。",
     "当季截至当前通道首日已到 0.84，第二、三日升到 0.99 与 1.00，它从季初累计笔数，"
     "对季内持续偏快更敏感；当年累计节奏通道三天内稳定在 0.58。"),
    ("首个越过冻结阈值 0.8356 的日期是 10-02，s_final 为 0.942，10-03 升到 0.991。",
     "首个越过冻结阈值 0.8415 的日期是 10-02，s_final 为 0.974，10-03 升到 0.995。"),
]

E161 = [
    ("A 的 AUPRC 均值 0.5312（区间 0.4666 至 0.5786）",
     "A 的 AUPRC 均值 0.5323（区间 0.4619 至 0.5934）"),
    ("B 为 0.4926（区间 0.4640 至 0.5343），跨机制跌幅 7.26%",
     "B 为 0.5175（区间 0.5094 至 0.5259），跨机制跌幅 2.77%"),
]


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
    for idx, edits in ((143, E143), (161, E161)):
        p = ps[idx]
        t = p.text
        new = t
        for old, rep in edits:
            if old not in new:
                print(f"MISS [{idx}] {old[:40]}")
                continue
            new = new.replace(old, rep, 1)
            applied += 1
        if new != t:
            set_text_keep_fmt(p, new)
    bak = os.path.join(HERE, f"MS-TRP_中文双栏论文_案例图前_{time.strftime('%Y%m%d-%H%M%S')}.docx")
    shutil.copy2(SRC, bak)
    d.save(SRC)
    print(f"applied {applied}; bak {bak}")


if __name__ == "__main__":
    main()
