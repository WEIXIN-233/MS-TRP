# -*- coding: utf-8 -*-
"""更新 Word 表10（季度误报）、叙述段落与验证集经验值 1.42→1.36。"""
import os
import re
import shutil
import time

import docx

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "MS-TRP_中文双栏论文 .docx")
NUM = re.compile(r"^[\d.,—\-]+$")

OLD137 = ("表10的正常支付总数为20,202，与测试集23,465条支付对应的阳性占比一致。"
          "四个季度的FAR/1000介于0.86至3.57。第二季度与第三季度低于验证集经验值1.42，"
          "第一季度为1.75，第四季度为3.57，首尾两个季度高于验证集经验值。"
          "正常样本量在季度间变化较大，单季度误报数量较少，季度差异不宜被过度解释为风险机制变化。")
NEW137 = ("表10的正常支付总数为20,202，与测试集23,465条支付对应的阳性占比一致。"
          "四个季度的FAR/1000介于0.57至2.56。第一至第三季度分别为0.88、0.57、0.73，"
          "均低于验证集经验值1.36，第四季度为2.56，高于该经验值。"
          "正常样本量在季度间变化较大，单季度误报数量较少，季度差异不宜被过度解释为风险机制变化。")


def set_text_keep_fmt(p, text):
    runs = p.runs
    if not runs:
        p.add_run(text)
        return
    first = runs[0]
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    first.text = text


def set_cell_number(cell, val):
    for p in cell.paragraphs:
        for r in p.runs:
            if NUM.match(r.text.strip()) and r.text.strip():
                r.text = val
                return True
    p = cell.paragraphs[0]
    if p.runs:
        p.runs[0].text = val
        for r in p.runs[1:]:
            r.text = ""
        return True
    p.add_run(val)
    return True


def main():
    d = docx.Document(SRC)
    ps = d.paragraphs

    # 叙述：优先按原文精确替换，找不到则整段替换（保留格式）
    p137 = ps[137]
    if OLD137 in p137.text:
        set_text_keep_fmt(p137, p137.text.replace(OLD137, NEW137))
        print("p137 replaced (exact)")
    else:
        print("p137 exact not found; head:", p137.text[:60])
        set_text_keep_fmt(p137, NEW137)
        print("p137 replaced (whole)")

    p138 = ps[138]
    n = p138.text.count("1.42")
    if n:
        set_text_keep_fmt(p138, p138.text.replace("1.42", "1.36"))
    print("p138 occurrences replaced:", n)

    # 表10：季度误报表（第 9 张表）
    t = d.tables[9]
    vals = {"2026Q1": "0.88", "2026Q2": "0.57", "2026Q3": "0.73", "2026Q4": "2.56"}
    hit = 0
    for row in t.rows:
        lab = row.cells[0].text.strip()
        if lab in vals:
            set_cell_number(row.cells[2], vals[lab])
            hit += 1
            print("table row", lab, "->", vals[lab])
    print("table rows updated:", hit)

    bak = os.path.join(HERE, f"MS-TRP_中文双栏论文_季度误报前_{time.strftime('%Y%m%d-%H%M%S')}.docx")
    shutil.copy2(SRC, bak)
    d.save(SRC)
    print("saved; bak", bak)


if __name__ == "__main__":
    main()
