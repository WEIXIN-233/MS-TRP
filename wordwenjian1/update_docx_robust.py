# -*- coding: utf-8 -*-
"""更新 Word 表9（鲁棒性）数字。"""
import os
import re
import shutil
import time

import docx

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "MS-TRP_中文双栏论文 .docx")
NUM = re.compile(r"^[\d.,—\-]+$")


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
    t = d.tables[8]
    rows = {"无扰动": ["0.6006", "0.1560"],
            "字段缺失10%": ["0.6026", "0.1560"],
            "字段缺失30%": ["0.6014", "0.1552"],
            "删除图边20%": ["0.5998", "0.1673"]}
    for row in t.rows:
        lab = row.cells[0].text.strip()
        if lab in rows:
            for i, v in enumerate(rows[lab]):
                set_cell_number(row.cells[1 + i], v)
            print("ok", lab)
        elif lab and lab != "扰动条件":
            print("skip", lab)
    bak = os.path.join(HERE, f"MS-TRP_中文双栏论文_鲁棒表前_{time.strftime('%Y%m%d-%H%M%S')}.docx")
    shutil.copy2(SRC, bak)
    d.save(SRC)
    print("saved; bak", bak)


if __name__ == "__main__":
    main()
