# -*- coding: utf-8 -*-
"""按加入"当季截至当前"尺度后的重跑结果，更新 Word 论文中的表4~表10。

只改数字文本（保留首个数字 run 的格式，如加粗与 * 标记），不动表格结构。
用法：python update_docx_tables.py [--dry]
"""
import os
import re
import sys
import shutil
import time

import docx

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "MS-TRP_中文双栏论文 .docx")
NUM = re.compile(r"^[\d.,—\-]+$")


def set_cell_number(cell, val):
    """把单元格里第一个纯数字 run 替换为 val（保留其余 run，如 * 标记）。"""
    for p in cell.paragraphs:
        for r in p.runs:
            if NUM.match(r.text.strip()) and r.text.strip():
                r.text = val
                return True
    # 没有纯数字 run：退化为整体替换首 run
    p = cell.paragraphs[0]
    if p.runs:
        p.runs[0].text = val
        for r in p.runs[1:]:
            r.text = ""
        return True
    p.add_run(val)
    return True


def update_row(table, label, values, col_start=1):
    for row in table.rows:
        if row.cells[0].text.strip() == label:
            for i, v in enumerate(values):
                set_cell_number(row.cells[col_start + i], v)
            return True
    print(f"  MISS row {label}")
    return False


def main():
    dry = "--dry" in sys.argv
    d = docx.Document(SRC)
    ts = d.tables

    print("表4 主表")
    update_row(ts[3], "MS-TRP", ["0.6006", "0.9447", "0.1624", "1.53", "0.1188"])
    update_row(ts[3], "Temporal-only", ["0.5734", "0.8601", "0.3203", "8.42", "0.1493"])

    print("表5 低误报召回")
    update_row(ts[4], "MS-TRP", ["0.4680", "0.6169"])
    update_row(ts[4], "Temporal-only", ["0.4615", "0.5440"])

    print("表6 六类事件召回")
    for label, v in [("金额突增", "0.1299"), ("频次突增", "0.7500"), ("进度滞后", "0.3333"),
                     ("年末集中", "0.3889"), ("历史偏移", "0.1875"), ("实体聚集", "0.4186")]:
        update_row(ts[5], label, [v], col_start=2)

    print("表7 消融")
    for label, vals in [
        ("完整模型", ["0.6006", "1.53", "0.4186"]),
        ("Temporal-only", ["0.5734", "1.58", "0.0930"]),
        ("时序与规则组合", ["0.5078", "5.84", "0.6279"]),
        ("时序与PPR组合", ["0.5944", "1.83", "0.0465"]),
        ("时序、规则与PPR组合", ["0.6005", "1.78", "0.3953"]),
        ("单尺度", ["0.4268", "1.19", "0.4186"]),
        ("固定边权", ["0.6011", "1.49", "0.4186"]),
        ("固定等权融合", ["0.6029", "2.67", "0.4419"]),
    ]:
        update_row(ts[6], label, vals)

    print("表8 冷启动")
    for label, vals in [
        ("≤30天", ["0.4070", "0.4819", "0.4503", "0.4887"]),
        ("31—180天", ["0.5473", "0.5288", "0.5676", "0.6757"]),
        (">180天", ["0.5635", "0.4344", "0.5697", "0.6154"]),
    ]:
        update_row(ts[7], label, vals)

    print("表10 季度FAR")
    for label, v in [("2026Q1", "0.88"), ("2026Q2", "0.57"),
                     ("2026Q3", "0.73"), ("2026Q4", "2.56")]:
        update_row(ts[9], label, [v], col_start=2)

    if dry:
        print("dry run, 未保存")
        return
    bak = os.path.join(HERE, f"MS-TRP_中文双栏论文_数字更新前_{time.strftime('%Y%m%d-%H%M%S')}.docx")
    shutil.copy2(SRC, bak)
    d.save(SRC)
    print("已保存:", SRC)
    print("备份:", bak)


if __name__ == "__main__":
    main()
