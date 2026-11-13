# -*- coding: utf-8 -*-
"""把新图并入 docx：图2升级为三联图、图3补样本量、新增消融图与跨机制图。

图号调整：图1框架 / 图2组合性能 / 图3消融 / 图4历史跨度 / 图5季度误报 / 图6跨机制。
版式：新增与升级的图用连续分节（1栏）做通栏，与图1、表4的处理方式一致；
      原图3（历史跨度）保留单栏宽度。
"""
import copy
import os
import shutil
import time

import docx
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "MS-TRP_中文双栏论文 .docx")
FIG = os.path.join(HERE, "redraw")

ts = time.strftime("%Y%m%d-%H%M%S")
BACKUP = os.path.join(HERE, f"MS-TRP_中文双栏论文_补图前备份_{ts}.docx")
shutil.copy2(SRC, BACKUP)
print("备份:", BACKUP)


def set_text_keep_fmt(p, text):
    runs = p.runs
    if not runs:
        p.add_run(text)
        return
    first = runs[0]
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    first.text = text


def insert_sect_before(anchor, sectpr_template):
    p = anchor.insert_paragraph_before()
    pPr = p._p.get_or_add_pPr()
    pPr.append(copy.deepcopy(sectpr_template))
    return p


def insert_figure_block(anchor, img, width_in, caption, tmpl_caption, sect2, sect1):
    insert_sect_before(anchor, sect2)                 # 结束前面的双栏节
    pimg = anchor.insert_paragraph_before()
    pimg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pimg.add_run().add_picture(img, width=Inches(width_in))
    new_el = copy.deepcopy(tmpl_caption._p)           # 沿用原图题的字体/居中
    anchor._p.addprevious(new_el)
    pcap = Paragraph(new_el, anchor._parent)
    set_text_keep_fmt(pcap, caption)
    insert_sect_before(anchor, sect1)                 # 结束通栏节，恢复双栏


def replace_picture(p, img, width_in):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(img, width=Inches(width_in))


def main():
    d = docx.Document(SRC)
    ps = d.paragraphs
    p92, p100, p101 = ps[92], ps[100], ps[101]
    p114, p115 = ps[114], ps[115]
    p119, p120 = ps[119], ps[120]
    p130 = ps[130]
    p144, p145 = ps[144], ps[145]
    tmpl_caption = ps[38]                     # 图1图题，用作格式模板

    sect2 = p93_sect = ps[93]._p.pPr.find(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr")
    sect1 = ps[97]._p.pPr.find(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr")
    assert sect2 is not None and sect1 is not None, "分节模板缺失"

    # ① 7.3 跨机制图（插在“鲁棒性结果缺…”之前）
    insert_figure_block(
        p145,
        os.path.join(FIG, "fig_generator.png"),
        7.0,
        "图6 两种生成机制下MS-TRP的逐种子结果",
        tmpl_caption, sect2, sect1,
    )

    # ② 原图4 → 图5
    set_text_keep_fmt(p130, "图5 测试季度误报及验证集经验参照")

    # ③ 6.3 消融图（插在表7注释段之前）
    insert_figure_block(
        p115,
        os.path.join(FIG, "fig_ablation.png"),
        7.0,
        "图3 组件变体的排序性能与实体聚集召回",
        tmpl_caption, sect2, sect1,
    )

    # ④ 原图3 → 图4，并换成补样本量版本
    replace_picture(p119, os.path.join(FIG, "fig3_cold_start2.png"), 3.35)
    set_text_keep_fmt(p120, "图4 历史跨度分组的Payment-AUPRC")

    # ⑤ 图2 升级为通栏三联图
    insert_sect_before(p100, sect2)
    replace_picture(p100, os.path.join(FIG, "fig2_combined.png"), 7.0)
    set_text_keep_fmt(p101, "图2 主要方法的排序概览、PR曲线与低误报召回")
    insert_sect_before(ps[102], sect1)

    # ⑥ 正文补图引用
    set_text_keep_fmt(
        p92,
        "表4报告全部21种方法或参照的支付级与事件级结果，表5报告相同方法的误报预算诊断。"
        "Payment-AUPRC按降序排列，图2给出主要方法的排序概览、PR曲线与低误报召回曲线。"
        "所有列的最优数值按指标方向加粗，随机与常数参照不参与最优值比较。",
    )
    set_text_keep_fmt(
        p114,
        p114.text + "各变体的排序性能与实体聚集召回对照见图3。",
    )
    set_text_keep_fmt(
        p144,
        p144.text + "逐种子的配对结果见图6。",
    )

    d.save(SRC)
    print("已保存:", SRC)


if __name__ == "__main__":
    main()
