# -*- coding: utf-8 -*-
"""新增 6.8 典型事件的证据链：图6 多尺度案例 + 图7 关系证据对照。

图号：图6（案例）与图7（关系）插在 6.8；原图6（跨机制）顺延为图8。
版式：与既有通栏图一致，用连续分节（1栏）包住图片与图题。
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
BACKUP = os.path.join(HERE, f"MS-TRP_中文双栏论文_案例图前备份_{ts}.docx")
shutil.copy2(SRC, BACKUP)
print("备份:", BACKUP)

SECT = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr"

HEADING = "6.8 典型事件的证据链"

P1 = (
    "前面的结果都是集合层面的。这里用测试期一个频次突增事件看证据链如何对齐。"
    "明细 DT00004556 在 2026-10-01 至 10-03 三天内对同一收款人 PY000473 支付 9 笔、"
    "合计 352.2 万元；此前 12 个月该明细只有 3 笔支付，历史单笔中位数为 11.0 万元。"
    "图6按日给出业务量与四个时序通道：金额长尺度通道首日即抬升到 0.61；"
    "7 日频次与 30 日趋势通道首日为 0，第二日才分别升到 0.91 与 0.76，"
    "因为窗口要先累积到观测，这是它天然的滞后；当年累计节奏通道三天内稳定在 0.58。"
    "图6下半部分给出 s_ts、s_g 与 s_final：首个越过冻结阈值 0.8356 的日期是 10-02，"
    "s_final 为 0.942，10-03 升到 0.991。图中横轴均为业务申请日期；"
    "数据不含入库、评分与更正的时点字段，因此这里不报告任何时延指标。"
)

P2 = (
    "再看关系证据。图7左侧是上述收款人 PY000473 的关系子图：67 笔支付、52 个项目、"
    "3 个账户共享同一收款人，本案 9 笔全部落在项目 P000458 上。"
    "图中边上的 a=0.9 与 a=0.7 是 paid_to 与 owns_account 两类关系的默认强度，"
    "实际传播还要再乘阶段因子、时间衰减与同质性。右侧是与它规模相当的对照收款人 PY000390，"
    "同样高度共享（60 笔、53 个项目、3 个账户），但 2026 年单日最多 2 笔。"
    "图7下半部分说明两者差别不在共享结构本身，而在实体级短窗口密度："
    "s_entity_freq 在案例上最高 0.993，在对照上最高 0.521。"
    "这与 7.2 节的判断一致，共享关系只用于解释风险关联，不单独构成异常证据。"
)

CAP_CASE = "图6 频次突增事件的业务量、时序通道响应与告警过程"
CAP_REL = "图7 共享收款人的关系证据与正常对照"


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
    p._p.get_or_add_pPr().append(copy.deepcopy(sectpr_template))
    return p


def clone_para_before(anchor, tmpl, text):
    el = copy.deepcopy(tmpl._p)
    anchor._p.addprevious(el)
    p = Paragraph(el, anchor._parent)
    set_text_keep_fmt(p, text)
    return p


def insert_figure_block(anchor, img, width_in, caption, tmpl_caption, sect2, sect1):
    insert_sect_before(anchor, sect2)
    pimg = anchor.insert_paragraph_before()
    pimg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pimg.add_run().add_picture(img, width=Inches(width_in))
    el = copy.deepcopy(tmpl_caption._p)
    anchor._p.addprevious(el)
    pcap = Paragraph(el, anchor._parent)
    set_text_keep_fmt(pcap, caption)
    insert_sect_before(anchor, sect1)


def main():
    d = docx.Document(SRC)
    ps = d.paragraphs
    tmpl_head = ps[139]
    tmpl_body = ps[141]
    tmpl_caption = ps[38]
    sect2 = ps[93]._p.pPr.find(SECT)
    sect1 = ps[97]._p.pPr.find(SECT)
    assert sect2 is not None and sect1 is not None, "分节模板缺失"
    assert ps[139].text.startswith("6.7"), ps[139].text[:20]
    assert ps[142].text.startswith("7 讨论"), ps[142].text[:20]

    anchor = ps[142]

    clone_para_before(anchor, tmpl_head, HEADING)
    clone_para_before(anchor, tmpl_body, P1)
    insert_figure_block(anchor, os.path.join(FIG, "fig_case_multiscale.png"),
                        6.8, CAP_CASE, tmpl_caption, sect2, sect1)
    clone_para_before(anchor, tmpl_body, P2)
    insert_figure_block(anchor, os.path.join(FIG, "fig_relation_evidence.png"),
                        6.8, CAP_REL, tmpl_caption, sect2, sect1)

    # 原图6 → 图8（跨机制）
    for p in d.paragraphs:
        t = p.text
        if t.startswith("图6 两种生成机制"):
            set_text_keep_fmt(p, t.replace("图6", "图8", 1))
        elif "逐种子的配对结果见图6" in t:
            set_text_keep_fmt(p, t.replace("见图6", "见图8"))

    d.save(SRC)
    print("已保存:", SRC)


if __name__ == "__main__":
    main()
