# -*- coding: utf-8 -*-
"""把 3 个 2025+ 新方法（DRL / FoMo-0D / TCCM）写进论文：
- §5.2 新增方法介绍段落
- 表4（支付级性能）、表5（召回@FAR）各插入 3 行
- §6.1 新增结果分析段落，并在召回@FAR 段落后补一句
- 表4 表注补充阈值漂移说明
- 参考文献新增 [16][17][18]
- 图2 替换为含 12 个方法的版本
新增内容统一标红（与文中既有修订标记一致）。
"""
import copy
import os
import zipfile

import docx
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from docx.text.paragraph import Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DOC = os.path.join(ROOT, "wordwenjian1", "MS-TRP_中文双栏论文.docx")
FIG2 = os.path.join(HERE, "fig2_auprc.png")

RED = RGBColor(0xFF, 0x00, 0x00)


W14 = "http://schemas.microsoft.com/office/word/2010/wordml"


def strip_ids(el):
    """移除克隆节点里重复的 w14:paraId / w14:textId，避免 Word 提示修复。"""
    for p in el.iter(qn("w:p")):
        for attr in (f"{{{W14}}}paraId", f"{{{W14}}}textId"):
            if attr in p.attrib:
                del p.attrib[attr]


def add_runs(paragraph, segments, size=None):
    """segments: [(text, kind)]，kind='t' 正文 / 'c' 上标引用。全部标红。"""
    for text, kind in segments:
        run = paragraph.add_run(text)
        run.font.color.rgb = RED
        if size is not None:
            run.font.size = Pt(size)
        if kind == "c":
            run.font.size = Pt(9)
            run.font.superscript = True


def clone_para_after(anchor_p, template_p, segments, size=None):
    new_p = copy.deepcopy(template_p._p)
    for child in list(new_p):
        if child.tag == qn("w:r") or child.tag == qn("w:hyperlink"):
            new_p.remove(child)
    strip_ids(new_p)
    anchor_p._p.addnext(new_p)
    para = Paragraph(new_p, anchor_p._parent)
    add_runs(para, segments, size=size)
    return para


def insert_table_row(table, anchor_text, cells, name_cite=None):
    """在首列以 anchor_text 开头的行之后插入一行；cells 为各列文本。"""
    idx = None
    for i, row in enumerate(table.rows):
        if row.cells[0].text.strip().startswith(anchor_text):
            idx = i
            break
    assert idx is not None, anchor_text
    src_tr = table.rows[idx]._tr
    new_tr = copy.deepcopy(src_tr)
    strip_ids(new_tr)
    src_tr.addnext(new_tr)
    row = table.rows[idx + 1]

    for ci, (cell, val) in enumerate(zip(row.cells, cells)):
        p = cell.paragraphs[0]
        runs = p.runs
        assert runs, "cell has no run"
        first = runs[0]
        for extra in runs[1:]:
            extra._r.getparent().remove(extra._r)
        first.text = val
        first.bold = False
        first.font.color.rgb = RED
        # 引用上标
        if ci == 0 and name_cite:
            first.text = name_cite[0]
            cite = p.add_run(name_cite[1])
            cite.font.size = Pt(7.5)
            cite.font.superscript = True
            cite.font.color.rgb = RED
            cite.bold = False
    return row


def main():
    d = docx.Document(DOC)
    ps = d.paragraphs

    # ---------- 1. §5.2 新增方法介绍（插在段末，5.3 标题之前）----------
    p79 = ps[79]
    assert p79.text.startswith("Concat-IF将派生表示"), p79.text[:30]
    intro = [
        ("为考察2025年公开方法在相同输入表示与评价口径下的表现，本文另行实测三种近期方法。DRL", "t"),
        ("[16]", "c"),
        ("在潜空间中以数据无关的正交基向量分解正常样本表示，并加入表示分离约束，采用128维隐空间、5个原型与200轮训练。FoMo-0D", "t"),
        ("[17]", "c"),
        ("是面向表格离群检测的零样本基础模型，以PFN式上下文推理直接输出离群概率，不在本文数据上更新参数，"
         "推理上下文取4,999条训练集正常样本。TCCM", "t"),
        ("[18]", "c"),
        ("学习时间条件收缩向量场，将正常样本在单步前向中收缩至原点，并以残差范数作为异常分数，训练100轮，"
         "学习率0.001，批大小64。三者均仅使用训练集正常样本拟合或构造上下文，并在与其他基线相同的验证集冻结阈值口径下评价。", "t"),
    ]
    clone_para_after(p79, p79, intro)

    # ---------- 2. “全部18种方法” → “21种” ----------
    p87 = ps[87]
    assert "18种方法" in p87.text
    run0 = p87.runs[0]
    text = run0.text
    pre, post = text.split("18种", 1)
    run0.text = pre
    r2 = p87.add_run("21")
    r2.font.color.rgb = RED
    r3 = p87.add_run("种" + post)

    # ---------- 3. 表4 插入 3 行 ----------
    t4 = d.tables[3]
    insert_table_row(t4, "Fixed Rule",
                     ["DRL", "0.4320", "0.4008", "0.3717", "93.11", "0.1332"],
                     name_cite=("DRL", "[16]"))
    insert_table_row(t4, "Robust MAD",
                     ["FoMo-0D", "0.3736", "0.7557", "0.1327", "7.19", "0.1962"],
                     name_cite=("FoMo-0D", "[17]"))
    insert_table_row(t4, "FoMo-0D",
                     ["TCCM", "0.3499", "0.1435", "1.0000", "1000.00", "0.0336"],
                     name_cite=("TCCM", "[18]"))

    # 最优值加粗更新：召回率最优由 Fixed Rule(0.2998) 变为 DRL(0.3717)
    for row in t4.rows:
        name = row.cells[0].text.strip()
        if name.startswith("Fixed Rule"):
            row.cells[3].paragraphs[0].runs[0].bold = False
        if name.startswith("DRL"):
            row.cells[3].paragraphs[0].runs[0].bold = True

    # ---------- 4. 表4 表注补充阈值漂移说明 ----------
    p90 = ps[90]
    assert p90.text.startswith("注：")
    note = ("DRL、FoMo-0D与TCCM为仅在正常样本上拟合或构造上下文的单分类与零样本方法。"
            "验证集冻结阈值在2026测试期发生分布偏移：DRL与FoMo-0D的FAR/1000高于其余方法；"
            "TCCM的测试分数全部高于冻结阈值，其召回率1.0000与FAR/1000为1000.00对应全量告警，"
            "精确率等于阳性占比，故三者的阈值类指标不宜与监督式排序方法直接比较，"
            "TCCM的全量告警不参与最优值加粗。三者的Payment-AUPRC与召回@FAR不受阈值影响。")
    r = p90.add_run(note)
    r.font.size = Pt(8)
    r.font.color.rgb = RED

    # ---------- 5. §6.1 新增分析段落（MS-TRP AUPRC 段之后、图2 之前）----------
    p92 = ps[92]
    assert p92.text.startswith("MS-TRP的Payment-AUPRC")
    analysis = [(
        "三种2025年方法均未超过MS-TRP。DRL的Payment-AUPRC为0.4320，在全部21种方法或参照中列第6位，"
        "高于Simple PPR的0.4242及全部PyOD方法，说明正交基分解与表示分离约束能够从本文的派生表示中提取部分异常信号；"
        "其冻结阈值下FAR/1000为93.11，代价是较高的告警负担。FoMo-0D未在本文数据上训练，Payment-AUPRC为0.3736，"
        "但Event-F1为0.1962，高于MS-TRP的0.0991，表明零样本基础模型在事件级聚合上具有一定互补性。"
        "TCCM的Payment-AUPRC为0.3499，低于FoMo-0D，且其冻结阈值在测试期失效，阈值类指标不能作为有效工作点。"
        "上述结果表明，在相同输入表示与评价口径下，2025年的通用表格异常检测方法仍低于本文的多尺度时序组合，"
        "但零样本方法的事件级覆盖值得进一步分析。", "t")]
    clone_para_after(p92, p92, analysis)

    # ---------- 6. 表5 插入 3 行 ----------
    t5 = d.tables[4]
    insert_table_row(t5, "Fixed Rule", ["DRL", "0.3398", "0.3744"], name_cite=("DRL", "[16]"))
    insert_table_row(t5, "Robust MAD", ["FoMo-0D", "0.2698", "0.3494"], name_cite=("FoMo-0D", "[17]"))
    insert_table_row(t5, "FoMo-0D", ["TCCM", "0.2258", "0.2824"], name_cite=("TCCM", "[18]"))

    # ---------- 7. 召回@FAR 段落补一句 ----------
    p97 = ps[97]
    assert p97.text.startswith("在10%误报率下")
    extra = ("三种2025年方法在5%与10%误报率下的召回均低于Temporal-only。DRL在5%误报率下的召回为0.3398，"
             "高于MS-TRP的0.2513，但在10%误报率下为0.3744，低于MS-TRP的0.5530；"
             "该交叉表明二者的相对优势随误报预算变化，不能由单一预算点推断整体排序。")
    r = p97.add_run(extra)
    r.font.color.rgb = RED

    # ---------- 8. 参考文献 [16][17][18] ----------
    p156 = ps[156]
    assert p156.text.startswith("[15]")
    refs = [
        "[16] YE H, ZHAO H, FAN W, et al. DRL: Decomposed representation learning for tabular anomaly "
        "detection[C]//The Thirteenth International Conference on Learning Representations (ICLR). 2025. "
        "https://openreview.net/forum?id=CJnceDksRd",
        "[17] SHEN Y, WEN H, AKOGLU L. FoMo-0D: A foundation model for zero-shot tabular outlier detection[J]. "
        "Transactions on Machine Learning Research, 2025. https://openreview.net/forum?id=XCQzwpR9jE",
        "[18] LI Z, HUANG Q, ZHU Y, et al. Scalable, explainable and provably robust anomaly detection with "
        "one-step flow matching[C]//Advances in Neural Information Processing Systems 38. 2025. arXiv: 2510.18328.",
    ]
    anchor = p156
    tmpl_ref = ps[147]  # [6] 带悬挂缩进的标准参考文献格式
    for ref in refs:
        anchor = clone_para_after(anchor, tmpl_ref, [(ref, "t")], size=8.0)

    d.save(DOC)
    print("docx 文本修改完成")

    # ---------- 9. 替换图2 并校正显示比例 ----------
    def find_fig2_para(doc):
        ps_ = doc.paragraphs
        for i, p_ in enumerate(ps_):
            if p_.text.strip().startswith("图2"):
                for j in range(i - 1, max(i - 4, -1), -1):
                    if list(ps_[j]._p.iter(qn("w:drawing"))):
                        return ps_[j]
        raise RuntimeError("未找到图2")

    d = docx.Document(DOC)  # 重新打开，确保 parts 一致
    fig_p = find_fig2_para(d)
    drawings = list(fig_p._p.iter(qn("w:drawing")))
    ext = drawings[0].find(".//" + qn("wp:extent"))
    cx = int(ext.get("cx"))
    from PIL import Image
    im = Image.open(FIG2)
    ratio = im.size[0] / im.size[1]
    d.save(DOC)  # 先保存，确保下面 zip 重写的是最终文件

    # 用 zipfile 直接替换 word/media/image2.png
    with zipfile.ZipFile(DOC, "r") as zin:
        items = [(i, zin.read(i.filename)) for i in zin.infolist()]
    newbytes = open(FIG2, "rb").read()
    with zipfile.ZipFile(DOC, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in items:
            if info.filename == "word/media/image2.png":
                data = newbytes
            zout.writestr(info, data)
    print("图2 已替换")

    # 校正显示比例（保持宽度，按新图比例调整高度）
    d = docx.Document(DOC)
    fig_p = find_fig2_para(d)
    drawings = list(fig_p._p.iter(qn("w:drawing")))
    ext = drawings[0].find(".//" + qn("wp:extent"))
    ext.set("cy", str(int(cx / ratio)))
    d.save(DOC)
    print(f"图2 显示尺寸校正: cx={cx}, cy={int(cx / ratio)}")


if __name__ == "__main__":
    main()
