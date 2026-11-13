# -*- coding: utf-8 -*-
"""修正新增图片段的段落属性：补 snapToGrid=0 / 固定行距，避免 Word 网格裁切图片。"""
import copy
import docx
from docx.oxml.ns import qn

SRC = "MS-TRP_中文双栏论文 .docx"
BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"

d = docx.Document(SRC)
ps = d.paragraphs
tmpl_ppr = ps[37]._p.pPr          # 图1 的图片段属性（已验证可正常显示）

fixed = []
for i, p in enumerate(ps):
    if p._p.findall(".//" + BLIP):
        ppr = p._p.pPr
        has_snap = ppr is not None and ppr.find(qn("w:snapToGrid")) is not None
        if not has_snap:
            old = p._p.pPr
            if old is not None:
                p._p.remove(old)
            p._p.insert(0, copy.deepcopy(tmpl_ppr))
            fixed.append(i)

d.save(SRC)
print("已修正图片段:", fixed)
