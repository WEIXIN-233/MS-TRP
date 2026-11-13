# -*- coding: utf-8 -*-
import pymupdf, docx, os, hashlib
from docx.oxml.ns import qn

SRC = r"C:\Users\魏新雨\PycharmProjects\pythonProject5\venv\MS-TRP\wordwenjian1\MS-TRP_中文双栏论文 .docx"
PDF = r"C:\Users\魏新雨\PycharmProjects\pythonProject5\venv\MS-TRP\wordwenjian1\latex\figures\MS-TRP_框架图_v12，最终.pdf"
PNG = r"C:\Users\魏新雨\Doubao\chats\2026-11-13\new-chat\fig1_framework_from_pdf_600dpi.png"

d = pymupdf.open(PDF); pg = d[0]
pm = pg.get_pixmap(dpi=600, alpha=False)   # 整页, 无 clip 裁切
pm.save(PNG)
print("rendered:", pm.w, "x", pm.h, os.path.getsize(PNG))

doc = docx.Document(SRC)
part = doc.part.related_parts["rId7"]
print("part:", part.partname, "old:", len(part.blob))
with open(PNG, "rb") as f:
    part._blob = f.read()
print("new:", len(part.blob))

# 尺寸保持等比, 与原 extent 一致
aspect = pg.rect.height / pg.rect.width
cx = 6477000
cy = int(round(cx * aspect))
target = None
for sh in doc.inline_shapes:
    el = sh._inline.element if hasattr(sh._inline, "element") else sh._inline
    blips = el.xpath(".//a:blip")
    if blips and blips[0].get(qn("r:embed")) == "rId7":
        target = el; break
assert target is not None
for e in target.xpath(".//wp:extent"):
    e.set("cx", str(cx)); e.set("cy", str(cy))
for e in target.xpath(".//pic:spPr/a:xfrm/a:ext"):
    e.set("cx", str(cx)); e.set("cy", str(cy))

doc.save(SRC)
print("saved", os.path.getsize(SRC), "aspect %.6f" % aspect)
