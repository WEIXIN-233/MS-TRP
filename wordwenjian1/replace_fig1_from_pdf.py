# -*- coding: utf-8 -*-
import pymupdf, docx, os
from docx.oxml.ns import qn

SRC = r"C:\Users\魏新雨\PycharmProjects\pythonProject5\venv\MS-TRP\wordwenjian1\MS-TRP_中文双栏论文 .docx"
PDF = r"C:\Users\魏新雨\PycharmProjects\pythonProject5\venv\MS-TRP\wordwenjian1\latex\figures\MS-TRP_框架图_v12，最终.pdf"
PNG = r"C:\Users\魏新雨\Doubao\chats\2026-11-13\new-chat\fig1_framework_from_pdf.png"

d = pymupdf.open(PDF); pg = d[0]
pm = pg.get_pixmap(dpi=300, alpha=False); pm.save(PNG)
print("rendered:", pm.w, "x", pm.h, os.path.getsize(PNG))

doc = docx.Document(SRC)
part = doc.part.related_parts["rId7"]
print("partname:", part.partname, "old:", len(part.blob))
with open(PNG, "rb") as f:
    part._blob = f.read()
print("new:", len(part.blob))

target = None
for sh in doc.inline_shapes:
    el = sh._inline.element if hasattr(sh._inline, "element") else sh._inline
    blips = el.xpath(".//a:blip")
    if blips and blips[0].get(qn("r:embed")) == "rId7":
        target = el
        break
assert target is not None, "未找到 rId7 图"

aspect = pg.rect.height / pg.rect.width
cx = 6477000
cy = int(round(cx * aspect))
for e in target.xpath(".//wp:extent"):
    e.set("cx", str(cx)); e.set("cy", str(cy))
for e in target.xpath(".//pic:spPr/a:xfrm/a:ext"):
    e.set("cx", str(cx)); e.set("cy", str(cy))
print("size emu:", cx, cy, "aspect %.6f" % aspect)

for blip in target.xpath(".//a:blip"):
    for extLst in blip.findall(qn("a:extLst")):
        for ext in list(extLst):
            uri = ext.get("uri")
            if uri and "96DAC541" in uri:
                extLst.remove(ext)
        if len(extLst) == 0:
            blip.remove(extLst)

doc.save(SRC)
print("saved:", SRC, os.path.getsize(SRC))
