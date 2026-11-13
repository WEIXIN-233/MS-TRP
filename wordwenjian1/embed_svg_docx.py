# -*- coding: utf-8 -*-
"""把 Word 稿件里的 8 张位图替换为「矢量 SVG 主图 + 600dpi PNG 兜底」。

Word 的图片部件不支持 PDF（PDF 只能作为 OLE 对象显示图标），
它支持的矢量格式是 SVG（Word 2016/365 与新版 WPS 均原生渲染）。
按 OOXML 规范，SVG 图片必须同时带一个位图 fallback：
  <a:blip r:embed="PNG的rId">
    <a:extLst><a:ext uri="{96DAC541-7B7A-43D3-8B79-37D633B846F1}">
      <asvg:svgBlip r:embed="SVG的rId"/>
    </a:ext></a:extLst>
  </a:blip>
本脚本只加 SVG 部件与引用，不改动原有 PNG（旧版阅读器回退时仍可用）。

用法：python embed_svg_docx.py
"""
import os
import re
import shutil
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX = os.path.join(HERE, "MS-TRP_中文双栏论文 .docx")
FIGDIR = os.path.join(HERE, "latex", "figures")

# docx 内 imageN.png -> 对应的 SVG
SVG_MAP = {
    1: "fig1-framework.svg",
    2: "fig2_combined.svg",
    3: "fig_ablation.svg",
    4: "fig3_cold_start2.svg",
    5: "image4.svg",
    6: "fig_case_multiscale.svg",
    7: "fig_relation_evidence.svg",
    8: "fig_generator.svg",
}
# 原 PNG 的 relationship id
PNG_RID = {i: "rId%d" % (i + 6) for i in range(1, 9)}
SVG_RID = {i: "rId%d" % (100 + i) for i in range(1, 9)}
SVG_EXT_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
SVG_NS = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"


def blip_xml(png_rid, svg_rid):
    return (
        '<a:blip r:embed="%s"><a:extLst>'
        '<a:ext uri="%s">'
        '<asvg:svgBlip xmlns:asvg="%s" r:embed="%s"/>'
        "</a:ext></a:extLst></a:blip>"
        % (png_rid, SVG_EXT_URI, SVG_NS, svg_rid)
    )


def main():
    backup = DOCX.replace(".docx", "_SVG前备份_%s.docx"
                          % time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(DOCX, backup)
    print("backup:", backup)

    zin = zipfile.ZipFile(DOCX)
    items = {n: zin.read(n) for n in zin.namelist()}
    zin.close()

    doc = items["word/document.xml"].decode("utf-8")
    rels = items["word/_rels/document.xml.rels"].decode("utf-8")
    ctypes = items["[Content_Types].xml"].decode("utf-8")

    for i in range(1, 9):
        png_rid, svg_rid = PNG_RID[i], SVG_RID[i]
        old = '<a:blip r:embed="%s"/>' % png_rid
        if doc.count(old) != 1:
            raise SystemExit("unexpected blip count for %s: %d"
                             % (png_rid, doc.count(old)))
        doc = doc.replace(old, blip_xml(png_rid, svg_rid))

        svg_path = os.path.join(FIGDIR, SVG_MAP[i])
        items["word/media/image%d.svg" % i] = open(svg_path, "rb").read()

        rels = rels.replace(
            "</Relationships>",
            '<Relationship Id="%s" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/image" Target="media/image%d.svg"/>'
            "</Relationships>" % (svg_rid, i))

    if 'Extension="svg"' not in ctypes:
        ctypes = ctypes.replace(
            "<Default ",
            '<Default Extension="svg" ContentType="image/svg+xml"/><Default ', 1)

    items["word/document.xml"] = doc.encode("utf-8")
    items["word/_rels/document.xml.rels"] = rels.encode("utf-8")
    items["[Content_Types].xml"] = ctypes.encode("utf-8")

    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in items.items():
            zout.writestr(name, data)
    os.replace(tmp, DOCX)
    print("written:", DOCX)
    print("svgBlip inserted:", doc.count("svgBlip"))


if __name__ == "__main__":
    main()
