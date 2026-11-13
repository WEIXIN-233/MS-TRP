# -*- coding: utf-8 -*-
"""把指定图的 PNG 兜底 + SVG 矢量两个媒体部件同步刷新进 docx。

用法：python refresh_docx_media.py
只替换 CHANGES 里列出的 docx 媒体部件，其余不动；改动前自动备份。
"""
import os
import shutil
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX = os.path.join(HERE, "MS-TRP_中文双栏论文 .docx")
FIG = os.path.join(HERE, "latex", "figures")
RED = os.path.join(HERE, "redraw")

# (docx PNG 媒体名, docx SVG 媒体名) -> (PNG 源文件, SVG 源文件)
# 注意：docx 经 Office 重存后媒体被交错重编号，图5=image9/10，图6=image11/12。
CHANGES = {
    ("word/media/image9.png", "word/media/image10.svg"):
        (os.path.join(RED, "fig4_quarter_far_cn.png"),
         os.path.join(FIG, "image4.svg")),
    ("word/media/image11.png", "word/media/image12.svg"):
        (os.path.join(RED, "fig_case_multiscale.png"),
         os.path.join(FIG, "fig_case_multiscale.svg")),
}


def main():
    backup = DOCX.replace(".docx", "_媒体刷新前备份_%s.docx"
                          % time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(DOCX, backup)
    print("backup:", backup)

    zin = zipfile.ZipFile(DOCX)
    items = {n: zin.read(n) for n in zin.namelist()}
    zin.close()

    for (media, svg_media), (png_src, svg_src) in CHANGES.items():
        if media not in items or svg_media not in items:
            raise SystemExit("docx 缺少媒体部件: %s / %s" % (media, svg_media))
        items[media] = open(png_src, "rb").read()
        items[svg_media] = open(svg_src, "rb").read()
        print("replaced:", media, len(items[media]), "bytes")
        print("replaced:", svg_media, len(items[svg_media]), "bytes")

    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in items.items():
            zout.writestr(name, data)
    os.replace(tmp, DOCX)
    print("written:", DOCX)


if __name__ == "__main__":
    main()
