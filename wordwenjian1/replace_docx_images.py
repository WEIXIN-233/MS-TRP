# -*- coding: utf-8 -*-
"""用重绘的 PNG 替换 Word 论文中的内嵌图片，并按新图纵横比调整高度。

映射（inline shape 序号 → 新图 → docx 内部 media 名）：
    1 图2 → redraw/fig2_combined.png        → word/media/image8.png
    2 图3 → redraw/fig_ablation.png         → word/media/image6.png
    3 图4 → redraw/fig3_cold_start2.png     → word/media/image7.png
    4 图5 → redraw/fig4_quarter_far_cn.png → word/media/image4.png
    5 图6 → redraw/fig_case_multiscale.png  → word/media/image9.png
    7 图8 → redraw/fig_generator.png        → word/media/image5.png
图1（框架图）与图7（关系对照）不替换。

用法：python replace_docx_images.py [--dry]
"""
import os
import shutil
import sys
import time
import zipfile

import docx
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "MS-TRP_中文双栏论文 .docx")
RED = os.path.join(HERE, "redraw")

SHAPE_MAP = {
    1: ("fig2_combined.png", "word/media/image8.png"),
    2: ("fig_ablation.png", "word/media/image6.png"),
    3: ("fig3_cold_start2.png", "word/media/image7.png"),
    4: ("fig4_quarter_far_cn.png", "word/media/image4.png"),
    5: ("fig_case_multiscale.png", "word/media/image9.png"),
    7: ("fig_generator.png", "word/media/image5.png"),
}


def main():
    dry = "--dry" in sys.argv
    for name, _ in SHAPE_MAP.values():
        p = os.path.join(RED, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少新图: {p}")

    # 1) 按新图纵横比调整每个 inline shape 的高度（宽度保持不变）
    d = docx.Document(SRC)
    shapes = list(d.inline_shapes)
    for idx, (name, _media) in SHAPE_MAP.items():
        w_px, h_px = Image.open(os.path.join(RED, name)).size
        sh = shapes[idx]
        old_w, old_h = sh.width, sh.height
        new_h = int(round(old_w * h_px / w_px))
        sh.height = new_h
        print(f"shape{idx}: {name}  {w_px}x{h_px}  height {old_h} -> {new_h}")

    if dry:
        print("[dry] 不改动文件")
        return

    tmp = SRC + ".tmp"
    d.save(tmp)

    # 2) 替换 media 二进制（zip 层重写）
    bak = os.path.join(HERE, f"MS-TRP_中文双栏论文_图前_{time.strftime('%Y%m%d-%H%M%S')}.docx")
    shutil.copy2(SRC, bak)
    zin = zipfile.ZipFile(tmp, "r")
    zout = zipfile.ZipFile(SRC, "w", zipfile.ZIP_DEFLATED)
    targets = {media: name for name, media in SHAPE_MAP.values()}
    replaced = 0
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename in targets:
            with open(os.path.join(RED, targets[item.filename]), "rb") as f:
                data = f.read()
            replaced += 1
        zout.writestr(item, data)
    zout.close()
    zin.close()
    os.remove(tmp)
    print(f"replaced {replaced} media files; bak {bak}")


if __name__ == "__main__":
    main()
