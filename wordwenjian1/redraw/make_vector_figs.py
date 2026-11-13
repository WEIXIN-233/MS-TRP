# -*- coding: utf-8 -*-
"""把全部 matplotlib 图同时导出为矢量 PDF 与 SVG（文字转路径）。

做法：不修改各绘图脚本，只在运行时给 Figure.savefig 打补丁，
       每次保存 .png 时顺带写出同名 .pdf / .svg，
       再把三者复制到 wordwenjian1/latex/figures/ 供 LaTeX 与 Word 使用。
用法：python make_vector_figs.py [脚本名 ...]
"""
import os
import runpy
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"   # 文字转路径，避免嵌入字体
matplotlib.rcParams["pdf.fonttype"] = 42       # PDF 用 TrueType，保持可检索

from matplotlib.figure import Figure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, "..", "latex", "figures"))
os.makedirs(FIGDIR, exist_ok=True)

_orig = Figure.savefig
made = []


def savefig(self, fname, *a, **kw):
    fname = os.fspath(fname)
    stem, ext = os.path.splitext(fname)
    if ext.lower() != ".png":
        return _orig(self, fname, *a, **kw)
    _orig(self, fname, *a, **kw)
    for out_ext in (".pdf", ".svg"):
        out = stem + out_ext
        _orig(self, out, *a, **kw)
        made.append(out)
        dst = os.path.join(FIGDIR, os.path.basename(out))
        if os.path.abspath(dst) != os.path.abspath(out):
            shutil.copy2(out, dst)
        print("vector:", out, "->", FIGDIR)


Figure.savefig = savefig

SCRIPTS = ["plot_extra_figs.py", "plot_case_fig.py",
           "plot_relation_fig.py", "plot_far_cn.py"]

if __name__ == "__main__":
    names = sys.argv[1:] or SCRIPTS
    for name in names:
        print("=== running", name)
        runpy.run_path(os.path.join(HERE, name), run_name="__main__")
    print("\n%d vector files written" % len(made))
