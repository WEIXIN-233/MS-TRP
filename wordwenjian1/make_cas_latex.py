# -*- coding: utf-8 -*-
"""把 main.tex（ctexart 双栏）重排为 Elsevier CAS 双栏模板（cas-dc）。

只改版式：导言区、题名/作者/摘要/关键词、浮动体写法、图表尺寸、参考文献标题。
正文文字、数据、结论一字不动。

用法：python make_cas_latex.py            # 生成 main.tex，旧版备份为 main_ctexart.tex
"""
import io
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
LATEX_DIR = os.path.join(HERE, "latex")
SRC = os.path.join(LATEX_DIR, "main.tex")
BAK = os.path.join(LATEX_DIR, "main_ctexart.tex")

PREAMBLE = r"""%% -*- coding: utf-8 -*-
%% 依 Elsevier CAS 双栏模板（cas-dc，CAS Bundle 2.4）重排。
%% 版式参考学长投稿用的 cas-dc 写法：题名/作者/摘要/关键词块、[pos=...] 浮动体、
%% booktabs + adjustbox 表格、natbib 数字引用。正文内容与原稿一致。
\documentclass[a4paper,fleqn]{cas-dc}
\let\printorcid\relax

\usepackage{graphicx,adjustbox}
\usepackage[numbers]{natbib}
\usepackage{xeCJK}
\setCJKmainfont{SimSun}[BoldFont=SimHei, ItalicFont=KaiTi]
\setCJKsansfont{SimHei}
\setCJKmonofont{FangSong}
% 数学字体：cas-dc 自带的 legacy stix 在 XeLaTeX 下符号错位（减号变星号），
% 已由同目录 stix.sty 占位跳过，这里换用 unicode-math 的 STIX Two Math。
\usepackage{unicode-math}
\setmathfont{STIX Two Math}

\graphicspath{{./}{figures/}}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\renewcommand{\refname}{参考文献}
% 正文沿用原稿的上标引用写法
\newcommand{\upcite}[1]{\textsuperscript{\cite{#1}}}

\begin{document}
\shortauthors{魏新雨}
\shorttitle{}

"""

FLOAT_SETUP = r"""% 浮动体参数：图表与题注同页，允许正文填满版面
\raggedbottom
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\setcounter{dbltopnumber}{2}
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dbltopfraction}{0.92}
\renewcommand{\dblfloatpagefraction}{0.75}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\setlength{\emergencystretch}{1em}

"""

# 单栏放不下、要跨栏的宽表（按题名匹配）
WIDE_CAPTIONS = (
    "异常模式及正常业务对照",
    "测试集支付级与事件级性能",
    "固定误报率下的支付召回",
)

# 每个图的目标：是否跨栏、宽度、高度上限
FIG_RULES = {
    "fig1-framework.pdf": (True, r"\textwidth", r"0.55\textheight"),
    "fig2_combined.png": (True, r"\textwidth", r"0.50\textheight"),
    "fig_ablation.png": (True, r"\textwidth", r"0.42\textheight"),
    "fig3_cold_start2.png": (False, r"\columnwidth", r"0.30\textheight"),
    "image4.png": (False, r"\columnwidth", r"0.32\textheight"),
    "fig_case_multiscale.png": (True, r"\textwidth", r"0.42\textheight"),
    "fig_relation_evidence.png": (True, r"0.90\textwidth", r"0.40\textheight"),
    "fig_generator.png": (True, r"0.90\textwidth", r"0.30\textheight"),
}


def read(path):
    with io.open(path, encoding="utf-8", newline="") as f:
        return f.read().replace("\r\n", "\n")


def write(path, s):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def split_document(s):
    """返回 (导言区, 正文)。正文从第一个 \\section 开始。"""
    i = s.index(r"\begin{document}")
    j = s.index(r"\section{引言}")
    return s[:i], s[j:]


def grab(s, start, end):
    a = s.index(start) + len(start)
    b = s.index(end, a)
    return s[a:b]


def convert_table(block):
    """把 \\begin{table*}[t]...\\end{table*} 转成 CAS 写法。

    宽表（列多或表头长，单栏放不下）保持 table*，用 tabular* 撑满版心；
    其余转单栏 table + tabular*{\\columnwidth}。
    """
    m = re.search(r"\\begin\{tabular\}\{([^}]*)\}", block)
    if not m:
        return block
    spec = m.group(1)
    ncols = len(re.sub(r"[^lcrpmbX]", "", spec.split("@")[0]))
    cap_m = re.search(r"\\caption\{([^\n]*)\}", block)
    caption = cap_m.group(1) if cap_m else ""
    wide = ncols > 4 or any(k in caption for k in WIDE_CAPTIONS)

    body = block
    body = re.sub(r"\\begin\{table\*\}\[[^\]]*\]", "", body, count=1)
    body = re.sub(r"\\end\{table\*\}", "", body, count=1)
    body = body.replace("\\centering\\small\n", "").replace("\\centering\\small", "")
    body = body.strip("\n")

    # 表格一律用 tabular*，按版心均分列宽
    body = body.replace("\\begin{tabular}{" + spec + "}",
                        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}" + spec + "@{}}", 1)
    body = body.replace("\\end{tabular}", "\\end{tabular*}", 1)

    if wide:
        head = "\\begin{table*}[pos=tbp]\n\\small\n"
        tail = "\n\\end{table*}\n"
    else:
        head = ("\\begin{table}[pos=tbp]\n"
                "\\scriptsize\n"
                "\\setlength{\\tabcolsep}{3pt}\n")
        tail = "\n\\end{table}\n"
    body = re.sub(r"\n{2,}", "\n", body)
    return head + body + tail


def move_table_notes(body):
    """把紧跟在表后面的“注：……”段落移进表格浮动体，避免表与注被分页拆开。"""
    # 注意：convert_table 与 convert_figure 会在块尾补一个换行，
    # 与源文件 \end{table*} 后的空行叠加成 2~3 个换行，故用 \n+ 匹配。
    pat = re.compile(r"(\\end\{table\*?\})\n+(注：[^\n]*)\n")

    def rep(m):
        return ("\n\\par\\smallskip\\noindent{\\scriptsize " + m.group(2).strip()
                + "}\n" + m.group(1) + "\n")

    return pat.sub(rep, body)


def convert_figure(block):
    m = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", block)
    if not m:
        return block
    fname = m.group(1).split("/")[-1]
    dbl, width, height = FIG_RULES.get(fname, (True, r"\textwidth", r"0.45\textheight"))

    env = "figure*" if dbl else "figure"
    out = re.sub(r"\\begin\{figure\*\}\[[^\]]*\]", "", block, count=1)
    tag = "\\end{%s}" % env
    out = re.sub(r"\\end\{figure\*\}", lambda _: tag, out, count=1)
    inc = "\\includegraphics[width=%s,height=%s,keepaspectratio]{%s}" % (
        width, height, m.group(1))
    out = re.sub(r"\\includegraphics\[[^\]]*\]\{[^}]+\}", lambda _: inc, out, count=1)
    return "\\begin{%s}[pos=tbp]\n" % env + out.strip("\n") + "\n"


def main():
    s = read(SRC)
    _, body = split_document(s)
    old = s

    # 摘要与关键词
    abstract = grab(old, r"\begin{abstract}", r"\end{abstract}").strip()
    keywords = grab(old, r"\keywords{", "}").strip()
    # 旧稿把 \keywords{...} 写在 abstract 里，这里摘掉，改用 CAS 的 keywords 环境
    abstract = re.sub(r"\s*\\keywords\{[^}]*\}", "", abstract).strip()
    kw = r" \sep ".join(k.strip() for k in keywords.split("；") if k.strip())

    # 浮动体
    body = re.sub(r"\\begin\{table\*\}.*?\\end\{table\*\}",
                  lambda m: convert_table(m.group(0)), body, flags=re.S)
    body = re.sub(r"\\begin\{figure\*\}.*?\\end\{figure\*\}",
                  lambda m: convert_figure(m.group(0)), body, flags=re.S)
    body = move_table_notes(body)

    body = body.replace("\n\\FloatBarrier\n", "\n")
    body = body.replace("\\section*{参考文献}\n", "")

    front = (r"\title[mode=title]{面向预算执行异常监控的时间可追溯多尺度风险建模与关系证据分析}"
             + "\n\n"
             + r"\author[1]{魏新雨}" + "\n\n"
             + r"\affiliation[1]{organization={（单位名称待补充）}," + "\n"
             + r"city={（城市）}," + "\n"
             + r"country={China}" + "\n}\n\n"
             + "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n"
             + "\\begin{keywords}\n" + kw + "\n\\end{keywords}\n\n"
             + "\\maketitle\n\n" + FLOAT_SETUP)

    data_avail = (
        "\n\\section*{数据可用性声明}\n"
        "本文全部实验基于作者构造的合成预算业务数据，不含真实财政业务数据；"
        "数据生成脚本、实验配置与逐样本预测保存在项目代码库中，"
        "按文中固定随机种子与参数可复算。\n")

    # 参考文献 8pt、行距 9pt（类文件默认 8pt/10pt）。类文件在导言区之后
    # 仍会重置 \bibfont，所以只能在文献环境之前就地覆盖。
    bib_tight = (
        "% 参考文献收紧：行距 10pt→9pt，把末条压回第 12 页\n"
        "\\renewcommand{\\bibfont}{\\fontsize{8pt}{9pt}\\selectfont}\n"
        "\\renewcommand{\\casbiographyfont}{\\fontsize{8pt}{9pt}\\selectfont}\n")

    # 数据可用性放在参考文献之前
    idx = body.index(r"\begin{thebibliography}")
    body = body[:idx] + data_avail + "\n" + bib_tight + body[idx:]

    out = PREAMBLE + front + "\n" + body
    # 参考文献环境与 \end{document} 之间空行整理
    out = re.sub(r"\n{3,}", "\n\n", out)

    if not os.path.exists(BAK):
        shutil.copy2(SRC, BAK)
    write(SRC, out)
    print("generated:", SRC)
    print("backup   :", BAK)


if __name__ == "__main__":
    main()
