# -*- coding: utf-8 -*-
import zipfile, re, os
from lxml import etree

SRC = '../MS-TRP_中文双栏论文 .docx'
OUTDIR = '../latex'
FIGDIR = os.path.join(OUTDIR, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

z = zipfile.ZipFile(SRC)
xml = z.read('word/document.xml')
root = etree.fromstring(xml)

M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

def wq(t): return '{%s}%s' % (W, t)
def aq(t): return '{%s}%s' % (A, t)
def rq(t): return '{%s}%s' % (R, t)

# ---- rId -> image filename ----
rels = z.read('word/_rels/document.xml.rels')
relroot = etree.fromstring(rels)
rid2img = {}
for rel in relroot:
    if rel.get('Type', '').endswith('/image'):
        rid2img[rel.get('Id')] = os.path.basename(rel.get('Target'))
for name in z.namelist():
    if name.startswith('word/media/') and not name.endswith('/'):
        with open(os.path.join(FIGDIR, os.path.basename(name)), 'wb') as f:
            f.write(z.read(name))

# ---- 15 corrected formulas ----
FORMULAS = [
    r'a_{max} \le t_{score}',
    r'q = \frac{h + v}{2}, \quad \lambda = \mathrm{clip}[0.1 + 0.8q,\ 0.1,\ 0.9]',
    r'\mu = \lambda \mu_{self} + [1 - \lambda]\mu_{peer}',
    r'z^{+} = \frac{\max[0,\ x - \mu]}{\max[\sigma,\ \sigma_{min}] + \varepsilon}',
    r's_{ts} = \sum \omega_{q,w} \tanh[\kappa_{q,w} z_{q,w}]',
    r'c_{ij,r} = b_{r}[\mathrm{stage}] \exp[-\beta_{r}\Delta t]\ h_{ij}',
    r'W = \sum \zeta_{r}P^{r}, \quad \sum \zeta_{r} = 1',
    r's_{g,k+1} = \rho s_{ts} + [1 - \rho]\hat{W}s_{g,k}',
    r'g_{ts} = \frac{\exp[r_{ts}/T]}{\exp[r_{ts}/T] + \exp[r_{g}/T]}',
    r's_{gate} = g_{ts}s_{ts} + (1 - g_{ts})s_{g}',
    r's_{core} = 1 - (1 - s_{gate})(1 - s_{mismatch})',
    r's_{rhythm\_ch} = \max(0.70\ s_{rhythm},\ 0.65\ s_{conc})',
    r's_{final} = \max(s_{core},\ s_{rhythm\_ch})',
    r"\theta'_{c} = \frac{n_c}{n_c + \gamma}\theta_c + \frac{\gamma}{n_c + \gamma}\theta_{global}",
    r'\mathrm{FAR}/1000 = 1000 \times \frac{FP}{FP + TN}',
]

def escape_text(s):
    """Escape ASCII LaTeX specials via placeholders, then convert Unicode math to inline math."""
    out = s
    out = (out.replace('&', '\x00a').replace('%', '\x00b').replace('#', '\x00c')
             .replace('_', '\x00d').replace('$', '\x00e').replace('{', '\x00f')
             .replace('}', '\x00g').replace('^', '\x00h').replace('~', '\x00i')
             .replace('\\', '\x00j'))
    out = (out.replace('\x00a', r'\&').replace('\x00b', r'\%').replace('\x00c', r'\#')
             .replace('\x00d', r'\_').replace('\x00e', r'\$').replace('\x00f', r'\{')
             .replace('\x00g', r'\}').replace('\x00h', r'\textasciicircum{}')
             .replace('\x00i', r'\textasciitilde{}').replace('\x00j', r'\textbackslash{}'))
    # variable names / multi-char tokens (longest first, on escaped form)
    for a, b in [
        ('s\_rhythm\_ch', r'$s_{rhythm\_ch}$'),
        ('s\_mismatch', r'$s_{mismatch}$'),
        ('s\_rhythm', r'$s_{rhythm}$'),
        ('s\_conc', r'$s_{conc}$'),
        ('s\_gate', r'$s_{gate}$'),
        ('s\_core', r'$s_{core}$'),
        ('s\_final', r'$s_{final}$'),
        ('s\_ts', r'$s_{ts}$'),
        ('s\_g', r'$s_g$'),
        ('r\_ts', r'$r_{ts}$'),
        ('r\_g', r'$r_g$'),
        ('a\_max', r'$a_{max}$'),
        ('t\_score', r'$t_{score}$'),
        ('apply\_date', r'\texttt{apply\_date}'),
        ('payee\_id', r'\texttt{payee\_id}'),
        ('σmin', r'$\sigma_{min}$'),
        ('⁻⁶', r'$^{-6}$'),
    ]:
        out = out.replace(a, b)
    # single unicode math chars (safe: render in math mode)
    for a, b in [
        ('≤', r'$\le$'), ('≥', r'$\ge$'),
        ('λ', r'$\lambda$'), ('μ', r'$\mu$'), ('σ', r'$\sigma$'),
        ('ζ', r'$\zeta$'), ('ρ', r'$\rho$'), ('γ', r'$\gamma$'),
        ('ε', r'$\varepsilon$'), ('θ', r'$\theta$'), ('β', r'$\beta$'),
        ('ω', r'$\omega$'), ('κ', r'$\kappa$'), ('ϱ', r'$\varrho$'),
        ('Ŵ', r'$\hat{W}$'), ('→', r'$\rightarrow$'),
        ('↑', r'$\uparrow$'), ('↓', r'$\downarrow$'),
        ('−', '-'),
    ]:
        out = out.replace(a, b)
    return out

def run_text(node):
    parts = []
    for c in node:
        tag = etree.QName(c).localname
        if tag == 't':
            parts.append(c.text or '')
        elif tag in ('tab', 'br'):
            parts.append(' ')
    return ''.join(parts)

def paragraph_content(p):
    """Direct-children runs -> list of (kind, text). text is escaped+formatted."""
    out = []
    for c in p:
        tag = etree.QName(c).localname
        if tag == 'oMath':
            out.append(('formula', ''))
        elif tag == 'r':
            blip = c.find('.//' + aq('blip'))
            if blip is not None:
                out.append(('image', blip.get(rq('embed'))))
                continue
            rpr = c.find(wq('rPr'))
            bold = rpr is not None and rpr.find(wq('b')) is not None
            italic = rpr is not None and rpr.find(wq('i')) is not None
            vert = None
            if rpr is not None:
                va = rpr.find(wq('vertAlign'))
                if va is not None:
                    vert = va.get(wq('val'))
            t = run_text(c)
            if t:
                esc = escape_text(t)
                if vert == 'superscript':
                    esc = r'\textsuperscript{%s}' % esc
                elif vert == 'subscript':
                    esc = r'\textsubscript{%s}' % esc
                if bold:
                    esc = r'\textbf{%s}' % esc
                if italic:
                    esc = r'\emph{%s}' % esc
                out.append(('text', esc))
        elif tag == 'hyperlink':
            for r in c.findall(wq('r')):
                t = run_text(r)
                if t:
                    out.append(('text', escape_text(t)))
    return out

def paragraph_plain(p):
    """Raw unformatted text of a paragraph (for detection)."""
    parts = []
    for c in p:
        tag = etree.QName(c).localname
        if tag == 'r':
            parts.append(run_text(c))
        elif tag == 'hyperlink':
            for r in c.findall(wq('r')):
                parts.append(run_text(r))
    return ''.join(parts)

def plain_text(p):
    return ''.join(x.text or '' for x in p.iter() if etree.QName(x).localname == 't')

def cell_text(tc):
    parts = []
    for p in tc.findall(wq('p')):
        t = plain_text(p).strip()
        if t:
            parts.append(t)
    return ' '.join(parts)

def is_numeric(s):
    s = s.replace(',', '').replace('%', '').strip()
    return s == '' or s == '—' or bool(re.match(r'^-?[\d.]+$', s))

def col_spec(rows):
    ncols = max(len(r) for r in rows)
    spec = ['l']
    for j in range(1, ncols):
        cells = [r[j] for r in rows[1:] if j < len(r)]
        numeric = all(is_numeric(x) for x in cells)
        spec.append('c' if numeric else 'l')
    return ''.join(spec)

def render_table(child):
    rows = []
    for tr in child.findall(wq('tr')):
        rows.append([cell_text(tc) for tc in tr.findall(wq('tc'))])
    if not rows:
        return ''
    spec = col_spec(rows)
    ncols = len(spec)
    lines = [r'\begin{tabular}{%s}' % spec, r'\toprule']
    for i, row in enumerate(rows):
        cells = [escape_text(x.replace('|', '').strip()) for x in row]
        while len(cells) < ncols:
            cells.append('')
        lines.append(' & '.join(cells) + r' \\')
        if i == 0:
            lines.append(r'\midrule')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)

def heading_level(st):
    return 1 if st == '3' else (2 if st == '4' else 0)

body = root.find(wq('body'))
lines = []
formula_i = 0
pending_table_caption = None
pending_image = None
p_seen = 0
abstract_open = False

def flush_table_caption():
    global pending_table_caption
    if pending_table_caption:
        lines.append(r'\begin{table*}[t]')
        lines.append(r'\centering\small')
        lines.append(r'\caption{%s}' % pending_table_caption)
        pending_table_caption = None

for child in body:
    tag = etree.QName(child).localname
    if tag == 'p':
        p_seen += 1
        if p_seen <= 2:  # skip title + subtitle
            continue
        pstyle = child.find(wq('pPr') + '/' + wq('pStyle'))
        st = pstyle.get(wq('val')) if pstyle is not None else ''
        content = paragraph_content(child)
        has_formula = any(k == 'formula' for k, _ in content)
        has_image = any(k == 'image' for k, _ in content)
        rawtext = ''.join(t for k, t in content if k == 'text')
        plain = paragraph_plain(child).strip()

        if has_image:
            rid = next(t for k, t in content if k == 'image')
            pending_image = rid2img.get(rid, '')
            continue

        if has_formula:
            if formula_i < len(FORMULAS):
                lines.append(r'\begin{equation}')
                lines.append(FORMULAS[formula_i])
                lines.append(r'\end{equation}')
                formula_i += 1
            continue

        if not plain:
            continue

        lvl = heading_level(st)
        if lvl == 1:
            heading = re.sub(r'^\d+(\.\d+)*\s*', '', plain).strip()
            if heading == '参考文献':
                lines.append(r'\section*{%s}' % escape_text(heading))
            else:
                lines.append(r'\section{%s}' % escape_text(heading))
            continue
        if lvl == 2:
            heading = re.sub(r'^\d+(\.\d+)*\s*', '', plain).strip()
            lines.append(r'\subsection{%s}' % escape_text(heading))
            continue

        # table caption (require space after number, so "表4报告..." prose is skipped)
        m = re.match(r'^表\s*(\d+)\s+(.+)$', plain)
        if m:
            pending_table_caption = escape_text(m.group(2).strip())
            continue

        # figure caption (image must be pending)
        m = re.match(r'^图\s*(\d+)\s+(.+)$', plain)
        if m and pending_image:
            wide = (pending_image == 'image1.png')
            env = 'figure*' if wide else 'figure'
            width = r'\textwidth' if wide else r'\linewidth'
            lines.append(r'\begin{%s}[htbp]' % env)
            lines.append(r'\centering')
            lines.append(r'\includegraphics[width=%s]{figures/%s}' % (width, pending_image))
            lines.append(r'\caption{%s}' % escape_text(m.group(2).strip()))
            lines.append(r'\end{%s}' % env)
            pending_image = None
            continue

        # references
        m = re.match(r'^\[(\d+)\]\s*(.*)$', plain)
        if m:
            lines.append(r'\bibitem{ref%s} %s' % (m.group(1), escape_text(m.group(2).strip())))
            continue

        # abstract (leave environment open; keywords nests inside)
        if plain.startswith('摘要'):
            ab = re.sub(r'^摘要[：:]', '', plain).strip()
            lines.append(r'\begin{abstract}')
            lines.append(escape_text(ab))
            abstract_open = True
            continue

        # keywords (inside abstract environment)
        if plain.startswith('关键词'):
            kw = re.sub(r'^关键词[：:]', '', plain).strip()
            lines.append(r'\keywords{%s}' % escape_text(kw))
            lines.append(r'\end{abstract}')
            abstract_open = False
            continue

        # normal paragraph
        lines.append(rawtext)
        lines.append('')

    elif tag == 'tbl':
        flush_table_caption()
        lines.append(render_table(child))
        lines.append(r'\end{table*}')
        lines.append('')

# wrap references in thebibliography
final = []
in_refs = False
for ln in lines:
    if ln.startswith(r'\bibitem{'):
        if not in_refs:
            final.append(r'\begin{thebibliography}{40}')
            in_refs = True
        final.append(ln)
    else:
        if in_refs:
            final.append(r'\end{thebibliography}')
            in_refs = False
        final.append(ln)
if in_refs:
    final.append(r'\end{thebibliography}')
bodytex = '\n'.join(final)

preamble = r"""\documentclass[twocolumn,11pt]{ctexart}

\usepackage[top=2.5cm,bottom=2.5cm,left=1.8cm,right=1.8cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=space}

\graphicspath{{figures/}}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\newcommand{\keywords}[1]{\par\noindent\textbf{关键词：}#1}

\begin{document}

\begin{center}
{\LARGE\bfseries 面向预算执行异常监控的时间可追溯多尺度风险建模与关系证据分析}\\[0.6em]
{\large\bfseries MS-TRP：多尺度时序风险建模与关系证据分析}
\end{center}

\medskip

"""

doc = preamble + bodytex + '\n\n\\end{document}\n'
with open(os.path.join(OUTDIR, 'main.tex'), 'w', encoding='utf-8') as f:
    f.write(doc)

print('formulas used:', formula_i)
print('tables:', bodytex.count(r'\begin{table}'))
print('figures:', bodytex.count(r'\includegraphics'))
print('references:', bodytex.count(r'\bibitem'))
print('written', os.path.join(OUTDIR, 'main.tex'), len(bodytex), 'chars')
