# -*- coding: utf-8 -*-
import zipfile, copy, shutil
from lxml import etree

SRC = '可编辑科研框架图_修正5.pptx'
DST = '可编辑科研框架图_修正6.pptx'

P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'

def P_(t): return '{%s}%s' % (P, t)
def A_(t): return '{%s}%s' % (A, t)

zin = zipfile.ZipFile(SRC)
root = etree.fromstring(zin.read('ppt/slides/slide1.xml'))

# ---------- DSL parser: _{...} = subscript, ^{...} = superscript ----------
def parse(s):
    runs = []
    i, n = 0, len(s)
    buf = ''
    def flush():
        nonlocal buf
        if buf:
            runs.append((buf, ''))
            buf = ''
    while i < n:
        if s[i:i+2] == '_{':
            flush()
            j = s.index('}', i)
            runs.append((s[i+2:j], 'sub'))
            i = j + 1
        elif s[i:i+2] == '^{':
            flush()
            j = s.index('}', i)
            runs.append((s[i+2:j], 'sup'))
            i = j + 1
        else:
            buf += s[i]
            i += 1
    flush()
    return runs

# ---------- shape lookup ----------
def find_sp(sid):
    for sp in root.iter(P_('sp')):
        nv = sp.find(P_('nvSpPr') + '/' + P_('cNvPr'))
        if nv is not None and nv.get('id') == sid:
            return sp
    return None

def shape_font(sp):
    sz, color = None, None
    for tag in ('defRPr', 'rPr'):
        el = sp.find('.//' + A_(tag))
        if el is None:
            continue
        if sz is None and el.get('sz'):
            sz = el.get('sz')
        f = el.find(A_('solidFill') + '/' + A_('srgbClr'))
        if f is not None and color is None:
            color = f.get('val')
    if sz is None:
        sz = '1600'
    if color is None:
        color = '101820'
    return int(sz), color

def make_rpr(sz, color, va=None):
    rpr = etree.Element(A_('rPr'))
    if va in ('sub', 'sup'):
        rpr.set('sz', str(max(int(sz * 0.6), 700)))
        rpr.set('baseline', '-25000' if va == 'sub' else '30000')
    else:
        rpr.set('sz', str(sz))
    rpr.set('b', '0')
    fill = etree.SubElement(rpr, A_('solidFill'))
    clr = etree.SubElement(fill, A_('srgbClr'))
    clr.set('val', color)
    for tf in ('latin', 'ea', 'cs'):
        e = etree.SubElement(rpr, A_(tf))
        e.set('typeface', 'Arial')
    return rpr

def set_shape_text(sp, paragraphs, sz=None):
    base_sz, color = shape_font(sp)
    if sz:
        base_sz = sz
    txBody = sp.find(P_('txBody'))
    if txBody is None:
        return
    for p in txBody.findall(A_('p')):
        txBody.remove(p)
    for para in paragraphs:
        runs = parse(para) if isinstance(para, str) else para
        ap = etree.SubElement(txBody, A_('p'))
        ppr = etree.SubElement(ap, A_('pPr'))
        ppr.set('algn', 'ctr')
        for txt, va in runs:
            ar = etree.SubElement(ap, A_('r'))
            ar.append(make_rpr(base_sz, color, va))
            at = etree.SubElement(ar, A_('t'))
            at.text = txt
        er = etree.SubElement(ap, A_('endParaRPr'))
        er.set('sz', str(base_sz))
        er.set('b', '0')

# ---------- Phase 1: text edits ----------
EDITS = {
    '18':  (['a_{max} ≤ t_{score}'], None),
    '110': (['Shrinkage Weight λ',
             'q = (h+v)/2,  λ = clip[0.1+0.8q, 0.1, 0.9]',
             'μ = λμ_{self} + (1−λ)μ_{peer}'], 1500),
    '127': (['z^{+} = max(0, x − μ) /',
             '(max(σ, σ_{min}) + ε)'], None),
    '150': (['s_{ts} =',
             'Σ ω_{q,w} · tanh(κ_{q,w} z_{q,w})'], None),
    '221': (['c_{ij,r} = b_{r}[stage] · exp(−β_{r}Δt) · h_{ij}'], 1400),
    '253': (['Relation Mixing',
             'W = Σ_{r} ζ_{r} P^{r},  Σ_{r} ζ_{r} = 1'], 1400),
    '255': (['W = Σ_{r} ζ_{r} P^{r}'], 1500),
    '271': (['Transition', 'Matrix Block', 'W = Σ_{r} ζ_{r} P^{r}'], 1500),
    '300': (['PPR Diffusion',
             's_{g,k+1} = ρ s_{ts} + (1−ρ) Ŵ s_{g,k}'], None),
    '319': (['s_{ts}'], None),
    '334': (['Temporal Quality Tensor  r_{ts}'], None),
    '335': (['(effective history ·',
             'feature coverage ·',
             'robust scale stability)'], None),
    '354': (['s_{g}'], None),
    '369': (['Graph Quality Tensor  r_{g}'], None),
    '370': (['(valid neighbours ·',
             'relation-type coverage ·',
             'topology stability)'], None),
    '371': (['Temperature Gate:'], None),
    '372': (['g_{ts} = exp(r_{ts}/T) / (exp(r_{ts}/T) + exp(r_{g}/T))'], 1500),
    '394': (['Fused Risk Score',
             's_{gate} = g_{ts}s_{ts} + (1−g_{ts})s_{g}'], None),
    '397': (['θ′_{c} = [n_{c}/(n_{c}+γ)] θ_{c} + [γ/(n_{c}+γ)] θ_{global},  γ=30'], 1500),
    '299': (['s_{g}'], None),
    '22':  (['X_{i,q,w}(t)'], None),
    '92':  (['Multi-Scale Feature Tensor', 'X_{i,q,w}(t)'], None),
}

edited = 0
for sid, (paras, sz) in EDITS.items():
    sp = find_sp(sid)
    if sp is None:
        print('MISS', sid)
        continue
    set_shape_text(sp, paras, sz)
    edited += 1
print('edited:', edited)

# ---------- Phase 2: new boxes (clone shape 150 as template) ----------
NEW = [
    # new_id, name, x, y, cx, cy, sz, paragraphs
    ('500', 'rhythm-note',  520, 378, 240, 30, 1400,
     ['s_{rhythm}, s_{conc}  →  Eq.(12)']),
    ('501', 'core-box',     565, 720, 235, 62, 1500,
     ['s_{core} = 1 − (1−s_{gate})(1−s_{mismatch})',
      's_{mismatch} ∈ {0,1}  (account consistency)']),
    ('502', 'rhythmch-box', 375, 720, 185, 55, 1500,
     ['s_{rhythm_ch} = max(0.70 s_{rhythm}, 0.65 s_{conc})']),
    ('503', 'final-box',    565, 790, 235, 50, 1500,
     ['s_{final} = max(s_{core}, s_{rhythm_ch})']),
]

spTree = root.find('.//' + P_('spTree'))
tpl = find_sp('150')
for new_id, name, x, y, cx, cy, sz, paras in NEW:
    sp = copy.deepcopy(tpl)
    nv = sp.find(P_('nvSpPr') + '/' + P_('cNvPr'))
    nv.set('id', new_id)
    nv.set('name', name)
    xfrm = sp.find(P_('spPr') + '/' + A_('xfrm'))
    xfrm.find(A_('off')).set('x', str(int(x * 12700)))
    xfrm.find(A_('off')).set('y', str(int(y * 12700)))
    xfrm.find(A_('ext')).set('cx', str(int(cx * 12700)))
    xfrm.find(A_('ext')).set('cy', str(int(cy * 12700)))
    set_shape_text(sp, paras, sz)
    spTree.append(sp)
print('added boxes:', len(NEW))

# ---------- write ----------
zin.close()
names = [n for n in zipfile.ZipFile(SRC).namelist()]
with zipfile.ZipFile(SRC) as zsrc, zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED) as zdst:
    for n in names:
        if n == 'ppt/slides/slide1.xml':
            zdst.writestr(n, etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True))
        else:
            zdst.writestr(n, zsrc.read(n))
print('written', DST)
