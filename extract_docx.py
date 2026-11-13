# -*- coding: utf-8 -*-
"""临时脚本：提取 docx 正文为 UTF-8 纯文本文件（输出到指定目录，避免终端编码问题）。"""
import sys, zipfile, os
from xml.etree import ElementTree as ET

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

def extract(path):
    with zipfile.ZipFile(path) as z:
        tree = ET.fromstring(z.read('word/document.xml'))
    lines = []
    for p in tree.iter(W + 'p'):
        texts = [t.text or '' for t in p.iter(W + 't')]
        line = ''.join(texts).strip()
        if line:
            lines.append(line)
    return '\n'.join(lines)

def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    for path in sys.argv[2:]:
        try:
            text = extract(path)
            base = os.path.basename(path)
            out = os.path.join(outdir, base + '.txt')
            with open(out, 'w', encoding='utf-8') as f:
                f.write(text)
            print('OK %d chars -> %s' % (len(text), out))
        except Exception as e:
            print('ERROR', path, repr(e))

if __name__ == '__main__':
    main()
