# -*- coding: utf-8 -*-
"""Convert in-text [n] / \\textsuperscript{[n]} citations to clickable \\cite/\\upcite.
Only touches the document body (between \\begin{document} and \\begin{thebibliography}).
Only converts bracket groups whose numbers are all in 1..40, so math like [0,1]
and LaTeX optional args are left alone.
"""
import re, sys, io

path = "main.tex"
with io.open(path, encoding="utf-8") as f:
    text = f.read()

i_doc = text.index("\\begin{document}")
i_bib = text.index("\\begin{thebibliography}")
head, body, tail = text[:i_doc], text[i_doc:i_bib], text[i_bib:]

n_sup = n_plain = 0

def valid(nums):
    return all(1 <= int(x) <= 40 for x in nums)

# 1) \textsuperscript{[1,2]} -> \upcite{ref1,ref2}
def repl_sup(m):
    global n_sup
    nums = m.group(1).split(",")
    if not valid(nums):
        return m.group(0)
    n_sup += 1
    return "\\upcite{" + ",".join("ref" + x for x in nums) + "}"

body = re.sub(r"\\textsuperscript\{\[([0-9]+(?:,[0-9]+)*)\]\}", repl_sup, body)

# 2) plain [1] -> \cite{ref1}   (leave \cite{...} untouched; it has no brackets)
def repl_plain(m):
    global n_plain
    nums = m.group(1).split(",")
    if not valid(nums):
        return m.group(0)
    n_plain += 1
    return "\\cite{" + ",".join("ref" + x for x in nums) + "}"

body = re.sub(r"\[([0-9]+(?:,[0-9]+)*)\]", repl_plain, body)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(head + body + tail)

print("superscript citations converted:", n_sup)
print("plain citations converted:", n_plain)
print("total:", n_sup + n_plain)
