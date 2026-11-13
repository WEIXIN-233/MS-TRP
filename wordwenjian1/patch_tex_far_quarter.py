# -*- coding: utf-8 -*-
"""更新表10（季度误报）及其叙述、验证集经验值 1.42→1.36。"""
import io

P = "latex/main.tex"

REPS = [
    (r"2026Q1 & 2,284 & 1.75 \\", r"2026Q1 & 2,284 & 0.88 \\"),
    (r"2026Q2 & 3,488 & 0.86 \\", r"2026Q2 & 3,488 & 0.57 \\"),
    (r"2026Q3 & 5,456 & 0.92 \\", r"2026Q3 & 5,456 & 0.73 \\"),
    (r"2026Q4 & 8,974 & 3.57 \\", r"2026Q4 & 8,974 & 2.56 \\"),
    ("四个季度的FAR/1000介于0.86至3.57。第二季度与第三季度低于验证集经验值1.42，第一季度为1.75，第四季度为3.57，首尾两个季度高于验证集经验值。",
     "四个季度的FAR/1000介于0.57至2.56。第一至第三季度分别为0.88、0.57、0.73，均低于验证集经验值1.36，第四季度为2.56，高于该经验值。"),
    ("本节FAR/1000与验证集经验值1.42均以分层收缩阈值计算",
     "本节FAR/1000与验证集经验值1.36均以分层收缩阈值计算"),
]


def main():
    with io.open(P, encoding="utf-8", newline="") as f:
        s = f.read().replace("\r\n", "\n")
    n = 0
    for old, new in REPS:
        if old not in s:
            print("MISS:", old[:40])
            continue
        s = s.replace(old, new, 1)
        n += 1
    with io.open(P, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(s)
    print(f"patched {n}/{len(REPS)}")


if __name__ == "__main__":
    main()
