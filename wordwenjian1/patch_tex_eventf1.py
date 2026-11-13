# -*- coding: utf-8 -*-
"""更新 Event-F1 与首警延迟叙述。"""
import io

P = "latex/main.tex"

REPS = [
    ("FoMo-0D 没在本文数据上训练，Payment-AUPRC 0.2793、Event-F1 0.0624，还是低于 MS-TRP 的 0.1269。",
     "FoMo-0D 没在本文数据上训练，Payment-AUPRC 0.2793、Event-F1 0.0624，还是低于 MS-TRP 的 0.1188。"),
    ("目标误报阈值下，MS-TRP 的 Event-F1 为 0.1269，低于 Concat-IF 的 0.2181 与 Isolation Forest 的 0.2208。",
     "目标误报阈值下，MS-TRP 的 Event-F1 为 0.1188，低于 Concat-IF 的 0.2181 与 Isolation Forest 的 0.2208。"),
    ("首警延迟汇总为 13.0 天", "首警延迟汇总为 12.9 天"),
]


def main():
    with io.open(P, encoding="utf-8", newline="") as f:
        s = f.read().replace("\r\n", "\n")
    n = 0
    for old, new in REPS:
        if old not in s:
            print("MISS:", old[:50])
            continue
        s = s.replace(old, new, 1)
        n += 1
    with io.open(P, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(s)
    print(f"patched {n}/{len(REPS)}")


if __name__ == "__main__":
    main()
