# -*- coding: utf-8 -*-
"""P0-2 事件跨集合审计脚本（可复现）。

加载当前 data_fixed 库，统计每个异常事件的成员落在哪些 split，
把跨集合事件写入 audit/event_split_audit.csv，并打印汇总。
验收判据①：跨集合事件数必须为 0。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, split

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "event_split_audit.csv")


def main():
    payments = data.load_payments()
    labels = data.load_labels()
    events = data.build_events(labels)
    tr, va, te = split.time_forward_split(payments)

    split_map = {}
    for name, ids in [("train", tr), ("val", va), ("test", te)]:
        for i in ids:
            split_map[str(payments.loc[i, "apply_id"])] = name

    rows = []
    unknown = 0
    for _, ev in events.iterrows():
        groups = {}
        n_unknown = 0
        for aid in ev["apply_ids"]:
            if aid not in split_map:
                n_unknown += 1
            else:
                groups[split_map[aid]] = True
        unknown += n_unknown
        if len(groups) > 1 or n_unknown > 0:
            rows.append({
                "event_id": ev["event_id"],
                "anomaly_type": ev["anomaly_type"],
                "n_members": len(ev["apply_ids"]),
                "splits": "+".join(sorted(groups)),
                "unknown_ids": n_unknown,
            })

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

    # 汇总
    cross = [r for r in rows if r["unknown_ids"] == 0 and "+" in r["splits"]]
    print(f"总事件数   : {len(events)}")
    print(f"跨集合事件 : {len(cross)}")
    print(f"未知ID事件 : {len([r for r in rows if r['unknown_ids'] > 0])} (未知成员 {unknown} 笔)")
    if cross:
        from collections import Counter
        by_type = Counter(r["anomaly_type"] for r in cross)
        print("按类型跨集合:")
        for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]):
            print(f"  {t}: {c}")
    print(f"审计已写入: {OUT}")
    # 以退出码表达判据：跨集合=0 才算通过
    sys.exit(0 if len(cross) == 0 and unknown == 0 else 1)


if __name__ == "__main__":
    main()
