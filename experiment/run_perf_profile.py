# -*- coding: utf-8 -*-
"""审稿意见第7条：实测运行开销（峰值内存、分阶段耗时、日终增量打分耗时）。

只做计时与内存采样，不改任何计算路径：
  · 阶段计时：用 monkeypatch 包裹 pipeline 调到的模块函数，记录累计耗时；
  · 峰值内存：后台线程每 50ms 采样当前进程 RSS（psutil），另记录各阶段结束时的 RSS；
  · 日终增量：asof 逐业务日重建转移矩阵并跑一次 PPR，统计「新增一个业务日」
    的平均/中位/最大耗时（不含数据入库与特征重算）。

输出 result_raw/perf_profile.json 与 perf_stage.csv。
用法：python -m experiment.run_perf_profile
"""
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import asof, config, data, graph_builder, pipeline, propagation  # noqa: E402
from src import temporal_baseline  # noqa: E402

SAMPLE_INTERVAL = 0.05


class RssSampler:
    def __init__(self):
        self.peak = 0.0
        self._stop = threading.Event()
        self._proc = psutil.Process(os.getpid())
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(SAMPLE_INTERVAL):
            try:
                rss = self._proc.memory_info().rss / 1024 ** 3
            except psutil.Error:
                return
            self.peak = max(self.peak, rss)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


class Timed:
    """包裹模块函数，记录累计耗时与调用次数。"""

    def __init__(self):
        self.acc = {}
        self.calls = {}

    def patch(self, module, name, label=None):
        label = label or name
        orig = getattr(module, name)
        self.acc.setdefault(label, 0.0)
        self.calls.setdefault(label, 0)

        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            self.acc[label] += time.perf_counter() - t0
            self.calls[label] += 1
            return out

        setattr(module, name, wrapper)
        return orig


def main():
    timed = Timed()
    timed.patch(data, "load_payments", "load_payments")
    timed.patch(data, "load_labels", "load_labels")
    timed.patch(data, "load_entity_tables", "load_entity_tables")
    timed.patch(temporal_baseline, "enrich_payments", "enrich_payments")
    timed.patch(temporal_baseline, "compute_multi_scale_features", "multi_scale_features")
    timed.patch(temporal_baseline, "compute_rhythm_features", "rhythm_features")
    timed.patch(temporal_baseline, "compute_peer_stats", "peer_stats")
    timed.patch(temporal_baseline, "build_self_peer_baseline", "self_peer_baseline")
    timed.patch(temporal_baseline, "compute_temporal_risk", "temporal_risk")
    timed.patch(graph_builder, "load_graph_tables", "load_graph_tables")
    timed.patch(graph_builder, "compute_structural_risk", "structural_risk")
    node_count = []
    orig_build_nodes = timed.patch(graph_builder, "build_nodes", "build_nodes")

    def build_nodes_record(*a, **kw):
        out = orig_build_nodes(*a, **kw)
        node_count.append(len(out[0]))
        return out

    graph_builder.build_nodes = build_nodes_record
    timed.patch(graph_builder, "build_edges", "build_edges")
    timed.patch(asof, "score_asof_diffuse", "asof_ppr_total")
    ppr_times = []
    mat_times = []
    orig_ppr = propagation.personalized_ppr

    def timed_ppr(*a, **kw):
        t0 = time.perf_counter()
        out = orig_ppr(*a, **kw)
        ppr_times.append(time.perf_counter() - t0)
        return out

    propagation.personalized_ppr = timed_ppr

    orig_mat = propagation.build_transition_matrix_from_arrays

    def timed_mat(*a, **kw):
        t0 = time.perf_counter()
        out = orig_mat(*a, **kw)
        mat_times.append(time.perf_counter() - t0)
        return out

    propagation.build_transition_matrix_from_arrays = timed_mat

    sampler = RssSampler()
    sampler.start()
    t0 = time.perf_counter()
    res, pred = pipeline.run_ms_trp()
    wall = time.perf_counter() - t0
    sampler.stop()

    payments = data.load_payments()
    from src import split as split_mod
    _tr, _val, test_idx = split_mod.time_forward_split(payments)
    test_days = payments.iloc[test_idx]["apply_date"].astype(str)
    n_test_days = test_days.nunique()
    per_test_day = test_days.value_counts()

    ppr = np.array(ppr_times) * 1000.0
    mat = np.array(mat_times) * 1000.0
    n_day = min(len(ppr), len(mat))
    day_ms = (mat[:n_day] + ppr[:n_day])
    stages = pd.DataFrame(
        [{"stage": k, "seconds": round(v, 3), "calls": timed.calls[k]}
         for k, v in timed.acc.items()]).sort_values("seconds", ascending=False)

    out = {
        "wall_seconds": round(wall, 3),
        "peak_rss_gb": round(sampler.peak, 3),
        "n_nodes": int(node_count[0]) if node_count else 0,
        "ppr_iters_mean": round(float(res["ppr_iters"]), 2),
        "ppr_days_scored": int(len(ppr)),
        "ppr_ms_mean": round(float(ppr.mean()), 2),
        "ppr_ms_median": round(float(np.median(ppr)), 2),
        "ppr_ms_p95": round(float(np.percentile(ppr, 95)), 2),
        "ppr_ms_max": round(float(ppr.max()), 2),
        "day_ms_mean": round(float(day_ms.mean()), 2),
        "day_ms_median": round(float(np.median(day_ms)), 2),
        "day_ms_p95": round(float(np.percentile(day_ms, 95)), 2),
        "day_ms_max": round(float(day_ms.max()), 2),
        "asof_total_s": round(timed.acc["asof_ppr_total"], 3),
        "n_test_days": int(n_test_days),
        "test_payments_per_day_mean": round(float(per_test_day.mean()), 1),
        "test_payments_per_day_max": int(per_test_day.max()),
        "stage_seconds": {k: round(v, 3) for k, v in timed.acc.items()},
    }

    json_path = os.path.join(config.RESULT_DIR, "perf_profile.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    stages.to_csv(os.path.join(config.RESULT_DIR, "perf_stage.csv"),
                  index=False, encoding="utf-8-sig")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(stages.to_string(index=False))
    print("written:", json_path)


if __name__ == "__main__":
    main()
