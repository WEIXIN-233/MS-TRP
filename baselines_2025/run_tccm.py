# -*- coding: utf-8 -*-
"""
TCCM（NeurIPS 2025，Time-Conditioned Contraction Matching，一步流匹配表格异常检测）
接入 MS-TRP 基线。
============================================================================
- 单分类：仅在训练集【正常】样本上训练（学一个把正常样本收缩到原点的向量场）。
- 异常分数 = ||f(x, t=1) + x||（越大越异常，单次前向）。
- 阈值在验证集冻结、测试集只评一次（与 _prep.report 同口径）。

用法：python -m baselines_2025.run_tccm [--epochs 100] [--lr 0.001] [--batch 64] [--seed 42] [--limit 0]
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TCCM_DIR = os.path.join(HERE, "TCCM")
sys.path.insert(0, ROOT)      # baselines_2025._prep
sys.path.insert(0, TCCM_DIR)  # FMAD.FlowMatchingAD

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from baselines_2025._prep import build_prep, report
from FMAD.FlowMatchingAD import TCCM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本自测（0=全部）")
    args = ap.parse_args()

    X, y, train_idx, val_idx, test_idx, payments, events = build_prep()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    # 单分类：训练集【正常】样本
    tr = train_idx[y[train_idx] == 0]
    if args.limit:
        tr = tr[:args.limit]
    X_train_norm = X[tr].astype(np.float64)

    # 预处理：StandardScaler 仅在训练正常样本上 fit（与 TCCM 官方 load_adbench_npz 一致）
    scaler = StandardScaler().fit(X_train_norm)
    Xtr = scaler.transform(X_train_norm).astype(np.float32)
    Xva = scaler.transform(X[val_idx]).astype(np.float32)
    Xte = scaler.transform(X[test_idx]).astype(np.float32)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = TCCM(n_features=X.shape[1], epochs=args.epochs,
                 learning_rate=args.lr, batch_size=args.batch)
    print(f"training TCCM on {len(Xtr)} normal samples, {args.epochs} epochs", flush=True)
    model.fit(Xtr)

    s_val = model.decision_function(Xva)
    s_test = model.decision_function(Xte)

    # 回填到全量分数（仅测试集参与最终指标；验证集用于冻结阈值）
    s_full = np.zeros(len(payments), dtype=np.float32)
    s_full[val_idx] = s_val
    s_full[test_idx] = s_test

    report("TCCM", s_full, payments)
    np.save(os.path.join(HERE, "scores_tccm.npy"), s_full)
    print("saved scores_tccm.npy")


if __name__ == "__main__":
    main()
