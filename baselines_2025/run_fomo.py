# -*- coding: utf-8 -*-
"""
FoMo-0D（ICML 2025，零样本表格离群检测）接入 MS-TRP 基线。
============================================================
- 权重：HuggingFace ``YuchenShen/FoMo-0D``（经 hf-mirror.com 镜像下载）。
- 输入：build_feature_matrix 的 21 维特征，按论文协议补零/缩放至 100 维。
- 输出：测试集每笔支付 P(离群) 分数，复用 _prep.report 算 AUPRC/P/R/F1/FAR。

用法：python -m baselines_2025.run_fomo [--context 4999] [--batch 64] [--limit 0]
"""
import os
import sys
import argparse

# 关键：先设 HF 镜像，再 import huggingface_hub（否则走被墙的 huggingface.co）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FOMO = os.path.join(HERE, "FoMo-0D")
sys.path.insert(0, ROOT)      # 供 from baselines_2025._prep import ...
sys.path.insert(0, FOMO)      # 供 import fomo_hub / pfns

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from baselines_2025._prep import build_prep, report


def preprocess(context, test, max_feat=100):
    """复刻 FoMo-0D pfn_inference_transform：z-score(按 context) → 分位数正态化 →
    特征尺度缩放 → 补零到 max_feat。返回 (context_padded, test_padded)。"""
    n_ctx = len(context)
    all_x = np.concatenate([context, test], axis=0).astype(np.float64)
    n_feat = all_x.shape[1]

    # 1) z-score（用 context 均值/标准差，test 一起标准化）
    mu = all_x[:n_ctx].mean(axis=0, keepdims=True)
    sd = all_x[:n_ctx].std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    all_x = (all_x - mu) / sd

    # 2) 逐列 QuantileTransformer(output='normal')，仅在 context 上 fit
    qt = QuantileTransformer(output_distribution="normal")
    for c in range(n_feat):
        try:
            qt.fit(all_x[:n_ctx, c:c + 1])
            all_x[:, c:c + 1] = qt.transform(all_x[:, c:c + 1])
        except Exception:
            pass

    # 3) 特征尺度缩放（num_feat / max_feat）
    all_x = all_x / (n_feat / max_feat)

    # 4) 补零到 max_feat
    if n_feat < max_feat:
        all_x = np.concatenate([all_x, np.zeros((len(all_x), max_feat - n_feat))], axis=1)

    # 5) 分位数正态化对常数列/极端值可能产生 NaN，统一置 0（与零填充协议一致的中性值）
    all_x = np.nan_to_num(all_x, nan=0.0, posinf=0.0, neginf=0.0)

    return all_x[:n_ctx].astype(np.float32), all_x[n_ctx:].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=4999)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条测试样本（0=全部），用于自测")
    args = ap.parse_args()

    X, y, train_idx, val_idx, test_idx, payments, events = build_prep()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # 载入模型
    from fomo_hub import FoMo0DHub
    model = FoMo0DHub.from_pretrained("YuchenShen/FoMo-0D", map_location=device).eval()
    model = model.to(device)
    print("FoMo-0D loaded")

    # 上下文：训练集【正常】无标签子样本（与 FoMo-0D 官方 semi 设置 normal=True 一致）
    normal_idx = train_idx[y[train_idx] == 0]
    rng = np.random.RandomState(0)
    ctx_idx = normal_idx if len(normal_idx) <= args.context else \
        rng.choice(normal_idx, args.context, replace=False)
    context = X[ctx_idx]

    # 评分对象：验证集 + 测试集（零样本方法对两者同等对待；验证集用于冻结阈值）
    score_idx = np.concatenate([val_idx, test_idx])
    if args.limit:
        score_idx = score_idx[:args.limit]
    test = X[score_idx]

    ctx_pad, test_pad = preprocess(context, test)

    # 推理：test 分批，batch 维并行（断点续跑：每 500 批存一次进度）
    ckpt = os.path.join(HERE, "scores_fomo_partial.npz")
    ctx_t = torch.from_numpy(ctx_pad).unsqueeze(1).to(device)  # (ctx, 1, 100)
    scores = np.empty(len(test_pad), dtype=np.float32)
    b0 = 0
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        scores = z["scores"]
        b0 = int(z["b0"])
        print(f"resume from batch offset {b0}", flush=True)
    with torch.no_grad():
        while b0 < len(test_pad):
            b1 = min(b0 + args.batch, len(test_pad))
            tb = test_pad[b0:b1]                                  # (B, 100)
            ctx_b = ctx_t.repeat(1, tb.shape[0], 1)               # (ctx, B, 100)
            test_b = torch.from_numpy(tb).unsqueeze(0).to(device) # (1, B, 100)
            out = model(train_x=ctx_b, test_x=test_b)             # (1, B, 2)
            prob = torch.softmax(out[0], dim=-1)[:, 1].cpu().numpy()
            scores[b0:b1] = prob
            if (b0 // args.batch) % 50 == 0:
                print(f"  {b1}/{len(test_pad)}", flush=True)
            if (b0 // args.batch) % 500 == 0:
                np.savez(ckpt, scores=scores, b0=b1)
            b0 = b1

    # 参考 FoMo-0D 官方协议：softmax 可能出 NaN，置 0（视为正常）
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # 回填到全量分数向量（val + test 都回填；其余填 0，不影响指标）
    s_full = np.zeros(len(payments), dtype=np.float32)
    s_full[score_idx] = scores

    report("FoMo-0D", s_full, payments)
    np.save(os.path.join(HERE, "scores_fomo.npy"), s_full)
    print("saved scores_fomo.npy")


if __name__ == "__main__":
    main()
