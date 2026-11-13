# -*- coding: utf-8 -*-
"""
DRL（ICLR 2025，Decomposed Representation Learning for Tabular Anomaly Detection）
接入 MS-TRP 基线。
============================================================================
- 单分类：仅在训练集【正常】样本上训练（one-class）。
- 异常分数 = 潜空间表示 h=encoder(x) 与其正交基重构 h_=weight@prototype 的 MSE（越大越异常）。
- 阈值在验证集冻结、测试集只评一次（与 _prep.report 同口径）。

用法：python -m baselines_2025.run_drl [--epochs 200] [--seed 42] [--batch 512] [--limit 0]
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DRL = os.path.join(HERE, "DRL")
sys.path.insert(0, ROOT)   # baselines_2025._prep
sys.path.insert(0, DRL)    # PrototypeNet.Model

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from baselines_2025._prep import build_prep, report
from PrototypeNet.Model import PN


def build_config(data_dim, device, seed, epochs, batch):
    # 全部超参采用论文默认/最常用设置，未在测试集上调参
    return {
        "data_dim": data_dim,
        "epochs": epochs,
        "learning_rate": 0.05,
        "sche_gamma": 0.98,
        "device": device,
        "batch_size": batch,
        "en_nlayers": 3,
        "de_nlayers": 3,
        "hidden_dim": 128,
        "z_dim": 128,
        "random_seed": seed,
        "preprocess": "standard",
        "diversity": True,          # 正交基（Gram-Schmidt）
        "plearn": False,            # 基向量训练无关（requires_grad=False）
        "input_info": True,
        "input_info_ratio": 0.1,
        "cl": True,
        "cl_ratio": 0.06,
        "prototype_num": 5,
    }


def train(model, X, cfg):
    """复刻 Trainer.training：分解损失 + 对齐损失(input_info) + 分离损失(cl)。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["sche_gamma"])
    model.train()
    n = len(X)
    bs = cfg["batch_size"]
    for epoch in range(cfg["epochs"]):
        running = 0.0
        nstep = 0
        for b0 in range(0, n, bs):
            x = torch.from_numpy(X[b0:b0 + bs]).float().to(cfg["device"])
            # 分解损失（forward 返回两种归一化 MSE 的 max）
            loss = model(x).mean()

            # 对齐损失
            if cfg["input_info"]:
                h = model.encoder(x)
                x_tilde = model.decoder(h)
                s_loss = (1 - F.cosine_similarity(x_tilde, x, dim=-1)).mean()
                loss = loss + cfg["input_info_ratio"] * s_loss

            # 分离损失
            if cfg["cl"]:
                h_ = model.phi(x)
                sel = np.random.choice(h_.shape[0], int(h_.shape[0] * 0.8), replace=False)
                h_ = h_[sel]
                matrix = h_ @ h_.T
                mol = torch.sqrt(torch.sum(h_ ** 2, dim=-1, keepdim=True)) @ \
                    torch.sqrt(torch.sum(h_.T ** 2, dim=0, keepdim=True))
                matrix = matrix / mol
                eye = torch.eye(h_.shape[0], device=h_.device)
                d_loss = ((1 - eye) * matrix).sum() / h_.shape[0] / h_.shape[0]
                loss = loss + cfg["cl_ratio"] * d_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            nstep += 1
        scheduler.step()
        if epoch % 20 == 0 or epoch == cfg["epochs"] - 1:
            print(f"  epoch {epoch}/{cfg['epochs']}  loss={running / max(nstep, 1):.5f}", flush=True)


@torch.no_grad()
def score(model, X, cfg, batch=1024):
    """复刻 Trainer.evaluate 的异常分数：MSE(h, weight@prototype) 按特征平均。"""
    model.eval()
    scores = np.empty(len(X), dtype=np.float32)
    for b0 in range(0, len(X), batch):
        b1 = min(b0 + batch, len(X))
        x = torch.from_numpy(X[b0:b1]).float().to(cfg["device"])
        h = model.encoder(x)
        weight = model.phi(x)
        h_ = weight @ model.prototype
        mse = F.mse_loss(h, h_, reduction="none").mean(dim=-1)
        scores[b0:b1] = mse.cpu().numpy()
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练/验证样本自测（0=全部）")
    args = ap.parse_args()

    X, y, train_idx, val_idx, test_idx, payments, events = build_prep()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    # 单分类：训练集【正常】样本
    tr = train_idx[y[train_idx] == 0]
    if args.limit:
        tr = tr[:args.limit]
    X_train_norm = X[tr].astype(np.float64)

    # 预处理：StandardScaler 仅在训练正常样本上 fit（与 DRL 一致）
    scaler = StandardScaler().fit(X_train_norm)
    Xtr = scaler.transform(X_train_norm).astype(np.float32)
    Xva = scaler.transform(X[val_idx]).astype(np.float32)
    Xte = scaler.transform(X[test_idx]).astype(np.float32)

    cfg = build_config(data_dim=X.shape[1], device=device, seed=args.seed,
                       epochs=args.epochs, batch=args.batch)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = PN(cfg).to(device)
    print(f"training on {len(Xtr)} normal samples, {cfg['epochs']} epochs", flush=True)
    train(model, Xtr, cfg)

    s_val = score(model, Xva, cfg)
    s_test = score(model, Xte, cfg)

    # 回填到全量分数（仅测试集参与最终指标；验证集用于冻结阈值）
    s_full = np.zeros(len(payments), dtype=np.float32)
    s_full[val_idx] = s_val
    s_full[test_idx] = s_test

    report("DRL", s_full, payments)
    np.save(os.path.join(HERE, "scores_drl.npy"), s_full)
    print("saved scores_drl.npy")


if __name__ == "__main__":
    main()
