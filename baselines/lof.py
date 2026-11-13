"""LOF 局部离群因子基线（密度基）。"""
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from src import feature_engineering
from baselines._common import _impute


def lof_scores(payments_df, ref_stats, train_idx, seed=10):
    F = feature_engineering.build_feature_matrix(payments_df, ref_stats)
    F = _impute(F)
    X = F.values
    # contamination 只影响 predict() 阈值、不影响 score_samples，故省略。
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True)
    clf.fit(X[train_idx])
    score = -clf.score_samples(X)   # 越大越异常
    return pd.Series(score, index=payments_df.index)
