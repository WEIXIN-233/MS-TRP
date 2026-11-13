"""Isolation Forest 无监督基线。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import feature_engineering
from baselines._common import _impute


def isolation_forest_scores(payments_df, ref_stats, train_idx, seed=10):
    F = feature_engineering.build_feature_matrix(payments_df, ref_stats)
    F = _impute(F)
    X = F.values
    # contamination 只影响 predict() 阈值、不影响 score_samples，故省略；
    # n_jobs=1 规避 Windows 中文路径下 joblib 多进程偶发 CreateProcess 失败。
    clf = IsolationForest(random_state=seed, n_estimators=200, n_jobs=1)
    clf.fit(X[train_idx])
    score = -clf.score_samples(X)   # 越大越异常
    return pd.Series(score, index=payments_df.index)
