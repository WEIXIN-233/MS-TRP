"""基线方法共享助手。"""
import numpy as np
import pandas as pd


def _amount(payments_df):
    """支付金额(万元)列。"""
    return payments_df["amount_wan"]


def _impute(F: pd.DataFrame) -> pd.DataFrame:
    """特征矩阵 NaN 填充。"""
    F = F.fillna(F.median())
    F = F.replace([np.inf, -np.inf], 0.0)
    return F
