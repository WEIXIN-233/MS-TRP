"""
MS-TRP 检测框架 —— 数据划分
========================================
1) time_forward_split: 按日期区间 train(2023~2024)/val(2025)/test(2026)，严格时间前向无泄漏
2) entity_disjoint_split: 按单位互斥划分(同一单位的支付链不跨集合)，用于补充验证
"""
import pandas as pd
import numpy as np

from src import config


def time_forward_split(payments_df: pd.DataFrame):
    """按日期区间切 train/val/test（指导书表5）。返回三个子集的索引(在原 DataFrame 上的位置)。"""
    d = payments_df["apply_date"]
    train_idx = payments_df.index[(d >= config.TRAIN_START) & (d <= config.TRAIN_END)].tolist()
    val_idx = payments_df.index[(d >= config.VAL_START) & (d <= config.VAL_END)].tolist()
    test_idx = payments_df.index[(d >= config.TEST_START) & (d <= config.TEST_END)].tolist()
    return train_idx, val_idx, test_idx


def entity_disjoint_split(payments_df: pd.DataFrame, entity_col: str = "unit_name",
                          test_ratio: float = 0.3, seed: int = 2026):
    """
    按实体(默认单位)互斥划分：test 抽取 test_ratio 比例的实体，其余做 train。
    保证同一实体的所有支付只落在一边，避免同一支付链跨集合。
    返回 train_idx, test_idx。
    """
    rng = np.random.RandomState(seed)
    entities = payments_df[entity_col].unique()
    n_test = max(1, int(len(entities) * test_ratio))
    test_entities = set(rng.choice(entities, n_test, replace=False))
    mask = payments_df[entity_col].isin(test_entities)
    test_idx = payments_df.index[mask].tolist()
    train_idx = payments_df.index[~mask].tolist()
    return train_idx, test_idx
