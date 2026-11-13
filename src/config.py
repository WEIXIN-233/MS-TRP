"""
MS-TRP 检测框架 —— 全局配置
========================================
统一管理路径、随机种子、类别常量、时间前向划分边界、业务阈值。
所有阈值/参数集中在 CFG，与生成器(仿真V3.py)解耦。
"""
import os

# ── 路径 ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # src/
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                        # MS-TRP/
DB_PATH = os.path.join(PROJECT_DIR, "data_fixed", "budget_data_v5.db")
RESULT_DIR = os.path.join(PROJECT_DIR, "result_raw")
os.makedirs(RESULT_DIR, exist_ok=True)

# ── 随机种子（指导书：≥3 个独立 data_seed；每个 data_seed 内换 model_seed 各跑 5 次）──
DATA_SEEDS = [10, 20, 30]          # 独立数据场景（生成器种子）
MODEL_SEEDS = [1, 2, 3, 4, 5]      # 每个数据种子内的模型随机种子（5 次）
SEEDS = list(DATA_SEEDS)           # 兼容旧引用（run_e08_seeds 等）


def data_dir(seed: int) -> str:
    """data_fixed/seed{N}/ 数据目录（多种子独立归档，不互相覆盖）。"""
    return os.path.join(PROJECT_DIR, "data_fixed", f"seed{seed}")


def set_data_seed(seed: int) -> str:
    """把当前 DB_PATH 切到 data_fixed/seed{N}/budget_data_v5.db，返回新路径。"""
    global DB_PATH
    DB_PATH = os.path.join(data_dir(seed), "budget_data_v5.db")
    return DB_PATH

# ── 时间前向划分（指导书表5：train < val < test，严格无泄漏）──
# train=2023-01-01~2024-12-31, val=2025 全年, test=2026 全年
TRAIN_START, TRAIN_END = "2023-01-01", "2024-12-31"
VAL_START, VAL_END     = "2025-01-01", "2025-12-31"
TEST_START, TEST_END   = "2026-01-01", "2026-12-31"

# ── 评分时点语义（指导书 §4.3；P0-3）──────────────────────────
# "day_end"：每笔按 apply_date 当日日终的可见快照评分（图分支日终批量；
#   时序分支 s_ts 严格时间前向，同日多笔互不进入对方历史统计）。
SCORE_TIME_MODE = "day_end"

# ── 类别常量 ───────────────────────────────────────────
ANOMALY_TYPES = [
    "AMOUNT_SPIKE",          # 单笔金额异常
    "FREQ_BURST",            # 支付频次突增
    "PROGRESS_LAG",          # 执行进度滞后
    "YEAREND_CONC",          # 年末集中支付
    "HISTORICAL_SHIFT",      # 历史节奏偏离
    "ENTITY_AGGREGATION",    # 关联实体异常聚集
]

# 点级异常（逐笔错）：适合"点级 F1"
POINT_ANOMALY_TYPES = ["AMOUNT_SPIKE", "FREQ_BURST"]
# 模式级异常（成组错）：适合"事件级 F1"
EVENT_ANOMALY_TYPES = ["PROGRESS_LAG", "YEAREND_CONC", "HISTORICAL_SHIFT", "ENTITY_AGGREGATION"]

# 4 种支付方式（一体化真实口径；工资统发=代发渠道非支付方式，财政专户拨款=教育收费的特殊操作，均不单列）
PAYMENT_TYPES = [
    "UNIT_ZERO_BALANCE", "FINANCE_ZERO_BALANCE", "SPECIAL_DEPOSIT",
    "DIRECT_PAYMENT",
]

# ── 业务阈值（与生成器一致的公开口径，用于固定规则基线）────────
LARGE_PAY_THRESHOLD_WAN = 100.0          # 区级大额支付阈值(万元)

# ── 固定规则基线的可配置参数 ────────────────────────────
FIXED_RULES = {
    "amount_abs_threshold_wan": 100.0,     # 绝对大额阈值
    "amount_median_mult": 5.0,             # 相对实体历史中位数的放大倍数
    "amount_mean_mult": 3.0,               # 相对实体历史均值的放大倍数
}

# ── 鲁棒统计(MAD)基线参数 ───────────────────────────────
MAD_K = 3.5                                # |x - median| / MAD > k → 异常

# ── 无监督基线通用参数 ─────────────────────────────────
CONTAMINATION = 0.10                       # IF/LOF 的异常比例先验(粗估)
