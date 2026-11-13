# MS-TRP：预算执行异常监控的多尺度时序建模与关系证据分析

本仓库是论文《面向预算执行异常监控的时间可追溯多尺度风险建模与关系证据分析》的研究代码、仿真数据生成器与制图脚本。

方法由三部分组成：**多尺度时序建模**（Self/Peer 参照的稳健方向评分）、**阶段感知的异构图传播**（关系亲和 + 个性化 PageRank 扩散）、**可靠度门控融合**，输出逐支付异常分数并配合分层收缩阈值给出告警与审计证据。

## 目录结构

```
src/                 方法实现（特征、图构建、PPR 传播、门控、校准、切分、指标）
experiment/          实验脚本（主流程、消融、冷启动、事件评估、多 seed、误报率等）
baselines/           项目内实现的无监督对比方法（MAD、LOF、iForest、规则、预测残差等）
baselines_2025/      外部/新近方法复现：DRL、FoMo-0D（零样本 PFN）、Outformer、TCCM
config/              数据契约、超参数、切分与实验协议（YAML/JSON）
tests/               单元测试（防未来信息泄漏、事件匹配、指标等）
仿真V3.py / 仿真V4.py  预算执行流水仿真生成器
result_summary/      论文用的小体量汇总表
wordwenjian1/        论文与制图：LaTeX 源、矢量图、DOCX 生成与修补脚本
```

## 数据说明

仓库**不包含** `data_fixed/`、`data_fixed_b/`（合计约 850 MB）与 `result_raw/` 下的大体量中间件（特征表、传播边、逐 seed 结果）。仿真数据由 `仿真V3.py` 按 `config/data.yaml` 中的 `data_seeds` 生成，可重建；论文中的全部数字均来自该生成器的固定种子。

为保证论文结果可核对，保留了小体量产物：

- `result_raw/metrics_main.csv`、`run_ablation/`、`run_robustness/`、`window_ablation/`、`cold_start.csv`、`frozen_threshold_far_by_period.csv`、`all_seed_results.csv`、`seed_diagnostics.csv`
- `result_raw/run_main/predictions.csv`：主模型逐支付打分（约 39 MB），图 2 / 图 6 直接读取
- `baselines_2025/scores_*.npy`：三种对比方法的逐支付分数

未入库：`baselines_2025/FoMo-0D/ckpt.zip`（预训练权重）与 `prep_*.pkl/npz` 预处理缓存，可按 `baselines_2025/_prep.py` 重新生成。

## 环境

Python 3.10，依赖见 `requirements.txt`。全流程可在 CPU 上运行；仅 FoMo-0D 基线需要 PyTorch。

```bash
pip install -r requirements.txt
git submodule update --init --recursive   # 拉取 FoMo-0D 上游代码（对比基线）
```

论文与配图的 PDF 文件不入库（`.gitignore` 中排除 `*.pdf`），仅在本地生成与保存。

## 复现

所有脚本均在项目根目录执行（脚本内部会自行把根目录加入 `sys.path`）：

```bash
python 仿真V3.py                     # 生成仿真数据到 data_fixed/
python experiment/run_main.py        # 主模型端到端评估 -> result_raw/run_main/
python experiment/run_ablation.py    # 消融实验
python experiment/run_cold_start.py  # 历史跨度（冷启动）实验
python experiment/run_events.py      # 事件级召回与匹配
python experiment/run_far_periods.py # 冻结阈值下的分季度误报
python experiment/run_e08_seeds.py   # data_seed × model_seed 多随机种子实验
python experiment/build_metrics_main.py   # 汇总主表 metrics_main.csv
```

测试：

```bash
python -m pytest tests/
```

## 论文与制图

- LaTeX 源：`wordwenjian1/latex/main_ctexart.tex`（Elsevier CAS 模板，XeLaTeX 编译三遍），`main.tex` 由 `wordwenjian1/make_cas_latex.py` 生成。
- 图：`wordwenjian1/redraw/` 下的脚本生成图；`make_vector_figs.py` 会同时输出 PDF/SVG 矢量版本，供 LaTeX 与 Word（svgBlip）使用。

## 说明

本仓库为研究原型，用于论文结果复现，未做工程化部署优化。数据为仿真生成，不含任何真实预算单位信息。
