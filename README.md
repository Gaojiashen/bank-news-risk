# 银行新闻语义与日度系统性风险预测

研究代码实现银行新闻目标级相关性与情绪分类、新闻共现图、BERT语义特征、日度 ΔCoVaR，以及图循环神经网络预测。

**本仓库只发布代码、配置、方法说明和合成测试。不提供原始或标注数据、语义向量、模型权重、实验结果。数据访问权限不可转授。**本地保有这些文件的研究者可按目录约定运行完整流程；仅下载本仓库不能直接复现实验分数。

## 环境

需要Python 3.13及以上。依赖和版本锁定见 `pyproject.toml`、`uv.lock`。

```powershell
uv sync --locked
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

当前锁定环境使用PyTorch CUDA 13.0发行源。CPU环境可先安装CPU版PyTorch，再安装其余依赖：

```text
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install openpyxl transformers scikit-learn numpy matplotlib scipy pyyaml
python -m unittest discover -s tests -t . -v
```

CPU安装方式不等同于锁定环境。基础测试只生成合成输入，不加载真实数据或下载BERT权重。三个依赖私有数据/权重/结果的集成检查默认跳过。已有私有资产的本地环境可显式启用：

```powershell
$env:RUN_PRIVATE_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
Remove-Item Env:RUN_PRIVATE_INTEGRATION
```

## 目录

| 目录 | 内容 |
|---|---|
| code/annotation/ | 抽样、文章级划分和标签校验 |
| code/text_models/ | 相关性和情绪模型训练 |
| code/news_features/ | 标注、向量提取、交易日聚合和新闻图 |
| code/risk_measure/ | 价格收益率清洗与日度 ΔCoVaR |
| code/forecasting/ | 风险网络、PCA、训练评估 |
| code/common/ | 公共文本、图、回归、配置与来源校验 |
| code/analysis/ | 结果出图和文本错例导出 |
| configs/ | 方法配置，路径指向用户本地资产 |
| docs/ | 方法及输入格式说明，不含实验结果 |
| tests/ | 合成测试和显式启用的私有集成检查 |

## 方法

1. 从按日期整理的新闻抽样，依据规则完成目标银行相关性和正负情绪标注。
2. 按原文章分组划分数据集，避免同篇文章跨集合。
3. 独立训练相关性BERT和情绪BERT。情绪模型支持CLS或目标银行提及位置池化，以及普通或温和平衡采样。
4. 标注新闻，并为每个文章—银行组合提取一个768维向量；按文章等权聚合到交易日。
5. 构造正负共现图，估计日度 ΔCoVaR。
6. 用60个交易日历史预测下一交易日风险，比较注意力偏置和扩散模型。可选语义经训练期PCA降至16维后加入融合GRU，不使用宏观特征。

参数和实验输出路径由YAML配置指定，实际输入需由有相应权限的使用者自行准备，格式见 `docs/data_dictionary.md`。

## 完整流程入口（需自行准备数据）

```powershell
.\.venv\Scripts\python.exe -m code.annotation.validate_annotations
.\.venv\Scripts\python.exe -m code.text_models.train_relevance --check
.\.venv\Scripts\python.exe -m code.text_models.train_sentiment --check
.\.venv\Scripts\python.exe -m code.news_features.annotate_daily_news --help
.\.venv\Scripts\python.exe -m code.news_features.extract_news_embeddings --help
.\.venv\Scripts\python.exe -m code.news_features.build_daily_news_graphs --help
.\.venv\Scripts\python.exe -m code.news_features.aggregate_daily_embeddings --help
.\.venv\Scripts\python.exe -m code.risk_measure.prepare_bank_prices
.\.venv\Scripts\python.exe -m code.risk_measure.compute_daily_delta_covar
.\.venv\Scripts\python.exe -m code.forecasting.train_risk --config configs/risk_graph.yaml --check
.\.venv\Scripts\python.exe -m code.forecasting.train_risk --config configs/risk_graph_semantic.yaml --check
```

这些命令不是无数据演示。文本训练/推理需要自行取得BERT基础模型及相应微调权重；推理入口优先从本地缓存读取架构配置，不提供项目训练权重的下载。

训练入口移除 `--check` 才训练。正式模型预期位于 `models/relevance_bert/` 和 `models/sentiment_bert_mention_balanced/`；训练产物需由使用者自行选定并部署到这些本地位置。新风险实验可通过 `--output` 指定新目录。

## 发布范围

`.gitignore` 排除 `data/`、`models/`、`experiments/`、标注裁决记录、本地迁移报告、数据文件及权重。代码中的路径仅是输入格式约定，仓库不附带对应资产。不得将代码说明理解为对第三方数据的访问或再分发许可。
