# 数据与文件约定

## 原始与最终标注

`data/raw/news/YYYY/MM/YYYYMMDD.csv` 保留 `NewsID,DeclareDate,Title,NewsContent,NewsSource`。SourceRow按CSV记录号定位，表头为第1条，正文换行不改变记录号。

`bank_prices_raw.xlsx` 为原45家金融机构价格表，只提取16家银行；`bank_aliases.xlsx` 为原银行别名表。工作簿内容未变。

`data/annotation/relevance/` 保留相关/不相关标签并清空情绪列。`data/annotation/sentiment/` 原样保留最终正负数据。兼容原数据，列名 `积极中性消极` 仍存在，但情绪只允许-1/+1；不包含中性0。

`data/labeled_news/` 是片段级自动标注。共现按文章去重计边；语义按原文章—目标银行组合编码一次。

## 特征与风险

| 文件 | 内容 |
|---|---|
| `data/embeddings/YYYY/MM/YYYYMMDD.npz` | `embeddings [K,768]`、文章来源和行号、银行索引、截断诊断、`daily_bank_mean [16,768]`、`article_counts [16]` |
| `data/features/daily_news_cooccurrence.npz` | 正负原始计数 `[T,16,16]`，零对角线，日期及银行顺序 |
| `data/features/daily_semantic_features.npz` | 交易日文章等权语义均值 `[T,16,768]` 和文章数 `[T,16]` |
| 实验内 `semantic_pca.npz` | 训练期均值、主成分、得分尺度、解释方差 |
| `data/risk/price_analysis/` | 清洗价格、对数收益率、固定16家等权系统收益率及审计图表 |
| `data/risk/daily_delta_covar/daily_delta_covar_16_banks.npz` | 风险 `[日期,16]`、VaR、回归参数、状态和配对数 |

T表示使用者提供的交易日数量，K表示某自然日的文章—银行组合数。日期覆盖范围由实际输入决定，仓库不提供示例真实数据。

新闻在交易日结束后可用，周末和节假日归入下一交易日再预测随后一个交易日。语义均值按文章数加权，无新闻银行为零并附标记。PCA仅在训练期可见且有新闻的银行日拟合。

损失为负对数收益率。95%系统损失分位数回归下，`delta = beta * (bank_VaR95 - bank_VaR50)`；负估计保留，缺失不填，数值乘100后为百分点。

训练后本地生成的 `predictions.csv` 记录输入结束日、目标日、银行、实际风险、预测及沿用昨日值；这些结果不随代码发布。

## 本地资产与输入责任

所有数据和权重路径均为本地约定，代码仓库不含相应文件。数据、向量、模型和实验目录均由Git忽略。

价格工作簿需提供银行对应的价格列，别名工作簿提供目标银行及别名；实际读取约定见 `prepare_bank_prices.py` 和 `common/banks.py`。使用者必须自行取得有权使用的数据。

迁移校验功能只服务于已经存在本地资产的项目，迁移清单、来源指纹和原始标注裁决均不发布；新提取任务可在新输出目录生成自己的来源记录。
