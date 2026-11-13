"""统一项目路径；原始文章标识仍以NEWS目录为相对基准。"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
NEWS = ROOT / "data/raw/news"
LABELED_NEWS = ROOT / "data/labeled_news"
EMBEDDINGS = ROOT / "data/embeddings"
FEATURES = ROOT / "data/features"
RISK = ROOT / "data/risk"
RELEVANCE_MODEL = ROOT / "models/relevance_bert"
SENTIMENT_MODEL = ROOT / "models/sentiment_bert_mention_balanced"
ALIASES = ROOT / "data/raw/bank_aliases.xlsx"
