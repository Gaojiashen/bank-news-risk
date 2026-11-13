"""使用已选模型进行两阶段文本推理，不执行训练或历史标注任务。"""
import torch
from transformers import AutoTokenizer, BertConfig
from . import text_training as base
from . import text_features as sentiment_code
from .text_models import MentionBertClassifier
from .banks import load_aliases, split_sentences, mentioned_banks
from .paths import RELEVANCE_MODEL as RELEVANCE_DIR, SENTIMENT_MODEL as SENTIMENT_DIR
def load_model(model_dir, model_type, device):
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    if checkpoint["num_labels"] != 2:
        raise ValueError(f"模型不是二分类：{model_dir}")
    expected = ["不相关", "相关"] if model_type == "relevance" else ["消极", "积极"]
    if checkpoint["label_names"] != expected:
        raise ValueError(f"模型标签顺序异常：{model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", use_fast=True, local_files_only=True)
    config = BertConfig.from_pretrained(base.MODEL_NAME, local_files_only=True)
    model = (base.BertClassifier(2, config=config) if model_type == "relevance"
             else MentionBertClassifier(2, config=config))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return tokenizer, model


@torch.inference_mode()
def classify(rows, tokenizer, model, model_type, device):
    inputs = [{"bank": row["BankName"], "content": row["NewsContent"]} for row in rows]
    features = (base.build_features(inputs, tokenizer, base.MAX_LEN) if model_type == "relevance"
                else sentiment_code.build_mention_features(inputs, tokenizer, base.MAX_LEN))
    width = max(len(feature["input_ids"]) for feature in features)
    ids = torch.full((len(rows), width), base.PAD_TOKEN_ID, dtype=torch.long)
    mask = torch.zeros_like(ids)
    segments = torch.zeros_like(ids)
    for index, feature in enumerate(features):
        length = len(feature["input_ids"])
        ids[index, :length] = torch.tensor(feature["input_ids"])
        mask[index, :length] = torch.tensor(feature["attention_mask"])
        segments[index, :length] = torch.tensor(feature["token_type_ids"])
    logits = model(ids.to(device), mask.to(device), segments.to(device))
    return logits.argmax(dim=1).tolist()
