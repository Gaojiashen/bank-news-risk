"""与已保存检查点兼容的BERT分类器结构。"""
import torch
from torch import nn
from transformers import BertModel
MODEL_NAME = "google-bert/bert-base-chinese"
DROPOUT = .1
class BertClassifier(nn.Module):
    """一套 BERT 编码器 + 一个分类层；每个任务各自实例化一份。"""

    def __init__(self, num_labels: int, config=None):
        super().__init__()
        self.bert = BertModel(config) if config is not None else BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        return self.classifier(self.dropout(out.pooler_output))
class MentionBertClassifier(nn.Module):
    """使用目标银行提及词元的末层平均表示进行分类。"""

    def __init__(self, num_labels, config=None):
        super().__init__()
        self.bert = BertModel(config) if config is not None else BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_segments = ((token_type_ids == 1) | (token_type_ids == 2)).long()
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=bert_segments).last_hidden_state
        target_mask = (token_type_ids >= 2).unsqueeze(-1)
        pooled = (hidden * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp_min(1)
        return self.classifier(self.dropout(pooled))
