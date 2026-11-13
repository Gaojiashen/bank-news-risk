"""日度风险基准及正负图分支、融合GRU模型。

输入风险形状为[B,L,N]，正负新闻计数为[B,L,N,N]。
风险值在每个图消息算子中使用一次，情绪分支GRU只接收图消息；
当天本行风险与分支状态、边强度共同进入融合投影层。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from code.common.graph_layers import EdgeBiasedAttention, NewsDiffusion, _off_diagonal


class DailyBenchmarkRiskGRU(nn.Module):
    def __init__(self, hidden: int = 32, semantic_dim: int = 0) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.project = nn.Linear(1 + semantic_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, risk, positive=None, negative=None, return_details=False, semantic=None):
        batch, steps, banks = risk.shape
        inputs = risk.unsqueeze(-1)
        if self.semantic_dim:
            if semantic is None or semantic.shape != (*risk.shape, self.semantic_dim):
                raise ValueError("Semantic benchmark requires aligned semantic features")
            inputs = torch.cat((inputs, semantic), dim=-1)
        inputs = F.relu(self.project(inputs))
        inputs = inputs.permute(0, 2, 1, 3).reshape(batch * banks, steps, -1)
        _, state = self.gru(inputs)
        prediction = risk[:, -1] + self.head(state[-1]).reshape(batch, banks)
        return (prediction, {}) if return_details else prediction


class DailyNewsNetworkRiskModel(nn.Module):
    def __init__(self, kind: str, hidden: int = 32, heads: int = 4, semantic_dim: int = 0) -> None:
        super().__init__()
        if kind not in ("attention_bias", "diffusion"):
            raise ValueError(f"Unknown graph operator: {kind}")
        self.kind, self.hidden = kind, hidden
        self.semantic_dim = semantic_dim
        graph = (lambda: EdgeBiasedAttention(hidden, heads)) if kind == "attention_bias" else (lambda: NewsDiffusion(hidden))
        self.positive_message = graph()
        self.negative_message = graph()
        self.positive_gru = nn.GRUCell(hidden, hidden)
        self.negative_gru = nn.GRUCell(hidden, hidden)
        self.fusion_project = nn.Linear(2 * hidden + 3 + semantic_dim, hidden)
        self.fusion_gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, risk, positive, negative, return_details=False, semantic=None):
        batch, steps, banks = risk.shape
        if self.semantic_dim and (semantic is None or semantic.shape != (*risk.shape, self.semantic_dim)):
            raise ValueError("News model requires aligned semantic features")
        pos_state = risk.new_zeros(batch, banks, self.hidden)
        neg_state = torch.zeros_like(pos_state)
        fusion_state = torch.zeros_like(pos_state)
        details = {}
        for t in range(steps):
            current = risk[:, t, :, None]
            if self.kind == "attention_bias" and return_details and t == steps - 1:
                pos_message, pos_weights = self.positive_message(current, pos_state, positive[:, t], True)
                neg_message, neg_weights = self.negative_message(current, neg_state, negative[:, t], True)
                details = {"positive_attention": pos_weights, "negative_attention": neg_weights}
            else:
                pos_message = self.positive_message(current, pos_state, positive[:, t])
                neg_message = self.negative_message(current, neg_state, negative[:, t])
            pos_state = self.positive_gru(pos_message.reshape(batch * banks, -1),
                                          pos_state.reshape(batch * banks, -1)).reshape(batch, banks, -1)
            neg_state = self.negative_gru(neg_message.reshape(batch * banks, -1),
                                          neg_state.reshape(batch * banks, -1)).reshape(batch, banks, -1)
            pos_strength = torch.log1p(_off_diagonal(positive[:, t]).sum(-1, keepdim=True))
            neg_strength = torch.log1p(_off_diagonal(negative[:, t]).sum(-1, keepdim=True))
            fused = torch.cat((pos_state, pos_strength, neg_state, neg_strength, current), dim=-1)
            if self.semantic_dim:
                fused = torch.cat((fused, semantic[:, t]), dim=-1)
            fused = F.relu(self.fusion_project(fused)).reshape(batch * banks, -1)
            fusion_state = self.fusion_gru(fused, fusion_state.reshape(batch * banks, -1)).reshape(batch, banks, -1)
        prediction = risk[:, -1] + self.head(fusion_state).squeeze(-1)
        return (prediction, details) if return_details else prediction
