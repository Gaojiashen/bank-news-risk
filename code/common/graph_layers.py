"""基于已观测新闻边的注意力与两步图扩散算子。"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def _off_diagonal(adjacency: torch.Tensor) -> torch.Tensor:
    n = adjacency.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=adjacency.device)
    return adjacency.masked_fill(eye, 0)


def diffusion_steps(values: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对当前时点的边计数取对数并按行归一化，计算一步与两步传播。"""
    weights = torch.log1p(_off_diagonal(adjacency))
    transition = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    one = torch.bmm(transition, values)
    two = torch.bmm(transition, one)
    return one, two


class EdgeBiasedAttention(nn.Module):
    """在全部已观测新闻边上执行四头注意力，以共现次数构造偏置。"""

    def __init__(self, hidden: int = 32, heads: int = 4) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden, self.heads, self.width = hidden, heads, hidden // heads
        self.query = nn.Linear(hidden + 1, hidden)
        self.key = nn.Linear(hidden + 1, hidden)
        self.value = nn.Linear(hidden + 1, hidden)
        self.output = nn.Linear(hidden, hidden, bias=False)
        # 此初值经softplus变换后约为1，保证新闻强度项的系数非负。
        self.raw_edge_bias = nn.Parameter(torch.full((heads,), 0.54132485))

    def forward(self, risk: torch.Tensor, previous: torch.Tensor,
                adjacency: torch.Tensor, return_weights: bool = False):
        batch, banks, _ = risk.shape
        node = torch.cat((risk, previous), dim=-1)
        def split(linear: nn.Linear) -> torch.Tensor:
            return linear(node).reshape(batch, banks, self.heads, self.width).transpose(1, 2)
        q, k, v = split(self.query), split(self.key), split(self.value)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.width)
        score = score + F.softplus(self.raw_edge_bias)[None, :, None, None] * torch.log1p(adjacency)[:, None]
        nondiagonal = ~torch.eye(banks, dtype=torch.bool, device=adjacency.device)
        mask = (adjacency > 0) & nondiagonal
        score = score.masked_fill(~mask[:, None], -1e9)
        weights = torch.softmax(score, dim=-1) * mask[:, None]
        # 无邻居的行输出严格为零的消息，避免产生均匀注意力。
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        combined = torch.matmul(weights, v).transpose(1, 2).reshape(batch, banks, self.hidden)
        message = self.output(combined)
        return (message, weights) if return_weights else message


class NewsDiffusion(nn.Module):
    """学习新闻网络一步和两步传播消息的组合。"""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.node = nn.Linear(hidden + 1, hidden)
        self.mix = nn.Linear(2 * hidden, hidden, bias=False)

    def forward(self, risk: torch.Tensor, previous: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        state = self.node(torch.cat((risk, previous), dim=-1))
        one, two = diffusion_steps(state, adjacency)
        return F.relu(self.mix(torch.cat((one, two), dim=-1)))
