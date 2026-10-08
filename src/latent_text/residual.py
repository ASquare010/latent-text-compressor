"""Selected plain residual-attention encoder, Branch Sigmoid FFNs and RoPE.

Attention selects among previous sublayer outputs at each token. Ordered,
factorized maps pack tokens into vectors; reconstruction has no feature skip.
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .model import Config as BaseConfig, Model as BaseModel

@dataclass
class Config(BaseConfig):
    max_tokens: int = 512
    span: int = 64
    variant: str = "depth_route"
    code_features: int = 125

    def __post_init__(self):
        super().__post_init__()
        if self.variant != "depth_route" or type(self.code_features) is not int or self.code_features < 1:
            raise ValueError("Expected plain depth_route and positive code_features")

class Pack(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.span = c.span
        self.width = c.width
        self.token = nn.Linear(c.width, c.code_features, bias=False)
        self.span_map = nn.Linear(c.span * c.code_features, c.width, bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(481)
            nn.init.orthogonal_(self.token.weight)
            nn.init.normal_(self.span_map.weight, std=0.02)

    def forward(self, x):
        x = x.reshape(*x.shape[:-1], self.span, self.width)
        h = self.token(x)
        return self.span_map(h.flatten(-2))


class Unpack(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.span = c.span
        self.features = c.code_features
        self.span_map = nn.Linear(c.width, c.span * c.code_features, bias=False)
        self.token = nn.Linear(c.code_features, c.width, bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(482)
            nn.init.normal_(self.span_map.weight, std=0.02)
            nn.init.orthogonal_(self.token.weight)

    def forward(self, z):
        h = self.span_map(z).reshape(*z.shape[:-1], self.span, self.features)
        return self.token(h).flatten(-2)


class Model(BaseModel):
    def __init__(self, config=None, seed=17):
        c = config or Config()
        super().__init__(c, seed)
        self.compress = Pack(c)
        self.expand = Unpack(c)
        self.depth_queries = nn.Parameter(torch.zeros(2 * c.encoder_layers, c.width))

    def encoder_parameters(self):
        return sum(
            p.numel()
            for name, p in self.named_parameters()
            if not name.startswith(("decoder.", "decoder_norm.", "expand."))
        )

    def mix_depth(self, values, index):
        if len(values) == 1:
            return values[0]
        stack = torch.stack(values).float()
        keys = stack * torch.rsqrt(stack.square().mean(-1, keepdim=True) + 1e-5)
        logits = torch.einsum(
            "nbtd,d->nbt", keys, self.depth_queries[index - 1 if index >= 0 else index].float()
        ) / math.sqrt(self.config.width)
        weights = logits.softmax(0)
        # At zero queries this recovers the usual sum, easing transfer from the parent.
        return (len(values) * (weights[..., None] * stack).sum(0)).to(values[0].dtype)

    def encode(self, tokens, mask):
        c = self.config
        if tokens.ndim != 2 or mask.shape != tokens.shape or mask.dtype != torch.bool:
            raise ValueError("Invalid tokens/mask")
        if (
            not 0 < tokens.shape[1] <= c.max_tokens
            or not mask.any(1).all()
            or (mask[:, 1:] & ~mask[:, :-1]).any()
        ):
            raise ValueError("Expected nonempty right padded inputs")
        values = [self.embedding(tokens)]
        for i, block in enumerate(self.encoder):
            h = block.attention_norm(self.mix_depth(values, 2 * i))
            values.append(block.attention(h, h, mask))
            values.append(block.ffn(block.ffn_norm(self.mix_depth(values, 2 * i + 1))))
        x = self.encoder_norm(self.mix_depth(values, -1)) * mask.unsqueeze(-1)
        x = F.pad(x, (0, 0, 0, (-x.shape[1]) % c.span))
        return self.compress(x.reshape(x.shape[0], -1, c.span * c.width)), mask.sum(1)

