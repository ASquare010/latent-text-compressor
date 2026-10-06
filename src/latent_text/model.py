"""Position-preserving span compression with a small, parallel reconstruction decoder.

There are no encoder-to-decoder skip connections or supplied output tokens.
Exact token lengths are explicit metadata and must be counted when storing memory.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return (normalized * self.weight).to(x.dtype)


def parameter_generator(seed: int, name: str) -> torch.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


class CurveFFN(nn.Module):
    """Mix features, apply learned curves, then mix features again."""

    def __init__(self, config):
        super().__init__()
        self.width = config.width
        self.up = nn.Linear(config.width, config.width, bias=False)
        self.down = nn.Linear(config.width, config.width, bias=False)
        self.initial_slopes = [0.5, 1.0, 1.5]
        self.initial_offsets = [-1.0, 0.0, 1.0]
        templates = torch.tensor(self.initial_slopes)[:, None].expand(3, config.width).clone()
        offsets = torch.tensor(self.initial_offsets)[:, None].expand(3, config.width).clone()
        # Keep parameter names compatible with the measured checkpoints.
        self.a = nn.Parameter(templates)
        self.b = nn.Parameter(offsets)
        self.c = nn.Parameter(torch.ones(3, config.width))
        self.e = nn.Parameter(torch.zeros(3, config.width))
        self.initial_mean, self.scale = calibration(config.width)
        self.projection_parameters = 2 * config.width**2

    def forward(self, x):
        z = self.up(x)  # Mix the token's features: width -> width.
        output_dtype = z.dtype
        # Keep curve arithmetic in FP32 during mixed-precision training.
        z = z if z.dtype == torch.float64 else z.float()

        value = torch.zeros_like(z)
        for branch in range(3):
            response = F.silu(self.a[branch] * z + self.b[branch])
            multiplier = self.c[branch] * z + self.e[branch]
            value = value + response * multiplier

        value = self.scale * (value / math.sqrt(3) - self.initial_mean)
        return self.down(value.to(output_dtype))  # Mix features back into the output.

    @torch.no_grad()
    def initialize(self, seed, prefix, residual_scale):
        self.up.weight.normal_(0, 0.02, generator=parameter_generator(seed, f"{prefix}.up.weight"))
        self.down.weight.normal_(
            0, 0.02 * residual_scale, generator=parameter_generator(seed, f"{prefix}.down.weight")
        )


def calibration(width, nodes=128):
    """Fixed starting mean and scale; preserve the measured initialization."""
    points, weights = np.polynomial.hermite.hermgauss(nodes)
    z = torch.from_numpy(points) * math.sqrt(2 * width * 0.02**2)
    w = torch.from_numpy(weights) / math.sqrt(math.pi)
    slopes = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    offsets = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    branches = F.silu(slopes[:, None] * z + offsets[:, None])
    variance = width * 0.02**2
    target = (1376 / 512) * variance * float((w * F.silu(z).square()).sum())
    raw = z * branches.sum(0) / math.sqrt(3)
    mean = float((raw * w).sum())
    initial_variance = float(((raw - mean).square() * w).sum())
    return mean, math.sqrt(target / initial_variance)


@dataclass
class Config:
    vocab_size: int = 4096
    width: int = 256
    heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 1
    max_tokens: int = 256
    span: int = 8
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def __post_init__(self):
        for name in (
            "vocab_size",
            "width",
            "heads",
            "encoder_layers",
            "decoder_layers",
            "max_tokens",
            "span",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.width % self.heads or self.span > self.max_tokens:
            raise ValueError("Width must divide into heads; span must fit max_tokens")
        if (self.pad_id, self.bos_id, self.eos_id) != (0, 1, 2):
            raise ValueError("Data format uses pad=0, bos=1, eos=2")


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.query = nn.Linear(config.width, config.width, bias=False)
        self.key = nn.Linear(config.width, config.width, bias=False)
        self.value = nn.Linear(config.width, config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x, context, valid, causal=False):
        def split(t):
            b, n, d = t.shape
            return t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(context)), split(self.value(context))
        allowed = valid[:, None, None, :]
        if causal:
            allowed = (
                allowed
                & torch.ones(x.shape[1], context.shape[1], device=x.device, dtype=torch.bool).tril()
            )
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.out(y.transpose(1, 2).reshape_as(x))


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.width)
        self.ffn = CurveFFN(config)

    def forward(self, x, mask):
        normalized = self.attention_norm(x)
        x = x + self.attention(normalized, normalized, mask)
        return x + self.ffn(self.ffn_norm(x))


class Model(nn.Module):
    """L token features -> ceil(L/span) memories -> L token predictions."""

    def __init__(self, config=None, seed=17):
        super().__init__()
        self.config = config or Config()
        c = self.config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(c.vocab_size, c.width)
            self.position = nn.Embedding(c.max_tokens, c.width)
            self.encoder = nn.ModuleList([EncoderBlock(c) for _ in range(c.encoder_layers)])
            self.encoder_norm = RMSNorm(c.width)
            # Concatenation preserves the order inside each span, unlike weighted averaging.
            self.compress = nn.Linear(c.span * c.width, c.width, bias=False)
            self.expand = nn.Linear(c.width, c.span * c.width, bias=False)
            self.decoder = nn.ModuleList([EncoderBlock(c) for _ in range(c.decoder_layers)])
            self.decoder_norm = RMSNorm(c.width)
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(module.weight, std=0.02)
            for name, module in self.named_modules():
                if isinstance(module, CurveFFN):
                    module.initialize(seed, name, 1 / math.sqrt(2 * c.encoder_layers))

    def encode(self, tokens, mask):
        c = self.config
        if tokens.ndim != 2 or mask.shape != tokens.shape or mask.dtype != torch.bool:
            raise ValueError("Expected token IDs and a same-shaped boolean mask")
        if not 0 < tokens.shape[1] <= c.max_tokens or not mask.any(1).all():
            raise ValueError("Input must have 1..max_tokens real tokens")
        if (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("Use right padding")
        x = self.embedding(tokens) + self.position(
            torch.arange(tokens.shape[1], device=tokens.device)
        )
        for block in self.encoder:
            x = block(x, mask)
        x = self.encoder_norm(x) * mask.unsqueeze(-1)
        x = F.pad(x, (0, 0, 0, (-x.shape[1]) % c.span))
        z = self.compress(x.reshape(x.shape[0], -1, c.span * c.width))
        return z, mask.sum(1)

    def decode(self, z, lengths):
        c = self.config
        if (
            z.ndim != 3
            or z.shape[2] != c.width
            or lengths.shape != (z.shape[0],)
            or lengths.dtype != torch.long
            or (lengths < 1).any()
            or (lengths > c.max_tokens).any()
            or (lengths > z.shape[1] * c.span).any()
        ):
            raise ValueError("Invalid memories or token-length metadata")
        x = self.expand(z).reshape(z.shape[0], -1, c.width)[:, : int(lengths.max())]
        positions = torch.arange(x.shape[1], device=x.device)
        mask = positions[None] < lengths[:, None]
        x = x + self.position(positions)
        for block in self.decoder:
            x = block(x, mask)
        return F.linear(self.decoder_norm(x), self.embedding.weight)

    def forward(self, tokens, mask):
        return self.decode(*self.encode(tokens, mask))

    @torch.no_grad()
    def generate(self, z, lengths):
        self.eval()
        logits = self.decode(z, lengths)
        logits[..., :3] = -torch.inf
        return logits.argmax(-1)
