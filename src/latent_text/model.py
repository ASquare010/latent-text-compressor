"""Position-preserving span compression with a small, parallel reconstruction decoder.

There are no encoder-to-decoder skip connections or supplied output tokens.
Exact token lengths are explicit metadata and must be counted when storing memory.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ffn import CompressorBranchFFN, parameter_generator


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return (normalized * self.weight).to(x.dtype)


@dataclass
class Config:
    vocab_size: int = 4096
    width: int = 256
    heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 1
    max_tokens: int = 256
    span: int = 32
    hidden: int = 1024
    ffn: str = "branch_sigmoid_v1"
    position_encoding: str = "rope_v1"
    rope_base: float = 10000.0
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    positional: str = "learned"
    rope_base: float = 10000.0
    residual_scale_reference_layers: int | None = None

    def __post_init__(self):
        if self.ffn != "branch_sigmoid_v1" or type(self.hidden) is not int or self.hidden < 1:
            raise ValueError("Expected branch_sigmoid_v1 and positive hidden calibration width")
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
<<<<<<< Updated upstream
        if self.position_encoding != "rope_v1" or self.width // self.heads % 2:
            raise ValueError("RoPE requires an even head dimension and rope_v1")
        if not math.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError("rope_base must be finite and greater than one")
=======
        if self.positional not in ("learned", "rope") or self.rope_base <= 0:
            raise ValueError("Expected learned or rope positions and positive RoPE base")
        if self.positional == "rope" and (self.width // self.heads) % 2:
            raise ValueError("RoPE requires even head dimensions")
        if self.residual_scale_reference_layers is not None and (
            type(self.residual_scale_reference_layers) is not int
            or self.residual_scale_reference_layers < 1
        ):
            raise ValueError("Residual scale reference must be positive")
>>>>>>> Stashed changes
        if (self.pad_id, self.bos_id, self.eos_id) != (0, 1, 2):
            raise ValueError("Data format uses pad=0, bos=1, eos=2")


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.rope_base = config.rope_base if config.positional == "rope" else None
        self.query = nn.Linear(config.width, config.width, bias=False)
        self.key = nn.Linear(config.width, config.width, bias=False)
        self.value = nn.Linear(config.width, config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        dimension = config.width // config.heads
        inverse = config.rope_base ** (-torch.arange(0, dimension, 2).float() / dimension)
        angles = torch.outer(torch.arange(config.max_tokens).float(), inverse)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def rotate(self, x):
        # Rotate query/key pairs in FP32; values retain their content representation.
        cosine, sine = self.cos[:, :, : x.shape[-2]], self.sin[:, :, : x.shape[-2]]
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        return (
            torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1)
            .flatten(-2)
            .to(x.dtype)
        )

    def forward(self, x, context, valid, causal=False):
        def split(t):
            b, n, d = t.shape
            return t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(context)), split(self.value(context))
        if self.rope_base is not None:
            q, k = rotary(q, self.rope_base), rotary(k, self.rope_base)
        allowed = valid[:, None, None, :]
        if causal:
            allowed = (
                allowed
                & torch.ones(x.shape[1], context.shape[1], device=x.device, dtype=torch.bool).tril()
            )
        y = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v, attn_mask=allowed)
        return self.out(y.transpose(1, 2).reshape_as(x))


def rotary(x, base=10000.0):
    """Rotate adjacent Q/K pairs; compute angles in FP32, retain activation dtype."""
    frequencies = base ** (-torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1])
    angles = torch.arange(x.shape[-2], device=x.device).float()[:, None] * frequencies
    cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.width)
        self.ffn = CompressorBranchFFN(config)

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
<<<<<<< Updated upstream
=======
            self.position = (
                nn.Embedding(c.max_tokens, c.width) if c.positional == "learned" else None
            )
>>>>>>> Stashed changes
            self.encoder = nn.ModuleList([EncoderBlock(c) for _ in range(c.encoder_layers)])
            self.encoder_norm = RMSNorm(c.width)
            # Concatenation preserves the order inside each span, unlike weighted averaging.
            self.compress = nn.Linear(c.span * c.width, c.width, bias=False)
            self.expand = nn.Linear(c.width, c.span * c.width, bias=False)
            self.decoder = nn.ModuleList([EncoderBlock(c) for _ in range(c.decoder_layers)])
            self.decoder_norm = RMSNorm(c.width)
            for name, module in self.named_modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    generator = (
                        parameter_generator(seed, name + ".weight")
                        if c.positional == "rope"
                        else None
                    )
                    nn.init.normal_(module.weight, std=0.02, generator=generator)
            for name, module in self.named_modules():
                if isinstance(module, CompressorBranchFFN):
                    reference = c.residual_scale_reference_layers or c.encoder_layers
                    module.initialize(seed, name, 1 / math.sqrt(2 * reference))

    def encode(self, tokens, mask):
        c = self.config
        if tokens.ndim != 2 or mask.shape != tokens.shape or mask.dtype != torch.bool:
            raise ValueError("Expected token IDs and a same-shaped boolean mask")
        if not 0 < tokens.shape[1] <= c.max_tokens or not mask.any(1).all():
            raise ValueError("Input must have 1..max_tokens real tokens")
        if (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("Use right padding")
        x = self.embedding(tokens)
<<<<<<< Updated upstream
=======
        if self.position is not None:
            x = x + self.position(torch.arange(tokens.shape[1], device=tokens.device))
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
        if self.position is not None:
            x = x + self.position(positions)
>>>>>>> Stashed changes
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
