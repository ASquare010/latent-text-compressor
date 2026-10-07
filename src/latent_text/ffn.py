"""Selected Step 1 model: four sigmoid-gated squared-ReLU branches."""

import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def parameter_generator(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


class SingleFFN(nn.Module):
    def __init__(self, config, kind, hidden):
        super().__init__()
        self.kind = kind
        self.up = nn.Linear(config.width, hidden, bias=False)
        self.down = nn.Linear(hidden, config.width, bias=False)
        points, weights = np.polynomial.hermite.hermgauss(128)
        variance = config.width * 0.02**2
        z = torch.from_numpy(points) * math.sqrt(2 * variance)
        w = torch.from_numpy(weights) / math.sqrt(math.pi)
        target = float((w * F.silu(z).square()).sum()) * variance * config.hidden / hidden
        value = F.relu(z).square()
        self.center = 0.0
        second = float((w * (value - self.center).square()).sum())
        self.scale = math.sqrt(target / second)

    @torch.no_grad()
    def initialize(self, seed, prefix, residual_scale):
        for name, p in (("up", self.up.weight), ("down", self.down.weight)):
            p.normal_(
                0,
                0.02 * (residual_scale if name == "down" else 1),
                generator=parameter_generator(seed, f"{prefix}.{name}.weight"),
            )


class BranchFFN(SingleFFN):
    widths = (384, 384, 383, 383)

    def __init__(self, config):
        super().__init__(config, "relu_squared", 1534)
        self.up = nn.Linear(config.width, 1538, bias=False)
        self.kind = "branch_sigmoid"
        self.route_scale = 1.0

    def components(self, z):
        values, logits = z.split((1534, 4), -1)
        positive = values.relu()
        hidden = positive * positive
        weights = (
            2
            * logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32).sigmoid()
        ).to(logits.dtype)
        return torch.cat(
            [
                part * weights[..., i : i + 1]
                for i, part in enumerate(hidden.split(self.widths, -1))
            ],
            -1,
        )

    def forward(self, x):
        return self.down(self.components(self.up(x))) * (self.scale * self.route_scale)

    @torch.no_grad()
    def initialize(self, seed, prefix, residual_scale):
        super().initialize(seed, prefix, residual_scale)
        x = torch.randn(512, self.up.in_features, generator=torch.Generator().manual_seed(9127))
        x = x / x.square().mean(-1, keepdim=True).sqrt()
        z = self.up(x)
        self.route_scale = math.sqrt(
            float(z[..., :1534].relu().pow(4).mean() / self.components(z).square().mean())
        )


class CompressorBranchFFN(BranchFFN):
    """Selected FFN with checkpointed calibration for seed-independent loading."""

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("output_scale", torch.ones((), dtype=torch.float64))

    @torch.no_grad()
    def initialize(self, seed, prefix, residual_scale):
        super().initialize(seed, prefix, residual_scale)
        self.output_scale.fill_(self.scale * self.route_scale)

    def forward(self, x):
        y = self.down(self.components(self.up(x)))
        return y * self.output_scale.to(dtype=y.dtype)
