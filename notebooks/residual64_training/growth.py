"""Keep the learned four-layer depth router; append residual refinements."""
import torch
from torch import nn
from dataclasses import dataclass
from latent_text.residual import Config as BaseConfig, Model as BaseModel
from latent_text.codec import EncoderOnly as BaseEncoder

@dataclass
class Config(BaseConfig):
    routed_layers: int = 4

class Model(BaseModel):
    def __init__(self, config, seed=17):
        super().__init__(config, seed)
        assert config.routed_layers == 4 and config.encoder_layers >= 4
        self.depth_queries = nn.Parameter(torch.zeros(8, config.width))

    def encode(self, tokens, mask):
        from torch.nn import functional as F
        c = self.config
        if tokens.ndim != 2 or mask.shape != tokens.shape or mask.dtype != torch.bool:
            raise ValueError('Invalid tokens/mask')
        assert 0 < tokens.shape[1] <= c.max_tokens and mask.any(1).all()
        assert not (mask[:, 1:] & ~mask[:, :-1]).any()
        values = [self.embedding(tokens)]
        for i, block in enumerate(self.encoder[:c.routed_layers]):
            h = block.attention_norm(BaseModel.mix_depth(self, values, 2*i))
            values.append(block.attention(h, h, mask))
            values.append(block.ffn(block.ffn_norm(BaseModel.mix_depth(self, values, 2*i+1))))
        x = BaseModel.mix_depth(self, values, -1)
        for block in self.encoder[c.routed_layers:]:
            x = block(x, mask)
        x = self.encoder_norm(x) * mask.unsqueeze(-1)
        x = F.pad(x, (0,0,0,(-x.shape[1]) % c.span))
        return self.compress(x.reshape(x.shape[0],-1,c.span*c.width)), mask.sum(1)

class EncoderOnly(BaseEncoder):
    def forward(self, tokens, mask):
        return Model.encode(self, tokens, mask)
