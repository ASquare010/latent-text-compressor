import pytest
import torch
from latent_text.model import Attention, Config


def test_rope_relative_position_and_norm():
    attention = Attention(Config(width=16, heads=2, max_tokens=16, span=4))
    torch.manual_seed(12)
    q, k = torch.randn(1,2,1,8), torch.randn(1,2,1,8)
    qr = attention.rotate(q.expand(1,2,8,8))
    kr = attention.rotate(k.expand(1,2,8,8))
    torch.testing.assert_close(qr.square().sum(-1), q.square().sum(-1).expand(1,2,8))
    # Equal query/key displacement has the same dot product at different offsets.
    torch.testing.assert_close((qr[:,:,0]*kr[:,:,3]).sum(-1),
                               (qr[:,:,2]*kr[:,:,5]).sum(-1),atol=2e-6,rtol=2e-6)
    assert not torch.allclose(qr[:,:,0],qr[:,:,3])
    with pytest.raises(ValueError,match="even head"):
        Config(width=12,heads=4)
