"""The selected mild margin recipe, now paired with fresh ordinary CE training."""
import torch
from torch.nn import functional as F

def loss_terms(logits,tokens,mask,kind,count,step,total):
    strength=min(1.,max(0.,(1.-(step-1)/(total-1))/.25))
    target=tokens.masked_fill(~mask,-100).flatten()
    scores=logits.float().flatten(0,1)
    if kind=='plain' or strength==0:
        value=F.cross_entropy(scores,target,reduction='sum')/count
        return value,value.detach()
    assert kind=='margin'
    with torch.no_grad():ordinary=F.cross_entropy(scores,target,reduction='sum')/count
    # BF16 conversion is a new tensor; do not modify FP32 caller-owned logits.
    if logits.dtype==torch.float32:scores=scores.clone()
    scores.scatter_add_(1,target.clamp_min(0)[:,None],-.25*strength*mask.flatten()[:,None].to(scores.dtype))
    return F.cross_entropy(scores,target,reduction='sum')/count,ordinary
