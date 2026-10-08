"""Exact continuation of the completed 10.46M model, including optimizer/RNG."""
from common import *
import torch
from growth import Config,Model

def initialize(kind):
    assert kind=='continue_plain' and digest(PARENT)==PARENT_SHA
    result=json.loads(PARENT_RESULT.read_text(encoding='utf-8'))
    audit=json.loads(PARENT.with_name('audit.json').read_text(encoding='utf-8'))
    assert digest(PARENT_RESULT)==audit['result_sha256'] and result['last_sha256']==PARENT_SHA
    for name,path in source_paths().items():
        if name in ('growth.py','selected_model.py','selected_residual.py','selected_ffn.py'):
            assert digest(path)==result['sources'][name],name
    saved=torch.load(PARENT,map_location='cpu',weights_only=True)
    assert saved['format']=='residual64-long-v1'
    assert saved['step']==10000 and saved['inherited_updates']==34000
    assert all(float(s['step'])==OPT_STEPS for s in saved['optimizer']['state'].values())
    assert saved['config']==RECIPE['model'] and saved['seed']==SEED
    assert saved['inherited_updates']+saved['step']==INHERITED
    c=Config(**saved['config'])
    assert (c.encoder_layers,c.decoder_layers,c.width,c.span,c.max_tokens,c.code_features)==(4,1,256,64,512,125)
    model=Model(c,saved['seed']);model.load_state_dict(saved['model'],strict=True)
    for n,v in model.state_dict().items():assert torch.equal(v,saved['model'][n]),n
    assert sum(p.numel() for p in model.parameters())==10456576
    receipt=dict(parent_sha256=PARENT_SHA,parent_result_sha256=digest(PARENT_RESULT),
        parent_updates=INHERITED,optimizer='restore full AdamW state, then constant lr3e-5',
        inherited_weights_exact=True,initial_logits_and_vectors_exact=True,
        exact_function_basis='identical computational sources, config and all state tensors',
        sampler='restore checkpoint Python sampler state and torch CPU/CUDA RNG',
        parent_target_schedule_sha256=saved['target_schedule_sha256'],
        inherited_optimizer_steps=OPT_STEPS,parameters=10456576,encoder_parameters=model.encoder_parameters())
    return model,receipt
