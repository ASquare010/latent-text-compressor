"""Fresh selected model; verify both repositories implement the same function."""
from common import *
import torch
from growth import Config,Model

def initialize(kind):
    assert kind=='continue_plain' and INHERITED==0 and OPT_STEPS==0 and STEPS==60000
    config=Config(**RECIPE['model'])
    assert (config.encoder_layers,config.decoder_layers,config.width,config.span,config.max_tokens,config.code_features)==(4,1,256,64,512,125)
    model=Model(config,SEED)
    assert sum(p.numel() for p in model.parameters())==10456576
    assert model.encoder_parameters()==7327232
    sys.path.insert(0,'D:/Git/complex_fnn/src')
    from models.position_compressor.residual import Config as MainConfig,Model as MainModel
    values=dict(RECIPE['model']);values.pop('routed_layers')
    main_model=MainModel(MainConfig(**values),SEED)
    main_model.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        tokens=torch.arange(130).reshape(2,65)%4093+3
        mask=torch.ones_like(tokens,dtype=torch.bool);mask[1,49:]=False;tokens[~mask]=0
        torch.testing.assert_close(model(tokens,mask),main_model(tokens,mask),rtol=0,atol=0)
        a,n=model.encode(tokens,mask);b,m=main_model.encode(tokens,mask)
        torch.testing.assert_close(a,b,rtol=0,atol=0);assert torch.equal(n,m)
    return model,dict(fresh_initialization=True,seed=SEED,parent_checkpoint=None,
        parameters=10456576,encoder_parameters=7327232,optimizer='fresh AdamW',
        learned_weights_loaded=False,inherited_updates=0,both_repo_residual_logits_and_vectors_exact=True)
