"""Shared immutable scaling protocol and standard-library orchestration helpers."""
from runtime_affinity import pin_process
AFFINITY=pin_process()
import os
for key in ('RAYON_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib,json,math,sys,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('D:/Git/latent-text-compressor'); ROOT=REPO
config_arg=sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else str(REPO/'config/train.json')
CONFIG_PATH=Path(config_arg).resolve()
RECIPE=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
assert RECIPE['mode']=='residual64_fresh'
def resolve(value):
    p=Path(value);return p if p.is_absolute() else REPO/p
OUT=resolve(RECIPE['output_dir'])
sys.path.insert(0,str(REPO/'src'))
STEPS=RECIPE['additional_updates'];SEED=RECIPE['sampler_seed'];KINDS=('continue_plain',);MICRO=2;ACCUM=4
assert type(STEPS) is int and STEPS>0
PARENT=None;PARENT_SHA=None
PARENT_RESULT=None
INHERITED=RECIPE['inherited_updates'];OPT_STEPS=RECIPE['inherited_optimizer_steps']
assert RECIPE['learning_rate']==6e-4 and INHERITED==0 and OPT_STEPS==0
# Four jobs plus CUDA contexts must fit 8GiB. Actual peaks are reported separately.
CAP=dict(allocated_mib=1300.,reserved_mib=1500.)

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # Readers on Windows can briefly deny destination replacement. Retry only
    # this metadata rename; never repeat model updates or retry native failures.
    import time
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(.05)
def source_paths():
    names=('common.py','runtime_affinity.py','transfer.py','growth.py','packed_data.py','objective.py','evaluate.py','memory.py','run.py')
    paths={n:HERE/n for n in names};paths['training_recipe.json']=CONFIG_PATH
    paths.update({'selected_'+p.name:p for p in (REPO/'src/latent_text').glob('*.py')})
    main=Path('D:/Git/complex_fnn/src')
    for name in ('models/position_compressor/residual.py','models/position_compressor/transformer.py','models/position_compressor/ffn.py','models/branch_sigmoid/transformer.py','models/components.py'):
        paths['main_'+name.replace('/','_')]=main/name
    return paths
def pins():return {n:digest(p) for n,p in sorted(source_paths().items())}
def rank(report):
    s=report['summary'];return (-s['exact'],-s['token_accuracy'],s['nll'])
def lr(step):
    if step<=1000:return 6e-4*step/1000
    fraction=(step-1000)/(STEPS-1000)
    return 1e-5+(6e-4-1e-5)*.5*(1+math.cos(math.pi*fraction))

def replay(a,b):
    assert a['payload']==b['payload']
    for k,v in a['summary'].items():assert abs(v-b['summary'][k])<1e-7,(k,v,b['summary'][k])
def tensor_hash(model):
    h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):
        v=v.detach().cpu().contiguous()
        h.update(n.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

WAIVERS=dict(separate_preflight=True,synthetic_resource_probes=True,initial_full_validation=True,
             explanation='Latest explicit user: start training directly. Checks waived, not passed. Native cause unresolved; this is one authorized attempt, no automatic retries.')


