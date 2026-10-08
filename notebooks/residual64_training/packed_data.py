"""Pack cached paragraphs without crossing train/validation/test boundaries."""
from runtime_affinity import pin_process
PREPARATION_AFFINITY=pin_process()
import hashlib,json,random
from pathlib import Path

from common import RECIPE,resolve
OLD=resolve(RECIPE['data_dir'])
DATA=resolve(RECIPE['packed_data_dir'])

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def identity(rows):
    h=hashlib.sha256()
    for r in rows:h.update(json.dumps([r['text'],r['ids']],ensure_ascii=False,separators=(',',':')).encode())
    return h.hexdigest()

def prepare():
    from tokenizers import Tokenizer
    import shutil
    if DATA.exists():
        raise RuntimeError('Packed data already exists; do not silently replace')
    source=json.loads((OLD/'manifest.json').read_text())
    assert digest(OLD/'tokenizer.json')==source['tokenizer_sha256']
    tokenizer=Tokenizer.from_file(str(OLD/'tokenizer.json'))
    tokenizer.encode_special_tokens=True
    sep=tokenizer.encode('\n',add_special_tokens=False).ids
    assert tokenizer.decode(sep,skip_special_tokens=False)=='\n'
    DATA.mkdir()
    shutil.copyfile(OLD/'tokenizer.json',DATA/'tokenizer.json')
    manifest=dict(format='packed-paragraph512-v1',source=source,separator_ids=sep,max_tokens=512,
                  packing='Greedy whole paragraphs in original source order; may span documents within a split only',
                  splits={},tokenizer_sha256=digest(DATA/'tokenizer.json'))
    docs={}
    for split in ('train','valid','test'):
        assert digest(OLD/f'{split}.jsonl')==source['split_sha256'][split]
        rows=[json.loads(s) for s in (OLD/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        docs[split]={r['document_id'] for r in rows}
        packed=[]; ids=[]; texts=[]; members=[]
        for i,r in enumerate(rows):
            assert 0<len(r['ids'])<=256
            if ids and len(ids)+len(sep)+len(r['ids'])>512:
                packed.append(dict(ids=ids,text='\n'.join(texts),members=members))
                ids=[]; texts=[]; members=[]
            if ids:ids+=sep
            ids+=r['ids']; texts.append(r['text']); members.append(i)
        if ids:packed.append(dict(ids=ids,text='\n'.join(texts),members=members))
        assert [i for r in packed for i in r['members']]==list(range(len(rows)))
        for r in packed:
            assert tokenizer.decode(r['ids'],skip_special_tokens=False)==r['text']
        out=DATA/f'{split}.jsonl'
        out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in packed),encoding='utf-8')
        manifest['splits'][split]=dict(rows=len(packed),source_rows=len(rows),sha256=digest(out),identity=identity(packed),
                                      tokens=sum(len(r['ids']) for r in packed),min_length=min(map(lambda r:len(r['ids']),packed)),
                                      max_length=max(map(lambda r:len(r['ids']),packed)))
    assert not docs['train'] & docs['valid'] and not docs['train'] & docs['test'] and not docs['valid'] & docs['test']
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['splits'],indent=2),flush=True)

def load():
    from tokenizers import Tokenizer
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert digest(DATA/'tokenizer.json')==manifest['tokenizer_sha256']
    tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    splits={}
    for split,entry in manifest['splits'].items():
        assert digest(DATA/f'{split}.jsonl')==entry['sha256']
        rows=[json.loads(s) for s in (DATA/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==entry['rows'] and identity(rows)==entry['identity']
        # Independent complete decoding of the newly packed representation.
        for r in rows:
            assert tokenizer.decode(r['ids'],skip_special_tokens=False)==r['text']
        splits[split]=rows
    return tokenizer,splits,manifest

def synthetic(count,seed,patterns=False):
    rng=random.Random(seed); result=[]
    for _ in range(count):
        length=rng.randint(32,512)
        if patterns:
            motif=[rng.randrange(3,4096) for _ in range(rng.randint(1,32))]
            ids=(motif*512)[:length]
        else:ids=[rng.randrange(3,4096) for _ in range(length)]
        result.append(dict(ids=ids))
    return result

def short_rows(tokenizer, manifest):
    result={}
    for split in ('train','valid'):
        assert digest(OLD/f'{split}.jsonl')==manifest['source']['split_sha256'][split]
        rows=[json.loads(s) for s in (OLD/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in rows:
            assert tokenizer.decode(row['ids'],skip_special_tokens=False)==row['text']
        result[split]=rows
    return result

if __name__=='__main__':
    prepare()
