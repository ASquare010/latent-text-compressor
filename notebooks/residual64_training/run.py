"""One exact 10.46M continuation,10000 additional updates at context512."""
from common import *
import argparse,faulthandler,shutil,subprocess,time,traceback
faulthandler.enable()

def load_checkpoint(path,device='cpu'):
    import torch
    from growth import Config,Model
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['format']=='residual64-fresh10m60k-v1'
    model=Model(Config(**saved['config']),saved['seed'])
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval(),saved

def load_encoder(path,device='cpu'):
    import torch
    from growth import Config,Model
    from growth import EncoderOnly
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['format']=='residual64-fresh10m60k-encoder-v1'
    encoder=EncoderOnly(Model(Config(**saved['config']),saved['seed']))
    encoder.load_state_dict(saved['encoder'],strict=True)
    return encoder.to(device).eval()

def worker(kind):
    folder=OUT/kind;folder.mkdir(exist_ok=False)
    write_json(folder/'worker.lock.json',dict(pid=os.getpid(),created=time.time()))
    step=0
    def status(state,**kw):write_json(folder/'status.json',dict(state=state,pid=os.getpid(),kind=kind,updates=step,affinity=AFFINITY,**kw))
    try:
        status('loading')
        import torch
        from dataclasses import asdict
        from growth import Model,Config
        from growth import EncoderOnly
        from transfer import initialize
        import packed_data
        from objective import loss_terms
        from evaluate import evaluate
        from memory import Tracker
        from latent_text.data import batch
        from latent_text.runtime import autocast,save_checkpoint
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert str(torch.__version__)=='2.14.0+cu132' and torch.cuda.is_available()
        manifest=json.loads((OUT/'manifest.json').read_text())
        assert pins()==manifest['sources']
        tokenizer,splits,data=packed_data.load()
        assert data==manifest['dataset']
        short_splits=packed_data.short_rows(tokenizer,data)
        write_json(folder/'data_receipt.json',dict(file_hashes=True,complete_packed_decoded_identities=True,dataset=data))
        model,transfer_receipt=initialize(kind)
        initial_sha=tensor_hash(model)
        write_json(folder/'initialization.json',dict(kind=kind,initial_state_sha256=initial_sha,transfer=transfer_receipt,sources=pins(),affinity=AFFINITY))
        status('waiting_for_peer_identities')
        deadline=time.monotonic()+180
        while not all((OUT/k/'initialization.json').exists() for k in KINDS):
            state=json.loads((OUT/'status.json').read_text(encoding='utf-8')) if (OUT/'status.json').exists() else {}
            if state.get('failed') or state.get('state')=='failed':raise RuntimeError('Peer initialization failed; no automatic retry')
            if time.monotonic()>deadline:raise TimeoutError('Peer initialization deadline exceeded; no automatic retry')
            time.sleep(1)
        assert transfer_receipt['fresh_initialization']
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        model=model.cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=lr(1),weight_decay=.01,fused=True)
        rng=random.Random(SEED)
        write_json(folder/'resume_receipt.json',dict(optimizer_restored=False,fresh_initialization=True,optimizer_steps=0))
        pools=[short_splits['train'],splits['train'],packed_data.synthetic(5000,241),packed_data.synthetic(5000,243,True)]
        assert all(0<len(r['ids'])<=512 and min(r['ids'])>=3 and max(r['ids'])<4096 for pool in pools for r in pool)
        short=short_splits['valid']
        tracker=Tracker(folder/'memory.json')
        schedule=hashlib.sha256();history=[];best=None;best_step=None
        def checkpoint(n):
            return dict(format='residual64-fresh10m60k-v1',kind=kind,config=asdict(model.config),seed=SEED,
                        protocol=dict(model=asdict(model.config)),model=model.state_dict(),optimizer=opt.state_dict(),step=n,budget=STEPS,inherited_updates=INHERITED,
                        tokenizer=(packed_data.DATA/'tokenizer.json').read_text(encoding='utf-8'),dataset=data,sources=pins(),
                        rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        parent_sha256=PARENT_SHA,transfer=transfer_receipt,affinity=AFFINITY,initial_state_sha256=initial_sha,target_schedule_sha256=schedule.hexdigest(),
                        waivers=WAIVERS,training=dict(microbatch=MICRO,accumulation=ACCUM,weight_decay=.01,margin=.25 if kind.endswith('margin') else 0.,fade_start=.75))
        def measured(rows,phase):
            tracker.begin()
            result=evaluate(model,rows,tokenizer,'cuda',microbatch=8 if rows is short else 2)
            tracker.capture(phase)
            tracker.begin()
            return result
        # User explicitly requested direct launch: no separate probe or initial full replay.
        write_json(folder/'initial_validation.json',dict(performed=False,waivers=WAIVERS))
        save_checkpoint(folder/'initial.pt',checkpoint(0))
        tracker.begin()
        for step in range(1,STEPS+1):
            started=time.perf_counter();model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=lr(step)
            choices=[];micros=[]
            for _ in range(ACCUM):
                micro=[]
                for __ in range(MICRO):
                    source=min(3,int(rng.random()*4))
                    index=rng.randrange(len(pools[source]));choices.append((source,index));micro.append(pools[source][index])
                micros.append(micro)
            schedule.update(json.dumps([choices,[[r['ids'] for r in micro] for micro in micros]],separators=(',',':')).encode())
            count=sum(len(r['ids']) for micro in micros for r in micro)
            ordinary=objective=0.
            for micro in micros:
                tokens,mask=batch(micro,'cuda')
                with autocast('cuda'):
                    logits=model(tokens,mask)
                    loss,nll=loss_terms(logits,tokens,mask,kind.split('_')[1],count,step,STEPS)
                loss.backward();ordinary+=nll.item();objective+=loss.item()
                del logits,loss,nll,tokens,mask
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
            tracker.capture('training')
            row=dict(updates=step,ordinary_nll=ordinary,objective=objective,lr=lr(step),tokens=count,
                     elapsed=time.perf_counter()-started,**tracker.peaks['training'])
            with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if step==1 or step%25==0:
                status('training',nll=ordinary,lr=lr(step),**tracker.peaks['training'])
                print(kind,step,ordinary,flush=True)
            if step%250==0:save_checkpoint(folder/'last.pt',checkpoint(step))
            if step%500==0:
                status('validating');opt.zero_grad(set_to_none=True)
                valid=measured(splits['valid'],'validation')
                short_report=measured(short,'short_validation')
                history.append(dict(updates=step,packed=valid,short=short_report))
                write_json(folder/'validation_history.json',history)
                if best is None or rank(valid)<rank(best):
                    best,best_step=valid,step;shutil.copyfile(folder/'last.pt',folder/'best.pt')
                write_json(folder/'best_validation.json',dict(updates=best_step,report=best))
                assert pins()==manifest['sources']
        status('final_audit');del model,opt;torch.cuda.empty_cache()
        model,last=load_checkpoint(folder/'last.pt','cuda')
        assert last['step']==STEPS and all(float(s['step'])==STEPS+OPT_STEPS for s in last['optimizer']['state'].values())
        final=measured(splits['valid'],'final_replay');replay(final,history[-1]['packed'])
        final_short=measured(short,'final_replay');replay(final_short,history[-1]['short'])
        del model,last;torch.cuda.empty_cache()
        model,saved=load_checkpoint(folder/'best.pt','cuda')
        selected=measured(splits['valid'],'best_replay');replay(selected,best)
        selected_short=measured(short,'best_replay')
        replay(selected_short,next(r['short'] for r in history if r['updates']==best_step))
        encoder=EncoderOnly(model)
        save_checkpoint(folder/'encoder.pt',dict(format='residual64-fresh10m60k-encoder-v1',seed=SEED,
             config=asdict(model.config),encoder={n:v.cpu() for n,v in encoder.state_dict().items()},
             tokenizer=(packed_data.DATA/'tokenizer.json').read_text(encoding='utf-8'),sources=pins(),selected_update=best_step,
             decoder_checkpoint_sha256=digest(folder/'best.pt')))
        fresh=load_encoder(folder/'encoder.pt','cuda')
        tokens,mask=batch(splits['valid'][:1],'cuda')
        with torch.no_grad(),autocast('cuda'):
            a,n=encoder(tokens,mask);b,m=fresh(tokens,mask)
            torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(n,m,rtol=0,atol=0)
        tracker.capture('export_parity')
        assert pins()==manifest['sources']
        result=dict(kind=kind,seed=SEED,updates=STEPS,inherited_updates=INHERITED,parameters=sum(p.numel() for p in model.parameters()),encoder_parameters=model.encoder_parameters(),
                    config=asdict(model.config),sources=pins(),dataset=data,initial_state_sha256=initial_sha,
                    parent_sha256=PARENT_SHA,transfer=transfer_receipt,affinity=AFFINITY,
                    target_schedule_sha256=schedule.hexdigest(),best_update=best_step,final_report=final,best_report=selected,
                    final_short=final_short,best_short=selected_short,memory=tracker.peaks,memory_cap=CAP,
                    waivers=WAIVERS,full_final_replay=True,full_best_replay=True,encoder_parity=True,complete_packed_decoded_identities=True,
                    last_sha256=digest(folder/'last.pt'),best_sha256=digest(folder/'best.pt'),encoder_sha256=digest(folder/'encoder.pt'))
        write_json(folder/'result.json',result)
        record=ROOT/f'artifacts/records/residual64-fresh10m60k-{kind}-v1.json';write_json(record,result)
        write_json(folder/'audit.json',dict(result_sha256=digest(folder/'result.json'),record_sha256=digest(record),sources=pins()))
        status('complete',best_update=best_step);(folder/'worker.lock.json').unlink()
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise

def finalize():
    manifest=json.loads((OUT/'manifest.json').read_text());assert pins()==manifest['sources']
    for n,sha in manifest['sources'].items():assert digest(OUT/'source'/n)==sha
    results=[]
    for kind in KINDS:
        folder=OUT/kind;result=json.loads((folder/'result.json').read_text());audit=json.loads((folder/'audit.json').read_text())
        record=ROOT/f'artifacts/records/residual64-fresh10m60k-{kind}-v1.json'
        assert result==json.loads(record.read_text()) and digest(record)==audit['record_sha256']
        assert digest(folder/'result.json')==audit['result_sha256']
        assert result['sources']==manifest['sources'] and result['dataset']==manifest['dataset']
        initial=json.loads((folder/'initialization.json').read_text(encoding='utf-8'))
        assert result['initial_state_sha256']==initial['initial_state_sha256']
        assert result['parent_sha256']==PARENT_SHA
        for n,key in (('last.pt','last_sha256'),('best.pt','best_sha256'),('encoder.pt','encoder_sha256')):assert digest(folder/n)==result[key]
        logs=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
        assert [s['updates'] for s in logs]==list(range(1,STEPS+1))
        history=json.loads((folder/'validation_history.json').read_text())
        assert [s['updates'] for s in history]==list(range(500,STEPS+1,500))
        assert all(result[k] for k in ('full_final_replay','full_best_replay','encoder_parity','complete_packed_decoded_identities'))
        assert all(all(v<=CAP[k] for k,v in peak.items()) for peak in result['memory'].values())
        results.append(result)
    assert len({r['target_schedule_sha256'] for r in results})==1
    results.sort(key=lambda r:rank(r['best_report']))
    write_json(OUT/'completion.json',dict(state='complete',winner=results[0]['kind'],results=results))
    write_json(OUT/'completion_verification.json',dict(passed=True,candidates=1,updates_each=STEPS,matched_targets=True,
                fresh_initialization=True,all_hashes_verified=True,full_replays=True))
    write_json(OUT/'status.json',dict(state='complete',finished=list(KINDS),failed=[],winner=results[0]['kind']))

def controller():
    OUT.mkdir(exist_ok=False);write_json(OUT/'controller.lock.json',dict(pid=os.getpid(),created=time.time()))
    source_hashes=pins();(OUT/'source').mkdir()
    for n,p in source_paths().items():shutil.copyfile(p,OUT/'source'/n)
    write_json(OUT/'status.json',dict(state='preparing_data',pid=os.getpid(),active={},finished=[],failed=[],waivers=WAIVERS))
    # Reuse immutable packed data; every worker independently checks all identities.
    assert pins()==source_hashes
    from packed_data import DATA
    dataset=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    manifest=dict(kinds=KINDS,seed=SEED,updates_each=STEPS,max_workers=1,
                  context=512,width=256,microbatch=MICRO,accumulation=ACCUM,memory_cap=CAP,
                  sources=source_hashes,dataset=dataset,initialization='fresh seed47 model and optimizer;zero inherited updates',
                  parent_sha256=PARENT_SHA,inherited_updates=INHERITED,waivers=WAIVERS,
                  parent_result_sha256=None,preflight_passed=False,controller_affinity=AFFINITY,worker_cpus=[16])
    write_json(OUT/'manifest.json',manifest)
    active={};finished=[];failed=[];handles=[]
    for kind in KINDS:
        log=(OUT/f'{kind}.log').open('w');handles.append(log)
        active[kind]=subprocess.Popen([sys.executable,'-u',__file__,'--config',str(CONFIG_PATH),'--kind',kind],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    while active:
        for kind,p in list(active.items()):
            code=p.poll()
            if code is not None:
                if code:write_json(OUT/f'{kind}-exit.json',dict(pid=p.pid,code=code,hex_code=hex(code & 0xffffffff)))
                (failed if code else finished).append(kind);del active[kind]
        write_json(OUT/'status.json',dict(state='failure_waiting' if failed else 'training',pid=os.getpid(),active={k:p.pid for k,p in active.items()},finished=finished,failed=failed))
        if active:time.sleep(5)
    for f in handles:f.close()
    if failed:raise RuntimeError(f'Worker failure: {failed}; no automatic retry')
    finalize();(OUT/'controller.lock.json').unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=KINDS);parser.add_argument('--config');parser.add_argument('--inspect',action='store_true');args=parser.parse_args()
    if args.inspect:
        import torch
        torch.set_num_threads(1)
        from transfer import initialize
        model,receipt=initialize('continue_plain')
        print(json.dumps(dict(recipe=RECIPE,transfer=receipt,training_started=False),indent=2))
    elif args.kind:worker(args.kind)
    else:
        try:controller()
        except BaseException as exc:
            lock=OUT/'controller.lock.json'
            if lock.exists() and json.loads(lock.read_text())['pid']==os.getpid():
                write_json(OUT/'controller_failure.json',dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
                write_json(OUT/'status.json',dict(state='failed',error=f'{type(exc).__name__}: {exc}'))
            raise



