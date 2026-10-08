import gc
import torch
from common import CAP,write_json

class Tracker:
    def __init__(self,path):self.path,self.peaks=path,{}
    def begin(self):
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    def capture(self,phase):
        torch.cuda.synchronize()
        value=self.peaks.setdefault(phase,dict(allocated_mib=0.,reserved_mib=0.))
        value['allocated_mib']=max(value['allocated_mib'],torch.cuda.max_memory_allocated()/2**20)
        value['reserved_mib']=max(value['reserved_mib'],torch.cuda.max_memory_reserved()/2**20)
        ok=all(v<=CAP[k] for k,v in value.items())
        write_json(self.path,dict(phases=self.peaks,cap=CAP,passed=ok))
        if not ok:raise RuntimeError(f'Memory guard exceeded: {phase} {value}; no automatic retry')
