"""Process-local CPU pinning, validated without changing Python/CUDA or data."""
import ctypes
import os
import sys

def pin_process():
    cpus={'continue_plain':16,'encoder_plain':17,'balanced_plain':18}
    kind=sys.argv[sys.argv.index('--kind')+1] if '--kind' in sys.argv else None
    cpu=cpus[kind] if kind is not None else 16
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t)
    kernel.GetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t))
    handle=kernel.GetCurrentProcess();mask=1<<cpu
    if not kernel.SetProcessAffinityMask(handle,mask):raise ctypes.WinError(ctypes.get_last_error())
    actual=ctypes.c_size_t();system=ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle,ctypes.byref(actual),ctypes.byref(system)):raise ctypes.WinError(ctypes.get_last_error())
    assert actual.value==mask
    return dict(pid=os.getpid(),logical_cpu=cpu,mask=actual.value,mitigation='process-local pinning; root cause unresolved')
