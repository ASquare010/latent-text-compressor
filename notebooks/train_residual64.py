"""Continue the selected model; --inspect loads/checks it without training."""
import subprocess,sys
from pathlib import Path
if __name__ == '__main__':
    root=Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call([sys.executable,str(root/'notebooks/residual64_training/run.py'),*sys.argv[1:]],cwd=root))
