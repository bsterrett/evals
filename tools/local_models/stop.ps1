param([string]$RuntimeRoot = "/home/ben/llm/recurrent")
$ErrorActionPreference = "Stop"
@'
from pathlib import Path
import os, signal, sys
root = Path(sys.argv[1])
lock = root / 'gpu.lock'
if not lock.exists() or not lock.read_text().strip():
    print('No worker PID recorded.')
    raise SystemExit(0)
pid = int(lock.read_text().strip())
try:
    args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
except FileNotFoundError:
    print('Worker already stopped.')
    raise SystemExit(0)
if not any(a.endswith(b'/tools/local_models/serve.py') or a.endswith(b'/eval-local-worker') for a in args) and b'eval_core.local.worker' not in args:
    raise SystemExit('Recorded PID is not a recurrent model server; refusing to stop it.')
os.kill(pid, signal.SIGTERM)
print(f'Stopped recurrent worker {pid}.')
'@ | & wsl -d Ubuntu-LLM -- python3 -B - $RuntimeRoot
exit $LASTEXITCODE
