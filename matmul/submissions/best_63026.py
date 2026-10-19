import hashlib,json,runpy
from pathlib import Path
p=Path(__file__).resolve().parent
r=json.loads((p/'best_63026.json').read_text())
s=p.parent/'matmul.py'
if hashlib.sha256(s.read_bytes()).hexdigest()!=r['scorer_sha256']:raise ValueError('scorer changed')
b=(p/'best_63026.ir').read_bytes()
if hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('source changed')
v=runpy.run_path(str(s),run_name='verification')['score_16x16'](b.decode())
if v!=r['score']:raise ValueError('score mismatch')
print(v)
