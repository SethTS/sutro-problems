"""Audit all original canonical records; never pool successful runs selectively."""
import sys,json,hashlib,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root));import mnist
folder=Path(sys.argv[1]);source=Path(sys.argv[2]);sha=hashlib.sha256(source.read_bytes()).hexdigest();rows=[]
for p in sorted(folder.glob('run-*.json')):
 v=json.loads(p.read_text());r=v.get('record') or {};calls=[mnist.Call(**c) for c in r.get('calls',[])];c=[x for x in calls if x.dataset=='mnist'];problems,rank,summary=mnist.judge(calls,270) if len(calls)==15 else (['missing calls'],None,{})
 e=r.get('energy') or {};energy=e.get('mj_per_call');energy_ok=False
 if energy is not None:
  rebuilt=mnist.energy_summary(e['windows'],e['device'],r['band_bp'],statistics.median(x.ms for x in c));energy_ok=not e.get('problems') and not rebuilt['problems'] and abs(rebuilt['mj_per_call']-energy)<1e-6
 checks=dict(returncode=v['returncode']==0,source_sha=r.get('source_sha256')==sha,sandbox=r.get('sandboxed') is True,calls=len(calls)==15 and len(c)==11,judge=not problems and r.get('ranked_ms') is not None and abs(rank-r['ranked_ms'])<1e-8,energy=energy_ok)
 rows.append(dict(file=p.name,passed=all(checks.values()),checks=checks,error_pct=100*(1-sum(x.correct for x in c)/sum(x.total for x in c)) if c else None,ranked_ms=r.get('ranked_ms'),mj_per_call=energy,problems=r.get('problems'),uuid=(e.get('device') or {}).get('uuid')))
manifest=json.loads((folder/'manifest.json').read_text())
assert manifest['source_sha256']==sha and manifest['source_bytes']==len(source.read_bytes())
for name,digest in manifest['raw_record_sha256'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
qualified=len(rows)==3 and all(x['passed'] for x in rows) and len({x['uuid'] for x in rows})==3
a=dict(source_sha256=sha,runs=rows,qualified=qualified,median_ms=statistics.median(x['ranked_ms'] for x in rows) if qualified else None,median_mj=statistics.median(x['mj_per_call'] for x in rows) if qualified else None)
(folder/'audit.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))

sys.exit(0 if qualified else 1)
