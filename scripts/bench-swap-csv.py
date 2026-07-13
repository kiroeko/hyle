#!/usr/bin/env python3
# Run each exe in both orders, N reps, print CSV: op,hyle_first_ratio,hw_first_ratio
# ratio = HYLE_ops / Highway_ops (>1 = HYLE faster). Usage: swap_csv.py N exe1 exe2 ...
import subprocess, re, sys, statistics, collections, os
N=int(sys.argv[1]); exes=sys.argv[2:]
row=re.compile(r'\|\s*[\d.]+%\s*\|\s*[\d.]+\s*\|\s*([\d,]+\.\d+)\s*\|[^|]*\|[^|]*\|\s*`(HYLE|Highway)`')
title=re.compile(r'\|\s*relative\s*\|.*\|\s*([^|]+?)\s*$')
def run(exe,hwf):
    env=dict(os.environ)
    if hwf: env['HYLE_BENCH_HWYFIRST']='1'
    else: env.pop('HYLE_BENCH_HWYFIRST',None)
    acc=collections.defaultdict(lambda:{'HYLE':[],'Highway':[]})
    for _ in range(N):
        out=subprocess.run([exe],env=env,capture_output=True,text=True).stdout
        cur=None
        for ln in out.splitlines():
            mt=title.match(ln)
            if mt: cur=mt.group(1).strip(); continue
            m=row.search(ln)
            if m and cur: acc[cur][m.group(2)].append(float(m.group(1).replace(',','')))
    return {t:statistics.median(d['HYLE'])/statistics.median(d['Highway']) for t,d in acc.items() if d['HYLE'] and d['Highway']}
allops={}
for exe in exes:
    hf=run(exe,False); wf=run(exe,True)
    for t in set(hf)|set(wf): allops[t]=(hf.get(t),wf.get(t))
for t in sorted(allops):
    r1,r2=allops[t]
    print(f'{t}\t{r1 if r1 else ""}\t{r2 if r2 else ""}')
