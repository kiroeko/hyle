#!/usr/bin/env python3
"""Paired ordinary-decode KL measurement; one row per worker, no logits files.

Workers keep two models resident but the controller serializes all GPU calls.
Every candidate sees the same BF16 greedy teacher prefixes. This is numerical
measurement, not a throughput benchmark or a general model-quality score.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import os
import queue
import struct
import subprocess
import threading
import time
import numpy as np

HEADER=struct.Struct('<8sIIQQQQiI')
MAGIC=b'HYQL0001'

def exact(stream,count):
    parts=[]
    while count:
        data=stream.read(count)
        if not data:raise EOFError('truncated quality frame')
        parts.append(data);count-=len(data)
    return b''.join(parts)

@dataclass
class Frame:
    kind:int;vocabulary:int;case:int;position:int;length:int;rng:int;top1:int;flags:int
    values:np.ndarray|None=None

def read_frame(stream):
    magic,kind,vocab,case,pos,length,rng,top1,flags=HEADER.unpack(exact(stream,HEADER.size))
    if magic!=MAGIC or kind not in (1,2,3,4,5) or not 2<=vocab<=1000000:
        raise ValueError('invalid quality frame header')
    frame=Frame(kind,vocab,case,pos,length,rng,top1,flags)
    if kind==2:
        frame.values=np.frombuffer(exact(stream,vocab*4),dtype='<f4')
        if not np.all(np.isfinite(frame.values)) or not 0<=top1<vocab or int(np.argmax(frame.values))!=top1:
            raise ValueError('nonfinite row or greedy/logits mismatch')
    elif kind==5:
        if not 1<=length<=512:raise ValueError('invalid tokenized prompt length')
        frame.values=np.frombuffer(exact(stream,length*4),dtype='<i4')
        if np.any(frame.values<0) or np.any(frame.values>=vocab):raise ValueError('invalid prompt tokens')
    return frame

class Worker:
    def __init__(self,command,runtime,folder,timeout=180):
        self.folder=Path(folder);self.folder.mkdir();self.timeout=timeout
        (self.folder/'command.txt').write_text(subprocess.list2cmdline(list(map(str,command)))+'\n',encoding='utf-8')
        self.error=(self.folder/'stderr.log').open('wb')
        env=os.environ.copy();env['PATH']=str(runtime)+os.pathsep+env.get('PATH','')
        self.process=subprocess.Popen(list(map(str,command)),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.error,env=env,bufsize=0)
        self.items=queue.Queue(maxsize=1);self.stopping=threading.Event()
        self.thread=threading.Thread(target=self._reader,daemon=True);self.thread.start()
        self.rows=0
        try:
            ready=self.next(1);self.vocabulary=ready.vocabulary
        except BaseException:
            self.close(False);raise

    def _reader(self):
        try:
            while not self.stopping.is_set():
                frame=read_frame(self.process.stdout)
                finished=frame.kind==4
                while not self.stopping.is_set():
                    try:self.items.put(frame,timeout=.2);break
                    except queue.Full:pass
                del frame
                if finished:return
        except BaseException as error:
            while not self.stopping.is_set():
                try:self.items.put(error,timeout=.2);break
                except queue.Full:pass

    def next(self,kind):
        try:item=self.items.get(timeout=self.timeout)
        except queue.Empty:raise TimeoutError(f'quality worker timeout;see {self.folder}') from None
        if isinstance(item,BaseException):raise RuntimeError(f'quality worker stream failed;see {self.folder}') from item
        if item.kind!=kind or item.rng:raise ValueError('quality kind/RNG mismatch')
        if hasattr(self,'vocabulary') and item.vocabulary!=self.vocabulary:raise ValueError('vocabulary changed')
        if kind==2:self.rows+=1
        return item

    def send(self,line):
        pending=memoryview((line+'\n').encode('ascii'))
        while pending:
            count=self.process.stdin.write(pending)
            if not count:raise BrokenPipeError('quality worker input closed')
            pending=pending[count:]
        self.process.stdin.flush()

    def reset(self):
        self.send('R');self.next(3)

    def close(self,success=False):
        try:
            if success and self.process.poll() is None:
                self.send('X');self.next(4)
                code=self.process.wait(timeout=30)
                if code:raise RuntimeError('quality worker failed on clean exit')
            elif self.process.poll() is None:self.process.kill();self.process.wait(timeout=30)
        finally:
            self.stopping.set()
            if self.process.poll() is None:self.process.kill();self.process.wait(timeout=30)
            self.process.stdin.close();self.process.stdout.close();self.thread.join(timeout=2)
            self.error.close()
            (self.folder/'result.txt').write_text(f'exit={self.process.returncode} rows={getattr(self,"rows",0)} clean={success}\n')

def metrics(oracle,candidate):
    if oracle.shape!=candidate.shape or oracle.ndim!=1:raise ValueError('logit shapes differ')
    p=oracle.astype(np.float64);q=candidate.astype(np.float64)
    p-=p.max();q-=q.max()
    p-=np.log(np.exp(p).sum());q-=np.log(np.exp(q).sum())
    kl=float(np.dot(np.exp(p),p-q))
    if not np.isfinite(kl) or kl < -1e-10:raise ValueError('invalid KL')
    pair=np.partition(oracle,-2)[-2:]
    return max(0.,kl),int(np.argmax(oracle)==np.argmax(candidate)),float(pair.max()-pair.min())

def load_corpus(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        cases=list(csv.DictReader(stream,delimiter='\t'))
    if not cases or any(set(c)!= {'id','group','prompt'} for c in cases):raise ValueError('corpus columns must be id/group/prompt')
    ids=[int(c['id']) for c in cases]
    if len(set(ids))!=len(ids) or len({c['prompt'] for c in cases})!=len(cases):raise ValueError('duplicate corpus case')
    if any(i<1 for i in ids) or any(not c['prompt'] or not c['group'] or '\0' in c['prompt'] for c in cases):
        raise ValueError('invalid or empty corpus field')
    return cases

def check_row(frame,case,position,length,vocab):
    if (frame.case,frame.position,frame.length,frame.vocabulary)!=(case,position,length,vocab):
        raise ValueError('teacher prefix metadata differs')

def run(args):
    if args.output.exists():raise ValueError('output already exists;use a fresh directory')
    paths=(args.hyle_worker,args.hyle_runtime,args.peer_worker,args.official_bin,args.bf16_model,
        args.q8_model,args.q4_model,args.corpus)
    if any(not p.exists() for p in paths):raise FileNotFoundError('a required local worker/runtime/model/corpus is missing')
    for p in (args.bf16_inputs,args.q8_inputs,args.q4_inputs):
        if not (p/'weights.tsv').is_file():raise FileNotFoundError(f'local prepared weights missing:{p}')
    if not 1<=args.positions<=1000000 or not 1<=args.tokens_per_prompt<=1024:raise ValueError('invalid row limits')
    cases=load_corpus(args.corpus);args.output.mkdir(parents=True)
    (args.output/'plan.txt').write_text(f'positions={args.positions};tokens_per_prompt={args.tokens_per_prompt};capacity={args.capacity}\n'
        f'corpus={args.corpus.resolve()};bootstrap={args.bootstrap};seed={args.seed}\n'
        f'hyle_source={args.hyle_version};peer_source={args.peer_version}\n'
        'Default model math,ordinary greedy teacher,BF16 KV;GPU calls serialized;two models resident;not timing evidence.\n'
        'No model downloads,no complete logits files;at most one row queued per worker.\n',encoding='utf-8')
    configurations=[('BF16-HYLE','hyle',args.bf16_inputs),('Q8_0-HYLE','hyle',args.q8_inputs),
        ('Q8_0-PEER','peer',args.q8_model),('Q4_K_M-PEER','peer',args.q4_model),('Q4_K_M-HYLE','hyle',args.q4_inputs)]
    schedule=[];vocabulary=None
    for label,engine,model in configurations:
        folder=args.output/label;folder.mkdir();oracle=candidate=None;passed=False
        try:
            oracle=Worker([args.peer_worker,args.official_bin,args.bf16_model,args.capacity],args.official_bin,folder/'oracle')
            command=[args.hyle_worker,model,args.capacity,args.positions] if engine=='hyle' else [args.peer_worker,args.official_bin,model,args.capacity]
            candidate=Worker(command,args.hyle_runtime if engine=='hyle' else args.official_bin,folder/'candidate')
            vocabulary=oracle.vocabulary
            if vocabulary!=candidate.vocabulary:raise ValueError('oracle/candidate vocabulary differs')
            total=0;started=time.monotonic()
            with (folder/'rows.tsv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream,delimiter='\t');writer.writerow(['id','group','position','prompt_tokens','context','teacher','candidate_top1','agree','kl','oracle_gap'])
                iterable=cases if not schedule else schedule
                new_schedule=[]
                for item in iterable:
                    case=int(item['id']);group=item['group']
                    if not schedule:
                        oracle.send(f'T {case} '+item['prompt'].encode('utf-8').hex())
                        tokenized=oracle.next(5)
                        if tokenized.case!=case:raise ValueError('tokenizer case differs')
                        prompt=tokenized.values.tolist();teacher=[]
                    else:prompt=item['ids'];teacher=item['teacher']
                    command=f'P {case} {len(prompt)} '+' '.join(map(str,prompt))
                    oracle.send(command);ref=oracle.next(2)
                    candidate.send(command);got=candidate.next(2)
                    count=args.tokens_per_prompt if not schedule else len(teacher)
                    for position in range(count):
                        check_row(ref,case,position,len(prompt)+position,vocabulary)
                        check_row(got,case,position,len(prompt)+position,vocabulary)
                        if not ref.flags&2:raise ValueError('oracle EOS metadata absent')
                        if schedule and ref.top1!=teacher[position]:raise ValueError('BF16 teacher changed between paired runs')
                        if not schedule:teacher.append(ref.top1)
                        kl,agree,gap=metrics(ref.values,got.values)
                        writer.writerow([case,group,position,len(prompt),ref.length,ref.top1,got.top1,agree,format(kl,'.17g'),format(gap,'.9g')])
                        total+=1
                        if (not schedule and (total==args.positions or ref.flags&1)) or position+1==count:break
                        token=teacher[position]
                        del ref,got
                        oracle.send(f'D {case} {token}');ref=oracle.next(2)
                        candidate.send(f'D {case} {token}');got=candidate.next(2)
                    oracle.reset();candidate.reset();stream.flush()
                    new_schedule.append({'id':case,'group':group,'ids':prompt,'teacher':teacher})
                    print(f'{label}:positions={total};prompts={len(new_schedule)};elapsed={time.monotonic()-started:.1f}s',flush=True)
                    if total>=args.positions:break
                if total!=args.positions:raise ValueError('corpus exhausted before the requested valid pre-EOS positions')
            if not schedule:
                schedule=new_schedule
                with (args.output/'teacher.tsv').open('w',encoding='utf-8',newline='') as stream:
                    writer=csv.writer(stream,delimiter='\t');writer.writerow(['id','group','prompt_ids','teacher_ids'])
                    for c in schedule:writer.writerow([c['id'],c['group'],' '.join(map(str,c['ids'])),' '.join(map(str,c['teacher']))])
            passed=True
        finally:
            try:
                if candidate:candidate.close(passed)
            finally:
                if oracle:oracle.close(passed)
    analyze(args.output,args.bootstrap,args.seed)

def analyze(output,bootstrap=4000,seed=20261105):
    output=Path(output);labels=['BF16-HYLE','Q8_0-HYLE','Q8_0-PEER','Q4_K_M-HYLE','Q4_K_M-PEER']
    all_rows={}
    for label in labels:
        with (output/label/'rows.tsv').open(encoding='utf-8') as stream:all_rows[label]=list(csv.DictReader(stream,delimiter='\t'))
    base=all_rows['BF16-HYLE'];keys=[(r['id'],r['group'],r['position'],r['context'],r['teacher']) for r in base]
    if not keys:raise ValueError('empty measured corpus')
    case_ids=list(dict.fromkeys(r['id'] for r in base));group={r['id']:r['group'] for r in base}
    aggregates={};summary=[]
    for label,data in all_rows.items():
        if keys!=[(r['id'],r['group'],r['position'],r['context'],r['teacher']) for r in data]:raise ValueError('paired rows do not match')
        kl=np.array([float(r['kl']) for r in data]);agree=np.array([int(r['agree']) for r in data])
        if not np.all(np.isfinite(kl)) or np.any(kl<0):raise ValueError('invalid stored metric')
        aggregates[label]=np.array([[sum(float(r['kl']) for r in data if r['id']==c),sum(r['id']==c for r in data)] for c in case_ids])
        summary.append(dict(arm=label,positions=len(data),prompts=len(case_ids),mean_kl=float(kl.mean()),top1_agree=float(agree.mean())))
    rng=np.random.default_rng(seed);strata={g:[i for i,c in enumerate(case_ids) if group[c]==g] for g in set(group.values())}
    draws=np.concatenate([rng.choice(indices,size=(bootstrap,len(indices)),replace=True) for _,indices in sorted(strata.items())],axis=1)
    baseline=next(r for r in summary if r['arm']=='BF16-HYLE')['mean_kl'];lines=[];passed=True
    for r in summary:lines.append(f'{r["arm"]}:positions={r["positions"]};prompts={r["prompts"]};mean_KL={r["mean_kl"]:.12g};top1_agree={r["top1_agree"]:.9g}')
    for kind in ('Q8_0','Q4_K_M'):
        h,p=[next(r for r in summary if r['arm']==kind+'-'+a) for a in ('HYLE','PEER')]
        hs=aggregates[kind+'-HYLE'][draws,0].sum(axis=1);ps=aggregates[kind+'-PEER'][draws,0].sum(axis=1)
        ratio=np.divide(hs,ps,out=np.full_like(hs,np.inf),where=ps>0);ratio[(hs==0)&(ps==0)]=1
        low,high=np.quantile(ratio,[.025,.975],method='nearest');limit=1.1*p['mean_kl']+baseline
        ok=h['mean_kl']<=limit and h['top1_agree']>=p['top1_agree']-.01;passed &=ok
        observed=h['mean_kl']/p['mean_kl'] if p['mean_kl']>0 else 1. if h['mean_kl']==0 else float('inf')
        lines.append(f'{kind}:KL_ratio={observed:.9g};prompt_cluster_bootstrap_95=[{low:.9g},{high:.9g}];KL_limit={limit:.12g};guard_PASS={ok}')
    lines+=['Bootstrap resamples whole prompts within corpus groups,not individual neighboring tokens.',
        'Synthetic held-out short-prompt corpus;finite numerical regression guard,not a universal110%quality proof.',f'NUMERICAL_GUARD_PASS={passed}']
    with (output/'summary.tsv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]),delimiter='\t');writer.writeheader();writer.writerows(summary)
    (output/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)
    if not passed:raise RuntimeError('expanded-corpus numerical guard failed;raw measurements retained')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    run_parser=sub.add_parser('run')
    for name in ('hyle-worker','hyle-runtime','peer-worker','official-bin','bf16-model','q8-model','q4-model',
        'bf16-inputs','q8-inputs','q4-inputs','corpus','output'):
        run_parser.add_argument('--'+name,type=Path,required=True)
    run_parser.add_argument('--positions',type=int,default=8192)
    run_parser.add_argument('--tokens-per-prompt',type=int,default=64)
    run_parser.add_argument('--capacity',type=int,default=32768)
    run_parser.add_argument('--bootstrap',type=int,default=4000)
    run_parser.add_argument('--seed',type=int,default=20261105)
    run_parser.add_argument('--hyle-version',default='unspecified')
    run_parser.add_argument('--peer-version',default='unspecified')
    report=sub.add_parser('analyze');report.add_argument('output',type=Path)
    report.add_argument('--bootstrap',type=int,default=4000);report.add_argument('--seed',type=int,default=20261105)
    args=parser.parse_args()
    if args.mode=='run':run(args)
    else:analyze(args.output,args.bootstrap,args.seed)

if __name__=='__main__':main()
