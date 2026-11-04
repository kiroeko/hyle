from pathlib import Path
import contextlib
import csv
import importlib.util
import io
import math
import struct
import sys
import unittest
import uuid
try:
    import numpy as np
except ImportError:
    print('SKIP:optional inference-quality tooling requires NumPy')
    raise SystemExit(77)
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('quality_under_test',ROOT/'scripts/inference-quality.py')
Q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=Q;spec.loader.exec_module(Q)
def packet(kind,values=(),case=7,pos=2,length=11,top=1,vocab=3):
    head=Q.HEADER.pack(Q.MAGIC,kind,vocab,case,pos,length,0,top,2)
    return head+np.asarray(values,dtype='<f4' if kind==2 else '<i4').tobytes()
class SmallReads(io.BytesIO):
    def read(self,n=-1):return super().read(min(n,3))
class QualityTests(unittest.TestCase):
    def folder(self):
        p=ROOT/'.scratch/inference-quality-unit-20261105'/uuid.uuid4().hex;p.mkdir(parents=True);return p
    def test_partial_frames_and_prefix_identity(self):
        f=Q.read_frame(SmallReads(packet(2,[1,4,-2])))
        np.testing.assert_array_equal(f.values,[1,4,-2]);Q.check_row(f,7,2,11,3)
        with self.assertRaises(ValueError):Q.check_row(f,7,3,11,3)
        t=Q.read_frame(SmallReads(packet(5,[0,1,2],length=3)))
        np.testing.assert_array_equal(t.values,[0,1,2])
    def test_bad_frames_do_not_silently_measure(self):
        with self.assertRaises(EOFError):Q.read_frame(io.BytesIO(packet(2,[1,4,-2])[:-1]))
        with self.assertRaises(ValueError):Q.read_frame(io.BytesIO(b'x'*56))
        with self.assertRaises(ValueError):Q.read_frame(io.BytesIO(packet(2,[1,float('nan'),2])))
        with self.assertRaises(ValueError):Q.read_frame(io.BytesIO(packet(2,[1,4,2],top=0)))
    def test_kl_matches_independent_binary_distribution(self):
        p=np.log(np.array([.25,.75],dtype=np.float64)).astype(np.float32)
        q=np.log(np.array([.5,.5],dtype=np.float64)).astype(np.float32)
        kl,agree,gap=Q.metrics(p,q)
        self.assertAlmostEqual(kl,.25*math.log(.5)+.75*math.log(1.5),places=7)
        self.assertEqual(agree,0);self.assertGreater(gap,0)
        self.assertEqual(Q.metrics(p,p)[0],0)
    def test_bad_worker_start_is_reaped(self):
        folder=self.folder()/'worker'
        code="import sys,time;sys.stdout.buffer.write(b'x'*56);sys.stdout.buffer.flush();time.sleep(30)"
        with self.assertRaises(RuntimeError):Q.Worker([sys.executable,'-c',code],Path(sys.executable).parent,folder,timeout=3)
        result=(folder/'result.txt').read_text();self.assertIn('clean=False',result);self.assertNotIn('exit=None',result)
    def test_cluster_report_uses_matched_rows_and_rejects_misalignment(self):
        root=self.folder();labels=['BF16-HYLE','Q8_0-HYLE','Q8_0-PEER','Q4_K_M-HYLE','Q4_K_M-PEER']
        for label in labels:
            folder=root/label;folder.mkdir()
            with (folder/'rows.tsv').open('w',newline='') as stream:
                writer=csv.writer(stream,delimiter='\t');writer.writerow(['id','group','position','context','teacher','agree','kl'])
                for case,count in ((1,1),(2,8),(3,2),(4,4)):
                    value=.0001 if label=='BF16-HYLE' else .01 if 'Q8' in label else .1
                    for pos in range(count):writer.writerow([case,'a' if case<=2 else 'b',pos,10+pos,7,1,value])
        with contextlib.redirect_stdout(io.StringIO()):Q.analyze(root,200,123)
        report=(root/'summary.txt').read_text();self.assertIn('positions=15;prompts=4',report)
        self.assertIn('prompt_cluster_bootstrap_95=[1,1]',report)
        self.assertIn('NUMERICAL_GUARD_PASS=True',report)
        p=root/'Q8_0-HYLE/rows.tsv';p.write_text(p.read_text().replace('1\ta\t0\t10\t7','1\ta\t0\t10\t8',1))
        with self.assertRaises(ValueError):Q.analyze(root,20,123)
    def test_corpus_is_frozen_unique_and_stratified(self):
        data=Q.load_corpus(ROOT/'tests/data/inference-quality-prompts.tsv')
        self.assertEqual(len(data),256)
        counts={g:sum(c['group']==g for c in data) for g in {c['group'] for c in data}}
        self.assertEqual(len(counts),8);self.assertEqual(set(counts.values()),{32})
if __name__=='__main__':unittest.main()
