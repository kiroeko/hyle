from pathlib import Path
import hashlib
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('draft_head_preparer',ROOT/'scripts/prepare-mtp-draft-head.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
LIB=ROOT/'.scratch/inference-upstream-head-to-head-20261103/official/ggml-base.dll'
VERSION='llama.cpp@2cbca389f515df455412dafa07e2b1359cc72a6d'


class DraftHeadPreparation(unittest.TestCase):
    def setUp(self):
        # Keep finite validation artifacts under the repository. No model is
        # copied/downloaded and this test does not recursively remove caches.
        self.work=Path(tempfile.mkdtemp(prefix='draft-head-prep-',dir=ROOT/'.scratch'))

    def source(self,dtype='BF16',rows=4):
        import numpy as np
        values=((np.arange(rows*256,dtype=np.float32)%31)-15)/8
        if dtype=='BF16': raw=(values.view(np.uint32)>>16).astype('<u2').tobytes()
        else: raw=values.astype('<f2' if dtype=='F16' else '<f4').tobytes()
        path=self.work/'head.bin';path.write_bytes(b'prefix!!'+raw+b'untouched suffix')
        text=('HYLE_QWEN_ROLE_FILES_V2\n'+'a'*64+'\ngeometry 256 512 '+str(rows)+'\n'
              f'model.language_model.embed_tokens.weight {dtype} 2 {rows} 256 "{path.as_posix()}" 8 {len(raw)}\n')
        (self.work/'weights.tsv').write_text(text,encoding='utf-8')
        return text,path,values

    def test_binds_original_source_offset_shape_and_identity(self):
        _,path,_=self.source()
        head=P.source_head(self.work)
        self.assertEqual((head['identity'],head['path'],head['offset'],head['bytes']),('a'*64,path,8,2048))
        self.assertEqual((head['vocabulary'],head['hidden'],head['dtype']),(4,256,'BF16'))

    def test_untied_head_does_not_quantize_embedding(self):
        text,path,_=self.source()
        text+='lm_head.weight BF16 2 4 256 "'+path.as_posix()+'" 8 2048\n'
        text=text.replace('embed_tokens.weight BF16','embed_tokens.weight U8')
        (self.work/'weights.tsv').write_text(text,encoding='utf-8')
        self.assertEqual(P.source_head(self.work)['dtype'],'BF16')

    def test_rejects_invalid_source_ranges_and_shapes(self):
        text,_,_=self.source()
        for old,new in [('a'*64,'not-an-identity'),('geometry 256','geometry 255'),
                        ('BF16 2 4 256','BF16 2 3 256'),('BF16 2','Q4_K 2'),
                        ('8 2048','-1 2048'),('8 2048','8 2049'),('8 2048','999999 2048')]:
            with self.subTest(new=new):
                (self.work/'weights.tsv').write_text(text.replace(old,new),encoding='utf-8')
                with self.assertRaises(ValueError):P.source_head(self.work)
        (self.work/'weights.tsv').write_text(text+text.splitlines()[-1]+'\n',encoding='utf-8')
        with self.assertRaises(ValueError):P.source_head(self.work)

    @unittest.skipUnless(LIB.is_file(),'local pinned ggml library unavailable; no implicit download')
    def test_real_quantizer_all_plain_types_tail_chunk_and_source_immutability(self):
        for dtype in ('BF16','F16','F32'):
            with self.subTest(dtype=dtype):
                self.source(dtype,1031)
                original=(self.work/'head.bin').read_bytes()
                output=self.work/dtype
                result=P.prepare(self.work,LIB,VERSION,output)
                payload=(output/'draft-head-q4_k.bin').read_bytes()
                lines=(output/'mtp-draft-head.tsv').read_text().splitlines()
                self.assertEqual(len(payload),1031*144)
                self.assertEqual(lines[:4],[P.SIGNATURE,'a'*64,hashlib.sha256(payload).hexdigest(),VERSION+':Q4_K:reference:no-imatrix'])
                self.assertEqual(lines[4:],["1031 256",'"draft-head-q4_k.bin" '+str(1031*144)])
                self.assertEqual(result['bytes'],len(payload))
                self.assertEqual(result['source_head_sha256'],hashlib.sha256(original[8:-16]).hexdigest())
                self.assertEqual((self.work/'head.bin').read_bytes(),original)

    @unittest.skipUnless(LIB.is_file(),'local pinned ggml library unavailable; no implicit download')
    def test_invalid_values_do_not_publish_metadata_and_existing_output_is_untouched(self):
        import numpy as np
        self.source('F32')
        path=self.work/'head.bin';raw=bytearray(path.read_bytes());raw[8:12]=np.array([np.nan],dtype='<f4').tobytes();path.write_bytes(raw)
        output=self.work/'invalid'
        with self.assertRaises(ValueError):P.prepare(self.work,LIB,VERSION,output)
        self.assertFalse((output/'mtp-draft-head.tsv').exists())
        self.source('F32')
        marker=output/'marker';marker.write_text('keep')
        with self.assertRaises(FileExistsError):P.prepare(self.work,LIB,VERSION,output)
        self.assertEqual(marker.read_text(),'keep')
        self.assertFalse((output/'mtp-draft-head.tsv').exists())

    def test_invalid_quantizer_identity_does_not_create_output(self):
        self.source()
        out=self.work/'bad-version'
        with self.assertRaises(ValueError):P.prepare(self.work,LIB,'unknown',out)
        self.assertFalse(out.exists())

    def default_references(self):
        # Exercise reference publication only; the runtime separately validates
        # the full mathematical inventory on the actual local 2B checkpoint.
        identity='a'*64
        geometry='geometry 2048 6144 248320 8 2 256 256 64 16 16 128 128 4 262144 10000000 1e-6 1 1 grouped "GGGAGGGAGGGAGGGAGGGAGGGA"'
        names=['mtp.fc.weight','mtp.norm.weight','mtp.pre_fc_norm_embedding.weight','mtp.pre_fc_norm_hidden.weight']
        names += ['mtp.layers.0.'+x+'.weight' for x in ['input_layernorm','post_attention_layernorm',
            'mlp.gate_proj','mlp.up_proj','mlp.down_proj','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj',
            'self_attn.o_proj','self_attn.q_norm','self_attn.k_norm']]
        (self.work/'borrowed weights.bin').write_bytes(b'12345678')
        text='\n'.join(['HYLE_QWEN_ROLE_FILES_V2',identity,geometry]+[
            name+' BF16 2 2 2 "borrowed weights.bin" 0 8' for name in names])+'\n'
        (self.work/'weights.tsv').write_text(text,encoding='utf-8')
        (self.work/'prompt.ids').write_bytes(b'1 2 3\n')
        mapping=self.work/'mapping.ids'
        mapping.write_text('hyle-draft-vocabulary-v1\n'+identity+'\n248320\n'+
                           ' '.join(map(str,reversed(range(114688))))+'\n',encoding='utf-8')
        return mapping,dict(identity=identity,dtype='BF16',hidden=2048,vocabulary=248320),text

    def test_ready_inputs_borrow_original_ranges_and_keep_source_ids(self):
        mapping,head,text=self.default_references()
        files=P.default_k3_inputs(self.work,mapping,head)
        self.assertEqual(set(files),{'weights.tsv',P.DEFAULT_K3_VOCABULARY,'prompt.ids'})
        self.assertEqual(files['prompt.ids'],b'1 2 3\n')
        lines=files['weights.tsv'].decode().splitlines()
        self.assertEqual(lines[:3],text.splitlines()[:3])
        self.assertEqual(sum(line.startswith('mtp.') for line in lines),15)
        self.assertIn('"'+(self.work/'borrowed weights.bin').as_posix()+'"',lines[-1])
        ids=files[P.DEFAULT_K3_VOCABULARY].decode().splitlines()
        self.assertEqual(ids[:3],['hyle-draft-vocabulary-v1','a'*64,'248320'])
        self.assertEqual(list(map(int,ids[3].split())),list(range(114688)))
        self.assertEqual((self.work/'weights.tsv').read_text(),text)

    def test_default_preparation_refuses_stale_duplicate_or_invalid_rows(self):
        mapping,head,_=self.default_references();text=mapping.read_text()
        for invalid in [text.replace('a'*64,'b'*64),text.replace('114687 114686','114686 114686'),
                        text.replace('114687 ','248320 '),text.replace('114687 ','-1 '),
                        '\n'.join(text.splitlines()[:3])+'\n']:
            with self.subTest(prefix=invalid[:80]):
                mapping.write_text(invalid)
                with self.assertRaises(ValueError):P.default_k3_inputs(self.work,mapping,head)

    def test_prepared_inputs_have_no_model_size_or_fixed_subset_gate(self):
        mapping,head,text=self.default_references()
        # Reference publication does not claim another model's performance.
        # The runtime retains full source-inventory and backend qualification.
        text=text.replace('geometry 2048 6144', 'geometry 4096 12288')
        (self.work/'weights.tsv').write_text(text,encoding='utf-8')
        mapping.write_text('hyle-draft-vocabulary-v1\n'+'a'*64+'\n248320\n9 3 0\n')
        files=P.default_k3_inputs(self.work,mapping,head)
        self.assertEqual(files[P.DEFAULT_K3_VOCABULARY].decode().splitlines()[3],'0 3 9')
        self.assertEqual(files['weights.tsv'].decode().splitlines()[2],text.splitlines()[2])

    def test_unqualified_defaults_fail_before_quantizer_or_output_creation(self):
        self.source();mapping=self.work/'not-read.ids';out=self.work/'unqualified'
        with patch.object(P,'Quantizer',side_effect=AssertionError('unqualified source reached quantizer')):
            with self.assertRaises(ValueError):P.prepare(self.work,LIB,VERSION,out,mapping)
        self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
