from pathlib import Path
import hashlib
import importlib.util
import tempfile
import unittest

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


if __name__=='__main__':unittest.main()
