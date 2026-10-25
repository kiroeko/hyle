"""Host-only GGUF input contract tests; no model payloads or network."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from engine.models.qwen35 import gguf_header as grammar
from engine.models.qwen35 import gguf_roles as roles


def fixture(embedding_code=14):
    metadata = {'general.architecture':'qwen35','general.type':'model','qwen35.block_count':25,
        'qwen35.context_length':262144,'qwen35.embedding_length':2048,'qwen35.feed_forward_length':6144,
        'qwen35.attention.head_count':8,'qwen35.attention.head_count_kv':2,
        'qwen35.attention.key_length':256,'qwen35.attention.value_length':256,
        'qwen35.attention.layer_norm_rms_epsilon':1.e-6,'qwen35.ssm.conv_kernel':4,
        'qwen35.ssm.state_size':128,'qwen35.ssm.group_count':16,'qwen35.ssm.time_step_rank':16,
        'qwen35.ssm.inner_size':2048,'qwen35.full_attention_interval':4,'qwen35.rope.dimension_count':64,
        'qwen35.rope.dimension_sections':[11,11,10,0],'qwen35.rope.freq_base':10000000.0,
        'qwen35.nextn_predict_layers':1}
    tensors, offset = [], 0
    g=roles.geometry.from_gguf(metadata,[dict(name='token_embd.weight',shape=[2048,248320])])
    for name,spec in roles.role_specs(g).items():
        # Shape inventory is generated, but the checked semantic/extent cases
        # below name their expected mappings independently.
        shape = [4,6144] if name.endswith('ssm_conv1d.weight') else list(reversed(spec['logical_shape']))
        code = embedding_code if name == 'token_embd.weight' else 8 if name.endswith('nextn.eh_proj.weight') else 0
        _,block,width = roles.CODECS[code]
        count = 1
        for dimension in shape:
            count *= dimension
        offset = (offset+31)//32*32
        tensors.append(dict(name=name,shape=shape,dtype_code=code,offset=offset))
        offset += count//block*width
    header = dict(tensor_count=335,tensor_data_offset=32768,metadata=metadata,tensors=tensors)
    source = dict(repository='test/Qwen3.5-2B-MTP',revision='1'*40,file='fixture.gguf',
                  size=32768+offset,lfs_sha256='2'*64)
    return header,source


def inventory(header=None,source=None):
    if header is None:
        header,source = fixture()
    return roles.typed_inventory(header,source,header_sha256='3'*64)


def string(value):
    raw = value.encode()
    return struct.pack('<Q',len(raw))+raw


def small_header():
    raw = bytearray(struct.pack('<4sIQQ',b'GGUF',3,1,1))
    raw += string('general.architecture')+struct.pack('<I',8)+string('qwen35')
    raw += string('blk.24.nextn.eh_proj.weight')+struct.pack('<I',2)+struct.pack('<QQIQ',4096,2048,8,0)
    raw += b'\0'*(-len(raw)%32)
    return bytes(raw)


class QwenGgufInputTest(unittest.TestCase):
    def test_untied_larger_source_geometry_and_original_iq_blocks_are_preserved(self):
        header,source=fixture();m=header['metadata']
        m.update({'qwen35.block_count':65,'qwen35.embedding_length':5120,
            'qwen35.feed_forward_length':17408,'qwen35.attention.head_count':24,
            'qwen35.attention.head_count_kv':4,'qwen35.ssm.time_step_rank':48,
            'qwen35.ssm.inner_size':6144})
        g=roles.geometry.from_gguf(m,[dict(name='token_embd.weight',shape=[5120,248320]),
                                     dict(name='output.weight',shape=[5120,248320])])
        tensors,offset=[],0;codecs=(11,23,20,21,12,13,14,8,40)
        for i,(name,spec)in enumerate(roles.role_specs(g).items()):
            shape=roles._shape_ne(spec)
            code=codecs[i%len(codecs)] if spec['semantic']in ('linear_weight','embedding_rows') else 0
            _,block,width=roles.CODECS[code]
            count=1
            for d in shape:count*=d
            offset=(offset+31)//32*32
            tensors.append(dict(name=name,shape=shape,dtype_code=code,offset=offset))
            offset+=count//block*width
        header.update(tensor_count=len(tensors),tensors=tensors)
        source['size']=header['tensor_data_offset']+offset
        result=inventory(header,source);by_name={r['physical_name']:r for r in result['roles']}
        self.assertEqual((result['physical_role_count'],result['main_role_count'],result['mtp_role_count']),(866,851,15))
        self.assertEqual(by_name['output.weight']['logical_name'],'lm_head.weight')
        self.assertEqual(by_name['blk.64.attn_q.weight']['storage']['logical_shape'],[12288,5120])
        self.assertEqual(by_name['blk.0.ssm_conv1d.weight']['storage']['logical_shape'],[10240,1,4])
        self.assertEqual(result['geometry']['gdn_layout'],'cyclic')
        self.assertEqual(result['vocabulary']['draft_head_physical'],'output.weight')
        for code,expected in ((11,('Q3_K',256,110)),(23,('IQ4_XS',256,136)),
                              (20,('IQ4_NL',32,18)),(21,('IQ3_S',256,110)),
                              (40,('NVFP4',64,36))):
            self.assertEqual(roles.CODECS[code],expected)
            self.assertIn(expected[0],result['physical_codecs'])

    def test_nvfp4_type40_preserves_64_value_blocks_and_selected_row_layout(self):
        header,source=fixture(40)
        result=inventory(header,source)
        embedding=next(row for row in result['roles'] if row['physical_name']=='token_embd.weight')
        self.assertEqual(embedding['dtype_code'],40)
        self.assertEqual(embedding['storage'],dict(kind='gguf_blocked',dtype='NVFP4',logical_dtype='F32',
            logical_shape=[248320,2048],original_ne=[2048,248320],block_axis=1,
            elements_per_block=64,bytes_per_block=36,byte_strides=[1152,36]))
        self.assertEqual(embedding['bytes'],286064640)
        self.assertEqual(result['vocabulary']['target_head_physical'],'token_embd.weight')
        # GGUF tensor type39 is MXFP4, not NVFP4; file_type39 cannot select it.
        bad=deepcopy(header);bad['tensors'][0]['dtype_code']=39
        with self.assertRaisesRegex(ValueError,'unsupported physical GGUF codec'):
            inventory(bad,source)
        bad=deepcopy(header)
        next(row for row in bad['tensors'] if row['name']=='output_norm.weight')['dtype_code']=40
        with self.assertRaisesRegex(ValueError,'plain floating representation'):
            inventory(bad,source)

    def test_payload_preparation_is_distinct_complete_and_uses_borrowed_ranges(self):
        header,source = fixture(1)
        typed = inventory(header,source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b'cached-header-identity'
            (root/'header.bin').write_bytes(raw)
            header_sha = hashlib.sha256(raw).hexdigest()
            source.update(header_bytes=len(raw),header_sha256=header_sha,api_xet_hash='4'*64)
            bindings = [dict(physical_name=row['physical_name'],path='borrowed payload.bin',
                offset=row['original_offset'],bytes=row['bytes'],sha256='5'*64,
                original_offset=row['absolute_inclusive'][0]) for row in typed['roles']]
            data = dict(signature=roles.PAYLOAD_SIGNATURE,complete=True,source=source,
                header=dict(path='header.bin',bytes=len(raw),sha256=header_sha),roles=bindings)
            path = root/'payload.json';path.write_text(json.dumps(data))
            with patch.object(roles,'ROOT',root),patch.object(roles,'inspect_local',return_value=typed),\
                    patch.object(roles,'_range_sha256',return_value='5'*64) as authenticates:
                lines,receipt = roles.prepare_role_files(path,False)
                self.assertEqual(authenticates.call_count,335)
                self.assertEqual(lines[0],roles.ROLE_SIGNATURE)
                self.assertNotEqual(lines[0],'HYLE_QWEN35_ROLE_FILES_V1')
                self.assertEqual(len(lines),339)
                self.assertEqual((receipt['metadata_tensor_count'],receipt['selected_tensor_count']),(335,320))
                self.assertTrue(any(line.startswith('mtp.fc.weight\tblk.24.nextn.eh_proj.weight\tQ8_0\tlinear_weight') for line in lines))
                self.assertTrue(any('\tgdn_negative_exp_A_log\t' in line for line in lines))
                data['roles'][-1]['bytes'] += 1
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,'range identity'):
                    roles.prepare_role_files(path,False)
            path.write_text(json.dumps(typed))
            with self.assertRaisesRegex(ValueError,'header-only'):
                roles.prepare_role_files(path,True)

    def test_original_range_hashing_is_bounded_and_rejects_short_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'range.bin';path.write_bytes(b'prefix-original-weight-suffix')
            self.assertEqual(roles._range_sha256(path,7,15),hashlib.sha256(b'original-weight').hexdigest())
            with self.assertRaises(ValueError):
                roles._range_sha256(path,7,999)

    def test_fidelity_embedding_keeps_plain_f16_instead_of_inventing_q8(self):
        header,source = fixture(1)
        result = inventory(header,source)
        embedding = next(row for row in result['roles'] if row['physical_name'] == 'token_embd.weight')
        self.assertEqual(embedding['storage']['kind'],'plain')
        self.assertEqual(embedding['storage']['dtype'],'F16')
        self.assertEqual(embedding['storage']['logical_dtype'],'F16')
        self.assertNotIn('block_axis',embedding['storage'])
        self.assertEqual(embedding['bytes'],1017118720)

    def test_roles_preserve_original_codec_layout_and_parameter_meaning(self):
        result = inventory()
        self.assertEqual((result['physical_role_count'],result['main_role_count'],result['mtp_role_count']),(335,320,15))
        by_name = {row['physical_name']:row for row in result['roles']}
        projection = by_name['blk.24.nextn.eh_proj.weight']
        self.assertEqual(projection['logical_name'],'mtp.fc.weight')
        self.assertEqual(projection['bytes'],8912896)
        self.assertEqual(projection['storage'],dict(kind='gguf_blocked',dtype='Q8_0',logical_dtype='F32',
            logical_shape=[2048,4096],original_ne=[4096,2048],block_axis=1,
            elements_per_block=32,bytes_per_block=34,byte_strides=[4352,34]))
        self.assertEqual(by_name['blk.0.ssm_a']['logical_name'],'model.language_model.layers.0.linear_attn.A_log')
        self.assertEqual(by_name['blk.0.ssm_a']['semantic'],'gdn_negative_exp_A_log')
        self.assertEqual(by_name['blk.24.nextn.enorm.weight']['semantic'],'rms_effective_gamma')
        self.assertEqual(by_name['blk.0.ssm_norm.weight']['semantic'],'gdn_effective_gamma')
        self.assertEqual(by_name['blk.0.ssm_conv1d.weight']['storage']['logical_shape'],[6144,1,4])
        self.assertEqual(by_name['blk.0.ssm_conv1d.weight']['bytes'],98304)
        self.assertEqual(result['vocabulary']['target_head_physical'],'token_embd.weight')
        self.assertEqual(result['vocabulary']['draft_head_physical'],'token_embd.weight')
        self.assertNotEqual(result['signature'],'HYLE_QWEN35_ROLE_FILES_V1')
        self.assertEqual(result['tensor_payload_bytes_read'],0)

    def test_missing_extra_duplicate_and_wrong_geometry_are_rejected(self):
        for kind in ('missing','extra','duplicate','geometry','split'):
            header,source = fixture()
            if kind == 'missing':
                header['tensors'].pop()
            elif kind == 'extra':
                header['tensors'].append(dict(name='output.weight',shape=[2048,248320],dtype_code=14,offset=0))
            elif kind == 'duplicate':
                header['tensors'][-1] = deepcopy(header['tensors'][0])
            elif kind == 'geometry':
                header['metadata']['qwen35.nextn_predict_layers'] = 0
            else:
                header['metadata']['split.count'] = 2
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                inventory(header,source)

    def test_codec_shape_range_overlap_and_norm_cast_are_rejected(self):
        for kind in ('codec','shape','range','overlap','norm_cast'):
            header,source = fixture()
            row = header['tensors'][1]
            if kind == 'codec':
                row['dtype_code'] = 999
            elif kind == 'shape':
                row['shape'] = [128]
            elif kind == 'range':
                row['offset'] = source['size']
            elif kind == 'overlap':
                row['offset'] = 0
            else:
                row['dtype_code'] = 8
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                inventory(header,source)

    def test_exact_structural_reader_never_requests_tensor_bytes(self):
        raw = small_header()
        calls = []
        def exact(url,first,last,*,total=None,etag=None):
            self.assertLess(last,len(raw))
            calls.append((first,last))
            return raw[first:last+1],len(raw)+8912896,'opaque-fixed-identity'
        parsed,metadata,tensors,count,total,etag = grammar.scan(exact,url='local-fixture',tensor_count=1,
            maximum_header_bytes=4096,chunk_bytes=64)
        self.assertEqual(parsed,raw)
        self.assertEqual(sum(last-first+1 for first,last in calls),len(raw))
        self.assertEqual(metadata['general.architecture'],'qwen35')
        self.assertEqual(tensors[0]['shape'],[4096,2048])
        self.assertEqual((count,total,etag),(1,len(raw)+8912896,'opaque-fixed-identity'))
        with self.assertRaises(ValueError):
            grammar.parse_header(raw[:-1],1,len(raw)-1,whole_file_bytes=total)

    def test_local_reader_does_not_read_payload_or_accept_a_short_header(self):
        raw = small_header()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.gguf'
            path.write_bytes(raw+b'TENSOR_PAYLOAD_MUST_NOT_BE_READ')
            parsed,received = grammar.read_local_header(path)
            self.assertEqual(received,raw)
            self.assertEqual(parsed['tensor_data_offset'],len(raw))
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                grammar.read_local_header(path,whole_file_bytes=len(raw)+8912896)


if __name__ == '__main__':
    unittest.main()
