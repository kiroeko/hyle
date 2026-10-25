"""Original HF physical-role preparation; no model, device or network work."""
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
sys.path.insert(0, str(ROOT))
from engine.models.qwen35 import prepare as roles


def ct_group(base, n=8, k=32, group=32, dtype="BF16"):
    result = {}
    for suffix, kind, shape in ((".weight_packed", "I32", [n, k//8]),
                              (".weight_scale", dtype, [n, k//group]),
                              (".weight_shape", "I64", [2])):
        name = base + suffix
        count = roles.ROLE_SIZES[kind]
        for dim in shape:
            count *= dim
        result[name] = dict(name=name, dtype=kind, shape=shape, bytes=count,
                            path="borrowed source.bin", offset=0, sha256="1"*64)
    return result


def source_config():
    return dict(text_config=dict(model_type="qwen3_5_text",attn_output_gate=True,
        attention_bias=False,hidden_act="silu",mtp_use_dedicated_embeddings=False,
        hidden_size=2048,intermediate_size=6144,vocab_size=248320,num_attention_heads=8,
        num_key_value_heads=2,head_dim=256,linear_num_key_heads=16,linear_num_value_heads=16,
        linear_key_head_dim=128,linear_value_head_dim=128,linear_conv_kernel_dim=4,
        max_position_embeddings=262144,num_hidden_layers=24,mtp_num_hidden_layers=1,
        layer_types=["full_attention" if i%4==3 else "linear_attention" for i in range(24)],
        rms_norm_eps=1.e-6,tie_word_embeddings=True,
        rope_parameters=dict(rope_type="default",rope_theta=10000000,
                             partial_rotary_factor=.25,mrope_section=[11,11,10])))


def source_fixture():
    shapes=roles.geometry.role_shapes(roles.geometry.from_hf(source_config()))
    result={}
    for name,shape in shapes.items():
        if name=="model.language_model.layers.0.mlp.gate_proj.weight" or (
                name.startswith("mtp.") and len(shape)==2 and name!="mtp.fc.weight"):
            result.update(ct_group(name[:-7],*shape))
        else:
            count=2
            for dimension in shape:count*=dimension
            result[name]=dict(name=name,dtype="BF16",shape=list(shape),bytes=count,
                path="borrowed source.bin",offset=0,sha256="1"*64)
    return dict(complete=True,repository="test/original",revision="a"*40,roles=list(result.values()))


class OriginalHfRolesTest(unittest.TestCase):
    def test_modelopt_static_fp8_has_one_explicit_reader_and_three_original_roles(self):
        config=source_config();config['text_config']['dtype']='bfloat16'
        base='model.language_model.layers.0.linear_attn.in_proj_qkv'
        nv='model.language_model.layers.0.mlp.gate_proj'
        config['quantization_config']=dict(quant_method='modelopt',quant_algo='MIXED_PRECISION',
            producer=dict(name='modelopt',version='0.47.0.dev80+g913f5e224'),
            quantized_layers={base:dict(quant_algo='FP8'),nv:dict(quant_algo='NVFP4',group_size=16)},
            config_groups=dict(group_0=dict(weights=dict(dynamic=False,num_bits=8,type='float'),
                input_activations=dict(dynamic=False,num_bits=8,type='float'),targets=[base])))
        model=roles.geometry.from_hf(config);physical=[]
        for name,shape in roles.geometry.role_shapes(model).items():
            if name==base+'.weight':
                entries=((name,'F8_E4M3',shape),(base+'.input_scale','F32',[]),(base+'.weight_scale','F32',[]))
            elif name==nv+'.weight':
                n,k=shape
                entries=((name,'U8',[n,k//2]),(nv+'.weight_scale','F8_E4M3',[n,k//16]),
                    (nv+'.weight_scale_2','F32',[]),(nv+'.input_scale','F32',[]))
            else:entries=((name,'BF16',shape),)
            for key,dtype,dims in entries:
                count=roles.ROLE_SIZES[dtype]
                for dim in dims:count*=dim
                physical.append(dict(name=key,dtype=dtype,shape=list(dims),bytes=count,
                    path='original static scales.bin',offset=0,sha256='8'*64))
        source=dict(complete=True,repository='test/modelopt-static',revision='c'*40,roles=physical)
        authenticated=[]
        def authenticate(role):
            authenticated.append(role['name'])
            return Path(role['path']),None
        for mtp in (False,True):
            lines,receipt=roles._prepare_role_records(source,mtp,config,authenticate,roles.STATIC_FP8_READER)
            self.assertIn(roles.STATIC_FP8_POLICY_LINE,lines)
            self.assertEqual(receipt['modelopt_static_fp8_groups'],1)
            self.assertEqual(receipt['modelopt_static_fp8_policy'],roles.STATIC_FP8_POLICY)
            self.assertNotIn('ct_mixed_policy',receipt);self.assertNotIn('fp8_block128_policy',receipt)
            self.assertEqual(receipt['physical_roles'],len(roles.geometry.role_shapes(model,include_mtp=mtp))+5)
            self.assertTrue(any(line.startswith(base+'.weight\tF8_E4M3\t2')for line in lines))
            self.assertTrue(any(line.startswith(nv+'.weight\tU8\t2')for line in lines))
        _,old_receipt=roles._prepare_role_records(source,False,config,authenticate,roles.STATIC_FP8_READER)
        future=deepcopy(config);future['quantization_config']['producer']['version']='future_same_physical_contract'
        _,future_receipt=roles._prepare_role_records(source,False,future,authenticate,roles.STATIC_FP8_READER)
        self.assertNotEqual(old_receipt['source_id'],future_receipt['source_id'])
        self.assertEqual(future_receipt['modelopt_static_fp8_source']['producer']['version'],'future_same_physical_contract')
        self.assertEqual(future_receipt['modelopt_static_fp8_policy'],old_receipt['modelopt_static_fp8_policy'])
        authenticated.clear()
        with self.assertRaisesRegex(ValueError,'explicit --fp8-reader'):
            roles._prepare_role_records(source,False,config,authenticate)
        self.assertFalse(authenticated)
        for change in ('missing','dtype','axis','ambiguous','source_dtype','dynamic','targets'):
            bad,policy=deepcopy(source),deepcopy(config)
            if change=='missing':bad['roles']=[r for r in bad['roles']if r['name']!=base+'.input_scale']
            elif change=='dtype':
                row=next(r for r in bad['roles']if r['name']==base+'.weight_scale');row.update(dtype='BF16',bytes=2)
            elif change=='axis':next(r for r in bad['roles']if r['name']==base+'.input_scale')['shape']=[1,1]
            elif change=='ambiguous':
                row=deepcopy(next(r for r in bad['roles']if r['name']==base+'.weight_scale'));row['name']=base+'.weight_scale_2';bad['roles'].append(row)
            elif change=='source_dtype':policy['text_config']['dtype']='float16'
            elif change=='dynamic':policy['quantization_config']['config_groups']['group_0']['input_activations']['dynamic']=True
            else:policy['quantization_config']['config_groups']['group_0']['targets']=[]
            authenticated.clear()
            with self.subTest(change=change),self.assertRaises(ValueError):
                roles._prepare_role_records(bad,False,policy,authenticate,roles.STATIC_FP8_READER)
            self.assertFalse(authenticated)

    def test_mlx_original_affine_reader_preserves_provenance_and_tied_head_math(self):
        for bits,tied in ((4,True),(8,False)):
            config=source_config();config['text_config']['tie_word_embeddings']=tied
            config['quantization']=dict(group_size=64,bits=bits,mode='affine')
            g=roles.geometry.from_hf(config);physical=[]
            for name,shape in roles.geometry.role_shapes(g,include_mtp=False).items():
                original='language_model.model.'+name[21:] if name.startswith('model.language_model.') else 'language_model.'+name
                if len(shape)==2:
                    n,k=shape
                    entries=((original,'U32',[n,k//32*bits]),(original[:-7]+'.scales','BF16',[n,k//64]),(original[:-7]+'.biases','BF16',[n,k//64]))
                else:
                    stored=[shape[0],shape[2],1] if name.endswith('.conv1d.weight') else list(shape)
                    entries=((original,'BF16',stored),)
                for key,dtype,dims in entries:
                    count=roles.ROLE_SIZES[dtype]
                    for d in dims:count*=d
                    physical.append(dict(name=key,dtype=dtype,shape=dims,bytes=count,path='original-mlx.bin',offset=0,sha256='3'*64))
            source=dict(complete=True,repository='test/original-mlx',revision='b'*40,roles=physical)
            lines,receipt=roles._prepare_role_records(source,False,config,lambda r:(Path(r['path']),None))
            self.assertIn('mlx_affine\t'+str(bits)+'\t64\tBF16\tF32\tBF16', '\n'.join(lines))
            self.assertTrue(any(x.startswith('language_model.model.embed_tokens.weight\tU32')for x in lines))
            self.assertEqual(receipt['original_role_records'][0]['name'].split('.')[0],'language_model')
            self.assertEqual(receipt['geometry']['tied_head'],tied)
            self.assertEqual(receipt['mlx_affine_policy']['producer_version_verified'],False)
            broken=deepcopy(source);broken['roles']=[r for r in broken['roles']if r['name']!='language_model.model.embed_tokens.biases']
            with self.assertRaisesRegex(ValueError,'incomplete MLX'):roles._prepare_role_records(broken,False,config,lambda r:(Path('x'),None))
            broken=deepcopy(source);conv=next(r for r in broken['roles']if r['name'].endswith('conv1d.weight'));conv['shape']=[conv['shape'][0],1,conv['shape'][1]]
            with self.assertRaisesRegex(ValueError,'already sanitized'):roles._prepare_role_records(broken,False,config,lambda r:(Path('x'),None))

    def test_ct_asymmetric_preserves_packed_zero_axis_and_requires_source_policy(self):
        # Original Q27 cyankiwi@6e134bae... uses I32 Z[ceil(N/8),K/32],
        # not its config's unpacked int8 zero dtype or AWQ/GPTQ axes.
        group = ct_group("x", 1024, 5120)
        group['x.weight_zero_point'] = dict(dtype='I32', shape=[128,160])
        actual = roles.ct_packed_groups(group)
        self.assertEqual(actual['x'], dict(logical_shape=[1024,5120],group_size=32,
                                          scale_dtype='BF16',asymmetric=True))
        config = dict(quantization_config=dict(quant_method='compressed-tensors',format='pack-quantized',
            config_groups=dict(group_0=dict(format='pack-quantized',input_activations=None,output_activations=None,
                targets=['Linear'],weights=dict(num_bits=4,type='int',strategy='group',group_size=32,
                    symmetric=False,dynamic=False,actorder=None,block_structure=None,scale_dtype=None,zp_dtype='torch.int8')))))
        self.assertEqual(roles.ct_asymmetric_policy(actual,config), roles.CT_ASYMMETRIC_POLICY)
        for dtype, shape in (('I64',[128,160]),('I32',[160,128]),('I32',[128,159])):
            bad = deepcopy(group);bad['x.weight_zero_point']=dict(dtype=dtype,shape=shape)
            with self.subTest(dtype=dtype,shape=shape), self.assertRaisesRegex(ValueError,'zero-point output-axis'):
                roles.ct_packed_groups(bad)
        for mutate in ('absent','symmetric','group','type','activation'):
            bad = deepcopy(config)
            if mutate == 'absent': bad = {}
            elif mutate == 'activation': bad['quantization_config']['config_groups']['group_0']['input_activations'] = {}
            else:
                key,value = {'symmetric':('symmetric',True),'group':('group_size',128),'type':('zp_dtype','torch.int32')}[mutate]
                bad['quantization_config']['config_groups']['group_0']['weights'][key]=value
            with self.subTest(config=mutate), self.assertRaisesRegex(ValueError,'reader config'):
                roles.ct_asymmetric_policy(actual,bad)
        no_zero = deepcopy(group);del no_zero['x.weight_zero_point']
        with self.assertRaisesRegex(ValueError,'every packed zero-point'):
            roles.ct_asymmetric_policy(roles.ct_packed_groups(no_zero),config)
        self.assertEqual(roles.logical_roles(group),{'x.weight'})

    def test_ct_asymmetric_ordinary_preparation_keeps_all_original_roles_and_zero_policy(self):
        original = source_fixture()
        original['roles']=[r for r in original['roles'] if not r['name'].startswith('mtp.')]
        config=source_config()
        config['quantization_config']=dict(quant_method='compressed-tensors',format='pack-quantized',config_groups=dict(
            group_0=dict(format='pack-quantized',targets=['Linear'],input_activations=None,output_activations=None,
                weights=dict(num_bits=4,type='int',strategy='group',group_size=32,symmetric=False,dynamic=False,
                    actorder=None,block_structure=None,scale_dtype=None,zp_dtype='torch.int8'))))
        base='model.language_model.layers.0.mlp.gate_proj'
        n,k=roles.ct_packed_groups({r['name']:r for r in original['roles']})[base]['logical_shape']
        name=base+'.weight_zero_point'
        original['roles'].append(dict(name=name,dtype='I32',shape=[(n+7)//8,k//32],bytes=4*((n+7)//8)*(k//32),
            path='original zero.bin',offset=0,sha256='3'*64))
        authenticate=lambda r:(Path(r['path']),struct.pack('<qq',n,k) if r['name']==base+'.weight_shape' else None)
        lines,receipt=roles._prepare_role_records(original,False,config,authenticate)
        self.assertEqual(lines[3],roles.CT_ASYMMETRIC_POLICY_LINE)
        self.assertEqual((receipt['ct_asymmetric_int4_groups'],receipt['metadata_ct_asymmetric_int4_groups']),(1,1))
        self.assertEqual(receipt['ct_asymmetric_int4_policy'],roles.CT_ASYMMETRIC_POLICY)
        self.assertEqual(receipt['selected_tensor_count'],len(original['roles']))
        self.assertTrue(any(line.startswith(name+'\tI32\t2\t768\t64\t') for line in lines))
        missing=deepcopy(original);missing['roles']=[r for r in missing['roles'] if r['name']!=name]
        with self.assertRaisesRegex(ValueError,'every packed zero-point'):
            roles._prepare_role_records(missing,False,config,authenticate)
        wrong=deepcopy(original);next(r for r in wrong['roles']if r['name']==name)['shape'].reverse()
        with self.assertRaisesRegex(ValueError,'zero-point output-axis'):
            roles._prepare_role_records(wrong,False,config,authenticate)

    def test_original_nf4_json_is_f16_bounded_and_not_a_smaller_component_shape(self):
        # unsloth/Qwen3.8-27B-unsloth-bnb-4bit@8aa5f05d26b7205477066e1449e0af13f762a299
        # model.safetensors bytes15264695436..15264695606; exact original171B.
        raw=(Path(__file__).parent/'fixtures/qwen_nf4_quant_state.json').read_bytes()
        state=roles.nf4_state.parse(raw,[1024,5120])
        self.assertEqual((len(raw),state['offset_f32_bits']),(171,'3d2fd550'))
        self.assertEqual(state['original_json_sha256'],'9210c25c0ca8632c7166cc3a92dc0cc43ffe528a45a11e3b8dde08941ed9d6c1')
        for bad in (raw+b' '*512,raw.replace(b'float16',b'bfloat16'),
                    raw.replace(b'"blocksize": 64',b'"blocksize": true'),
                    raw.replace(b'0.04292804002761841',b'NaN'),
                    raw.replace(b'"quant_type": "nf4"',b'"quant_type": "nf4", "quant_type": "nf4"')):
            with self.subTest(raw=bad[:60]),self.assertRaises(ValueError):roles.nf4_state.parse(bad)
        with self.assertRaisesRegex(ValueError,'logical shape'):roles.nf4_state.parse(raw,[3,5120])

    def test_nf4_all_six_roles_and_each_original_state_are_required_before_selection(self):
        config=source_config();config['text_config']['dtype']='float16'
        config['quantization_config']=dict(quant_method='bitsandbytes',bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_quant_storage='uint8',bnb_4bit_compute_dtype='float16')
        shapes=roles.geometry.role_shapes(roles.geometry.from_hf(config))
        selected={'model.language_model.layers.3.self_attn.k_proj.weight','mtp.layers.0.self_attn.k_proj.weight'}
        original=dict(complete=True,repository='test/nf4',revision='f'*40,roles=[]);raw_states={}
        def add(name,dtype,shape,raw=None):
            count=roles.ROLE_SIZES[dtype]
            for dim in shape:count*=dim
            original['roles'].append(dict(name=name,dtype=dtype,shape=list(shape),bytes=count,path='original.bin',offset=0,
                sha256=hashlib.sha256(raw).hexdigest()if raw is not None else '1'*64))
            if raw is not None:raw_states[name]=raw
        for name,shape in shapes.items():
            if name not in selected:add(name,'F16',shape);continue
            n,k=shape;g=(n*k+63)//64
            for suffix,dtype,dims in (('','U8',[n*k//2,1]),('.absmax','U8',[g]),
                    ('.nested_absmax','F32',[(g+255)//256]),('.nested_quant_map','F32',[256]),('.quant_map','F32',[16])):
                add(name+suffix,dtype,dims)
            raw=json.dumps(dict(quant_type='nf4',blocksize=64,dtype='float16',shape=list(shape),nested_blocksize=256,
                nested_dtype='float32',nested_offset=.04292804002761841)).encode()
            add(name+roles.nf4_state.STATE_SUFFIX,'U8',[len(raw)],raw)
        seen=[]
        def authenticate(role):
            seen.append(role['name']);return Path(role['path']),raw_states.get(role['name'])
        lines,receipt=roles._prepare_role_records(original,False,config,authenticate)
        self.assertEqual(lines[3],roles.nf4_state.POLICY_LINE)
        self.assertEqual((receipt['metadata_nf4_double_quant_groups'],receipt['nf4_double_quant_groups']),(2,1))
        self.assertEqual((receipt['metadata_tensor_count'],receipt['physical_roles']),(345,325))
        self.assertEqual(len(seen),345) # Each state is checked once, including unselected MTP.
        self.assertEqual(receipt['nf4_double_quant_policy']['compute_dtype'],'F16')
        self.assertIn('metadata preparation only',receipt['nf4_execution_scope'])
        self.assertIn('retains six original NF4 operands',receipt['nf4_execution_scope'])
        self.assertIn('cold planner selects',receipt['nf4_execution_scope'])
        self.assertIn('reported workspace',receipt['nf4_execution_scope'])
        self.assertIn('full artifact execution is not verified',receipt['nf4_execution_scope'])
        ordinary=deepcopy(original);ordinary['roles']=[r for r in ordinary['roles']if not r['name'].startswith('mtp.')]
        _,plain=roles._prepare_role_records(ordinary,False,config,authenticate)
        self.assertEqual((plain['geometry']['mtp_layers'],plain['metadata_tensor_count'],plain['physical_roles']),(1,325,325))
        with self.assertRaisesRegex(ValueError,'no original NextN'):
            roles._prepare_role_records(ordinary,True,config,authenticate)
        partial=deepcopy(original);partial['roles']=[r for r in partial['roles']if r['name']!='mtp.norm.weight']
        with self.assertRaisesRegex(ValueError,'logical role inventory'):
            roles._prepare_role_records(partial,False,config,authenticate)
        for suffix in roles.nf4_state.SUFFIXES:
            bad=deepcopy(original);bad['roles']=[r for r in bad['roles']if r['name']!='mtp.layers.0.self_attn.k_proj.weight'+suffix]
            with self.subTest(suffix=suffix),self.assertRaises(ValueError):roles._prepare_role_records(bad,False,config,authenticate)
        wrong=dict(raw_states);state_name='mtp.layers.0.self_attn.k_proj.weight'+roles.nf4_state.STATE_SUFFIX
        wrong[state_name]=wrong[state_name].replace(b'512',b'513',1)
        with self.assertRaises(ValueError):roles._prepare_role_records(original,False,config,lambda r:(Path('x'),wrong.get(r['name'])))
        bad=deepcopy(config);bad['quantization_config']['bnb_4bit_compute_dtype']='bfloat16'
        with self.assertRaisesRegex(ValueError,'compute F16'):roles._prepare_role_records(original,False,bad,authenticate)
        bad=deepcopy(original);r=next(r for r in bad['roles']if r['name']=='model.language_model.norm.weight');r['dtype']='BF16'
        with self.assertRaisesRegex(ValueError,'retain F16'):roles._prepare_role_records(bad,False,config,authenticate)

    def test_ct_channel_local_and_bf16_cache_omission_are_separate_contracts(self):
        config=source_config()
        common=dict(num_bits=8,type='float',symmetric=True,group_size=None)
        local=dict(num_bits=4,type='float',symmetric=True,group_size=16,strategy='tensor_group')
        config['quantization_config']=dict(quant_method='compressed-tensors',version='0.17.2.a20260716',
            config_groups=dict(channel=dict(format='float-quantized',weights=dict(common,strategy='channel',dynamic=False),
                input_activations=dict(common,strategy='token',dynamic=True)),
                local=dict(format='nvfp4-pack-quantized',weights=dict(local,dynamic=False),input_activations=dict(local,dynamic='local'))))
        channel='model.language_model.layers.3.self_attn.k_proj'
        packed='model.language_model.layers.0.mlp.gate_proj'
        original=dict(complete=True,repository='test/ct-mixed',revision='b'*40,roles=[])
        def add(name,dtype,shape):
            count=roles.ROLE_SIZES[dtype]
            for d in shape:count*=d
            original['roles'].append(dict(name=name,dtype=dtype,shape=list(shape),bytes=count,path='original-range.bin',offset=0,sha256='1'*64))
        for name,shape in roles.geometry.role_shapes(roles.geometry.from_hf(config)).items():
            if name==channel+'.weight':
                add(name,'F8_E4M3',shape);add(channel+'.weight_scale','BF16',[shape[0],1])
            elif name==packed+'.weight':
                n,k=shape
                add(packed+'.weight_packed','U8',[n,k//2]);add(packed+'.weight_scale','F8_E4M3',[n,k//16])
                add(packed+'.weight_global_scale','F32',[1]);add(packed+'.input_global_scale','F32',[1])
            else:add(name,'BF16',shape)
        for which in ('k','v'):add(f'model.language_model.layers.3.self_attn.{which}_scale','BF16',[1])
        authenticate=lambda r:(Path(r['path']),None)
        with self.assertRaisesRegex(ValueError,'explicit --kv-cache-dtype'):
            roles._prepare_role_records(original,False,config,authenticate,roles.CT_MIXED_READER)
        lines,receipt=roles._prepare_role_records(original,False,config,authenticate,roles.CT_MIXED_READER,'bf16')
        self.assertEqual(lines[3:5],[roles.CT_MIXED_POLICY_LINE,roles.BF16_CACHE_POLICY_LINE])
        self.assertEqual((receipt['metadata_tensor_count'],receipt['physical_roles']),(341,324))
        self.assertEqual((receipt['ct_channel_token_groups'],receipt['ct_local_nvfp4_groups']),(1,1))
        self.assertEqual(receipt['cache_calibration_omission']['bytes'],4)
        self.assertEqual(len(receipt['cache_calibration_omission']['original_roles']),2)
        self.assertEqual(receipt['ct_mixed_policy']['cache_dtype'],'BF16')
        # Omission affects IO only. Its original hash still fixes source identity.
        changed=deepcopy(original)
        changed['roles'][-1]['sha256']='2'*64
        _,other=roles._prepare_role_records(changed,False,config,authenticate,roles.CT_MIXED_READER,'bf16')
        self.assertNotEqual(receipt['source_id'],other['source_id'])
        for bad in ('missing-global','bad-channel-scale','unknown-cache-role','bad-cache-shape'):
            changed=deepcopy(original)
            if bad=='missing-global':changed['roles']=[r for r in changed['roles']if r['name']!=packed+'.input_global_scale']
            elif bad=='bad-channel-scale':
                r=next(r for r in changed['roles']if r['name']==channel+'.weight_scale');r['dtype']='F32';r['bytes']*=2
            elif bad=='unknown-cache-role':changed['roles'][-1]['name']='model.language_model.layers.3.self_attn.unknown_scale'
            else:changed['roles'][-1].update(shape=[2],bytes=4)
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                roles._prepare_role_records(changed,False,config,authenticate,roles.CT_MIXED_READER,'bf16')

        # Original calibration is metadata, not another device weight operand.
        e4_config=deepcopy(config)
        e4_config['quantization_config']['kv_cache_scheme']=dict(dynamic=False,num_bits=8,
            type='float',strategy='tensor',symmetric=True,group_size=None)
        e4=deepcopy(original)
        known={r['name'] for r in e4['roles']}
        raw_scales={}
        for index,kind in enumerate(roles.geometry.from_hf(e4_config)['layers']):
            if kind!='A':continue
            for side,bits in (('k',0x3d00),('v',0x3e00)):
                name=f'model.language_model.layers.{index}.self_attn.{side}_scale'
                raw=struct.pack('<H',bits);raw_scales[name]=raw
                if name not in known:
                    e4['roles'].append(dict(name=name,dtype='BF16',shape=[1],bytes=2,
                        path='original-calibration.bin',offset=0,sha256=hashlib.sha256(raw).hexdigest()))
                else:next(r for r in e4['roles'] if r['name']==name)['sha256']=hashlib.sha256(raw).hexdigest()
        def e4_auth(role):return Path(role['path']),raw_scales.get(role['name'])
        e4_lines,e4_receipt=roles._prepare_role_records(e4,False,e4_config,e4_auth,roles.CT_MIXED_READER,'e4m3fn')
        self.assertEqual(e4_lines[3:5],[roles.CT_MIXED_POLICY_LINE,roles.FLOAT8_CACHE_POLICY_LINE])
        self.assertEqual(e4_receipt['selected_weight_bytes'],receipt['selected_weight_bytes'])
        self.assertEqual(e4_receipt['physical_roles'],receipt['physical_roles'])
        self.assertEqual(e4_receipt['ct_mixed_policy']['cache_dtype'],'E4M3FN')
        self.assertEqual(e4_receipt['cache_calibration']['bytes'],24)
        self.assertEqual(len(e4_receipt['cache_calibration']['original_roles']),12)
        self.assertNotIn('cache_calibration_omission',e4_receipt)
        _,same_source_bf16=roles._prepare_role_records(e4,False,e4_config,e4_auth,roles.CT_MIXED_READER,'bf16')
        self.assertNotEqual(e4_receipt['source_id'],same_source_bf16['source_id'])
        self.assertEqual({r['f32_bits'] for r in e4_receipt['cache_calibration']['original_roles']},
                         {'3d000000','3e000000'})
        with self.assertRaisesRegex(ValueError,'ordinary inference only'):
            roles._prepare_role_records(e4,True,e4_config,e4_auth,roles.CT_MIXED_READER,'e4m3fn')
        incomplete=deepcopy(e4)
        incomplete['roles']=[r for r in incomplete['roles']if r['name']!='model.language_model.layers.7.self_attn.v_scale']
        with self.assertRaisesRegex(ValueError,'calibration pair'):
            roles._prepare_role_records(incomplete,False,e4_config,e4_auth,roles.CT_MIXED_READER,'e4m3fn')
        wrong_scheme=deepcopy(e4_config);wrong_scheme['quantization_config']['kv_cache_scheme']['dynamic']=True
        with self.assertRaisesRegex(ValueError,'cache scheme'):
            roles._prepare_role_records(e4,False,wrong_scheme,e4_auth,roles.CT_MIXED_READER,'e4m3fn')
        with self.assertRaisesRegex(ValueError,'requires original CT mixed weights'):
            roles.ct_mixed_groups({r['name']:r for r in e4['roles'] if r['name'] in raw_scales},
                e4_config,roles.geometry.from_hf(e4_config),roles.CT_MIXED_READER,'e4m3fn')
        for bits in (0,0x8000,0xbf80,0x7f80,0x7fc0):
            with self.subTest(bits=bits),self.assertRaisesRegex(ValueError,'positive finite'):
                roles.cache_scale_bits(struct.pack('<H',bits),'original K/V fixture')
        bad_scale=deepcopy(e4);bad_name='model.language_model.layers.3.self_attn.k_scale'
        next(r for r in bad_scale['roles']if r['name']==bad_name)['sha256']=hashlib.sha256(b'\0\0').hexdigest()
        with self.assertRaisesRegex(ValueError,'positive finite'):
            roles._prepare_role_records(bad_scale,False,e4_config,
                lambda r:(Path(r['path']),b'\0\0'if r['name']==bad_name else raw_scales.get(r['name'])),
                roles.CT_MIXED_READER,'e4m3fn')
        _,bf16_mtp=roles._prepare_role_records(original,True,config,authenticate,roles.CT_MIXED_READER,'bf16')
        self.assertEqual(bf16_mtp['physical_roles'],339)

    def test_ct_bf16_block128_and_local_reader_follow_source_config(self):
        config=source_config();config['text_config']['dtype']='bfloat16'
        fp=dict(num_bits=8,type='float',symmetric=True,scale_dtype=None)
        local=dict(num_bits=4,type='float',symmetric=True,group_size=16,
                   strategy='tensor_group',scale_dtype='torch.float8_e4m3fn')
        config['quantization_config']=dict(quant_method='compressed-tensors',format='mixed-precision',
            version='0.17.2.dev32+g46196ce',config_groups=dict(
                block=dict(format='float-quantized',targets=[r're:.*linear_attn\.in_proj_a$'],
                    weights=dict(fp,strategy='block',block_structure=[128,128],group_size=None,dynamic=False),
                    input_activations=dict(fp,strategy='group',group_size=128,block_structure=None,dynamic=True)),
                local=dict(format='nvfp4-pack-quantized',targets=[r're:.*mlp\..*'],
                    weights=dict(local,dynamic=False),input_activations=dict(local,dynamic='local'))))
        block='model.language_model.layers.0.linear_attn.in_proj_a'
        packed='model.language_model.layers.0.mlp.gate_proj'
        original=dict(complete=True,repository='fixture/ct-block128',revision='d'*40,roles=[])
        def add(name,dtype,shape):
            count=roles.ROLE_SIZES[dtype]
            for d in shape:count*=d
            original['roles'].append(dict(name=name,dtype=dtype,shape=list(shape),bytes=count,
                path='original-range.bin',offset=0,sha256='4'*64))
        for name,shape in roles.geometry.role_shapes(roles.geometry.from_hf(config)).items():
            if name==block+'.weight':
                n,k=shape;add(name,'F8_E4M3',shape);add(block+'.weight_scale','BF16',[(n-1)//128+1,k//128])
            elif name==packed+'.weight':
                n,k=shape;add(packed+'.weight_packed','U8',[n,k//2]);add(packed+'.weight_scale','F8_E4M3',[n,k//16])
                add(packed+'.weight_global_scale','F32',[1]);add(packed+'.input_global_scale','F32',[1])
            else:add(name,'BF16',shape)
        authenticate=lambda r:(Path(r['path']),None)
        lines,receipt=roles._prepare_role_records(original,False,config,authenticate)
        self.assertEqual(lines[3],roles.CT_BLOCK128_POLICY_LINE)
        self.assertEqual(receipt['ct_mixed_policy']['local_nvfp4_input_rounding'],'BF16')
        self.assertEqual((receipt['ct_block128_groups'],receipt['ct_channel_token_groups'],receipt['ct_local_nvfp4_groups']),(1,0,1))
        self.assertNotIn('fp8_block128_policy',receipt)
        for change in ('version','target','activation-group','f32-sf'):
            bad=deepcopy(config);wrong=deepcopy(original)
            if change=='version':bad['quantization_config']['version']='0.17.2.a20260716'
            elif change=='target':bad['quantization_config']['config_groups']['block']['targets']=['unrelated.projection']
            elif change=='activation-group':bad['quantization_config']['config_groups']['block']['input_activations']['group_size']=64
            else:
                sf=next(r for r in wrong['roles']if r['name']==block+'.weight_scale');sf['dtype']='F32';sf['bytes']*=2
            with self.subTest(change=change),self.assertRaises(ValueError):
                roles._prepare_role_records(wrong,False,bad,authenticate)
        with self.assertRaisesRegex(ValueError,'ordinary inference only'):
            roles._prepare_role_records(original,True,config,authenticate)
        # The same 1x1 SF extent does not choose between channel and block math.
        principal=dict(name=block+'.weight',dtype='F8_E4M3',shape=[1,128])
        sf=dict(name=block+'.weight_scale',dtype='BF16',shape=[1,1])
        pair={r['name']:r for r in (principal,sf)}
        self.assertIn(block,roles.ct_mixed_groups(pair,config,dict(layers=''),None,None)['block128'])
        with self.assertRaises(ValueError):
            roles.ct_mixed_groups(pair,{},dict(layers=''),None,None)

    def test_geometry_comes_from_axes_and_layers_not_model_name(self):
        config=source_config();c=config['text_config']
        c.update(hidden_size=5120,intermediate_size=17408,num_attention_heads=24,
                 num_key_value_heads=4,linear_num_value_heads=48,num_hidden_layers=64,
                 tie_word_embeddings=False,
                 layer_types=['full_attention' if i%4==3 else 'linear_attention' for i in range(64)])
        g=roles.geometry.from_hf(config);shapes=roles.geometry.role_shapes(g)
        self.assertEqual((len(shapes),g['layers'].count('G'),g['layers'].count('A')),(866,48,16))
        self.assertEqual(shapes['lm_head.weight'],(248320,5120))
        self.assertEqual(shapes['model.language_model.layers.0.linear_attn.conv1d.weight'],(10240,1,4))
        self.assertEqual(shapes['mtp.layers.0.self_attn.q_proj.weight'],(12288,5120))
        self.assertEqual(shapes['mtp.layers.0.self_attn.o_proj.weight'],(5120,6144))
        self.assertTrue(roles.geometry.line(g).endswith('grouped\t"'+g['layers']+'"'))

    def test_unsafe_or_ambiguous_source_mathematics_is_rejected(self):
        for field,value in [('hidden_size',True),('mtp_use_dedicated_embeddings',True),
                            ('rms_norm_eps',float('nan')),('linear_num_value_heads',17),
                            ('output_gate_type','sigmoid')]:
            config=source_config();config['text_config'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                roles.geometry.from_hf(config)
        config=source_config();config['text_config']['rope_parameters']['mrope_section']=[1]
        with self.assertRaisesRegex(ValueError,'sections'):
            roles.geometry.from_hf(config)
        original=source_fixture();by_name={r['name']:r for r in original['roles']}
        by_name['model.language_model.layers.0.linear_attn.in_proj_qkv.weight']['shape'][0]+=1
        with self.assertRaisesRegex(ValueError,'source geometry'):
            roles.validate_role_geometry(by_name,roles.geometry.from_hf(source_config()))

    def test_local_shards_emit_one_v2_manifest_for_ordinary_and_mtp_without_visual_reads(self):
        config=source_config();c=config['text_config']
        c.update(hidden_size=8,intermediate_size=16,vocab_size=32,num_attention_heads=2,
            num_key_value_heads=1,head_dim=4,linear_num_key_heads=1,linear_num_value_heads=2,
            linear_key_head_dim=2,linear_value_head_dim=2,linear_conv_kernel_dim=2,
            max_position_embeddings=32,num_hidden_layers=2,
            layer_types=['linear_attention','full_attention'])
        c['rope_parameters'].update(partial_rotary_factor=.5,mrope_section=[1])
        shapes=roles.geometry.role_shapes(roles.geometry.from_hf(config))
        header,body={},bytearray()
        for name,shape in shapes.items():
            count=4
            for d in shape:count*=d
            first=len(body);body.extend(b'\0'*count)
            header[name]=dict(dtype='F32',shape=list(shape),data_offsets=[first,len(body)])
        start=len(body);body.extend(b'VISUAL_PAYLOAD_NOT_READ_BY_ROLE_PREPARER')
        header['model.visual.fixture']=dict(dtype='U8',shape=[len(body)-start],data_offsets=[start,len(body)])
        encoded=json.dumps(header,separators=(',',':')).encode()
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);(root/'model.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+body)
            with patch.object(roles,'_read_role_source',wraps=roles._read_role_source)as reads:
                ordinary,receipt=roles.prepare_safetensors(root,False,config)
                self.assertEqual(reads.call_count,len(shapes))
                self.assertTrue(all(not call.args[0]['name'].startswith('model.visual.')for call in reads.call_args_list))
            mtp,_=roles.prepare_safetensors(root,True,config)
            self.assertEqual(ordinary,mtp)
            self.assertEqual(ordinary[0],'HYLE_QWEN_ROLE_FILES_V2')
            self.assertEqual(receipt['metadata_tensor_count'],len(shapes))
            self.assertEqual(len(list(root.iterdir())),1)

    def test_ct_geometry_uses_original_axes_scale_type_and_groups(self):
        for dtype in ("F16", "BF16", "F32"):
            group=ct_group("mtp.test",8,128,64,dtype)
            self.assertEqual(roles.ct_packed_groups(group)["mtp.test"],
                             dict(logical_shape=[8,128],group_size=64,scale_dtype=dtype))

    def test_actual_mtp_logical_geometry_has_seven_ct_and_eight_plain_roles(self):
        original=source_fixture();all_roles={r["name"]:r for r in original["roles"]}
        mtp={n:r for n,r in all_roles.items()if n.startswith("mtp.")}
        self.assertEqual((len(mtp),len(roles.logical_roles(mtp)),len(roles.ct_packed_groups(mtp))),(29,15,7))
        self.assertEqual(roles.logical_roles(mtp),{n for n in roles.geometry.role_shapes(roles.geometry.from_hf(source_config())) if n.startswith("mtp.")})

    def test_incomplete_ambiguous_or_wrong_ct_geometry_is_rejected(self):
        for suffix in (".weight_packed", ".weight_scale", ".weight_shape"):
            group=ct_group("x");del group["x"+suffix]
            with self.assertRaisesRegex(ValueError,"incomplete"):
                roles.ct_packed_groups(group)
        group=ct_group("x");group["x.weight_packed"]["shape"]=[4,8]
        with self.assertRaisesRegex(ValueError,"group geometry"):
            roles.ct_packed_groups(group)
        for field,value in (("dtype","I32"),("shape",[1,2])):
            group=ct_group("x");group["x.weight_shape"][field]=value
            with self.assertRaisesRegex(ValueError,"shape descriptor"):
                roles.ct_packed_groups(group)
        group=ct_group("x");group["x.weight"]={}
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            roles.ct_packed_groups(group)

    def test_actual_shape_values_are_required_in_original_little_endian_order(self):
        geometry=dict(logical_shape=[8,32])
        roles.validate_ct_shape_values(struct.pack("<qq",8,32),geometry,"x")
        for raw in (struct.pack("<qq",32,8),struct.pack("<qq",8,-1),b"0"*15):
            with self.assertRaisesRegex(ValueError,"weight_shape values differ"):
                roles.validate_ct_shape_values(raw,geometry,"x")

    def test_original_shape_range_is_hashed_without_copying_the_weight_bank(self):
        raw=struct.pack("<qq",8,32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"borrowed source.bin").write_bytes(b"prefix"+raw+b"suffix")
            role=ct_group("x")["x.weight_shape"]
            role.update(offset=6,sha256=hashlib.sha256(raw).hexdigest())
            with patch.object(roles,"ROOT",root):
                source,values=roles.verify_role_source(role)
                self.assertEqual((source,values),(root/"borrowed source.bin",raw))
                role["sha256"]="0"*64
                with self.assertRaisesRegex(ValueError,"SHA differs"):
                    roles.verify_role_source(role)

    def test_original_cache_scalar_range_retains_exact_bf16_bytes(self):
        raw=struct.pack('<H',0x3ce1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scales.bf16').write_bytes(b'xx'+raw+b'yy')
            role=dict(name='model.language_model.layers.3.self_attn.k_scale',dtype='BF16',
                shape=[1],bytes=2,path='scales.bf16',offset=2,sha256=hashlib.sha256(raw).hexdigest())
            with patch.object(roles,'ROOT',root):
                source,values=roles.verify_role_source(role)
                self.assertEqual((source,values),(root/'scales.bf16',raw))
                self.assertEqual(roles.cache_scale_bits(values,role['name']),
                    dict(bf16_bits='3ce1',f32_bits='3ce10000'))
                bad=dict(role,dtype='F32')
                with self.assertRaisesRegex(ValueError,r'BF16\[1\]'):
                    roles.verify_role_source(bad)
                bad=dict(role,sha256='0'*64)
                with self.assertRaisesRegex(ValueError,'SHA differs'):
                    roles.verify_role_source(bad)

    def test_full_source_shape_checks_precede_mtp_selection(self):
        original=source_fixture();groups=roles.ct_packed_groups({r["name"]:r for r in original["roles"]})
        seen=[]
        def authenticates(role):
            seen.append(role["name"])
            raw=struct.pack("<qq",*groups[role["name"][:-13]]["logical_shape"]) if role["name"].endswith(".weight_shape") else None
            return Path("borrowed source.bin"),raw
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"source.json";path.write_text(json.dumps(original))
            with patch.object(roles,"verify_role_source",side_effect=authenticates):
                lines,receipt=roles.prepare_role_files(path,False,source_config())
            self.assertEqual((len(lines),receipt["physical_roles"]),(354,322))
            self.assertTrue(any(line.startswith("mtp.")for line in lines))
            self.assertEqual(lines[0],"HYLE_QWEN_ROLE_FILES_V2")
            self.assertTrue(lines[2].startswith("geometry\t2048\t6144\t248320\t"))
            self.assertEqual(sum(n.endswith(".weight_shape")for n in seen),8)
            self.assertTrue(any(n.startswith("mtp.")for n in seen))
            self.assertTrue(any("\tI64\t1\t2\t"in line for line in lines))
            def corrupts(role):
                source,raw=authenticates(role)
                return source,struct.pack("<qq",1,1) if role["name"].startswith("mtp.") and raw is not None else raw
            with patch.object(roles,"verify_role_source",side_effect=corrupts):
                with self.assertRaisesRegex(ValueError,"weight_shape values differ"):
                    roles.prepare_role_files(path,False,source_config())
            invalid=deepcopy(original);invalid["roles"].append(deepcopy(invalid["roles"][-1]))
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError,"duplicate"):
                roles.prepare_role_files(path,False,source_config())

    def test_fp8_block128_requires_complete_physical_groups_and_explicit_reader(self):
        config=source_config()
        config['quantization_config']=dict(quant_method='fp8',fmt='e4m3',activation_scheme='dynamic',weight_block_size=[128,128])
        original=source_fixture()
        # Dense fixture with one main and one NextN group; no source payload IO.
        shapes=roles.geometry.role_shapes(roles.geometry.from_hf(config))
        original['roles']=[]
        selected={'model.language_model.layers.0.mlp.gate_proj.weight','mtp.layers.0.self_attn.k_proj.weight'}
        for name,shape in shapes.items():
            dtype='F8_E4M3' if name in selected else 'BF16'
            count=roles.ROLE_SIZES[dtype]
            for d in shape:count*=d
            original['roles'].append(dict(name=name,dtype=dtype,shape=list(shape),bytes=count,path='x',offset=0,sha256='1'*64))
            if name in selected:
                n,k=shape
                sf=dict(name=name[:-7]+'.weight_scale_inv',dtype='BF16',shape=[(n+127)//128,k//128],bytes=2*((n+127)//128)*(k//128),path='y',offset=0,sha256='2'*64)
                original['roles'].append(sf)
        authenticate=lambda r:(Path(r['path']),None)
        with self.assertRaisesRegex(ValueError,'explicit --fp8-reader'):
            roles._prepare_role_records(original,False,config,authenticate)
        lines,receipt=roles._prepare_role_records(original,False,config,authenticate,roles.FP8_READER)
        self.assertEqual(lines[3],roles.FP8_POLICY_LINE)
        self.assertEqual((receipt['metadata_tensor_count'],receipt['physical_roles'],receipt['fp8_block128_groups']),(337,321,1))
        self.assertEqual(receipt['fp8_block128_policy']['activation_scale'],'divide')
        self.assertEqual(receipt['fp8_block128_policy']['activation_floor_f32_bits'],'2edbe6ff')
        # Full inventory is checked when only text IO will be selected.
        broken=deepcopy(original);broken['roles']=[r for r in broken['roles'] if r['name']!='mtp.layers.0.self_attn.k_proj.weight_scale_inv']
        with self.assertRaisesRegex(ValueError,'incomplete original FP8'):
            roles._prepare_role_records(broken,False,config,authenticate,roles.FP8_READER)
        broken=deepcopy(original)
        next(r for r in broken['roles'] if r['name'].endswith('weight_scale_inv'))['shape'][0]+=1
        with self.assertRaisesRegex(ValueError,'byte extent differs'):
            roles._prepare_role_records(broken,False,config,authenticate,roles.FP8_READER)
        broken=deepcopy(config);broken['quantization_config']['activation_scheme']='static'
        with self.assertRaisesRegex(ValueError,'config differs'):
            roles._prepare_role_records(original,False,broken,authenticate,roles.FP8_READER)
        # Choosing a reader changes the authenticated source/math identity; it
        # is not silently added to non-FP8 records.
        ordinary=source_fixture()
        old_lines,old_receipt=roles._prepare_role_records(ordinary,False,source_config(),lambda r:(Path('x'),struct.pack('<qq',*roles.ct_packed_groups({x['name']:x for x in ordinary['roles']})[r['name'][:-13]]['logical_shape']) if r['name'].endswith('.weight_shape') else None))
        self.assertNotIn('fp8_block128_policy',old_receipt)
        self.assertFalse(old_lines[3].startswith('fp8_block128'))

    def test_ct_auxiliary_roles_do_not_masquerade_as_parameters(self):
        group=ct_group("x")
        self.assertEqual(roles.logical_roles(group),{"x.weight"})
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            roles.logical_roles([*group,"x.weight"])


if __name__=="__main__":
    unittest.main()
