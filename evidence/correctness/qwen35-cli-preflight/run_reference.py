"""Local text generation reference. This is not HYLE execution."""
from pathlib import Path
import argparse, ctypes, hashlib, importlib.metadata, json, os, sys, time

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
if sys.platform=='win32': ctypes.windll.kernel32.SetErrorMode(0x8003)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--prompt',default='Please calculate 2 + 3. Answer briefly.')
parser.add_argument('--max-new-tokens',type=int,default=24)
args=parser.parse_args()
assert not args.output.exists(), 'output must be new'
assert 1 <= args.max_new_tokens <= 128

root=Path.cwd()
sys.path.insert(0,str(root/'tests/compute_models'))
from local_inputs import read_inventory,verify
inventory=read_inventory(root/'docs/implementation/tasks/compute/qwen35-2b-local-input.json')
print('Verify original model input',flush=True)
verification=verify(args.model,inventory)

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as impl
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
assert torch.cuda.is_available(), 'reference CUDA required'
config=Qwen3_5TextConfig(**json.loads((args.model/'config.json').read_bytes())['text_config'])
config._attn_implementation='eager'
start=time.perf_counter()
with torch.device('meta'): model=impl.Qwen3_5ForCausalLM(config)
model.model.rotary_emb=impl.Qwen3_5TextRotaryEmbedding(config,device='cpu')
weights={}; weight_records=[]
for shard in inventory['shards']:
    with safe_open(args.model/shard['path'],framework='pt',device='cpu') as f:
        for name in f.keys():
            if name.startswith('model.language_model.'):
                tensor=f.get_tensor(name)
                weights[name.replace('model.language_model.','model.',1)]=tensor
                weight_records.append({'name':name,'shape':list(tensor.shape),'dtype':str(tensor.dtype),'bytes':tensor.numel()*tensor.element_size()})
weights['lm_head.weight']=weights['model.embed_tokens.weight']
model.load_state_dict(weights,strict=True,assign=True)
model.lm_head.weight=model.model.embed_tokens.weight
del weights
assert not any(p.is_meta for p in model.parameters())
free,total=torch.cuda.mem_get_info()
weight_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
assert free>=weight_bytes+1024**3, 'weight bytes + 1GiB reserve must fit before transfer'
model.eval().to('cuda')
for layer in model.model.layers:
    if hasattr(layer,'linear_attn'):
        a=layer.linear_attn
        a.causal_conv1d_fn=None
        a.causal_conv1d_update=impl.torch_causal_conv1d_update
        a.chunk_gated_delta_rule=impl.torch_chunk_gated_delta_rule
        a.recurrent_gated_delta_rule=impl.torch_recurrent_gated_delta_rule
torch.cuda.synchronize();load_seconds=time.perf_counter()-start
tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
encoded=tokenizer.apply_chat_template([{'role':'user','content':args.prompt}],add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)
ids=encoded['input_ids'].to('cuda')
stops={int(config.eos_token_id),int(tokenizer.convert_tokens_to_ids('<|im_end|>'))}
tokens=[];steps=[];cache=None;current=ids
torch.cuda.reset_peak_memory_stats()
print(f'Text-only reference loaded: {weight_bytes} bytes, prompt tokens={ids.shape[1]}',flush=True)
with torch.inference_mode():
    for step in range(args.max_new_tokens):
        torch.cuda.synchronize();stamp=time.perf_counter()
        output=model(input_ids=current,past_key_values=cache,use_cache=True,logits_to_keep=1)
        cache=output.past_key_values
        logits=output.logits[:,-1]
        assert torch.isfinite(logits).all(), 'non-finite logits'
        token=int(logits.argmax(-1).item())
        torch.cuda.synchronize();elapsed=time.perf_counter()-stamp
        tokens.append(token);steps.append({'step':step,'input_tokens':int(current.shape[1]),'token_id':token,'seconds':elapsed})
        if token in stops:break
        current=torch.tensor([[token]],device='cuda',dtype=torch.long)
generated=tokenizer.decode(tokens,skip_special_tokens=True)
state_shapes={}
for family in ('conv_states','recurrent_states','key_cache','value_cache'):
    value=getattr(cache,family,None)
    if value is not None:
        state_shapes[family]=[{'layer':i,'shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.numel()*x.element_size()} for i,x in enumerate(value) if x is not None]
receipt={'kind':'external_reference_generation_not_hyle_verdict','input_verification':verification,
 'model_branch':'text-only 24 layers; vision/MTP excluded; tied head','prompt':args.prompt,
 'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','safetensors','tokenizers')},
 'model_source_sha256':hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(),
 'reference_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'gpu':torch.cuda.get_device_name(),'device_free_before_load':free,'device_total_bytes':total,
 'weights':weight_records,'text_weight_bytes':weight_bytes,'state_shapes':state_shapes,
 'prompt_ids':ids[0].tolist(),'generated_ids':tokens,'generated_text':generated,
 'load_seconds':load_seconds,'steps':steps,'diagnostic_only_not_formal_engine_benchmark':True,
 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
 'result':'REFERENCE_GENERATED'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'result':receipt['result'],'generated_text':generated,'new_tokens':len(tokens),'load_seconds':load_seconds,'prefill_seconds':steps[0]['seconds'],'decode_seconds':sum(s['seconds'] for s in steps[1:]),'output':str(args.output)},ensure_ascii=True),flush=True)
