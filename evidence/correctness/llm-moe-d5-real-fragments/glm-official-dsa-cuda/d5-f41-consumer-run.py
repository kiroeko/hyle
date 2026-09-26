from pathlib import Path
import os
os.environ['GTEST_FILTER']='ModelFloat8Decode.*:DeepSeekFp8Decode.*:*MoeFragments.D5Fp8WeightPreparation:*MoeFragments.D5GlmDsaPreparedKvAndNextN:*MoeFragments.D5GlmOfficialMixedRemaining:*MoeFragments.D5GlmGgufMla*:*MoeFragments.D5GlmGgufNextN*'
source=Path('.scratch/d5-kimi-owner-windows.py').read_text(encoding='utf-8')
source=source.replace('d5-kimi-owner','d5-f41-complete-consumer')
source=source.replace("['compute_provider_ggml_cuda', 'graph_provider_ggml_cuda', 'compute_route_cuda', 'graph_route_cuda']", "['compute_ops', 'compute_provider_cuda', 'compute_route_cuda', 'graph_route_cuda']")
source=source.replace("'steps': []}","'steps': [], 'test_filter': os.environ['GTEST_FILTER']}")
exec(compile(source,str(Path(__file__)),'exec'))
