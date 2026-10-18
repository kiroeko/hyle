from pathlib import Path
import os
os.environ['GTEST_FILTER']='*MoeFragments.D5GlmOfficialMixedRemaining:*MoeFragments.D5GlmFp8AccumulationBoundaries'
source=Path('.scratch/d5-kimi-owner-windows.py').read_text(encoding='utf-8')
source=source.replace('d5-kimi-owner','d5-f40-declared-accumulation')
source=source.replace("['compute_provider_ggml_cuda', 'graph_provider_ggml_cuda', 'compute_route_cuda', 'graph_route_cuda']", "['compute_route_cuda', 'graph_route_cuda']")
source=source.replace("'steps': []}","'steps': [], 'test_filter': os.environ['GTEST_FILTER']}")
exec(compile(source,str(Path(__file__)),'exec'))
