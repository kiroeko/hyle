from pathlib import Path
import os
import sys

route = sys.argv[1]
os.environ['VSLANG'] = '1033'
os.environ['HYLE_TEST_VULKAN_VALIDATE'] = '1'
os.environ['VK_LAYER_PATH'] = 'C:/public/tools/hyle-vulkan/1.4.357.0/Bin'
source = Path('.scratch/d5-moe-semantic-windows.py').read_text(encoding='utf-8')
source = source.replace("'.scratch/validation/d5-moe-semantic'", "'.scratch/validation/d5-glm-corrected'")
source = source.replace('--gtest_filter=*MoeFragments.*:D5ExactMoeVocabulary/*',
                        '--gtest_filter=*MoeFragments.D5Glm*')
source = source.replace('hyle_graph_route_{route}_tests -j 2',
    'hyle_graph_route_{route}_tests hyle_compute_provider_ggml_{route}_tests -j 2')
source = source.replace('raise SystemExit(1 if failed else 0)', '''
exe = tree / 'bin' / f'hyle_compute_provider_ggml_{route}_test.exe'
filter = ('GgmlCudaTensor.*Concat*:GgmlCudaTensor.UnsupportedNumericDomains*:GgmlCudaAttention.*:GgmlCuda.ColdAllocations*'
          if route == 'cuda' else 'GgmlVulkanRegion.*:GgmlVulkanStorage.*:GgmlVulkanMetadata.*')
with (out / 'provider.log').open('w', encoding='utf-8') as log:
    result = subprocess.run([str(exe), '--gtest_filter=' + filter], cwd=exe.parent,
        stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
output = (out / 'provider.log').read_text(encoding='utf-8', errors='replace')
print('provider', result.returncode, output[-1200:], flush=True)
failed = failed or result.returncode != 0 or '[  SKIPPED ]' in output
raise SystemExit(1 if failed else 0)
''')
exec(compile(source, str(Path(__file__)), 'exec'))
