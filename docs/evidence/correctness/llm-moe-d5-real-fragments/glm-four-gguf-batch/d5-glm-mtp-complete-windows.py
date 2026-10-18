from pathlib import Path
import json
import os
import subprocess
import sys

os.environ['VSLANG'] = '1033'
os.environ['HYLE_TEST_VULKAN_VALIDATE'] = '1'
os.environ['VK_LAYER_PATH'] = 'C:/public/tools/hyle-vulkan/1.4.357.0/Bin'
route = sys.argv[1]
folder = '.scratch/validation/d5-glm-mtp-qualified'
source = Path('.scratch/d5-moe-semantic-windows.py').read_text(encoding='utf-8')
source = source.replace("'.scratch/validation/d5-moe-semantic'", repr(folder))
filter_ = '*MoeFragments.D5Glm*:*MoeFragments.D5SixProfilesBoundedKdaActualStateFeedback:*MoeFragments.D5KimiAbsorbedMlaWithCache:D5ExactQwenGguf/*HeadRunsEntireLogicalVocabulary/1:D5ExactQwenGguf/*HeadRunsEntireLogicalVocabulary/3'
source = source.replace('*MoeFragments.*:D5ExactMoeVocabulary/*', filter_)
source = source.replace('output[-8000:]', 'output[-3000:]')
out = Path(folder) / route
out.mkdir(parents=True, exist_ok=True)
(out / 'candidate.json').write_text(json.dumps({'candidate': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip()}, indent=2) + '\n', encoding='utf-8')
exec(compile(source, str(Path(__file__)), 'exec'))
