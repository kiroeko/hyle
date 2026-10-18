from pathlib import Path
import json
import os
import subprocess
import sys
os.environ['VSLANG']='1033'
os.environ['HYLE_TEST_VULKAN_VALIDATE']='1'
os.environ['VK_LAYER_PATH']='C:/public/tools/hyle-vulkan/1.4.357.0/Bin'
route=sys.argv[1]
source=Path('.scratch/d5-moe-semantic-windows.py').read_text(encoding='utf-8')
source=source.replace("'.scratch/validation/d5-moe-semantic'", "'.scratch/validation/d5-glm-nextn-corrected'")
source=source.replace('--gtest_filter=*MoeFragments.*:D5ExactMoeVocabulary/*',
    '--gtest_filter=*MoeFragments.D5Glm*:*MoeFragments.D5EightGgufProfilesFullMoe:*MoeFragments.*OfficialKimi*')
source=source.replace('output[-8000:]', 'output[-2000:]')
out=Path('.scratch/validation/d5-glm-nextn-corrected')/route
out.mkdir(parents=True,exist_ok=True)
(out/'candidate.json').write_text(json.dumps({'candidate':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},indent=2)+'\n',encoding='utf-8')
exec(compile(source,str(Path(__file__)),'exec'))
