from pathlib import Path
import json
import os
import subprocess
os.environ['VSLANG'] = '1033'
source = Path('.scratch/d5-moe-semantic-windows.py').read_text(encoding='utf-8')
source = source.replace("'.scratch/validation/d5-moe-semantic'", "'.scratch/validation/d5-glm-qualified'")
source = source.replace('--gtest_filter=*MoeFragments.*:D5ExactMoeVocabulary/*', '--gtest_filter=*MoeFragments.D5Glm*')
candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
out = Path('.scratch/validation/d5-glm-qualified/cpu')
out.mkdir(parents=True, exist_ok=True)
(out/'candidate.json').write_text(json.dumps({'candidate': candidate}, indent=2)+'\n', encoding='utf-8')
exec(compile(source, str(Path(__file__)), 'exec'))
