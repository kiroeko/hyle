from pathlib import Path
import os
import sys

os.environ['VSLANG'] = '1033'
os.environ['HYLE_TEST_VULKAN_VALIDATE'] = '1'
os.environ['VK_LAYER_PATH'] = 'C:/public/tools/hyle-vulkan/1.4.357.0/Bin'
source = Path('.scratch/d5-moe-semantic-windows.py').read_text(encoding='utf-8')
source = source.replace("'.scratch/validation/d5-moe-semantic'", "'.scratch/validation/d5-glm-checkpoint'")
source = source.replace('--gtest_filter=*MoeFragments.*:D5ExactMoeVocabulary/*',
                        '--gtest_filter=*MoeFragments.D5Glm*:*MoeFragments.D5Kimi*:*MoeFragments.D5Hyper*')
exec(compile(source, str(Path(__file__)), 'exec'))
