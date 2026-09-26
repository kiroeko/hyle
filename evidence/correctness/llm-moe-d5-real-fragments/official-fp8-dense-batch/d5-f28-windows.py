from pathlib import Path
import os
os.environ['VSLANG'] = '1033'
source = Path('.scratch/d5-moe-semantic-windows.py').read_text(encoding='utf-8')
source = source.replace("'.scratch/validation/d5-moe-semantic'", "'.scratch/validation/d5-f27/correction'")
source = source.replace('--gtest_filter=*MoeFragments.*:D5ExactMoeVocabulary/*',
                        '--gtest_filter=*MoeFragments.D5OfficialFp8*:VulkanIree.*:VulkanGraphIree.*')
exec(compile(source, str(Path(__file__)), 'exec'))
