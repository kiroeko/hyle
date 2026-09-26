from pathlib import Path
import os
os.environ['VSLANG'] = '1033'
os.environ['HYLE_D5_QWEN35_FIXTURE'] = str(Path('.scratch/d5-qwen35-six-profile-fixture').resolve())
source = Path('.scratch/d5-moe-semantic-windows.py').read_text(encoding='utf-8')
source = source.replace("'.scratch/validation/d5-moe-semantic'", "'.scratch/validation/d5-kda-full'")
source = source.replace('--gtest_filter=*MoeFragments.*:D5ExactMoeVocabulary/*',
                        '--gtest_filter=*MoeFragments.D5SixProfiles*Kda*:*QwenRecurrent.D5NineProfilesUseCompleteGateAndActualStateFeedback')
exec(compile(source, str(Path(__file__)), 'exec'))
