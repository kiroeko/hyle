from pathlib import Path
source = Path('.scratch/d5-moe-semantic-apple.py').read_text(encoding='utf-8')
source = source.replace('.scratch/validation/d5-moe-semantic/apple', '.scratch/validation/d5-kda-full/apple')
source = source.replace('877c9f5b..HEAD', '3f8caf58..HEAD')
source = source.replace('d5-moe-semantic.bundle', 'd5-kda-full.bundle')
source = source.replace("command += f'cmake", "command += 'export HYLE_D5_QWEN35_FIXTURE=\"$HOME/hyle-scratch/d5-qwen35-six-profile-fixture\"; '\n    command += f'cmake")
source = source.replace('*MoeFragments.*:D5ExactMoeVocabulary/*',
                        '*MoeFragments.D5SixProfiles*Kda*:*QwenRecurrent.D5NineProfilesUseCompleteGateAndActualStateFeedback')
exec(compile(source, str(Path(__file__)), 'exec'))
