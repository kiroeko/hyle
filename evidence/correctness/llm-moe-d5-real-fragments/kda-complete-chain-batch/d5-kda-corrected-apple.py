from pathlib import Path
source = Path('.scratch/d5-moe-semantic-apple.py').read_text(encoding='utf-8')
source = source.replace('.scratch/validation/d5-moe-semantic/apple', '.scratch/validation/d5-kda-corrected/apple')
source = source.replace('877c9f5b..HEAD', '9bb6be90..HEAD')
source = source.replace('d5-moe-semantic.bundle', 'd5-kda-corrected.bundle')
source = source.replace('*MoeFragments.*:D5ExactMoeVocabulary/*',
                        '*MoeFragments.D5SixProfiles*Kda*:*MoeFragments.D5OfficialKimiMixedStorageFullMoe:*MoeFragments.D5EightGgufProfilesFullMoe')
exec(compile(source, str(Path(__file__)), 'exec'))
