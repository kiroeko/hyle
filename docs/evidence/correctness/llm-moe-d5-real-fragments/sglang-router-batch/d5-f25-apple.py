from pathlib import Path

source = Path('.scratch/d5-moe-semantic-apple.py').read_text(encoding='utf-8')
source = source.replace('.scratch/validation/d5-moe-semantic/apple',
                        '.scratch/validation/d5-f25/apple')
source = source.replace('877c9f5b..HEAD', 'a11aac6a..HEAD')
source = source.replace('d5-moe-semantic.bundle', 'd5-f25.bundle')
source = source.replace('*MoeFragments.*:D5ExactMoeVocabulary/*',
                        '*MoeFragments.D5ThirteenProfilesModelRouters:*MoeFragments.D5EightGgufProfilesFullMoe')
exec(compile(source, str(Path(__file__)), 'exec'))
