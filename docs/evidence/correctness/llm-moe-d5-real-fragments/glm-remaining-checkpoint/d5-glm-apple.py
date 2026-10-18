from pathlib import Path

source = Path('.scratch/d5-moe-semantic-apple.py').read_text(encoding='utf-8')
source = source.replace('.scratch/validation/d5-moe-semantic/apple', '.scratch/validation/d5-glm-checkpoint/apple')
source = source.replace('877c9f5b..HEAD', '9d8d90e4..HEAD')
source = source.replace('d5-moe-semantic.bundle', 'd5-glm-checkpoint.bundle')
source = source.replace('*MoeFragments.*:D5ExactMoeVocabulary/*', '*MoeFragments.D5Glm*:*MoeFragments.D5Kimi*')
exec(compile(source, str(Path(__file__)), 'exec'))
