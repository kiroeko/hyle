from pathlib import Path
source = Path('.scratch/d5-moe-semantic-apple.py').read_text(encoding='utf-8')
source = source.replace('.scratch/validation/d5-moe-semantic/apple', '.scratch/validation/d5-f27/apple')
source = source.replace('877c9f5b..HEAD', '4c862164..HEAD')
source = source.replace('d5-moe-semantic.bundle', 'd5-f27.bundle')
source = source.replace('*MoeFragments.*:D5ExactMoeVocabulary/*', '*MoeFragments.D5OfficialFp8*')
exec(compile(source, str(Path(__file__)), 'exec'))
