from pathlib import Path
source=Path('.scratch/d5-moe-semantic-apple.py').read_text(encoding='utf-8')
source=source.replace('.scratch/validation/d5-moe-semantic/apple','.scratch/validation/d5-moe-execution/balanced3/apple')
source=source.replace('877c9f5b..HEAD','40699b3d7..HEAD')
source=source.replace('d5-moe-semantic.bundle','d5-moe-execution.bundle')
source=source.replace('*MoeFragments.*:D5ExactMoeVocabulary/*','*MoeFragments.D5MatureClamp*:*MoeFragments.D5Thirteen*:*MoeFragments.D5Eight*:*MoeFragments.D5KimiThreeProfilesSituActivation')
exec(compile(source,str(Path(__file__)),'exec'))
