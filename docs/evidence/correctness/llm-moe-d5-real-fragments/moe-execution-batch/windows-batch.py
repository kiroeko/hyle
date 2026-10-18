from pathlib import Path
source=Path('.scratch/d5-moe-semantic-windows.py').read_text(encoding='utf-8')
source=source.replace("'.scratch/validation/d5-moe-semantic'", "'.scratch/validation/d5-moe-execution/balanced3'")
source=source.replace('--gtest_filter=*MoeFragments.*:D5ExactMoeVocabulary/*','--gtest_filter=*MoeFragments.D5MatureClamp*:*MoeFragments.D5Thirteen*:*MoeFragments.D5Eight*:*MoeFragments.D5KimiThreeProfilesSituActivation')
exec(compile(source,str(Path(__file__)),'exec'))
