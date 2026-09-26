from pathlib import Path

source = Path('.scratch/d5-glm-mtp-complete-windows.py').read_text(encoding='utf-8')
source = source.replace('d5-glm-mtp-qualified', 'd5-glm-expert-variants')
lines=source.splitlines()
lines=["filter_ = '*MoeFragments.D5GlmLayerSpecificExpertFormats'" if line.startswith('filter_ = ') else line for line in lines]
exec(compile('\n'.join(lines)+'\n', str(Path(__file__)), 'exec'))
