from pathlib import Path

source = Path('.scratch/d5-kimi-owner-windows.py').read_text(encoding='utf-8')
source = source.replace('d5-kimi-owner', 'd5-glm-owner')
exec(compile(source, str(Path(__file__)), 'exec'))
