from pathlib import Path
import json
import re
import subprocess

root = Path('C:/hv/d5cd')
repo = Path.cwd().resolve()
summary = []
for stage in ['source', 'static', 'shared']:
    build = root / ('source-build' if stage == 'source' else stage+'-consumer')
    out = root/'receipts'/(stage+'-provider-runtime')
    cache = (build/'CMakeCache.txt').read_text()
    ninja = re.search(r'^CMAKE_MAKE_PROGRAM:FILEPATH=(.+)$', cache, re.M).group(1)
    result = subprocess.run([ninja, '-C', str(build), '-t', 'commands', 'd5_cuda_delivery'],
                            capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    (out/'compile-link-commands.log').write_bytes(result.stdout)
    text = result.stdout.decode(errors='replace').replace('\\', '/').lower()
    canonical_source_references = str(repo).replace('\\', '/').lower() in text
    if stage != 'source':
        assert not canonical_source_references, stage+' leaks canonical product paths'
        assert str(root/(stage+'-relocated')).replace('\\', '/').lower() in text
        assert str(root/'source-build').replace('\\', '/').lower() not in text
    summary.append({'stage': stage, 'canonical_product_path_in_commands': canonical_source_references,
                    'relocated_prefix_required': stage != 'source'})
(root/'receipts/link-provenance.json').write_text(json.dumps(summary, indent=2)+'\n')
print('source/static/shared compile and link provenance captured; installed consumers use only relocated SDKs')
