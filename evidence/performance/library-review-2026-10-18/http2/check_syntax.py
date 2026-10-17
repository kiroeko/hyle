import hashlib
import json
import pathlib
import re
import subprocess

root = pathlib.Path('/mnt/c/public/project-hyle/hyle')
out = root / '.scratch/perf-review-2026-10-18/http2-syntax'
source = pathlib.Path('/root/.cache/hyle/dependencies/v1/sources/nghttp2-85e300c79fb6-ca7aa79a25b3')
version = re.search(r'project\(nghttp2 VERSION ([0-9.]+)', (source / 'CMakeLists.txt').read_text())[1]
parts = list(map(int, version.split('.')))
template = source / 'lib/includes/nghttp2/nghttp2ver.h.in'
generated = out / 'include/nghttp2/nghttp2ver.h'
generated.parent.mkdir(parents=True, exist_ok=True)
generated.write_text(template.read_text().replace('@PACKAGE_VERSION@', version).replace(
    '@PACKAGE_VERSION_NUM@', hex((parts[0] << 16) | (parts[1] << 8) | parts[2])))
command = ['/usr/bin/g++-14', '-std=c++23', '-DHYLE_IO_HTTP2=1', '-DNDEBUG',
           '-I' + str(root / 'include'), '-I/root/hyle/third_party/nanobench',
           '-I' + str(out / 'include'), '-I' + str(source / 'lib/includes'),
           '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-fsyntax-only',
           str(root / 'bench/io_http2/http2_p1_bench.cc')]
result = subprocess.run(command, capture_output=True, text=True)
(out / 'stdout.txt').write_text(result.stdout)
(out / 'stderr.txt').write_text(result.stderr)
files = [root / 'bench/io_http2/http2_p1_bench.cc', source / 'lib/includes/nghttp2/nghttp2.h',
         template, generated, pathlib.Path('/root/hyle/third_party/nanobench/nanobench.h')]
record = {'command': command, 'exit_code': result.returncode,
          'compiler': subprocess.check_output(['/usr/bin/g++-14', '--version'], text=True),
          'nghttp2_cached_version': version,
          'scope': 'Current Windows product source and public headers through /mnt/c. Syntax-only Linux compile; no linking, benchmark execution, rebuild or SDK download.',
          'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'exit_code': result.returncode, 'stderr': result.stderr[:6000]}, indent=2))
raise SystemExit(result.returncode)
