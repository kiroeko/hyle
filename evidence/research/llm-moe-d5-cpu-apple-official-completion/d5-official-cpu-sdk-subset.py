import hashlib
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path

root = Path('.scratch/validation/d5-cpu-apple-official-sources')
index = json.loads((root / 'libtorch-cpu-zip-index.json').read_text())
out = root / 'cpu-sdk-subset/lib'
out.mkdir(parents=True, exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:7897'}))
names = {'torch_cpu.dll', 'torch_cpu.lib', 'c10.dll', 'c10.lib',
         'fbgemm.dll', 'asmjit.dll', 'libiomp5md.dll'}
installed = Path('C:/Program Files/Python312/Lib/site-packages/torch/lib')
records = []
for entry in index['entries']:
    name = entry['name'].rsplit('/', 1)[-1]
    if name not in names or not entry['name'].startswith('libtorch/lib/'): continue
    target = out / name
    local = installed / name
    if local.exists() and local.stat().st_size == entry['bytes'] and zlib.crc32(local.read_bytes()) == entry['crc32']:
        body = local.read_bytes()
        transport = 'reused exact archive-entry bytes from installed 2.8.0'
    else:
        begin = entry['offset']
        request = urllib.request.Request(index['url'], headers={'User-Agent': 'Mozilla/5.0',
            'Range': f'bytes={begin}-{begin + 30 + len(entry["name"].encode()) + entry["compressed"] + 65535 - 1}'})
        with opener.open(request, timeout=30) as response:
            assert response.status == 206
            prefix = response.read(30)
            header = struct.unpack('<4s5H3L2H', prefix)
            assert header[0] == b'PK\x03\x04'
            response.read(header[-2] + header[-1])
            stream = zlib.decompressobj(-15)
            chunks = []
            remaining = entry['compressed']
            started = time.monotonic()
            while remaining:
                data = response.read(min(1048576, remaining))
                assert data
                remaining -= len(data)
                chunks.append(stream.decompress(data))
                if time.monotonic() - started > 20:
                    print(name, 'remaining compressed bytes', remaining, flush=True)
                    started = time.monotonic()
            chunks.append(stream.flush())
            body = b''.join(chunks)
        transport = 'official CPU ZIP compressed-entry range; CRC32 checked'
    assert len(body) == entry['bytes'] and zlib.crc32(body) == entry['crc32']
    target.write_bytes(body)
    records.append(dict(name=name,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),transport=transport))
    print(name,len(body),flush=True)
assert {r['name'] for r in records} == names
(out.parent / 'manifest.json').write_text(json.dumps(dict(url=index['url'],archive_bytes=index['bytes'],
    scope='Qualification CPU runtime subset; whole archive SHA256 not claimed; C++ headers from same 2.8.0 installed SDK',files=records),indent=2)+'\n')
