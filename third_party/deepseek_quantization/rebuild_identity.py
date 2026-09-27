"""Compare pinned CUDA ELF bytes, excluding only NVCC internal symbol entropy."""
import hashlib
import re
import struct


def identity(blob):
    assert blob[:6] == b'\x7fELF\x02\x01'
    offset, = struct.unpack_from('<Q', blob, 40)
    size, count, names = struct.unpack_from('<HHH', blob, 58)
    assert size == 64 and offset + count * size <= len(blob) and names < count
    records = [struct.unpack_from('<IIQQQQIIQQ', blob, offset + i * size) for i in range(count)]
    names_record = records[names]
    names_blob = blob[names_record[4]:names_record[4] + names_record[5]]
    result = bytearray(blob)
    changed = 0
    for section in records:
        name = names_blob[section[0]:].split(b'\0', 1)[0]
        if name != b'.strtab':
            continue
        assert section[1] == 3 and section[4] + section[5] <= len(blob)
        start, end = section[4], section[4] + section[5]
        result[start:end], changed = re.subn(
            rb'(_INTERNAL_|_GLOBAL__N__)[0-9a-f]{8}(?=_16_fp4_[0-9]{4}_[0-9]{4}_cu_)',
            lambda match: match[1] + b'00000000', blob[start:end])
    return bytes(result), changed


def compare(blob, job, embedded):
    pinned = re.search(r'\b' + re.escape(job['symbol']) + r'\[\] = \{(.*?)\};', embedded, re.S)
    assert pinned is not None
    expected = bytes(int(value, 16) for value in re.findall(r'0x([0-9a-f]{2})', pinned[1]))
    assert hashlib.sha256(expected).hexdigest() == job['sha256']
    exact = blob == expected
    a, na = identity(blob)
    b, nb = identity(expected)
    same = exact or (job.get('indexed') is True and na == nb == 11 and a == b)
    return {'matches_pinned': exact, 'matches_program': same,
            'internal_symbol_hashes_ignored': na if not exact and same else 0,
            'identity_sha256': hashlib.sha256(a).hexdigest()}
