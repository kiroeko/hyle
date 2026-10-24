"""GGUF structural fields used by the Qwen input preparer; no payload/HTTP IO.

The exact-range grammar is shared with bounded original-header acquisition.
Only the consumer of these fields chooses a supported model/representation.
"""
from pathlib import Path
import struct

FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
           7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class MetadataReader:
    """Fetch structural bytes and conservative still-unread lower bounds."""
    def __init__(self, exact, url, maximum_header_bytes, chunk_bytes):
        require(0 < chunk_bytes <= maximum_header_bytes, 'invalid structural reader bound')
        self.exact, self.url = exact, url
        self.limit, self.chunk = maximum_header_bytes, chunk_bytes
        self.raw, self.offset, self.total, self.etag = bytearray(), 0, None, None

    def take(self, size, minimum_after=0):
        require(0 <= size <= self.limit and minimum_after >= 0, 'invalid structural field extent')
        end = self.offset + size
        require(end <= self.limit, 'GGUF metadata exceeds its explicit limit')
        if self.total is not None:
            require(end <= self.total, 'GGUF field exceeds authenticated file EOF')
        while len(self.raw) < end:
            first = len(self.raw)
            stop = min(end + minimum_after, first + self.chunk, self.limit)
            require(stop > first, 'invalid structural lower bound')
            if self.total is not None:
                require(stop <= self.total, 'GGUF structural lower bound exceeds authenticated file EOF')
            part, total, etag = self.exact(self.url, first, stop - 1,
                                          total=self.total, etag=self.etag)
            require(len(part) == stop - first and total >= stop, 'short structural source range')
            require(self.total is None or self.total == total, 'GGUF source size changed')
            require(self.etag is None or self.etag == etag, 'GGUF source entity changed')
            self.total, self.etag = total, etag
            self.raw.extend(part)
        result = self.raw[self.offset:end]
        self.offset = end
        return result

    def number(self, fmt, minimum_after=0):
        return struct.unpack('<' + fmt, self.take(struct.calcsize('<' + fmt), minimum_after))[0]

    def string(self, minimum_after=0, keep=True):
        size = self.number('Q', minimum_after)
        raw = self.take(size, minimum_after)
        return raw.decode('utf-8') if keep else None


class CachedReader(MetadataReader):
    def __init__(self, raw, whole_file_bytes):
        require(type(whole_file_bytes) is int and 0 <= len(raw) <= whole_file_bytes,
                'invalid cached-header original size')
        self.raw, self.offset, self.total, self.etag = raw, 0, whole_file_bytes, None
        self.limit, self.chunk = len(raw), 0

    def take(self, size, minimum_after=0):
        require(size >= 0 and minimum_after >= 0, 'invalid cached structural extent')
        end = self.offset + size
        require(end <= len(self.raw), 'GGUF metadata exceeds the exact cached header')
        result = self.raw[self.offset:end]
        self.offset = end
        return result


def _parse(reader, tensor_count, retained_metadata):
    require(type(tensor_count) is int and tensor_count >= 0, 'invalid GGUF tensor count')
    retained_metadata = set() if retained_metadata is None else set(retained_metadata)
    magic, version, count, metadata_count = struct.unpack('<4sIQQ', reader.take(24))
    require(magic == b'GGUF' and version in (2, 3) and count == tensor_count, 'invalid GGUF preamble')
    require(0 < metadata_count <= 1000, 'GGUF metadata count exceeds the structural domain')

    def value(kind, after, keep):
        if kind in FORMATS:
            result = reader.number(FORMATS[kind], after)
            return result if keep else None
        if kind == 8:
            return reader.string(after, keep)
        require(kind == 9, 'unknown GGUF metadata type')
        element = reader.number('I', after + 8)
        length = reader.number('Q', after)
        require(element != 9 and length <= reader.limit, 'invalid GGUF metadata array')
        if element in FORMATS:
            width = struct.calcsize('<' + FORMATS[element])
            raw = reader.take(length * width, after)
            return list(struct.unpack('<' + FORMATS[element] * length, raw)) if keep else None
        require(element == 8, 'unknown GGUF metadata array type')
        result = [] if keep else None
        for index in range(length):
            item = reader.string(after + (length - index - 1) * 8, keep)
            if keep:
                result.append(item)
        return result

    metadata, seen = {}, set()
    for index in range(metadata_count):
        after = (metadata_count - index - 1) * 13 + tensor_count * 32
        key = reader.string(after + 5)
        require(key not in seen, 'duplicate GGUF metadata key')
        seen.add(key)
        kind = reader.number('I', after + 1)
        keep = not key.startswith('tokenizer.') or key in retained_metadata
        result = value(kind, after, keep)
        if keep:
            metadata[key] = result
    tensors = []
    for index in range(tensor_count):
        after = (tensor_count - index - 1) * 32
        name = reader.string(after + 24)
        rank = reader.number('I', after + 20)
        require(0 < rank <= 4, 'unsupported GGUF tensor rank')
        shape = list(struct.unpack('<' + 'Q' * rank, reader.take(rank * 8, after + 12)))
        require(all(n > 0 for n in shape), 'nonpositive GGUF tensor shape')
        code = reader.number('I', after + 8)
        offset = reader.number('Q', after)
        tensors.append(dict(name=name, shape=shape, dtype_code=code, offset=offset))
    alignment = metadata.get('general.alignment', 32)
    require(type(alignment) is int and 0 < alignment <= 4096, 'invalid GGUF data alignment')
    if tensor_count == 0 and reader.offset == reader.total:
        end = reader.offset
    else:
        end = (reader.offset + alignment - 1) // alignment * alignment
        if tensor_count == 0:
            require(end == reader.total, 'zero-tensor GGUF has unparsed trailing bytes')
    reader.take(end - reader.offset)
    require(len(reader.raw) == reader.offset == end, 'GGUF source extends past its exact structural header')
    require(len({row['name'] for row in tensors}) == tensor_count, 'duplicate GGUF tensor name')
    return metadata, tensors, metadata_count


def scan(exact, *, url, tensor_count, maximum_header_bytes, chunk_bytes=256 * 1024,
         retained_metadata=None):
    reader = MetadataReader(exact, url, maximum_header_bytes, chunk_bytes)
    metadata, tensors, count = _parse(reader, tensor_count, retained_metadata)
    return bytes(reader.raw), metadata, tensors, count, reader.total, reader.etag


def parse_header(raw, tensor_count, header_bytes, retained_metadata=None, *, whole_file_bytes):
    require(len(raw) == header_bytes, 'cached GGUF header byte extent differs')
    return _parse(CachedReader(raw, whole_file_bytes), tensor_count, retained_metadata)


def read_local_header(path, *, whole_file_bytes=None, maximum_header_bytes=16 * 1024 * 1024):
    """Read only an original local file's header, or an exact cached header.

    A cached header needs its separately pinned whole_file_bytes. The returned
    raw bytes stop at tensor_data_offset; no tensor payload is read or decoded.
    """
    path = Path(path)
    available = path.stat().st_size
    whole_file_bytes = available if whole_file_bytes is None else whole_file_bytes
    require(type(whole_file_bytes) is int and whole_file_bytes >= available >= 24,
            'invalid local/original GGUF size')
    with path.open('rb') as source:
        prefix = source.read(24)
        magic, version, count, _ = struct.unpack('<4sIQQ', prefix)
        require(magic == b'GGUF' and version in (2, 3) and 0 < count <= 4096,
                'unsupported local GGUF preamble')

        def exact(url, first, last, *, total=None, etag=None):
            require(0 <= first <= last < min(available, maximum_header_bytes),
                    'short local GGUF structural source')
            source.seek(first)
            return source.read(last - first + 1), whole_file_bytes, None

        raw, metadata, tensors, metadata_count, _, _ = scan(exact,
            url=str(path), tensor_count=count, maximum_header_bytes=maximum_header_bytes)
    return dict(gguf_version=version, tensor_count=count, metadata_count=metadata_count,
                tensor_data_offset=len(raw), metadata=metadata, tensors=tensors), raw
