"""Cold, immutable filesystem assets. No model, device or request authority."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Iterator


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    # OS locks disappear after a crashed preparer. The persistent lock file is
    # never interpreted as ownership, so no PID guessing or stale-lock delete.
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b'\0'); stream.flush()
        if os.name == 'nt':
            import msvcrt
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.05)
            try:
                yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class AssetStore:
    """Publish once; a live program can retain the returned paths unchanged."""
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check(directory: Path, recipe: str) -> None:
        lines = (directory / 'assets.tsv').read_text(encoding='utf-8').splitlines()
        if len(lines) < 3 or lines[0] != 'HYLE_COLD_ASSETS_V1' or lines[1] != recipe:
            raise ValueError('prepared asset recipe differs; existing entries are never overwritten')
        seen = set()
        for line in lines[2:]:
            name, size, digest = line.split('\t')
            if not name or Path(name).name != name or name in seen or name == 'assets.tsv':
                raise ValueError('invalid prepared asset file name')
            seen.add(name)
            path = directory / name
            if not path.is_file() or path.stat().st_size != int(size) or file_digest(path) != digest:
                raise ValueError('prepared asset is incomplete or modified: ' + name)

    def prepare(self, recipe: str, build: Callable[[Path], None]) -> tuple[Path, bool]:
        if not recipe or '\n' in recipe or '\r' in recipe:
            raise ValueError('asset recipe must be one nonempty line')
        key = hashlib.sha256(recipe.encode('utf-8')).hexdigest()
        final = self.directory / key
        with _lock(self.directory / (key + '.lock')):
            if final.exists():
                self._check(final, recipe)
                return final, False
            with tempfile.TemporaryDirectory(prefix=key + '.', dir=self.directory) as temporary:
                staged = Path(temporary) / 'ready'
                staged.mkdir()
                build(staged)
                files = sorted(staged.iterdir())
                if not files or any(not p.is_file() or p.name == 'assets.tsv' for p in files):
                    raise ValueError('asset preparer must produce bounded regular files')
                metadata = ['HYLE_COLD_ASSETS_V1', recipe]
                for path in files:
                    with path.open('r+b') as payload:
                        os.fsync(payload.fileno())
                    metadata.append(f'{path.name}\t{path.stat().st_size}\t{file_digest(path)}')
                with (staged / 'assets.tsv').open('x', encoding='utf-8', newline='\n') as stream:
                    stream.write('\n'.join(metadata) + '\n'); stream.flush(); os.fsync(stream.fileno())
                self._check(staged, recipe)
                staged.rename(final)  # Same filesystem; never replace a live entry.
            return final, True
