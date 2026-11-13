"""Collect original IDs through the ordinary token generation contract."""
from __future__ import annotations

from .text.contracts import TokenWorker


def generated_ids(worker: TokenWorker, prompt: list[int], maximum: int, vocabulary: int) -> list[int]:
    """The caller cold-prepares an ordinary greedy worker on its chosen backend."""
    request = worker.submit(prompt, maximum)
    result = []
    try:
        while True:
            event = request.read(timeout=120)
            if event.kind == 'token':
                if type(event.token) is not int or not 0 <= event.token < vocabulary or len(result) >= maximum:
                    raise ValueError('calibration returned invalid or excessive token IDs')
                result.append(event.token)
                request.acknowledge()
            elif event.kind == 'completed':
                if not result:
                    raise ValueError('calibration returned no generated IDs')
                return result
            else:
                raise ValueError('calibration did not retire successfully: ' + event.kind + ' ' + event.message)
    finally:
        request.close()


def write_record(output, source: str, ids: list[int]) -> None:
    output.write('HYLE_TOKEN_CALIBRATION_V1\n' + source + '\n' + ' '.join(map(str, ids)) + '\n')


def read_record(text: str, source: str, vocabulary: int) -> list[int]:
    lines = text.splitlines()
    if len(lines) != 3 or lines[0] != 'HYLE_TOKEN_CALIBRATION_V1' or lines[1] != source:
        raise ValueError('recorded token calibration source or format differs')
    ids = list(map(int, lines[2].split()))
    if not ids or any(not 0 <= token < vocabulary for token in ids):
        raise ValueError('invalid recorded token calibration IDs')
    return ids
