"""Shared text preparation and delivery, independent of wire protocols."""
from __future__ import annotations

import threading
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .codec import TextCodec
from .contracts import TokenEvent, TokenRequest, TokenWorker, WorkerError
from .configuration import PreparedChoices, Selection, SessionPolicy
from .sessions import TextSessions


@dataclass(frozen=True)
class PreparedText:
    ids: list[int]
    limit: int
    selection: Selection = Selection()


@dataclass
class GenerationProgress:
    count: int = 0
    last: int | None = None
    stop_sequence: str | None = None


class _TextStops:
    """Hold only a possible delimiter prefix; never publish part of a stop."""
    def __init__(self, sequences: Sequence[str]):
        self.sequences = tuple(sequences)
        if any(not isinstance(stop, str) or not stop for stop in self.sequences):
            raise ValueError('text stop sequences must be nonempty strings')
        self.pending = ''
        self.states = [0] * len(self.sequences)
        self.prefixes = []
        for stop in self.sequences:
            prefix = array('I', [0]) * len(stop)
            for index in range(1, len(stop)):
                length = prefix[index - 1]
                while length and stop[index] != stop[length]: length = prefix[length - 1]
                if stop[index] == stop[length]: length += 1
                prefix[index] = length
            self.prefixes.append(prefix)

    def push(self, chunk: str, *, final: bool = False) -> tuple[str, str | None]:
        text = self.pending + chunk
        for position, character in enumerate(chunk):
            matches = []
            for index, stop in enumerate(self.sequences):
                length, prefix = self.states[index], self.prefixes[index]
                while length and character != stop[length]: length = prefix[length - 1]
                if character == stop[length]: length += 1
                self.states[index] = length
                if length == len(stop): matches.append(stop)
            if matches:
                stop = max(matches, key=len)
                visible = text[:len(self.pending) + position + 1 - len(stop)]
                self.pending = ''
                return visible, stop
        keep = 0 if final else max(self.states, default=0)
        self.pending = text[-keep:] if keep else ''
        return text[:-keep] if keep else text, None


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests = self.completed = self.failed = self.output_tokens = 0

    def begin(self) -> None:
        with self._lock:
            self.requests += 1

    def end(self, completed: bool, tokens: int) -> None:
        with self._lock:
            self.completed += int(completed)
            self.failed += int(not completed)
            self.output_tokens += tokens

    def render(self) -> bytes:
        with self._lock:
            return (f"hyle_requests_total {self.requests}\n"
                    f"hyle_requests_completed_total {self.completed}\n"
                    f"hyle_requests_failed_total {self.failed}\n"
                    f"hyle_output_tokens_total {self.output_tokens}\n").encode("ascii")


class TextEngine:
    def __init__(self, worker: TokenWorker, codec: TextCodec, model_name: str, *,
                 choices: PreparedChoices | None = None, session_policy: SessionPolicy | None = None):
        self.worker, self.codec, self.model_name = worker, codec, model_name
        self.stop_ids = frozenset(worker.stop_ids | codec.stop_ids)
        self.extra_stop_ids = tuple(sorted(self.stop_ids - worker.stop_ids))
        self.metrics = Metrics()
        self.choices = choices or PreparedChoices()
        self.sessions = TextSessions(worker, session_policy, self.extra_stop_ids,
                                    getattr(codec, 'history_message', None)) if session_policy else None

    def prepare(self, messages: Sequence[dict[str, Any]], limit: int, selection: Selection = Selection(),
                previous_tape: Sequence[int] | None = None, *, continue_final_message: bool = False) -> PreparedText:
        if type(limit) is not int or limit <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        try:
            continuation = getattr(self.codec, 'continue_messages', None)
            ids = continuation(previous_tape, messages) if continuation and previous_tape and not continue_final_message else None
            if ids is None:
                ids = self.codec.encode_messages(messages, continue_final_message=True) if continue_final_message else self.codec.encode_messages(messages)
        except Exception as error:
            raise ValueError("chat template could not render the supplied messages: " + str(error)) from error
        if len(ids) > self.worker.context or limit - 1 > self.worker.context - len(ids):
            raise ValueError("prompt and output limit exceed the loaded model context")
        return PreparedText(ids, limit, selection)

    def submit(self, prepared: PreparedText) -> TokenRequest:
        return self.worker.submit(prepared.ids, prepared.limit, stop_ids=self.extra_stop_ids,
                                  **prepared.selection.worker_options())

    def begin_close(self) -> None:
        if self.sessions: self.sessions.begin_close()

    @property
    def healthy(self) -> bool:
        return self.worker.healthy and (self.sessions is None or self.sessions.healthy)

    def close(self) -> None:
        if self.sessions: self.sessions.close()

    def generate(self, request: TokenRequest, on_text: Callable[[str], None], progress: GenerationProgress,
                 read: Callable[[], TokenEvent] | None = None, *, stop_sequences: Sequence[str] = ()) -> str:
        decoder = self.codec.new_decoder()
        read_next = read if read is not None else request.read
        chunks: list[str] = []
        stops = _TextStops(stop_sequences) if stop_sequences else None
        while True:
            event = read_next()
            if event.kind == "token":
                assert event.token is not None
                progress.count += 1
                progress.last = event.token
                chunk = "" if event.token in self.stop_ids else decoder.push(event.token)
                if stops: chunk, progress.stop_sequence = stops.push(chunk)
                if chunk:
                    on_text(chunk)
                    chunks.append(chunk)
                if progress.stop_sequence is not None: return ''.join(chunks)
                # The consumer must accept the text before returning credit.
                request.acknowledge()
            elif event.kind == "completed":
                tail = decoder.finish()
                if stops: tail, progress.stop_sequence = stops.push(tail, final=True)
                if tail:
                    on_text(tail)
                    chunks.append(tail)
                return "".join(chunks)
            elif event.kind == "error":
                raise WorkerError(event.error_kind or "generation", event.message)
            else:
                raise WorkerError("generation", "generation " + event.kind)
