"""Shared text preparation and delivery, independent of wire protocols."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .codec import TextCodec
from .contracts import TokenEvent, TokenRequest, TokenWorker, WorkerError


@dataclass(frozen=True)
class PreparedText:
    ids: list[int]
    limit: int


@dataclass
class GenerationProgress:
    count: int = 0
    last: int | None = None


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
    def __init__(self, worker: TokenWorker, codec: TextCodec, model_name: str):
        self.worker, self.codec, self.model_name = worker, codec, model_name
        self.stop_ids = frozenset(worker.stop_ids | codec.stop_ids)
        self.extra_stop_ids = tuple(sorted(self.stop_ids - worker.stop_ids))
        self.metrics = Metrics()

    def prepare(self, messages: Sequence[dict[str, Any]], limit: int) -> PreparedText:
        if type(limit) is not int or limit <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        try:
            ids = self.codec.encode_messages(messages)
        except Exception as error:
            raise ValueError("chat template could not render the supplied messages: " + str(error)) from error
        if len(ids) > self.worker.context or limit - 1 > self.worker.context - len(ids):
            raise ValueError("prompt and output limit exceed the loaded model context")
        return PreparedText(ids, limit)

    def submit(self, prepared: PreparedText) -> TokenRequest:
        return self.worker.submit(prepared.ids, prepared.limit, stop_ids=self.extra_stop_ids)

    def generate(self, request: TokenRequest, on_text: Callable[[str], None], progress: GenerationProgress,
                 read: Callable[[], TokenEvent] | None = None) -> str:
        decoder = self.codec.new_decoder()
        read_next = read if read is not None else request.read
        chunks: list[str] = []
        while True:
            event = read_next()
            if event.kind == "token":
                assert event.token is not None
                progress.count += 1
                progress.last = event.token
                chunk = "" if event.token in self.stop_ids else decoder.push(event.token)
                if chunk:
                    on_text(chunk)
                    chunks.append(chunk)
                # The consumer must accept the text before returning credit.
                request.acknowledge()
            elif event.kind == "completed":
                tail = decoder.finish()
                if tail:
                    on_text(tail)
                    chunks.append(tail)
                return "".join(chunks)
            elif event.kind == "error":
                raise WorkerError("generation", event.message)
            else:
                raise WorkerError("generation", "generation " + event.kind)
