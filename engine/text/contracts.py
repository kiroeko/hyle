"""Transport-neutral token request and delivery contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class WorkerError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class TokenEvent:
    kind: str
    token: int | None = None
    message: str = ""
    turn: int | None = None


class TokenRequest(Protocol):
    def read(self, timeout: float | None = None) -> TokenEvent: ...
    def acknowledge(self) -> None: ...
    def close(self) -> None: ...


class TokenWorker(Protocol):
    context: int
    stop_ids: frozenset[int]

    @property
    def healthy(self) -> bool: ...
    def submit(self, tokens: Sequence[int], max_output_tokens: int, *,
               stop_ids: Sequence[int] = ()) -> TokenRequest: ...


class LiveTokenRequest(TokenRequest, Protocol):
    """One session with ordered, acknowledged turn events and explicit input EOF."""
    def append(self, tokens: Sequence[int], max_output_tokens: int, *, timeout: float = 30) -> None: ...
    def finish_input(self, *, timeout: float = 30) -> None: ...
    def cancel(self) -> bool: ...


class LiveTokenWorker(TokenWorker, Protocol):
    def open_session(self, tokens: Sequence[int], max_output_tokens: int, *,
                     stop_ids: Sequence[int] = ()) -> LiveTokenRequest: ...
