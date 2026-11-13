"""Transport-neutral token request and delivery contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ImageInput:
    media_type: str
    data: bytes
    detail: str = 'auto'


@dataclass(frozen=True)
class PromptAttachment:
    format: str
    begin: int
    rows: int
    shape: tuple[int, ...]
    payload: bytes


@dataclass(frozen=True)
class PreparedPrompt:
    ids: list[int]
    attachments: tuple[PromptAttachment, ...]


@dataclass(frozen=True)
class ExecutionProfileFacts:
    """Immutable facts from the actual prepared executor, not requested options."""
    name: str
    algorithm: str
    max_draft_tokens: int
    input_modalities: tuple[str, ...] = ('text',)
    co_resident_with: tuple[int, ...] = ()


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
    error_kind: str = ""  # Request-local error category, separate from display text.


class TokenRequest(Protocol):
    def read(self, timeout: float | None = None) -> TokenEvent: ...
    def acknowledge(self) -> None: ...
    def close(self) -> None: ...


class TokenWorker(Protocol):
    context: int
    stop_ids: frozenset[int]
    execution_profiles: tuple[ExecutionProfileFacts, ...]

    @property
    def healthy(self) -> bool: ...
    def submit(self, tokens: Sequence[int], max_output_tokens: int, *,
               stop_ids: Sequence[int] = (), sampling_index: int | None = None,
               seed: int | None = None, max_draft_tokens: int | None = None,
               execution_profile: int | None = None, cache_sharing_domain: int | None = None,
               attachments: Sequence[PromptAttachment] = ()) -> TokenRequest: ...


class LiveTokenRequest(TokenRequest, Protocol):
    """One session with ordered, acknowledged turn events and explicit input EOF."""
    def append(self, tokens: Sequence[int], max_output_tokens: int, *, timeout: float = 30,
               attachments: Sequence[PromptAttachment] = ()) -> None: ...
    def finish_input(self, *, timeout: float = 30) -> None: ...
    def cancel(self) -> bool: ...
    def retire(self, *, timeout: float = 120) -> TokenEvent: ...


class LiveTokenWorker(TokenWorker, Protocol):
    def allocate_cache_domain(self) -> int: ...
    def open_session(self, tokens: Sequence[int], max_output_tokens: int, *,
                     stop_ids: Sequence[int] = (), sampling_index: int | None = None,
                     seed: int | None = None, max_draft_tokens: int | None = None,
                     execution_profile: int | None = None, cache_sharing_domain: int | None = None,
                     attachments: Sequence[PromptAttachment] = ()) -> LiveTokenRequest: ...
