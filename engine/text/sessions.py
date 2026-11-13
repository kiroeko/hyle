"""Bounded text continuations over the original worker's live request owner."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import copy
import hashlib
import threading
import time

from .configuration import Selection, SessionPolicy
from .contracts import LiveTokenRequest, LiveTokenWorker, TokenEvent, WorkerError


class SessionNotFound(ValueError):
    pass


@dataclass
class NativeSession:
    request: LiveTokenRequest
    selection: Selection
    domain: int
    tape: tuple[int, ...]
    turn: int = 0
    released: bool = False
    conditioning: tuple = ()


@dataclass
class StoredTurn:
    key: str
    messages: tuple[dict, ...]
    result: dict
    domain: int
    expires: float
    tape: tuple[int, ...]
    native: NativeSession | None = None


class SessionTurn:
    """One HTTP owner's loan. State remains in the original native request."""
    def __init__(self, owner, key, messages, ids, selection, save, domain):
        self.owner, self.key, self.messages = owner, key, messages
        self.ids, self.selection, self.save, self.domain = tuple(ids), selection, save, domain
        self.native: NativeSession | None = None
        self.generated: list[int] = []
        self.cached_tokens = 0
        self.done = self.succeeded = self.committed = self.closed = False

    def read(self, timeout=None) -> TokenEvent:
        if self.native is None or self.closed or self.done:
            raise RuntimeError('text turn is not readable')
        event = self.native.request.read(timeout)
        if event.kind == 'token':
            self.generated.append(event.token)
        elif event.kind == 'turn_end':
            if event.turn != self.native.turn:
                raise WorkerError('protocol', 'native turn boundary is out of order')
            self.native.request.acknowledge()
            self.native.turn += 1
            self.native.tape = self.ids + tuple(self.generated)
            self.done = True
            self.succeeded = True
            return TokenEvent('completed')
        elif event.kind in ('completed', 'cancelled', 'error'):
            self.done = True
        return event

    def acknowledge(self) -> None:
        self.native.request.acknowledge()

    def commit(self, text: str, result: dict) -> None:
        if not self.succeeded or self.closed or self.committed:
            raise RuntimeError('only a completed text turn can be stored')
        self.owner._commit(self, text, result)
        self.committed = True

    def close(self) -> None:
        if self.closed or self.committed:
            return
        self.closed = True
        try:
            if self.native is not None:
                self.owner._retire(self.native)
        finally:
            with self.owner._condition:
                self.owner._active.discard(self)
                self.owner._condition.notify_all()


class TextSessions:
    """History records, native KV residency and ingress bounds are separate.

    A parent record is immutable. Its native tip can be loaned once; concurrent
    forks rebuild their own text input. Eviction drops native residency only,
    retaining bounded history for later replay. No KV is serialized to disk.
    """
    def __init__(self, worker: LiveTokenWorker, policy: SessionPolicy, stop_ids=(), history_message=None):
        self.worker, self.policy, self.stop_ids = worker, policy, tuple(stop_ids)
        self._history_message = history_message or (lambda ids, text: {'role': 'assistant', 'content': text})
        self._condition = threading.Condition()
        self._records: OrderedDict[str, StoredTurn] = OrderedDict()
        self._active: set[SessionTurn] = set()
        self._native_count = self._waiting = 0
        self._closing = False
        self._failure: str | None = None
        self._shared_domain = worker.allocate_cache_domain() if policy.shared_prefix else None
        self._reaper = threading.Thread(target=self._expire, name='hyle-session-expiry', daemon=True)
        self._reaper.start()

    def _check_open(self):
        if self._closing or self._failure:
            raise WorkerError('closed', self._failure or 'text session owner is closing')

    @property
    def healthy(self) -> bool:
        with self._condition:
            return not self._closing and self._failure is None

    def parent(self, key: str) -> StoredTurn:
        expired = None
        with self._condition:
            self._check_open()
            value = self._records.get(key)
            if value is None:
                raise SessionNotFound('stored response was not found or expired')
            if value.expires <= time.monotonic():
                expired = self._records.pop(key)
            else:
                value.expires = time.monotonic() + self.policy.ttl_seconds
                self._records.move_to_end(key)
                return value
        if expired.native:
            self._retire(expired.native)
        raise SessionNotFound('stored response was not found or expired')

    def retrieve(self, key: str) -> dict:
        return copy.deepcopy(self.parent(key).result)

    def delete(self, key: str) -> None:
        with self._condition:
            value = self._records.pop(key, None)
        if value is None:
            raise SessionNotFound('stored response was not found or expired')
        if value.native:
            self._retire(value.native)

    def _release_reservation(self):
        with self._condition:
            if self._native_count <= 0:
                raise RuntimeError('native session reservation underflow')
            self._native_count -= 1
            self._condition.notify_all()

    def _retire(self, state: NativeSession):
        if state.released:
            return
        try:
            event = state.request.retire(timeout=120)
            if event.kind not in ('completed', 'cancelled') and not (
                    event.kind == 'error' and event.error_kind in ('invalid', 'admission')):
                raise WorkerError('retirement', 'original native retirement is unconfirmed: ' + event.message)
            state.request.close()
            state.released = True
            self._release_reservation()
        except Exception as error:
            # Never recycle a reservation based on a timeout or native failure.
            # The worker remains the real resource/retirement authority.
            with self._condition:
                self._failure = 'native session retirement failed: ' + str(error)
                self._condition.notify_all()
            raise

    def _reserve(self):
        while True:
            victim = None
            with self._condition:
                self._check_open()
                if self._native_count < self.policy.kv_capacity:
                    self._native_count += 1
                    return
                for record in self._records.values():
                    if record.native:
                        victim, record.native = record.native, None
                        break
                if victim is None:
                    if self._waiting >= self.policy.max_waiting:
                        raise WorkerError('busy', 'text session waiting capacity is full')
                    self._waiting += 1
                    try:
                        self._condition.wait()
                    finally:
                        self._waiting -= 1
            if victim is not None:
                self._retire(victim)

    def start(self, key: str, messages, ids, limit: int, selection: Selection, *,
              parent: StoredTurn | None = None, save: bool = True, attachments=()) -> SessionTurn:
        state = None
        with self._condition:
            self._check_open()
            if parent is not None and self._records.get(parent.key) is parent:
                state, parent.native = parent.native, None
        domain = parent.domain if parent is not None else self._shared_domain
        if domain is None:
            domain = self.worker.allocate_cache_domain()
        conditioning=tuple((part.begin,part.rows,part.format,part.shape,hashlib.sha256(part.payload).digest()) for part in attachments)
        reusable = (state is not None and state.selection == selection and tuple(ids[:len(state.tape)]) == state.tape and
                    state.conditioning==tuple(item for item in conditioning if item[0]<len(state.tape) and item[0]+item[1]<=len(state.tape)) and
                    not any(item[0]<len(state.tape)<item[0]+item[1] for item in conditioning))
        if state is not None and not reusable:
            self._retire(state); state = None
        if state is None:
            self._reserve()
        # Structured content is mutable even though image bytes are immutable.
        # Retain our own history tree without copying the underlying byte data.
        turn = SessionTurn(self, key, tuple(copy.deepcopy(m) for m in messages), ids, selection, save, domain)
        try:
            with self._condition:
                self._check_open(); self._active.add(turn)
            if state is None:
                request = self.worker.open_session(ids, limit, stop_ids=self.stop_ids,
                    cache_sharing_domain=domain, **selection.worker_options(),
                    **({'attachments': attachments} if attachments else {}))
                state = NativeSession(request, selection, domain, tuple(ids),conditioning=conditioning)
                turn.native = state
            else:
                turn.native = state
                turn.cached_tokens = max(0, len(state.tape) - 1)
                suffix_attachments = tuple(replace(part, begin=part.begin-len(state.tape))
                    for part in attachments if part.begin >= len(state.tape))
                if any(part.begin < len(state.tape) < part.begin+part.rows for part in attachments):
                    raise ValueError('session continuation splits an image input')
                state.request.append(ids[len(state.tape):], limit, timeout=120,
                    **({'attachments': suffix_attachments} if suffix_attachments else {}))
                state.conditioning=conditioning
            with self._condition:
                self._check_open()
            return turn
        except Exception:
            if turn.native is None:
                if state is None: self._release_reservation()
                else: self._retire(state)
                with self._condition:
                    self._active.discard(turn); self._condition.notify_all()
            else:
                turn.close()
            raise

    def _commit(self, turn: SessionTurn, text: str, result: dict):
        evicted = []
        if not turn.save:
            self._retire(turn.native)
        with self._condition:
            self._check_open()
            self._active.discard(turn)
            if turn.save:
                history = turn.messages + (self._history_message(turn.generated, text),)
                self._records[turn.key] = StoredTurn(turn.key, history, copy.deepcopy(result), turn.domain,
                    time.monotonic() + self.policy.ttl_seconds, turn.native.tape, turn.native)
                while len(self._records) > self.policy.capacity:
                    _, victim = self._records.popitem(last=False)
                    if victim.native: evicted.append(victim.native)
            self._condition.notify_all()
        for state in evicted:
            self._retire(state)

    def _expire(self):
        while True:
            victims = []
            with self._condition:
                if self._closing: return
                now = time.monotonic()
                for key in list(self._records):
                    if self._records[key].expires <= now:
                        victim = self._records.pop(key)
                        if victim.native: victims.append(victim.native)
                delay = min((record.expires - now for record in self._records.values()), default=self.policy.ttl_seconds)
                if not victims:
                    self._condition.wait(max(.01, delay)); continue
            try:
                for state in victims: self._retire(state)
            except Exception:
                return

    def begin_close(self):
        with self._condition:
            self._closing = True
            states = [record.native for record in self._records.values() if record.native]
            states += [turn.native for turn in self._active if turn.native]
            self._condition.notify_all()
        for state in states:
            try: state.request.cancel()
            except WorkerError: pass  # close() still requires a terminal observation.

    def close(self):
        self.begin_close()
        self._reaper.join(timeout=125)
        if self._reaper.is_alive():
            raise WorkerError('shutdown', 'session expiry owner did not stop')
        with self._condition:
            if self._active:
                raise WorkerError('shutdown', 'active text owners must stop before session shutdown')
            states = [record.native for record in self._records.values() if record.native]
            self._records.clear()
        for state in states:
            self._retire(state)
        with self._condition:
            if self._native_count:
                raise WorkerError('shutdown', 'native session retirement remains unconfirmed')
