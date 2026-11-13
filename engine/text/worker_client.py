"""Bounded token transport to one persistent Hyle model process.

This module knows neither model mathematics nor HTTP schemas. A request has one
event credit; callers acknowledge after consuming each token or turn boundary.
"""
from __future__ import annotations

import math
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Sequence

from .contracts import TokenEvent, WorkerError


def _stop_owned_process(process: subprocess.Popen) -> bool:
    """Stop and reap this worker after graceful protocol use is impossible."""
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return False
    return True


class WorkerRequest:
    """One reader; cancellation may come from another thread."""
    def __init__(self, client: WorkerClient, request_id: int, events: queue.Queue, *, live: bool = False):
        self._client, self._id, self._events = client, request_id, events
        self._lock = threading.Lock()
        self._input_lock = threading.Lock()
        self._live = live
        self._input_closed = False
        self._input_uncertain = False
        self._outstanding = False
        self._terminal = False
        self._terminal_event: TokenEvent | None = None
        self._cancelled = False

    def read(self, timeout: float | None = None) -> TokenEvent:
        with self._lock:
            if self._outstanding:
                raise RuntimeError("acknowledge the previous event before reading another")
            if self._terminal:
                raise RuntimeError("request has already terminated")
        event = self._client._read_event(self._events, timeout)
        with self._lock:
            if event.kind in ("token", "turn_end"):
                self._outstanding = True
            else:
                self._terminal = True
                self._terminal_event = event
                self._client._forget(self._id)
        return event

    def acknowledge(self) -> None:
        with self._lock:
            if not self._outstanding or self._terminal:
                raise RuntimeError("there is no unacknowledged event")
            self._client._send(f"K\t{self._id}\t1")
            self._outstanding = False

    def cancel(self) -> bool:
        with self._lock:
            if self._terminal or self._cancelled:
                return False
            self._client._send(f"C\t{self._id}")
            self._cancelled = True
            self._outstanding = False
            return True

    def append(self, tokens: Sequence[int], max_output_tokens: int, *, timeout: float = 30) -> None:
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("output limit must be positive")
        if any(type(token) is not int or token < 0 or token > 2147483647 for token in tokens):
            raise ValueError("prompt contains an invalid token ID")
        with self._input_lock:
            with self._lock:
                if self._input_uncertain:
                    raise WorkerError("uncertain", "session input outcome is unknown; cancel this session")
                if not self._live or self._input_closed or self._terminal or self._cancelled:
                    raise WorkerError("closed", "session input is closed")
            try:
                self._client._input_command(self._id, "I", f"{max_output_tokens}\t" + ",".join(map(str, tokens)), timeout)
            except WorkerError as error:
                if error.kind == "timeout":
                    with self._lock:
                        self._input_uncertain = True
                raise

    def finish_input(self, *, timeout: float = 30) -> None:
        with self._input_lock:
            with self._lock:
                if self._input_uncertain:
                    raise WorkerError("uncertain", "session input outcome is unknown; cancel this session")
                if not self._live or self._input_closed or self._terminal or self._cancelled:
                    raise WorkerError("closed", "session input is closed")
            try:
                self._client._input_command(self._id, "Z", "", timeout)
            except WorkerError as error:
                if error.kind == "timeout":
                    with self._lock:
                        self._input_uncertain = True
                raise
            with self._lock:
                self._input_closed = True

    def close(self) -> None:
        try:
            self.cancel()
        except WorkerError:
            pass
        finally:
            self._client._forget(self._id)

    def retire(self, *, timeout: float = 120) -> TokenEvent:
        """Cancel and drain to the original owner's terminal observation.

        One reader, after delivery has stopped. Already emitted tokens need no
        credit after cancellation; do not send late acknowledgements to a
        reaped worker request. A timeout/error is never retirement success.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('retirement timeout must be finite and positive')
        self.cancel()
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._terminal_event is not None:
                    return self._terminal_event
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerError('timeout', 'original request retirement is unconfirmed')
            try:
                event = self._client._read_event(self._events, remaining)
            except queue.Empty as error:
                raise WorkerError('timeout', 'original request retirement is unconfirmed') from error
            if event.kind in ('token', 'turn_end'):
                continue
            with self._lock:
                self._terminal = True; self._terminal_event = event; self._outstanding = False
            self._client._forget(self._id)
            return event


class WorkerClient:
    def __init__(self, command: Sequence[str | Path], *, startup_timeout: float = 60):
        if not command:
            raise ValueError("model worker command must not be empty")
        self._process = subprocess.Popen(list(map(str, command)), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=None, text=True, encoding="utf-8", errors="replace", bufsize=1)
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[int, queue.Queue] = {}
        self._controls: dict[tuple[int, int], tuple[str, queue.Queue]] = {}
        self._next_id = 1
        self._next_domain = 1
        self._next_control = 1
        self._closed = False
        self._failure: str | None = None
        self._stream_end: str | None = None
        self.vocabulary = 0
        self.context = 0
        self.stop_ids: frozenset[int] = frozenset()
        self._ready = threading.Event()
        self._bye = threading.Event()
        self._reader = threading.Thread(target=self._read_frames, name="hyle-worker-output", daemon=True)
        self._reader.start()
        if not self._ready.wait(startup_timeout):
            self._cleanup_failed_start()
            raise WorkerError("startup", "model worker did not become ready")
        if self._failure:
            self._cleanup_failed_start()
            raise WorkerError("startup", self._failure)

    def _cleanup_failed_start(self) -> None:
        if not _stop_owned_process(self._process):
            raise WorkerError("startup", "failed model worker could not be stopped")
        self._reader.join(timeout=5)
        if self._reader.is_alive():
            raise WorkerError("startup", "failed model worker output reader did not stop")
        if self._process.stdin:
            self._process.stdin.close()
        if self._process.stdout:
            self._process.stdout.close()

    @property
    def healthy(self) -> bool:
        with self._state_lock:
            return not self._closed and self._failure is None and self._process.poll() is None

    def allocate_cache_domain(self) -> int:
        """Trusted frontend identity, never supplied by an HTTP client."""
        with self._state_lock:
            if self._next_domain >= 1 << 64:
                raise WorkerError('closed', 'cache sharing identity space is exhausted')
            value = self._next_domain
            self._next_domain += 1
            return value

    def submit(self, tokens: Sequence[int], max_output_tokens: int, *, stop_ids: Sequence[int] = (),
               sampling_index: int | None = None, seed: int | None = None,
               max_draft_tokens: int | None = None, execution_profile: int | None = None,
               cache_sharing_domain: int | None = None) -> WorkerRequest:
        return self._submit_turns(((tokens, max_output_tokens),), report_turns=False, stop_ids=stop_ids,
                                  sampling_index=sampling_index, seed=seed, max_draft_tokens=max_draft_tokens,
                                  execution_profile=execution_profile, cache_sharing_domain=cache_sharing_domain)

    def submit_turns(self, turns: Sequence[tuple[Sequence[int], int]], *, stop_ids: Sequence[int] = (),
                     sampling_index: int | None = None, seed: int | None = None,
                     max_draft_tokens: int | None = None, execution_profile: int | None = None,
                     cache_sharing_domain: int | None = None) -> WorkerRequest:
        """Submit one finite session with acknowledged, zero-based turn-end events."""
        return self._submit_turns(turns, report_turns=True, stop_ids=stop_ids,
                                  sampling_index=sampling_index, seed=seed, max_draft_tokens=max_draft_tokens,
                                  execution_profile=execution_profile, cache_sharing_domain=cache_sharing_domain)

    def open_session(self, tokens: Sequence[int], max_output_tokens: int, *, stop_ids: Sequence[int] = (),
                     sampling_index: int | None = None, seed: int | None = None,
                     max_draft_tokens: int | None = None, execution_profile: int | None = None,
                     cache_sharing_domain: int | None = None) -> WorkerRequest:
        """Open one live session; append turns and explicitly finish its input."""
        return self._submit_turns(((tokens, max_output_tokens),), report_turns=True, live=True, stop_ids=stop_ids,
                                  sampling_index=sampling_index, seed=seed, max_draft_tokens=max_draft_tokens,
                                  execution_profile=execution_profile, cache_sharing_domain=cache_sharing_domain)

    def _submit_turns(self, turns: Sequence[tuple[Sequence[int], int]], *, report_turns: bool,
                      live: bool = False, stop_ids: Sequence[int] = (), sampling_index: int | None = None,
                      seed: int | None = None, max_draft_tokens: int | None = None,
                      execution_profile: int | None = None, cache_sharing_domain: int | None = None) -> WorkerRequest:
        if not turns:
            raise ValueError("at least one turn is required")
        for value in (sampling_index, seed, max_draft_tokens, execution_profile, cache_sharing_domain):
            if value is not None and (type(value) is not int or not 0 <= value <= 18446744073709551615):
                raise ValueError("request configuration must be an unsigned 64-bit integer")
        if any(type(token) is not int or token < 0 or token >= self.vocabulary for token in stop_ids):
            raise ValueError("stop token is outside the model vocabulary")
        stops = sorted(set(stop_ids) - self.stop_ids)
        frames = []
        for index, (tokens, max_output_tokens) in enumerate(turns):
            if (index == 0 and not tokens) or type(max_output_tokens) is not int or max_output_tokens <= 0:
                raise ValueError("initial prompt and each output limit must be nonempty")
            if any(type(token) is not int or token < 0 or token > 2147483647 for token in tokens):
                raise ValueError("prompt contains an invalid token ID")
            frames.extend((str(max_output_tokens), ",".join(map(str, tokens))))
        events: queue.Queue = queue.Queue(maxsize=2)
        with self._state_lock:
            if self._closed or self._failure:
                raise WorkerError("closed", self._failure or "model worker is closed")
            if self._next_id > 18446744073709551615:
                raise WorkerError("closed", "request identity space is exhausted")
            request_id = self._next_id
            self._next_id += 1
            self._pending[request_id] = events
        try:
            kind = 'O' if live else 'S' if report_turns else 'G'
            if cache_sharing_domain is not None or execution_profile is not None or any(value is not None for value in (sampling_index, seed, max_draft_tokens)):
                choices = (sampling_index, seed, max_draft_tokens)
                fields = [str(value) if value is not None else '-' for value in choices]
                profile = ((str(execution_profile or 0), str(cache_sharing_domain)) if cache_sharing_domain is not None else
                           (str(execution_profile),) if execution_profile is not None else ())
                prefix = "\t".join(('D' if cache_sharing_domain is not None else 'P' if profile else 'R', str(request_id), kind, *profile,
                                     *fields, ','.join(map(str, stops)) or '-'))
            else:
                prefix = f"{kind.lower()}\t{request_id}\t{','.join(map(str, stops))}" if stops else f"{kind}\t{request_id}"
            self._send(prefix + "\t" + "\t".join(frames))
        except Exception:
            self._forget(request_id)
            raise
        return WorkerRequest(self, request_id, events, live=live)

    def _input_command(self, request_id: int, kind: str, payload: str, timeout: float) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("input acknowledgement timeout must be finite and positive")
        replies: queue.Queue = queue.Queue(maxsize=1)
        with self._state_lock:
            if self._closed or self._failure or request_id not in self._pending:
                raise WorkerError("closed", self._failure or "request is closed")
            if self._next_control > 18446744073709551615:
                raise WorkerError("closed", "input command identity space is exhausted")
            sequence = self._next_control
            self._next_control += 1
            self._controls[(request_id, sequence)] = (kind, replies)
        try:
            self._send(f"{kind}\t{request_id}\t{sequence}" + ("\t" + payload if payload else ""))
            try:
                status = replies.get(timeout=timeout)
            except queue.Empty as error:
                raise WorkerError("timeout", "session input acknowledgement timed out") from error
            if status != "accepted":
                raise WorkerError(status, "session input was not accepted")
        finally:
            with self._state_lock:
                self._controls.pop((request_id, sequence), None)

    def _send(self, frame: str) -> None:
        with self._write_lock:
            if self._process.stdin is None or self._process.poll() is not None:
                raise WorkerError("closed", "model worker is not running")
            try:
                self._process.stdin.write(frame + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as error:
                raise WorkerError("closed", "model worker input closed") from error

    def _forget(self, request_id: int) -> None:
        with self._state_lock:
            self._pending.pop(request_id, None)

    def _read_event(self, events: queue.Queue, timeout: float | None) -> TokenEvent:
        # Drain previously received events in order. If the output stream has
        # ended, retain its failure even when a full queue could not accept the
        # notification in _read_frames. Otherwise a reader could wait forever
        # after consuming those already queued events.
        with self._state_lock:
            try:
                return events.get_nowait()
            except queue.Empty:
                if self._stream_end is not None:
                    return TokenEvent("error", message=self._stream_end)
        return events.get(timeout=timeout)

    def _publish(self, request_id: int, event: TokenEvent) -> None:
        with self._state_lock:
            events = self._pending.get(request_id)
        if events is None:
            return
        try:
            events.put_nowait(event)
        except queue.Full as error:
            raise WorkerError("protocol", "worker exceeded the request output credit") from error

    def _read_frames(self) -> None:
        ready = False
        try:
            assert self._process.stdout is not None
            for raw in self._process.stdout:
                if not ready:
                    fields = raw.rstrip("\r\n").split("\t")
                    legacy = len(fields) == 5 and fields[:2] == ['READY', '2']
                    current = len(fields) == 6 and fields[:2] == ['READY', '3']
                    if not legacy and not current:
                        raise WorkerError("protocol", "invalid model worker handshake")
                    self.source_identity = fields[5] if current and fields[5] != '-' else None
                    if self.source_identity is not None and (not self.source_identity or
                            any(not 32 < ord(char) < 127 for char in self.source_identity)):
                        raise WorkerError('protocol', 'invalid model worker source identity')
                    self.vocabulary = int(fields[2])
                    self.context = int(fields[3])
                    self.stop_ids = frozenset() if fields[4] == "-" else frozenset(map(int, fields[4].split(",")))
                    if self.vocabulary <= 0 or self.context <= 0 or any(
                            token < 0 or token >= self.vocabulary for token in self.stop_ids):
                        raise WorkerError("protocol", "invalid model worker metadata")
                    ready = True
                    self._ready.set()
                    continue
                fields = raw.rstrip("\r\n").split("\t", 3)
                kind = fields[0]
                if kind == "BYE" and len(fields) == 1:
                    self._bye.set()
                    break
                if kind == "A" and len(fields) == 2 and int(fields[1]) > 0:
                    continue
                if kind == "C" and len(fields) == 3 and int(fields[1]) > 0 and fields[2] in ("accepted", "stale"):
                    continue
                if kind in ("I", "Z") and len(fields) == 4 and int(fields[1]) > 0 and int(fields[2]) > 0 and fields[3] in ("accepted", "closed", "invalid", "capacity", "failed"):
                    with self._state_lock:
                        pending = self._controls.get((int(fields[1]), int(fields[2])))
                    if pending and pending[0] == kind:
                        pending[1].put_nowait(fields[3])
                    continue
                if kind == "T" and len(fields) == 3:
                    request_id, token = int(fields[1]), int(fields[2])
                    if request_id <= 0 or token < 0 or token >= self.vocabulary:
                        raise WorkerError("protocol", "worker returned an invalid token frame")
                    self._publish(request_id, TokenEvent("token", token))
                elif kind == "B" and len(fields) == 3 and int(fields[1]) > 0 and int(fields[2]) >= 0:
                    self._publish(int(fields[1]), TokenEvent("turn_end", turn=int(fields[2])))
                elif kind == "F" and len(fields) == 3 and int(fields[1]) > 0 and fields[2] in ("completed", "cancelled", "closed"):
                    self._publish(int(fields[1]), TokenEvent(fields[2]))
                elif kind == "E" and len(fields) == 4:
                    request_id = int(fields[1])
                    if request_id == 0:
                        raise WorkerError("protocol", fields[3])
                    if request_id < 0:
                        raise WorkerError("protocol", "worker returned an invalid error frame")
                    self._publish(request_id, TokenEvent("error", message=fields[3], error_kind=fields[2]))
                else:
                    raise WorkerError("protocol", "invalid model worker response")
        except Exception as error:
            self._failure = str(error)
        finally:
            if not self._failure and not self._closed:
                self._failure = "model worker exited unexpectedly"
            # The stream can no longer be consumed safely. Stop only this
            # client's owned worker, including when it closed stdout but kept
            # running. Never report its in-flight work as successful.
            if self._failure and self._process.poll() is None:
                try:
                    self._process.terminate()
                except OSError:
                    pass
            self._ready.set()
            with self._state_lock:
                self._stream_end = self._failure or "model worker closed"
                pending = list(self._pending.values())
                controls = [entry[1] for entry in self._controls.values()]
            for events in pending:
                try:
                    events.put_nowait(TokenEvent("error", message=self._failure or "model worker closed"))
                except queue.Full:
                    pass
            for replies in controls:
                try:
                    replies.put_nowait("closed")
                except queue.Full:
                    pass

    def close(self, *, timeout: float = 30) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        if self._process.poll() is None:
            try:
                self._send("Q")
            except WorkerError:
                pass
        timed_out = False
        try:
            exit_code = self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Graceful shutdown was not observed. Ensure the subprocess we
            # own is not orphaned, then report the failed graceful shutdown.
            if not _stop_owned_process(self._process):
                raise WorkerError("shutdown", "model worker could not be stopped")
        self._reader.join(timeout=5)
        if self._reader.is_alive():
            raise WorkerError("shutdown", "model worker output reader did not stop")
        close_error = None
        for stream in (self._process.stdin, self._process.stdout):
            if stream:
                try:
                    stream.close()
                except OSError as error:
                    close_error = close_error or error
        if timed_out:
            raise WorkerError("shutdown", "model worker did not finish graceful shutdown before timeout")
        if self._failure:
            raise WorkerError("shutdown", self._failure)
        if exit_code != 0:
            raise WorkerError("shutdown", f"model worker exited with status {exit_code}")
        if not self._bye.is_set():
            raise WorkerError("shutdown", "model worker exited without shutdown acknowledgement")
        if close_error:
            raise WorkerError("shutdown", "model worker pipe could not be closed") from close_error
