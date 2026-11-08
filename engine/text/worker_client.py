"""Bounded token transport to one persistent Hyle model process.

This module knows neither model mathematics nor HTTP schemas. A request has one
outstanding token credit; callers acknowledge only after consuming that token.
"""
from __future__ import annotations

import queue
import subprocess
import threading
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
    def __init__(self, client: WorkerClient, request_id: int, events: queue.Queue):
        self._client, self._id, self._events = client, request_id, events
        self._lock = threading.Lock()
        self._outstanding = False
        self._terminal = False
        self._cancelled = False

    def read(self, timeout: float | None = None) -> TokenEvent:
        with self._lock:
            if self._outstanding:
                raise RuntimeError("acknowledge the previous token before reading another")
            if self._terminal:
                raise RuntimeError("request has already terminated")
        event = self._events.get(timeout=timeout)
        with self._lock:
            if event.kind == "token":
                self._outstanding = True
            else:
                self._terminal = True
                self._client._forget(self._id)
        return event

    def acknowledge(self) -> None:
        with self._lock:
            if not self._outstanding or self._terminal:
                raise RuntimeError("there is no unacknowledged token")
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

    def close(self) -> None:
        try:
            self.cancel()
        except WorkerError:
            pass
        finally:
            self._client._forget(self._id)


class WorkerClient:
    def __init__(self, command: Sequence[str | Path], *, startup_timeout: float = 60):
        if not command:
            raise ValueError("model worker command must not be empty")
        self._process = subprocess.Popen(list(map(str, command)), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=None, text=True, encoding="utf-8", errors="replace", bufsize=1)
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[int, queue.Queue] = {}
        self._next_id = 1
        self._closed = False
        self._failure: str | None = None
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

    def submit(self, tokens: Sequence[int], max_output_tokens: int) -> WorkerRequest:
        if not tokens or type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("prompt and output limit must be nonempty")
        if any(type(token) is not int or token < 0 or token > 2147483647 for token in tokens):
            raise ValueError("prompt contains an invalid token ID")
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
            self._send(f"G\t{request_id}\t{max_output_tokens}\t" + ",".join(map(str, tokens)))
        except Exception:
            self._forget(request_id)
            raise
        return WorkerRequest(self, request_id, events)

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
                    if len(fields) != 5 or fields[:2] != ["READY", "2"]:
                        raise WorkerError("protocol", "invalid model worker handshake")
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
                if kind == "T" and len(fields) == 3:
                    self._publish(int(fields[1]), TokenEvent("token", int(fields[2])))
                elif kind == "F" and len(fields) == 3 and fields[2] in ("completed", "cancelled", "closed"):
                    self._publish(int(fields[1]), TokenEvent(fields[2]))
                elif kind == "E" and len(fields) == 4:
                    request_id = int(fields[1])
                    if request_id == 0:
                        raise WorkerError("protocol", fields[3])
                    self._publish(request_id, TokenEvent("error", message=fields[2] + ": " + fields[3]))
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
                pending = list(self._pending.values())
            for events in pending:
                try:
                    events.put_nowait(TokenEvent("error", message=self._failure or "model worker closed"))
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
        if self._process.stdin:
            self._process.stdin.close()
        if self._process.stdout:
            self._process.stdout.close()
        if timed_out:
            raise WorkerError("shutdown", "model worker did not finish graceful shutdown before timeout")
        if self._failure:
            raise WorkerError("shutdown", self._failure)
        if exit_code != 0:
            raise WorkerError("shutdown", f"model worker exited with status {exit_code}")
        if not self._bye.is_set():
            raise WorkerError("shutdown", "model worker exited without shutdown acknowledgement")
