"""Text CLI and a text-only OpenAI Responses adapter for a local Hyle worker."""
from __future__ import annotations

import argparse
import json
import queue
import select
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .codec import HuggingFaceTextCodec, TextCodec
from .worker_client import WorkerClient, WorkerError, WorkerRequest


class InvalidRequest(ValueError):
    pass


class ClientDisconnected(ConnectionError):
    pass


def parse_responses_input(payload: Any, model_name: str) -> tuple[list[dict[str, str]], int, bool, str | None]:
    if not isinstance(payload, dict):
        raise InvalidRequest("request body must be a JSON object")
    supported = {"model", "input", "instructions", "max_output_tokens", "stream", "store"}
    unknown = set(payload) - supported
    if unknown:
        raise InvalidRequest("unsupported request fields: " + ", ".join(sorted(unknown)))
    if payload.get("model") != model_name:
        raise InvalidRequest("model does not match the loaded model")
    if "input" not in payload:
        raise InvalidRequest("input is required")
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise InvalidRequest("stream must be a boolean")
    if "store" in payload and payload["store"] is not False:
        raise InvalidRequest("stored responses are not supported; store must be false")
    limit = payload.get("max_output_tokens", 256)
    if type(limit) is not int or limit <= 0:
        raise InvalidRequest("max_output_tokens must be a positive integer")
    instructions = payload.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise InvalidRequest("instructions must be a string")
    messages: list[dict[str, str]] = []
    if instructions is not None:
        messages.append({"role": "system", "content": instructions})
    source = payload["input"]
    if isinstance(source, str):
        messages.append({"role": "user", "content": source})
    elif isinstance(source, list) and source:
        for item in source:
            if not isinstance(item, dict) or set(item) - {"type", "role", "content"}:
                raise InvalidRequest("input items must be text messages")
            if item.get("type", "message") != "message" or item.get("role") not in ("system", "user", "assistant"):
                raise InvalidRequest("only system, user and assistant text messages are supported")
            content = item.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if (not isinstance(part, dict) or set(part) != {"type", "text"} or
                            part["type"] not in ("input_text", "output_text") or not isinstance(part["text"], str)):
                        raise InvalidRequest("only text content parts are supported")
                    parts.append(part["text"])
                text = "".join(parts)
            else:
                raise InvalidRequest("message content must be text")
            messages.append({"role": item["role"], "content": text})
    else:
        raise InvalidRequest("input must be a string or a nonempty message list")
    return messages, limit, stream, instructions


@dataclass(frozen=True)
class PreparedText:
    ids: list[int]
    limit: int
    instructions: str | None


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
    def __init__(self, worker: WorkerClient, codec: TextCodec, model_name: str):
        self.worker, self.codec, self.model_name = worker, codec, model_name
        self.metrics = Metrics()

    def prepare(self, payload: Any) -> tuple[PreparedText, bool]:
        messages, limit, stream, instructions = parse_responses_input(payload, self.model_name)
        try:
            ids = self.codec.encode_messages(messages)
        except Exception as error:
            raise InvalidRequest("chat template could not render the supplied messages: " + str(error)) from error
        if len(ids) > self.worker.context or limit - 1 > self.worker.context - len(ids):
            raise InvalidRequest("prompt and output limit exceed the loaded model context")
        return PreparedText(ids, limit, instructions), stream

    def submit(self, prepared: PreparedText) -> WorkerRequest:
        return self.worker.submit(prepared.ids, prepared.limit)


def response_object(response_id: str, item_id: str, model: str, created: int, prepared: PreparedText,
                    status: str, text: str, output_tokens: int) -> dict[str, Any]:
    part = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
    item = {"id": item_id, "type": "message", "status": status, "role": "assistant", "content": [part]}
    usage = {"input_tokens": len(prepared.ids), "input_tokens_details": {"cached_tokens": 0},
             "output_tokens": output_tokens, "output_tokens_details": {"reasoning_tokens": 0},
             "total_tokens": len(prepared.ids) + output_tokens}
    return {"id": response_id, "object": "response", "created_at": created, "status": status,
            "error": None, "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "instructions": prepared.instructions, "max_output_tokens": prepared.limit, "model": model,
            "output": [item] if status != "in_progress" else [], "parallel_tool_calls": False,
            "previous_response_id": None, "reasoning": {"effort": None, "summary": None},
            "store": False, "text": {"format": {"type": "text"}}, "tool_choice": "none", "tools": [],
            "truncation": "disabled", "usage": usage if status != "in_progress" else None, "metadata": {}}


class ResponsesHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Hyle/0.1"
    MAX_BODY = 1024 * 1024

    @property
    def engine(self) -> TextEngine:
        return self.server.engine  # type: ignore[attr-defined]

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": {"message": message, "type": "invalid_request_error" if status < 500 else "server_error",
                                     "param": None, "code": None}})

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            healthy = self.engine.worker.healthy
            self._json(200 if healthy else 503, {"status": "ok" if healthy else "unavailable"})
        elif path == "/metrics" and self.server.metrics_enabled:  # type: ignore[attr-defined]
            data = self.engine.metrics.render()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers(); self.wfile.write(data); self.close_connection = True
        else:
            self._error(404, "endpoint not found")

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/v1/responses":
            self._error(404, "endpoint not found"); return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length <= 0 or length > self.MAX_BODY:
                raise InvalidRequest("invalid or oversized request body")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise InvalidRequest("incomplete request body")
            payload = json.loads(raw.decode("utf-8"))
            prepared, streaming = self.engine.prepare(payload)
        except (ValueError, UnicodeError, json.JSONDecodeError) as error:
            self._error(400, str(error)); return
        self.engine.metrics.begin()
        try:
            request = self.engine.submit(prepared)
        except Exception as error:
            self.engine.metrics.end(False, 0); self._error(503, str(error)); return
        if streaming:
            self._stream(prepared, request)
        else:
            self._complete(prepared, request)

    def _disconnected(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True

    def _next(self, request: WorkerRequest):
        while True:
            try:
                return request.read(timeout=.25)
            except queue.Empty:
                if self._disconnected():
                    raise ClientDisconnected("client disconnected")

    def _generate(self, request: WorkerRequest, on_text, progress: dict[str, Any]):
        decoder = self.engine.codec.new_decoder()
        chunks: list[str] = []
        while True:
            event = self._next(request)
            if event.kind == "token":
                assert event.token is not None
                progress["count"] += 1; progress["last"] = event.token
                chunk = "" if event.token in self.engine.worker.stop_ids else decoder.push(event.token)
                if chunk:
                    on_text(chunk); chunks.append(chunk)
                request.acknowledge()
            elif event.kind == "completed":
                tail = decoder.finish()
                if tail:
                    on_text(tail); chunks.append(tail)
                return "".join(chunks)
            elif event.kind == "error":
                raise WorkerError("generation", event.message)
            else:
                raise WorkerError("generation", "generation " + event.kind)

    def _complete(self, prepared: PreparedText, request: WorkerRequest) -> None:
        progress: dict[str, Any] = {"count": 0, "last": None}; success = False
        try:
            text = self._generate(request, lambda _: None, progress)
            count, last = progress["count"], progress["last"]
            status = "incomplete" if count == prepared.limit and last not in self.engine.worker.stop_ids else "completed"
            result = response_object("resp_" + uuid.uuid4().hex, "msg_" + uuid.uuid4().hex,
                                     self.engine.model_name, int(time.time()), prepared, status, text, count)
            self._json(200, result); success = True
        except ClientDisconnected:
            pass
        except Exception as error:
            try: self._error(400 if str(error).startswith("invalid:") else 503, str(error))
            except (BrokenPipeError, ConnectionResetError, OSError): pass
        finally:
            request.close(); self.engine.metrics.end(success, progress["count"])

    def _stream(self, prepared: PreparedText, request: WorkerRequest) -> None:
        response_id, item_id, created = "resp_" + uuid.uuid4().hex, "msg_" + uuid.uuid4().hex, int(time.time())
        sequence = 0; progress: dict[str, Any] = {"count": 0, "last": None}; success = False
        def send(kind: str, **values: Any) -> None:
            nonlocal sequence
            event = {"type": kind, "sequence_number": sequence, **values}; sequence += 1
            data = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.wfile.write(b"event: " + kind.encode("ascii") + b"\ndata: " + data + b"\n\n")
            self.wfile.flush()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers(); self.close_connection = True
            empty = response_object(response_id, item_id, self.engine.model_name, created, prepared, "in_progress", "", 0)
            send("response.created", response=empty)
            send("response.in_progress", response=empty)
            send("response.output_item.added", output_index=0,
                 item={"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []})
            send("response.content_part.added", item_id=item_id, output_index=0, content_index=0,
                 part={"type": "output_text", "text": "", "annotations": [], "logprobs": []})
            def on_text(chunk: str) -> None:
                send("response.output_text.delta", item_id=item_id, output_index=0, content_index=0,
                     delta=chunk, logprobs=[])
            text = self._generate(request, on_text, progress)
            count, last = progress["count"], progress["last"]
            status = "incomplete" if count == prepared.limit and last not in self.engine.worker.stop_ids else "completed"
            result = response_object(response_id, item_id, self.engine.model_name, created, prepared, status, text, count)
            send("response.output_text.done", item_id=item_id, output_index=0, content_index=0, text=text, logprobs=[])
            send("response.content_part.done", item_id=item_id, output_index=0, content_index=0,
                 part=result["output"][0]["content"][0])
            send("response.output_item.done", output_index=0, item=result["output"][0])
            send("response." + status, response=result); success = True
        except (ClientDisconnected, BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as error:
            try: send("error", code="server_error", message=str(error), param=None)
            except (BrokenPipeError, ConnectionResetError, OSError): pass
        finally:
            request.close(); self.engine.metrics.end(success, progress["count"])


class ResponsesServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, engine: TextEngine, metrics_enabled: bool):
        super().__init__(address, ResponsesHandler)
        self.engine, self.metrics_enabled = engine, metrics_enabled


def make_worker_command(args) -> list[str]:
    command = [str(args.engine), "--engine-worker", "--inputs", str(args.inputs), "--backend", args.backend,
               "--capacity", str(args.capacity), "--max-sessions", str(args.max_sessions),
               "--active-sequences", str(args.active_sequences), "--sampling", args.sampling]
    if args.draft_count is not None:
        command += ["--draft-count", str(args.draft_count)]
    if args.overlap is not None:
        command += ["--overlap", "on" if args.overlap else "off"]
    if args.sampling == "random":
        command += ["--temperature", str(args.temperature), "--top-k", str(args.top_k),
                    "--top-p", str(args.top_p), "--seed", str(args.seed)]
    if args.draft_vocabulary:
        command += ["--draft-vocabulary", str(args.draft_vocabulary)]
    if args.draft_head:
        command += ["--draft-head", str(args.draft_head)]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hyle local text CLI and Responses service")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--engine", type=Path, required=True, help="built Hyle executable")
    common.add_argument("--inputs", type=Path, required=True, help="prepared local model directory")
    common.add_argument("--tokenizer-dir", type=Path, required=True, help="local tokenizer and model metadata")
    common.add_argument("--template", type=Path, help="explicit local Jinja template")
    common.add_argument("--model-name", default="hyle")
    common.add_argument("--backend", choices=("auto", "cpu", "cuda", "apple", "vulkan"), default="auto")
    common.add_argument("--capacity", type=int, default=4096)
    common.add_argument("--max-sessions", type=int, default=2)
    common.add_argument("--active-sequences", type=int, default=2)
    common.add_argument("--draft-count", type=int)
    common.add_argument("--draft-vocabulary", type=Path)
    common.add_argument("--draft-head", type=Path)
    common.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=None)
    common.add_argument("--sampling", choices=("greedy", "random"), default="greedy")
    common.add_argument("--temperature", type=float, default=1.)
    common.add_argument("--top-k", type=int, default=0)
    common.add_argument("--top-p", type=float, default=1.)
    common.add_argument("--seed", type=int, default=0)
    serve = sub.add_parser("serve", parents=[common], help="serve the supported Responses subset")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=2048)
    serve.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True)
    generate = sub.add_parser("generate", parents=[common], help="generate from one text prompt")
    generate.add_argument("--text", required=True)
    generate.add_argument("--max-output-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    worker = WorkerClient(make_worker_command(args))
    try:
        codec = HuggingFaceTextCodec(args.tokenizer_dir, template=args.template,
                                     expected_vocabulary=worker.vocabulary)
        engine = TextEngine(worker, codec, args.model_name)
        if args.command == "serve":
            server = ResponsesServer((args.host, args.port), engine, args.metrics)
            try:
                print(f"Hyle Responses listening on {args.host}:{args.port}", file=sys.stderr, flush=True)
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            prepared, _ = engine.prepare({"model": args.model_name, "input": args.text,
                                          "max_output_tokens": args.max_output_tokens})
            request = engine.submit(prepared); decoder = codec.new_decoder()
            try:
                while True:
                    event = request.read()
                    if event.kind == "token":
                        assert event.token is not None
                        if event.token not in worker.stop_ids:
                            print(decoder.push(event.token), end="", flush=True)
                        request.acknowledge()
                    elif event.kind == "completed":
                        print(decoder.finish(), end="", flush=True); print(); break
                    else:
                        raise WorkerError("generation", event.message or event.kind)
            finally:
                request.close()
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
