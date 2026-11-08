"""Compose the local text CLI, model worker and supported HTTP adapters."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .codec import HuggingFaceTextCodec
from .responses import ResponsesAdapter
from .runtime import GenerationProgress, TextEngine
from .server import TextHttpServer
from .worker_client import WorkerClient


class ResponsesServer(TextHttpServer):
    """Current application composition; future protocols supply other routes."""
    def __init__(self, address, engine: TextEngine, metrics_enabled: bool):
        super().__init__(address, engine, metrics_enabled, {'/v1/responses': ResponsesAdapter()})


def make_worker_command(args) -> list[str]:
    command = [str(args.engine), '--engine-worker', '--inputs', str(args.inputs), '--backend', args.backend,
               '--capacity', str(args.capacity), '--max-sessions', str(args.max_sessions),
               '--active-sequences', str(args.active_sequences), '--sampling', args.sampling]
    if args.draft_count is not None:
        command += ['--draft-count', str(args.draft_count)]
    if args.overlap is not None:
        command += ['--overlap', 'on' if args.overlap else 'off']
    if args.sampling == 'random':
        command += ['--temperature', str(args.temperature), '--top-k', str(args.top_k),
                    '--top-p', str(args.top_p), '--seed', str(args.seed)]
    if args.draft_vocabulary:
        command += ['--draft-vocabulary', str(args.draft_vocabulary)]
    if args.draft_head:
        command += ['--draft-head', str(args.draft_head)]
    return command


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyle local text CLI and Responses service')
    sub = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--engine', type=Path, required=True, help='built Hyle executable')
    common.add_argument('--inputs', type=Path, required=True, help='prepared local model directory')
    common.add_argument('--tokenizer-dir', type=Path, required=True, help='local tokenizer and model metadata')
    common.add_argument('--template', type=Path, help='explicit local Jinja template')
    common.add_argument('--model-name', default='hyle')
    common.add_argument('--backend', choices=('auto', 'cpu', 'cuda', 'apple', 'vulkan'), default='auto')
    common.add_argument('--capacity', type=int, default=4096)
    common.add_argument('--max-sessions', type=int, default=2)
    common.add_argument('--active-sequences', type=int, default=2)
    common.add_argument('--draft-count', type=int)
    common.add_argument('--draft-vocabulary', type=Path)
    common.add_argument('--draft-head', type=Path)
    common.add_argument('--overlap', action=argparse.BooleanOptionalAction, default=None)
    common.add_argument('--sampling', choices=('greedy', 'random'), default='greedy')
    common.add_argument('--temperature', type=float)
    common.add_argument('--top-k', type=int)
    common.add_argument('--top-p', type=float)
    common.add_argument('--seed', type=int)
    serve = sub.add_parser('serve', parents=[common], help='serve the supported Responses subset')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=2048)
    serve.add_argument('--metrics', action=argparse.BooleanOptionalAction, default=True)
    generate = sub.add_parser('generate', parents=[common], help='generate from one text prompt')
    generate.add_argument('--text', required=True)
    generate.add_argument('--max-output-tokens', type=int, default=256)
    args = parser.parse_args(argv)
    if args.sampling != 'random' and any(value is not None for value in
                                          (args.temperature, args.top_k, args.top_p, args.seed)):
        parser.error('--temperature, --top-k, --top-p and --seed require --sampling random')
    args.temperature = 1. if args.temperature is None else args.temperature
    args.top_k = 0 if args.top_k is None else args.top_k
    args.top_p = 1. if args.top_p is None else args.top_p
    args.seed = 0 if args.seed is None else args.seed
    if args.sampling == 'random':
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            parser.error('--temperature requires a finite positive number')
        if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
            parser.error('--top-p requires a finite number in (0, 1]')
        if args.top_k < 0:
            parser.error('--top-k requires a nonnegative integer')
        if not 0 <= args.seed <= (1 << 64) - 1:
            parser.error('--seed requires an unsigned 64-bit integer')
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    worker = WorkerClient(make_worker_command(args))
    try:
        codec = HuggingFaceTextCodec(args.tokenizer_dir, template=args.template,
                                     expected_vocabulary=worker.vocabulary)
        engine = TextEngine(worker, codec, args.model_name)
        if args.command == 'serve':
            server = ResponsesServer((args.host, args.port), engine, args.metrics)
            try:
                print(f'Hyle Responses listening on {args.host}:{args.port}', file=sys.stderr, flush=True)
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        else:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            prepared = engine.prepare([{'role': 'user', 'content': args.text}], args.max_output_tokens)
            request = engine.submit(prepared)
            try:
                engine.generate(request, lambda chunk: print(chunk, end='', flush=True), GenerationProgress())
                print()
            finally:
                request.close()
    finally:
        worker.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
