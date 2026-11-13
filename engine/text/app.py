"""Compose the local text CLI, model worker and supported HTTP adapters."""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

from .codec import HuggingFaceTextCodec
from .responses import ResponsesAdapter
from .runtime import GenerationProgress, TextEngine
from .server import TextHttpServer
from .worker_client import WorkerClient
from .configuration import PreparedChoices, Selection, read_configuration
from ..model_preparation import prepare_local_assets, validate_local_tokenizer


class ResponsesServer(TextHttpServer):
    """Current application composition; future protocols supply other routes."""
    def __init__(self, address, engine: TextEngine, metrics_enabled: bool):
        adapter = ResponsesAdapter()
        super().__init__(address, engine, metrics_enabled, {'/v1/responses': adapter, '/v1/responses/*': adapter})


def make_worker_command(args, choices: PreparedChoices | None = None) -> list[str]:
    command = [str(args.engine), '--engine-worker', '--inputs', str(args.inputs), '--backend', args.backend,
               '--capacity', str(args.capacity), '--max-sessions', str(args.max_sessions),
               '--active-sequences', str(args.active_sequences), '--sampling', args.sampling]
    if args.sampling == 'random':
        command += ['--temperature', str(args.temperature), '--top-k', str(args.top_k),
                    '--top-p', str(args.top_p), '--seed', str(args.seed)]
    if choices is None:
        if args.draft_count is not None: command += ['--draft-count', str(args.draft_count)]
        if args.overlap is not None: command += ['--overlap', 'on' if args.overlap else 'off']
        if args.draft_vocabulary: command += ['--draft-vocabulary', str(args.draft_vocabulary)]
        if args.draft_head: command += ['--draft-head', str(args.draft_head)]
    else:
        for index, profile in enumerate(choices.executions):
            if index: command += ['--profile', profile.name]
            if profile.drafts is not None: command += ['--draft-count', str(profile.drafts)]
            if profile.overlap is not None: command += ['--overlap', 'on' if profile.overlap else 'off']
            for role, path in profile.assets: command += ['--profile-asset', role + '=' + str(path)]
        for sampler in choices.samplers[1:]:
            command += ['--additional-sampling', f'random,{sampler.temperature},{sampler.top_k},{sampler.top_p}' if sampler.random else 'greedy']
        if choices.prefix_entries: command += ['--prefix-cache-entries', str(choices.prefix_entries)]
    return command


def prepare_choices(args, codec, choices: PreparedChoices) -> PreparedChoices:
    profiles = []
    for profile in choices.executions:
        if profile.generate is not None:
            directory = args.asset_cache or Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.cache'))) / 'hyle' / 'assets'
            assets = prepare_local_assets(args.inputs, args.tokenizer_dir, codec.preparation_tokenizer,
                directory, template=args.template, **profile.generate)
            print('runtime assets ' + ('created ' if assets.created else 'reused ') + str(assets.directory), file=sys.stderr, flush=True)
            profile = replace(profile, assets=assets.execution_assets)
        profiles.append(profile)
    return replace(choices, executions=tuple(profiles))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyle local text CLI and Responses service')
    sub = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--engine', type=Path, required=True, help='built Hyle executable')
    common.add_argument('--inputs', type=Path, required=True, help='prepared local model directory')
    common.add_argument('--tokenizer-dir', type=Path, required=True, help='local tokenizer and model metadata')
    common.add_argument('--template', type=Path, help='explicit local Jinja template')
    common.add_argument('--model-name', default='hyle')
    common.add_argument('--config', type=Path, help='cold execution/sampling and session policy TOML')
    common.add_argument('--asset-cache', type=Path, help='immutable store for configured automatic asset recipes')
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
    generate.add_argument('--execution-profile', default='default')
    generate.add_argument('--sampling-profile', default='default')
    generate.add_argument('--request-seed', type=int)
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
    choices = read_configuration(args)
    codec = HuggingFaceTextCodec(args.tokenizer_dir, template=args.template)
    validate_local_tokenizer(args.inputs, codec.preparation_tokenizer)
    choices = prepare_choices(args, codec, choices)
    worker = WorkerClient(make_worker_command(args, choices))
    engine = None
    try:
        codec.validate_vocabulary(worker.vocabulary)
        engine = TextEngine(worker, codec, args.model_name, choices=choices,
                            session_policy=choices.sessions if args.command == 'serve' else None)
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
            selection = choices.select(dict(execution_profile=args.execution_profile, sampling_profile=args.sampling_profile,
                                            **({'seed': args.request_seed} if args.request_seed is not None else {})))
            prepared = engine.prepare([{'role': 'user', 'content': args.text}], args.max_output_tokens, selection)
            request = engine.submit(prepared)
            try:
                engine.generate(request, lambda chunk: print(chunk, end='', flush=True), GenerationProgress())
                print()
            finally:
                request.close()
    finally:
        try:
            if engine: engine.close()
        finally:
            worker.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
