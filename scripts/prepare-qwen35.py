#!/usr/bin/env python3
"""Prepare reusable local Qwen inputs and optional MTP assets for Hyle."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.models.qwen35 import draft_vocabulary_preparation, prepare
from engine.vocabulary import tokenizer_default_ids
from engine.model_preparation import validate_local_tokenizer


def source_vocabulary(inputs: Path) -> tuple[str, int]:
    lines = (inputs / 'weights.tsv').read_text(encoding='utf-8').splitlines()
    if len(lines) < 3 or lines[0] not in ('HYLE_QWEN_ROLE_FILES_V2', 'HYLE_QWEN_GGUF_ROLE_FILES_V2'):
        raise ValueError('prepared inputs require a Qwen source weights manifest')
    geometry = lines[2].split()
    if not re.fullmatch('[0-9a-f]{64}', lines[1]) or len(geometry) < 4 or geometry[0] != 'geometry':
        raise ValueError('invalid prepared source identity or geometry')
    vocabulary = int(geometry[3])
    if not 0 < vocabulary <= (1 << 31) - 1:
        raise ValueError('invalid prepared model vocabulary size')
    return lines[1], vocabulary


def head_preparer():
    path = ROOT / 'scripts/prepare-mtp-draft-head.py'
    spec = importlib.util.spec_from_file_location('hyle_prepare_mtp_draft_head', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', type=Path, help='local Qwen model/config directory')
    source.add_argument('--inputs', type=Path, help='existing Hyle inputs; generate separate reusable MTP assets')
    parser.add_argument('--output', type=Path, required=True, help='new directory for prepared inputs or MTP assets')
    parser.add_argument('--tokenizer-dir', type=Path, help='local tokenizer directory; defaults to --model')
    source_files = parser.add_mutually_exclusive_group()
    source_files.add_argument('--gguf', type=Path, help='existing local single-file Qwen GGUF')
    source_files.add_argument('--gguf-role-files', type=Path, help='existing authenticated GGUF role record')
    source_files.add_argument('--role-files', type=Path, help='existing authenticated HF role record')
    parser.add_argument('--mtp', action='store_true', help='prepare the original MTP layer and optional draft assets')
    parser.add_argument('--draft-vocabulary', choices=('auto','tokenizer','calibrated','none'), default='auto',
                        help='automatic, tokenizer-only, model-calibrated, or no draft subset')
    parser.add_argument('--draft-rows', type=int, help='selected original token rows; defaults to half the vocabulary')
    parser.add_argument('--hyle', type=Path, help='local Hyle token-worker executable for model calibration')
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'apple', 'vulkan'), help='calibration backend')
    parser.add_argument('--calibration-run', type=Path, action='append', default=[],
                        help='reuse source-bound calibration records; repeat to combine')
    parser.add_argument('--draft-head-q4', type=Path, metavar='GGML_LIBRARY',
                        help='also prepare a Q4_K draft head using this local ggml library')
    parser.add_argument('--quantizer-version', help='actual local library version, llama.cpp@<40 hex>')
    parser.add_argument('--prompt', help='initial text prompt when preparing model inputs')
    parser.add_argument('--continuation', help='optional raw continuation when preparing model inputs')
    parser.add_argument('--fp8-reader', choices=[prepare.FP8_READER,prepare.CT_MIXED_READER,prepare.STATIC_FP8_READER])
    parser.add_argument('--kv-cache-dtype', choices=['bf16','e4m3fn'])
    args = parser.parse_args(argv)

    if args.output.exists(): parser.error('output directory must be new')
    if args.inputs and (args.gguf or args.gguf_role_files or args.role_files or args.prompt is not None or
                        args.continuation is not None or args.fp8_reader or args.kv_cache_dtype):
        parser.error('--inputs reuses an existing source; model preparation options require --model')
    if args.inputs and not args.mtp: parser.error('separate MTP assets require --mtp')
    if not args.mtp and (args.draft_vocabulary != 'auto' or args.draft_rows is not None or args.hyle or
                         args.calibration_run or args.draft_head_q4 or args.quantizer_version):
        parser.error('draft asset options require --mtp')
    if args.draft_rows is not None and args.draft_rows <= 0: parser.error('--draft-rows must be positive')
    if args.draft_vocabulary == 'none' and args.draft_rows is not None:
        parser.error('--draft-rows cannot be used with no draft vocabulary')
    if args.draft_vocabulary != 'calibrated' and (args.hyle or args.calibration_run):
        parser.error('--hyle and --calibration-run require --draft-vocabulary calibrated')
    if args.draft_vocabulary == 'calibrated' and bool(args.hyle) == bool(args.calibration_run):
        parser.error('calibration requires either --hyle or one or more --calibration-run directories')
    if args.backend is not None and (args.draft_vocabulary != 'calibrated' or args.hyle is None):
        parser.error('--backend requires live model calibration with --hyle')
    if len({path.resolve() for path in args.calibration_run}) != len(args.calibration_run) or \
            any(not path.is_dir() for path in args.calibration_run):
        parser.error('calibration directories must exist and must not be repeated')
    if bool(args.draft_head_q4) != bool(args.quantizer_version):
        parser.error('--draft-head-q4 and --quantizer-version must be provided together')
    if args.draft_head_q4 and (not args.draft_head_q4.is_file() or
            not re.fullmatch(r'llama\.cpp@[0-9a-f]{40}',args.quantizer_version)):
        parser.error('Q4 draft head requires an existing local library and its exact llama.cpp revision')
    if args.draft_head_q4 and (args.gguf or args.gguf_role_files):
        parser.error('the offline Q4 head tool requires a dense HF source')
    if args.inputs and args.draft_vocabulary == 'none' and not args.draft_head_q4:
        parser.error('no MTP asset was requested')
    tokenizer_dir = args.tokenizer_dir or args.model
    needs_tokenizer = (args.inputs and args.draft_vocabulary in ('auto','tokenizer')) or \
        (args.draft_vocabulary == 'calibrated' and args.hyle is not None)
    if needs_tokenizer and (tokenizer_dir is None or not tokenizer_dir.is_dir()):
        parser.error('an existing local --tokenizer-dir is required for this draft vocabulary')
    if args.hyle and not args.hyle.is_file(): parser.error('--hyle must be an existing local executable')
    if args.inputs and not (args.inputs/'weights.tsv').is_file(): parser.error('--inputs requires weights.tsv')
    if args.inputs:
        manifest = (args.inputs/'weights.tsv').read_text(encoding='utf-8').splitlines()
        _, source_rows = source_vocabulary(args.inputs)
        if args.draft_rows is not None and args.draft_rows > source_rows:
            parser.error('--draft-rows exceeds the source vocabulary')
        if not any(line.startswith('mtp.') for line in manifest):
            parser.error('the prepared source has no MTP roles')
        if args.draft_head_q4:
            head_preparer().source_head(args.inputs.resolve())

    output = args.output.resolve()
    if args.model:
        arguments = ['--model',str(args.model.resolve()),'--output',str(output)]
        if args.tokenizer_dir: arguments += ['--tokenizer-dir',str(args.tokenizer_dir.resolve())]
        if args.mtp: arguments.append('--mtp')
        mode = 'none' if args.draft_vocabulary == 'calibrated' else args.draft_vocabulary
        if args.mtp: arguments += ['--draft-vocabulary',mode]
        if args.draft_rows is not None and mode != 'none': arguments += ['--draft-rows',str(args.draft_rows)]
        for option,value in (('--gguf',args.gguf),('--gguf-role-files',args.gguf_role_files),
                             ('--role-files',args.role_files),('--prompt',args.prompt),
                             ('--continuation',args.continuation),('--fp8-reader',args.fp8_reader),
                             ('--kv-cache-dtype',args.kv_cache_dtype)):
            if value is not None: arguments += [option,str(value.resolve() if isinstance(value,Path) else value)]
        prepare.main(arguments)
        inputs = output
    else:
        inputs = args.inputs.resolve()
        output.mkdir(parents=True)

    identity, vocabulary = source_vocabulary(inputs)
    mapping = output / 'mtp-draft-vocabulary.ids'
    if args.mtp and args.inputs and args.draft_vocabulary in ('auto','tokenizer'):
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                          HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir.resolve(), local_files_only=True,
                                                 trust_remote_code=False, use_fast=True)
        validate_local_tokenizer(inputs, tokenizer)
        ids,_ = tokenizer_default_ids(tokenizer,vocabulary,args.draft_rows)
        with mapping.open('x',encoding='ascii',newline='\n') as stream:
            draft_vocabulary_preparation.write_vocabulary(stream,identity,vocabulary,ids,None)
    if args.mtp and args.draft_vocabulary == 'calibrated':
        rows = args.draft_rows if args.draft_rows is not None else (vocabulary + 1)//2
        command = [sys.executable,str(ROOT/'scripts/prepare-mtp-draft-vocabulary.py'),
                   '--inputs',str(inputs),'--output',str(mapping),'--work-dir',str(output/'draft-calibration'),
                   '--rows',str(rows)]
        if args.hyle: command += ['--hyle',str(args.hyle.resolve()),'--tokenizer-dir',str(tokenizer_dir.resolve())]
        if args.backend is not None: command += ['--backend', args.backend]
        for path in args.calibration_run: command += ['--calibration-run',str(path.resolve())]
        subprocess.run(command,cwd=ROOT,check=True)

    metadata = None
    if args.draft_head_q4:
        head = head_preparer()
        mapping_lines = mapping.read_text(encoding='utf-8').splitlines() if mapping.is_file() else None
        if mapping_lines is not None and (len(mapping_lines) < 6 or mapping_lines[1] != identity or
                                          mapping_lines[4] != 'head original'):
            raise ValueError('prepared vocabulary is not bound to the selected original head')
        mapping_rows = len(' '.join(mapping_lines[5:]).split()) if mapping_lines is not None else 0
        compact = mapping if 0 < mapping_rows < vocabulary else None
        head.prepare(inputs,args.draft_head_q4.resolve(),args.quantizer_version,output/'draft-head',
                     compact_vocabulary=compact)
        lines = (output/'draft-head/mtp-draft-head.tsv').read_text(encoding='utf-8').splitlines()
        payload = shlex.split(lines[5])
        if len(payload) != 2 or payload[0] != 'draft-head-q4_k.bin':
            raise ValueError('prepared draft head payload metadata differs')
        lines[5] = '"draft-head/draft-head-q4_k.bin" ' + payload[1]
        metadata = output/'mtp-draft-head.tsv'
        metadata.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        if mapping_lines is not None:
            mapping_lines[4] = 'head prepared'
            mapping.write_text('\n'.join(mapping_lines)+'\n',encoding='utf-8')

    print(f'Hyle inputs: {inputs}',flush=True)
    if mapping.is_file(): print(f'Draft vocabulary: {mapping}',flush=True)
    if metadata: print(f'Draft Q4 head: {metadata}',flush=True)
    if args.inputs:
        command=['hyle','--inputs',str(inputs)]
        if mapping.is_file(): command += ['--mtp-draft-vocabulary',str(mapping)]
        if metadata: command += ['--mtp-draft-head',str(metadata)]
        print('Load with: '+subprocess.list2cmdline(command),flush=True)
    elif metadata and not mapping.is_file():
        print('Enable the prepared full head with --mtp on; no draft subset was requested.',flush=True)


if __name__ == '__main__':
    main()
