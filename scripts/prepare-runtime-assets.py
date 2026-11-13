#!/usr/bin/env python3
"""Prepare/reuse immutable local runtime assets; no model execution or download."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.model_preparation import prepare_local_assets
from engine.text.codec import HuggingFaceTextCodec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='immutable filesystem asset store')
    parser.add_argument('--rows', type=int)
    parser.add_argument('--languages', help='comma-separated language preference, e.g. zh,en,hu')
    parser.add_argument('--template', type=Path)
    args = parser.parse_args()
    codec = HuggingFaceTextCodec(args.tokenizer_dir, template=args.template)
    result = prepare_local_assets(args.inputs, args.tokenizer_dir, codec.preparation_tokenizer,
                                  args.output, rows=args.rows, languages=args.languages, template=args.template)
    print('created' if result.created else 'reused', result.directory)
    for role, path in result.artifacts: print(role, path)


if __name__ == '__main__':
    main()
