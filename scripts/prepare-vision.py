"""Prepare reusable local image assets for a prepared Hyle model; no downloads."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.model_preparation import prepare_local_vision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True,help='prepared language model')
    parser.add_argument('--model',type=Path,required=True,help='local original model and processor configuration')
    parser.add_argument('--mmproj',type=Path,help='optional local vision GGUF; otherwise borrow original HF vision tensors')
    parser.add_argument('--output',type=Path,required=True,help='output vision.tsv asset, reusable at subsequent launches')
    args=parser.parse_args()
    print(prepare_local_vision(args.inputs,args.model,args.output,mmproj=args.mmproj))


if __name__=='__main__': main()
