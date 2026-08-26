#!/usr/bin/env python3
"""Canonical command-line entry point for HYLE performance evidence v1."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys


SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from hyle_perf.contract import load_contract, make_contract_lock  # noqa: E402
from hyle_perf.evidence import load_and_validate_manifest  # noqa: E402
from hyle_perf.model import ValidationError  # noqa: E402
from hyle_perf.runner import (  # noqa: E402
    certify_manifests,
    run_contract,
    validate_runner_extension,
)


def _peer(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("peer must use ID=PATH")
    peer_id, path = value.split("=", 1)
    if not peer_id or not path:
        raise argparse.ArgumentTypeError("peer must use non-empty ID=PATH")
    return peer_id, pathlib.Path(path)


def _default_path(root: pathlib.Path, category: str, contract_id: str) -> pathlib.Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_id = contract_id.replace("/", "-").replace(":", "-")
    return root / ".scratch" / category / f"{safe_id}-{stamp}"


def _run_mode(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    contract = load_contract(args.contract)
    peers = dict(args.peer)
    if len(peers) != len(args.peer):
        raise ValidationError("peer ids must be unique")
    build = (args.build.resolve() if args.build else
             _default_path(source, "perf-build", contract["id"]))
    output = (args.output.resolve() if args.output else
              _default_path(source, "perf-evidence", contract["id"]))
    result = run_contract(
        source=source, contract_path=args.contract, machine_id=args.machine,
        peer_paths=peers, build=build, output=output, mode=args.command,
        formal=getattr(args, "formal", False))
    print(json.dumps({
        "manifest": result["manifest_path"],
        "qualification_gate": result["qualification_gate"],
        "performance_gate": result["performance_gate"],
    }, sort_keys=True, indent=2))
    if args.command == "qualify-platform" and (
            result["qualification_gate"] != "PASS" or
            result["performance_gate"] != "PASS"):
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="contract-driven HYLE performance instrument")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate a contract and print its immutable lock hash")
    validate.add_argument("--contract", type=pathlib.Path, required=True)

    verify = subparsers.add_parser(
        "verify", help="strictly replay one manifest and every referenced file")
    verify.add_argument("--contract", type=pathlib.Path, required=True)
    verify.add_argument("--manifest", type=pathlib.Path, required=True)

    for name in ("probe", "qualify-platform"):
        command = subparsers.add_parser(name)
        command.add_argument("--contract", type=pathlib.Path, required=True)
        command.add_argument("--machine", required=True)
        command.add_argument("--peer", action="append", type=_peer, default=[],
                             metavar="ID=PATH")
        command.add_argument("--source", type=pathlib.Path,
                             default=SCRIPT_DIRECTORY.parent)
        command.add_argument("--build", type=pathlib.Path)
        command.add_argument("--output", type=pathlib.Path)
        if name == "qualify-platform":
            command.add_argument(
                "--formal", action="store_true",
                help="apply this machine profile's Formal role/claim; required "
                     "for manifests later consumed by certify")

    certify = subparsers.add_parser(
        "certify", help="compose already-qualified platform manifests")
    certify.add_argument("--contract", type=pathlib.Path, required=True)
    certify.add_argument("--manifest", action="append", type=pathlib.Path,
                         required=True)
    certify.add_argument("--output", type=pathlib.Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "validate":
            contract = load_contract(args.contract)
            validate_runner_extension(contract)
            lock = make_contract_lock(contract)
            print(json.dumps({
                "contract_id": contract["id"],
                "module": contract["module"],
                "contract_sha256": lock["contract_sha256"],
                "runner_schema": contract["extensions"]["runner"]["schema"],
            }, sort_keys=True, indent=2))
            return 0
        if args.command == "verify":
            contract = load_contract(args.contract)
            validate_runner_extension(contract)
            manifest = load_and_validate_manifest(
                args.manifest, contract, verify_paths=True)
            print(json.dumps({
                "contract_sha256": manifest["contract_sha256"],
                "manifest": str(args.manifest.resolve()),
                "paths_verified": True,
                "qualification": manifest["qualification"],
            }, sort_keys=True, indent=2))
            return 0
        if args.command in {"probe", "qualify-platform"}:
            return _run_mode(args)
        certificate = certify_manifests(
            contract_path=args.contract, manifest_paths=args.manifest,
            output_path=args.output)
        print(json.dumps(certificate, sort_keys=True, indent=2))
        return 0
    except (ValidationError, RuntimeError, OSError, ValueError) as error:
        print(f"hyle-perf: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
