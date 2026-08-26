#!/usr/bin/env python3
"""Canaries for raw nanobench log indexing and independent replay."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from hyle_perf.controls import SUPPORTED_TIMER_BATCHES  # noqa: E402
from hyle_perf.model import ValidationError  # noqa: E402
from hyle_perf.raw_replay import (  # noqa: E402
    build_raw_log_index,
    replay_final_stage,
    validate_raw_log_index,
)


def fixture():
    contract = {
        "id": "raw-replay",
        "workloads": [{
            "id": "work", "required_strata": ["candidate-first"],
            "pair": {"candidate": "candidate", "peer": "peer"},
            "metric": {"name": "operations-per-second", "unit": "op/s",
                       "direction": "higher-is-better"},
        }],
        "strata": [{
            "id": "candidate-first", "target_variant": "primary",
            "dimensions": {"execution_order": "candidate-first"},
        }],
    }
    runner = {
        "executables": [{
            "id": "product", "target_variant": "primary",
            "measurement_batch": 1,
            "telemetry_prefixes": ["# HYLE_PRODUCT_RUN "],
            "workloads": [{"id": "work", "title": "consumer work"}],
        }],
    }
    return contract, runner


def table(title: str, candidate: str, candidate_ops: float,
          peer: str, peer_ops: float) -> str:
    return (
        f"| ns/op | op/s | err% | total | {title}\n"
        "|------:|-----:|-----:|------:|:---\n"
        f"| 1 | {candidate_ops:.9f} | 0.1% | 0.01 | `{candidate}`\n"
        f"| 1 | {peer_ops:.9f} | 0.1% | 0.01 | `{peer}`\n")


def control_text(order: str, nonce: str) -> str:
    text = (f"# HYLE_PERF_CONTROL_RUN execution_order={order} "
            f"run_nonce={nonce}\n")
    text += table("instrument A/A N=4096", "A/A candidate", 1000,
                  "A/A peer", 1000)
    text += table("instrument known delta N=4096", "delta5 candidate", 1050,
                  "delta5 peer slower", 1000)
    text += table("instrument known delta 10 percent N=4096",
                  "delta10 candidate", 1100, "delta10 peer slower", 1000)
    for batch in SUPPORTED_TIMER_BATCHES:
        text += table(f"instrument timer floor batch={batch}",
                      "timer floor candidate", 1000 * batch,
                      "timer floor peer", 1000 * batch)
    return text


class RawReplayTest(unittest.TestCase):
    def materialize(self, root: pathlib.Path):
        contract, runner = fixture()
        stage = root / "n-1"
        stage.mkdir()
        product = stage / "product-candidate-first-product-run-01.log"
        product.write_text(
            "# HYLE_PRODUCT_RUN execution_order=candidate-first "
            "run_nonce=00000000000000000000000000000001\n" +
            table("consumer work", "candidate", 2000, "peer", 1000),
            encoding="utf-8")
        controls = []
        for nonce_index, order in enumerate(
                ("candidate-first", "peer-first"), start=2):
            path = stage / f"control-{order}-run-01.log"
            path.write_text(
                control_text(order, f"{nonce_index:032x}"), encoding="utf-8")
            controls.append(path)
        paths = [product, *controls]
        index = build_raw_log_index(contract, runner, [1], paths)
        references = [entry["file"] for entry in index["entries"]]
        return contract, runner, product, index, references

    def test_reconstructs_product_and_controls_from_raw_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, runner, _, index, references = self.materialize(
                pathlib.Path(temporary))
            normalized = validate_raw_log_index(
                index, contract, runner, [1], references)
            observations, controls = replay_final_stage(
                normalized, contract, runner, platform_name="windows")
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0].candidate_value, 2000.0)
            self.assertEqual(controls["orders"]["candidate-first"]["delta5"],
                             [1.05])

    def test_changed_telemetry_and_missing_log_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, runner, product, index, references = self.materialize(
                pathlib.Path(temporary))
            product.write_text(
                product.read_text(encoding="utf-8").replace(
                    "candidate-first", "peer-first"), encoding="utf-8")
            # Rebuild the reference so this attacks semantics, not only SHA.
            index = build_raw_log_index(
                contract, runner, [1], [
                    pathlib.Path(reference["path"]) for reference in references])
            with self.assertRaisesRegex(ValidationError, "telemetry"):
                replay_final_stage(index, contract, runner,
                                   platform_name="windows")
            pathlib.Path(references[-1]["path"]).unlink()
            with self.assertRaisesRegex(ValidationError, "missing or changed"):
                validate_raw_log_index(
                    index, contract, runner, [1],
                    [entry["file"] for entry in index["entries"]])

    def test_copied_process_log_is_not_a_new_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, runner, _, index, references = self.materialize(
                pathlib.Path(temporary))
            candidate_control = pathlib.Path(references[1]["path"])
            peer_control = pathlib.Path(references[2]["path"])
            peer_control.write_bytes(candidate_control.read_bytes())
            with self.assertRaisesRegex(ValidationError, "repeat"):
                build_raw_log_index(
                    contract, runner, [1], [
                        pathlib.Path(reference["path"])
                        for reference in references])


if __name__ == "__main__":
    unittest.main()
