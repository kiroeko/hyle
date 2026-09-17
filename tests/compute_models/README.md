# Independent model verification consumer

These tools prepare the local input and independent oracle for the
[LLM/MoE goal](../../docs/implementation/tasks/compute/llm-moe-coverage-completion-plan.md).
They are test utilities, never installed SDK APIs or an inference server. C++ route consumers are added with their
owning capability packages; the current Python reference does not prove any HYLE route passed.

- `local_inputs.py`: verify every supplied file, copy through a verified `.partial` directory, reuse valid caches,
  or repair only damaged files. Python 3.9+ standard library; no network/model download.
- `qwen35_cases.json`: fixed tokens for prompt 1/17/128 plus 8 teacher-forced steps; 257-token state case.
- `qwen35_reference.py`: strict text-only local weight mapping with Transformers 5.3.0, Torch 2.8.0+cu129 and
  Safetensors 0.8.0. Saves each layer's last-token hidden state, full last-token logits and final conv/recurrent/KV
  state; preserves the executed recipe/source hashes. Eager attention and auditable Torch delta formulas.
- `qwen35_quantization.json`: exact 132 linear tensors and 188 preserved tensors, Q4-S128/Q8-B32/F8-B128 formulas.
- `prepare_quantized.py` / `quantization.py`: canonical CPU-only offline codes/scales, input/recipe hashes and
  per-tensor weight error; NumPy 2.1.2. Preserve the original high-precision tensors by source identity.
- `inputs_test.py`: small filesystem, path, matrix, rounding/format/scale tests. It does not need model weights or
  run inference. NumPy codec checks apply on the canonical Windows generator; standard-library checks run on Macs.

## Commands

```powershell
python tests/compute_models/inputs_test.py
python tests/compute_models/local_inputs.py verify --inventory docs/implementation/tasks/compute/qwen35-2b-local-input.json --source C:\Users\kiro\Downloads\Qwen3.5-2B --receipt C:\hyle-scratch\model-input.json
python tests/compute_models/qwen35_reference.py --source C:\Users\kiro\Downloads\Qwen3.5-2B --inventory docs/implementation/tasks/compute/qwen35-2b-local-input.json --cases tests/compute_models/qwen35_cases.json --output C:\hyle-scratch\new-b16-reference --device cuda
python tests/compute_models/prepare_quantized.py --source C:\Users\kiro\Downloads\Qwen3.5-2B --inventory docs/implementation/tasks/compute/qwen35-2b-local-input.json --contract tests/compute_models/qwen35_quantization.json --profile Q8-B32 --output C:\hyle-cache\new-q8-artifact
```

Use a new output directory; existing derivatives are reused by receipt and hash, not regenerated per machine.
Synchronize the same generated artifacts when their capability package needs them. GPU reference and HYLE runs are
serial and have independent process lifetimes. The generator holds compressed output plus a bounded per-matrix
floating workspace on the host; this is not HYLE's shipping memory policy.

Q8 uses W8A16 and F16 weight dequantization with F32 accumulation. It does not quantize activations to INT8.
Scales that underflow to F16 zero canonically have zero codes; every decoded value is zero. E4M3FN codes preserve
signed zero and saturate finite values; the local FP8 profile uses F32 scales, independently of UE8M0 target-model
profiles. Report weight representation error, quantized model quality and HYLE implementation error separately.

The installed reference's cached multi-token DeltaNet path resets initial state; do not use it as an oracle for
chunk continuation. C3 must compare the explicit chunk/recurrent formulas with nonzero state and full-sequence
results. These local B16 reference cases use a single initial prefill followed only by single-token updates.
