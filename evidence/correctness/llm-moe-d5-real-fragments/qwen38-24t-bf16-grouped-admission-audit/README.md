# Qwen3.8-2.4T BF16 grouped expert admission audit

## Verdict

`AUDITED_FIX_REQUIRED`。固定官方 profile 需要 BF16 dense expert weight + F32 activation/output and
accumulation + I32 expert ids。成熟 ggml 四后端已有 `GGML_OP_MUL_MAT_ID` 的该域；HYLE 公共 schema
以及 CPU、CUDA、Vulkan adapter 在进入 upstream 前把 weight 人为限制为 F32。

## Fixed consumer

- model: `Qwen/Qwen3.8-2.4T-A95B`
- revision: `207bd685a7e3696cfaff12ded7c6a7ea0f88c996`
- config SHA-256: `4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987`
- `model.safetensors.index.json` SHA-256:
  `e36c40d4e99b2714fff821218a0433bda2dec46afdb1ebee8ce96ced997928ee`
- official header: 1,609 BF16 tensors / 2,446,182,725,504 elements
- shape: H=8192, 512 experts, top-10, routed/shared expert intermediate=2048
- source: `docs/implementation/tasks/compute/llm-moe-model-weight-operator-audit.md`

## HYLE rejection boundary

- `include/hyle/compute/model_ops.h`: `GroupedMatMul` requires F32 weight/F32 activation/I32 ids and
  always infers F32 output.
- `src/compute_provider_ggml/ggml_cpu_provider.cc`
- `src/compute_provider_ggml_cuda/tensor_program.cc`
- `src/compute_provider_ggml_metal/native_program.cc`
- `src/compute_provider_ggml_vulkan/region.cc`

All four adapters lower the accepted operation directly to `ggml_mul_mat_id`. CPU, CUDA and Vulkan duplicate the
F32 dense-weight restriction; Metal already maps plain BF16 but is stopped by the shared public inference call.
This is a schema/adapter capability gap, not missing grouped-matmul math.

## Pinned upstream evidence

Pinned source: `ggml-456172ec733a-41b7e10fd424`.

- `src/ggml.c:3392-3413`: `ggml_mul_mat_id` accepts the expert tensor, activation tensor and I32 ids and
  creates an F32 result.
- `tests/test-backend-ops.cpp:10029-10030`: upstream registers BF16 weight + F32 activation
  `test_mul_mat_id` cases.
- `src/ggml-cpu/ggml-cpu.c:2892-2903`: CPU selects the BF16 weight type trait's vector-dot type and
  converts the activation only when needed.
- `src/ggml-cuda/ggml-cuda.cu:5124-5132`: CUDA admits contiguous `MUL_MAT_ID` with F32 activation; the
  BF16-weight/F32-activation pair does not hit the F16-only rejection.
- `src/ggml-metal/ggml-metal-ops.cpp:2604-2608`: Metal's native matrix dispatch explicitly includes BF16
  source weights; `GGML_OP_MUL_MAT_ID` uses that operation.
- `src/ggml-vulkan/ggml-vulkan.cpp:19220-19269`: Vulkan explicitly lists BF16 source weights and accepts
  F32 activation; only BF16-weight/F16-activation is rejected.

## Frozen boundary

The repair may admit **plain dense** BF16 expert weights while retaining F32 activation/output/accumulation and
I32 ids. Existing GGUF packed weights keep their F32 logical type contract. It may add provider and real-profile
tests. It may not change upstream ggml, add conversions or fallback, broaden activation/output dtype, alter
routing, or introduce HYLE math/kernel/shader code.
