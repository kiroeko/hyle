# CPU final F4 bounded provider consumer

Owning task: [F4](../../../docs/implementation/tasks/compute/cpu-final-f4-best-of.md).
This is development qualification code; ordinary users still select CPU only.
The development consumer explicitly links the SDK's already exported ggml
implementation target to receive its factory feature definition; ordinary
CPU routing already includes that implementation through its private closure.

`--probe` prepares the exact same descriptions and data with independently named
Native/oneDNN/ggml/ATen factories, records real declines and resource facts, runs
accepted programs and prints pairwise numerical checks. A decline is not a
performance loss. MatMul/SDPA have no self candidate; borrowed Native programs
are tracked separately by the task. This consumer does not add operator math.

Normal/holdout use real inner widths 2048/4096, token/small prefill 1/32, F2
F32 projection K=16384/N=4 or 24, and Qwen BF16 projection K=2048/N=6144.
Dense and physically transposed weights are separate requests. All conversion,
copy, temporary storage, checks and true terminal completion remain charged.

`--pair <case> <candidate> <peer>` measures one prequalified pair with the existing
nanobench paired adapter. Allocation/order strata use existing environment
controls. It may only run under the frozen F4 contract and unified runner after
preflight; the CLI alone does not produce a WIN, PASS or product certificate.
Single-carrier domains receive correctness/resources/P0, not a timing race.

The final same-process batch reads a fixed pair-list input asset. Its exact path
and SHA are bound by the contract and module validator, as with a model fixture;
it is not an online provider search or measurement-control override. Common
runner policy still owns OMP/MKL and caller affinity. The runner copies the
already qualified batch image/DLLs and validates their hashes before sampling.
The retired per-pair process transport is retained only in evidence; its changed
process lifecycle and 99.281-second cost did not acquire performance qualification.
