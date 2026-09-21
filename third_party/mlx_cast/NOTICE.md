# MLX BF16 conversion integration

Upstream MLX (MIT; see LICENSE):
https://github.com/ml-explore/mlx at
`1f8e74e3f12f31365464a6867c6579f0e9b29d85` (0.32.2).

The exact `float_to_bfloat_bits` function is borrowed without changes from
`mlx/backend/metal/kernels/bf16.h` at upstream
`726dbd926770b7c78cd23480e71f1a21c07a9bc0` (v0.20.0):
https://github.com/ml-explore/mlx/blob/726dbd926770b7c78cd23480e71f1a21c07a9bc0/mlx/backend/metal/kernels/bf16.h
The source file SHA-256 is
`a56b2cc67ad44e8faa1a20418feae0af1e00eeed76c84016de58e0c06f4f236c`.
The current upstream host BF16 type uses the same integer rounding method.

On the qualified Apple GPU, native F32-to-BF16 conversion flushes subnormal
inputs even with safe math. Returning native bfloat after software conversion
also loses subnormal results. The patch adds the original conversion function
and connects only the F32-to-BF16 copy pair to it with an integer storage port.
The original copy loops, array dtype, layout and owner remain unchanged.
Ordinary numeric F32-to-UInt16 casts have a different native library identity.

Compiled fusion stops at this F32-to-BF16 storage boundary so the original copy
primitive executes. This may add a native dispatch and materialized storage;
all other operations remain eligible for upstream fusion. There is no host
payload transfer, new HYLE kernel or new conversion algorithm. The source and
cost change is explicitly part of the owning Apple route's validation scope.

The owning Apple CMake authenticates this manifest and upstream source before
using the existing exact-patch materializer on a build-local copy. Shared cache
originals are never changed. Edited files carry before/after hashes;
upgrades require re-audit. Source, static and shared consumers receive the same
patched runtime, original MIT license and this notice.
