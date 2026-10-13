# CPU ragged two-request comparator

Finite G01 consumer: ordinary CPU `BatchProgram`, BF16 packed NHD, GQA8/2;
mixed Q1/17 KV1024/768 D256 causal, prefill Q64/128 at the same KV, and the
Q1/17 KV31/47 D64 noncausal holdout. Capacity tail13 is a 128 sentinel.
The independent double oracle has the original absolute bound8e-3, finite
output check and readonly K/V check; it is outside the timed region.

Both arms use the original `BatchProgram`/`BatchInvocation`/`PreparedBatch`,
the same host resource budgets, two independent slots, exact layout owner,
ready Q/K/V leases, original metadata staging/upload, F1 uses, CPU Queue,
checked Completion and returned output lease. Each measured episode explicitly
updates an identical full topology before running it; both retain the compiled
native plan and release the completion/token/output before slot reuse.

The private comparator decorates the original CPU provider only at its batch
extension. Ordinary leaf/copy configuration, creation and teardown are original.
It constructs pinned ggml `FLASH_ATTN_EXT` graphs independently and never calls
the candidate compile/lower or its native batch math. Packed original K/V are
permuted as views and read directly, Q is cast to F32, accumulation is F32,
the causal mask is F16 and output is BF16. Original `CpuExecution` supplies one
native thread. Known arena/payload/mask/workspace is reported cold; native private
pool bounds are UNKNOWN in both arms. No custom attention arithmetic is added.

The authenticated peer checkout is ggml `ffa4e8b80930029a35991f94e7c8a93cd67730ab`;
the product's locked CPU runtime uses this original flash implementation with
unrelated IQ1 extensions. The manifest binds the actual linked runtime archive,
compile recipe and source identities, rather than calling a separate executable
with a different ownership contract. This establishes a numerical/ownership
comparison, not whole-engine throughput or arbitrary paged support.

The frozen contract uses16 paired rounds, one minimum epoch iteration, 4ms
minimum epoch time, original machine policy, both execution/allocation orders
and a code placement artifact. A/A and the original qualified controls precede
product samples n15 and, only if required by the existing stop rule, n31; 5%
noninferiority does not erase smaller directional differences. Preflight cost
and its raw failures are kept separately from qualified product evidence.
The min2 diagnostic process cost15.641s; a long prefill costs169ms, already
above the4ms floor. That diagnostic helper overwrote the min2 raw log on its
second run; its actual15.641s tool result remains recorded, and it cannot be
replayed as qualification. The min1 raw process cost9.422s is retained and the
helper now appends attempts. The min1 recipe is fixed before qualification, preserving
all16 rounds, original process counts and controls rather than repeating two
long executions merely to fill an epoch whose timer floor is already met.
