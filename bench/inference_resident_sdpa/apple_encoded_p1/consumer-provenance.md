# Apple encoded-page attention fragment

Finite Q8_0 K/V, F16 query/output, GQA8/2, page16: Q1/17 and KV1024/768,
D256 window16; Q64/128 with those KV and D256 unbounded causal; Q17/1,
KV7/47, Dqk64/Dv128 full noncausal including Q>KV. These are named inference
fragments, not measured model frequencies. Shared/reversed physical pages,
inactive poison, exact valid rows and stored Q8 scales/codes are identical.
Both arms require the independent double oracle within2e-3, finite output,
four replays/two slots and exact readonly bytes before measurement.

Both arms retain the original native controller, exact per-page F1 guards,
F2 table upload and owned native completion/output. Every measured operation
includes prepare/version, native Program/Invocation replacement, allocation,
upload/wait, binding, submit, native completion and output return/release.
Only initial immutable placement/schema/layout and outer slot creation are
outside timing. Initial device library/pipeline compilation and cache warmup\nare inside the mandatory untimed numerical/warmup phase for both arms. Each\nsubsequent complete episode still charges native preparation. There is no\nprivate flag reset or warm replay shortcut.

The independent peer instantiates the unchanged original456 general Q8 Metal
template, pad and mask-block kernels. Its original64/64,128/128,256/256 exports
are retained, with one explicitly disclosed64/128 template instance. Arithmetic
is unchanged: half tiles and F32 score/softmax/output accumulation. This is
explicit general-kernel selection, not the original host's best route; vector,
split/reduce and full-history F16 expansion routes are not selected.

The original MLX CommandEncoder binds actual arrays/buffers/offsets, tracks
dependencies and retains scratch before the first accepted dispatch. Original
HYLE synchronization remains charged; after it succeeds the peer additionally
checks retained real command-buffer status/error. No host-pointer remapping,
unchecked async status, CPU fallback or hidden ingress synchronization is used.

Every peer operation charges Q-to-F32, per-request logical-row index creation,
GPU opaque Q8 gather, contiguous head-major K/V, a padded F16 causal/window
mask, original tail pad/mask-block/general launches, F32-to-F16 output and join.
Gathered bytes are1949696 for each D256 case and22032 for the holdout. F16 mask
storage is53248bytes for the window case and327680 for prefill; query rows round
to8 for the original block kernel. Noncausal has no mask. The short-K holdout
uses original K/V tail scratch,52224bytes across two requests. Native private
pool cache/rounding/retained bytes/hard upper remain UNKNOWN.

Use only the independently tested relocated Apple static SDK. Its28 HYLE
archives,6 native archives,10 recipes and1 metallib are authenticated. Each
native archive differs whole-file from its producer, with exact ordered
member payloads (9/15/28/2/1/120 objects). No cause for metadata differences is
inferred. This GPU-only comparison does not qualify that SDK's older CPU path.

Original machine policy,16 paired rounds,16 workload-matched A/A processes,
execution/allocation orders, placement replicas, n15/31 and5% noninferiority
remain fixed. Actual numerical preflight, immutable source and A/A are required
before P1. No producer/growth, capture, G06, whole goal or token/s claim follows.
