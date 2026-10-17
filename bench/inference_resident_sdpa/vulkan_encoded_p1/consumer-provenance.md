# Vulkan encoded-page attention fragment

Finite G04/G05 Q8_0 K/V, F16 Q/output, page16 and GQA8/2: Q1/17 KV1024/768
D256 window16; Q64/128 with the same KV/D256 and unbounded causal attention;
Q17/1 KV7/47 Dqk64/Dv128 full noncausal, including Q>KV. These are provisional
single-machine inference archetypes, not measured whole-model frequencies.
Physical pages are reversed with one shared page; unused pages/tails are poisoned.
The double oracle decodes the actual independent stored Q8 scales/codes and
valid region. Each arm checks finite output within2e-3, four replays over two
slots, and exact readonly bytes. Numerical qualification is required before A/A.

Both arms use the original native controller, exact per-page F1 permissions,
F2 table upload and checked completion/output owners. In the current source,
every operation includes checked prepare/version, idle input release, layout
update, bind/submit/native wait and output owner return/release. The controller
creates a native Program/Invocation on first prepare or query-topology change;
this benchmark repeats one fixed layout and reuses that plan. Metadata upload
initializes both table banks once and then transfers changed page-index bytes
only, with a checked wait when a transfer occurs. An unchanged table needs no
new transfer. Submit still consumes the prepared episode.
Only initial immutable payload placement, schema/layout and outer slot creation
are outside timing; final context close occurs after all results are released.
The reference embeds logical gather indices in its prepared recipe; this fixed
layout benchmark does not qualify reference reuse with changing KV layouts.
These current-source notes do not alter historical contracts or raw results.

Candidate attention directly reads physical Q8 pages. Peer uses unchanged
original456 non-MMQ scalar Q8 flash shader and existing original GetRows I32
opaque bit-copy, with independent K/V encoded row widths. Each request creates
and uploads one logical-row to physical-row index array during cold plan
preparation. Both K/V normalization and generated output Copy remain charged, as do compressed
gather, F16 causal/window masks, Q/output conversions, split/reduce and joining.
This explicitly selected original direct-Q8 pipeline is not a best-of original
host comparison; original host may select MMQ or a full-history F16 expansion.

Per operation the two D256 cases gather1949696 encoded bytes; their masks are
28160 and327680 bytes. The D64/Dv128 holdout gathers22032 bytes without a mask.
Exact generated dispatch/allocation counts await actual preflight. Native
private-pool retained bytes, bucket rounding and hard upper remain UNKNOWN.

The qualified relocated static SDK is used without rebuilding product libraries.
Its29 HYLE artifacts match sealed G04/G05 delivery hashes. Three native archives
retain both unequal whole-file hashes and identical ordered member payloads
(152/9/15 objects); container/header/TOC equality is not claimed. SDK CPU provider
predates CE12; this GPU comparison does not qualify CE12 CPU execution. Static
benchmark linkage makes actual executable hashes part of the unified evidence.

The original instrument, machine policy,16 paired rounds, workload-matched
16-process A/A, execution/allocation orders, placement replicas, n15/31 schedule
and5% noninferiority remain fixed. P1 starts only after numerical preflight,
immutable inputs and A/A qualify. No capture, incremental producer/growth,
G06, whole G04/G05 or end-to-end token/s claim follows from this contract.
