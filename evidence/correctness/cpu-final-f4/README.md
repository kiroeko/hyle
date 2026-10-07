# CPU final F4 named-provider preflight

Task: [F4](../../../docs/implementation/tasks/compute/cpu-final-f4-best-of.md).
This page contains capability/numerical preflight, not a performance WIN or
self-retirement verdict.

The current Windows static SDK was exercised with four named factories using
the same descriptions, bytes and terminal completion: 453 requests, 767
admitted pairs, no numerical mismatch. The 33 single-carrier requests all use
ATen and do not require a timing race. Current versions: oneDNN3.13.4, ggml
ffa4e8b8 plus original IQ1 port, locked LibTorch2.8 CPU. All actual declines and
caller/private resource facts remain in the raw log.

Two early consumer limitations are retained: buffered stdout/stderr split a
metadata line, then the explicit ggml consumer lacked its already exported SDK
implementation target. Ordinary CPU routing already used ggml; the developer
consumer was corrected without changing product/build governance or math.
Those incomplete rows never qualified a WIN or ggml loss.

[Receipt and exact scope](windows-named-provider-preflight-receipt.json),
[initial/current raw logs, admissions/pairs, consumer and drivers](windows-named-provider-preflight.zip).
F4 A/A/controls, immutable P1 and full region/other-platform verdicts remain
pending. F1/F3 proof supplies the existing carrier correctness boundary;
this short preflight does not expand it or qualify arbitrary input values.
