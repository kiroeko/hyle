# Apple parity: public source/static/shared delivery sample

Consumer candidate `8e95478c822326b0ddcb3ecb43acb511b8bef7f6`,
Windows canonical, complete Git bundle forwarded to MacBook Air Mac16,12.
Air M4/macOS27.0.1/AppleClang21/SDK27.0, deployment14.4, AC and lowpower0.
This is a targeted current delivery witness, **not whole Apple goal closure**.
Official FP4 and two official DSpark numerical witnesses remain open.

The same [public consumer](../../../tests/compute_route_apple/parity_delivery/main.cc)
and [CMake](../../../tests/compute_route_apple/parity_delivery/CMakeLists.txt) link
only `hyle::graph` and stage the Metal resource through the public helper.
There are no private HYLE/vendor include paths, mock providers or CPU model
fallbacks. Both ordinary Direct and Graph execute nonzero original Q4_K
MatMul with two changed-input replays, exact selected rows, checked index
failure and retained results after close, plus original mature NF4 decode.
This SDK sample complements owning full numerical matrices; it does not
rerun559/34/45 or certify untested official FP4.

| Delivery | Authoritative successful evidence |
|---|---|
| shared relocated SDK | [receipt](receipt-shared-confirmed.json), [configure](shared-consumer-configure.log), [build](shared-consumer-build.log), [runtime](shared-runtime.log), [commands](shared-commands.log) |
| embedded source | [receipt](receipt.json), [configure](source-configure.log), [build](source-build.log), [runtime](source-runtime.log), [commands](source-commands.log) |
| static relocated SDK | [receipt](receipt.json), [install](static-install.log), [configure](static-consumer-configure.log), [build](static-consumer-build.log), [runtime](static-runtime.log), [commands](static-commands.log) |

Both installed SDKs are moved to independent prefixes before consumer
configure. Configure uses disconnected FetchContent and unreachable local
proxy endpoints; no SDK source/cache download is needed. Their compiler/link
commands exclude canonical include/src and producer paths. Runtime hides
producer bin and all producer Metal libraries, clears inherited DYLD settings,
uses the staged Metal resource and records loaded dylibs. All hidden assets
are restored in finally. The SDK and staged `mlx.metallib` hashes match:
`2ae3af693d8e941861063c9f3ae5249e17665e8761837a9d87ab6e07002f568c`.
All new NF4/AWQ/CT/Q4_K/IQ1 license/source notices are present in the receipts.
The [task-local batch driver](run.py) is recorded; public governance/performance
tools were not rewritten.

Shared producer was already built from unchanged current product sources.
The consumer-only commits after `a8809428` do not change include/src/cmake/
third_party/root CMake inputs, so the installed SDK is reused for those
corrections; no repeat producer build or install. Source embedding reuses the
existing source producer; its actual incremental build took58.95s, installed
static consumer build0.96s. Heavy work stayed serial on Air, build2.

Preserved failures and one corresponding correction per named issue:

- [tests-on configure](shared-configure-tests-on.log): shipping install requires
  tests/instrumentation OFF. [Shipping configure](shared-configure.log) and
  [producer build](shared-build.log)/[install](shared-install.log) then pass.
- [first consumer compile](shared-consumer-build-first.log): missing public MLX
  transforms declaration and explicit Operation construction. Corrected only
  the consumer; SDK untouched.
- [first runtime](shared-runtime-dimensions-first.log): consumer reversed public
  GGUF K/N parameter order; no operator ran. Corrected only that argument pair.
- [source compiler refresh](source-configure-compiler-refresh.log): CMake changed
  compiler paths and refreshed the old cache, clearing first-call variables.
  One same-argument configure on the refreshed cache passed; no product fix.

These fixture/tooling failures remain evidence; none is mislabeled a product
math failure or counted as a passing whole model. A4 full header/boundary
matrix/final cleanup and A3 default/cost verdicts still need their own evidence.

## Directed CPU + Metal shared-core compatibility

Candidate21bcb41f, same current product inputs. [Receipt](mixed-core-21bcb41f-receipt.json),
[configure](mixed-core-21bcb41f-configure.log), [build](mixed-core-21bcb41f-build.log),
[actual commands](mixed-core-21bcb41f-commands.log), [CPU](mixed-core-21bcb41f-cpu.log)/
[XML](mixed-core-21bcb41f-cpu.xml)9/9 PASS and
[Metal](mixed-core-21bcb41f-metal.log)/[XML](mixed-core-21bcb41f-metal.xml)8/8 PASS.
The task-local [driver](run-mixed-core.py) selects only runtime+ggml Metal
maintainer factors, CPU/Apple ON, other GPUs OFF, shared/instrumentation OFF,
build2 and serial tests. No CPU product code/review or P1 was reopened.

The six actual selected header/core/quant/Metal source hashes exactly match
Apple-only. CPU tests execute original three narrow IQ1 MatMul/GetRows and
metadata/workspace/lifetimes; Metal tests execute original native packed
MatMul/grouped/recurrent programs and checked resource boundaries. The new
Metal IQ1 actual eight contracts retain earlier matching-source numerical
evidence, while mixed compilation and native admission are confirmed here.
This is directed source/backend compatibility, not whole mixed-model support.
The GgmlVulkanMetadata suite name refers only to shared arena tests; no Vulkan
GPU verification is claimed.

The [first configure](mixed-core-21bcb41f-configure-first.log) succeeded;
the script then rejected multiple cached source directories before compiling.
Source identity now comes from actual ggml-base Ninja commands, not every
old cache directory. This task-local correction did not change product input.
