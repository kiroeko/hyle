# Unified Compute C3 partial delivery receipts

This is ongoing validation for the [active delivery card](../../../docs/implementation/tasks/build/compute-unified-delivery.md).
It is not a terminal verdict for all platforms/routes, hybrid execution, or asynchronous expert transfers.

[Delivery matrix](delivery-matrix.json) fixes the declared applicable scope: 15 platform/route groups,
45 source/shared/static shapes, each with the existing Direct and Graph consumer. Consumer entries
are documented per actual receipt; the current matrix records 68 PASS and 22 UNRUN.
Earlier Vulkan failures remain archived. A mixed SDK entry runs each selected native route independently
and cannot certify the later joint execution or asynchronous transfer work.

[Conditional source-asset producer counterexample](apple-conditional-producer/receipt.json) records a
C2-01 build-graph defect: an inactive link suppresses copying but still builds its source asset owner.
This is a CMake fixture, not a Metal or math verdict. The [applied product correction](apple-conditional-producer/product-fix-receipt.json)
passed the metadata regression and actual conditional cross-directory asset builds on Windows Ninja and
Air Unix Makefiles. Existing non-Apple math receipts retain their original product identities; actual
Apple native delivery remains unrun.
The [disposable correction experiment](apple-conditional-producer/design-experiment/receipt.json) passed
inactive/active source asset cases on Windows Ninja and Air Unix Makefiles, with the real library owner
and generated asset in a separate directory. It validates the finite dependency approach, not an applied
product fix or native Metal execution.

- [Windows CPU receipt](windows-cpu-receipt.json): six actual public math consumers passed,
  covering Direct and Graph in embedded source, relocated shared SDK and relocated static SDK.
- [Per-entry result](windows-cpu-partial.json): exact observed test logs and distinct product identities;
  the later shipping-test-default correction does not change these explicitly tests-OFF math recipes.
- [Archived logs and package metadata](windows-cpu/): CTest logs, real HyleConfig outputs and three bare configuration checks.
- [Actual SDK requirement checks](windows-sdk-requirements.json): shared/static installed SDKs accept CPU,
  reject a requested absent CUDA route and reject a retired Provider option. Exported target route/recipe
  properties agree with package variables; these six configure verdicts required no rebuild or P1.
- [Actual SDK offline and missing-dependency checks](windows-sdk-offline.json): eight configure cases passed
  across the same relocated shared/static CPU SDKs. A populated offline cache is accepted; an empty offline
  cache, invalid explicit Torch root and missing original ggml runtime are rejected with their specific
  diagnostics. [Logs](windows-sdk-offline/) preserve the observed outputs; no native rebuild or P1 was run.
- [Windows CUDA receipt](windows-cuda-partial.json): the embedded-source and relocated shared/static SDK
  Direct and Graph FFN consumers passed on the actual RTX 4070. [Logs and package metadata](windows-cuda/)
  preserve their six CTest verdicts. Required DLLs are adjacent to the EXEs; the CUDA-only producer excludes
  the HYLE CPU backend and oneDNN. The batch completed with exit 0. Shared export hygiene passed for
  18 HYLE libraries and 985 exports.

The producer used MSVC 19.50.35728.0, full CPU route, locked oneDNN 3.13.4, ggml CPU core/port and
LibTorch 2.8.0+cpu. Source/static/shared share capability semantics, while recipe identities reflect linkage/configuration.
Actual runtime staging placed DLLs beside each EXE. Static HYLE still requires the qualified dynamic Torch runtime.
The required Torch runtime remains a locked machine SDK acquisition, not part of the HYLE archive; offline
consumption needs its populated cache or a qualified local root/archive. Route requirements do not remove it.
The shared SDK export check inspected 17 HYLE libraries and 1289 exports; no new P1 was run.

Windows CPU and CUDA three-delivery consumer batches are complete. Remaining route/platform/mixed SDK entries have no PASS here.
[Windows Vulkan source failure receipt](windows-vulkan-failure/receipt.json) preserves both actual failures:
the sample's old native-dispatch count and the 16-row F32 FFN numerical mismatch. An independent host
rounding bill matches the observed GPU result after narrowing the down-projection activation to F16;
disabling cooperative-matrix variants does not resolve it. The [bounded completion receipt](windows-vulkan-partial.json)
now signs source Direct/Graph PASS after binding the original full-F32 shaders and preserving explicitly
requested F32 operands before cooperative-path conversion. The original numerical gate is retained.
The relocated shared/static SDK Direct/Graph FFN consumers also passed; the resumed SDK loop exited 0.
[Additional F32 range/transposed/replay receipt](windows-vulkan-range.json) covers all three deliveries after
only the needed small consumers were rebuilt. SDKs were not rebuilt for that test addition. Existing native
resource and other-dtype regressions passed in [22 selected cases](windows-vulkan-regressions.json);
[eight matrix/ID and quantized regression cases](windows-vulkan-quant-regressions.json) also passed, including
the new plain-F32 grouped matrix path with non-F16-exact and out-of-F16-range operands. These subsets had
no fixture skips. Other platforms remain unrun. No self multiplication is added.
Read-only verification copies use a 42,956,800-byte source-only archive from the Windows product candidate
79137d31, SHA256 `6ca08a1ea6728ee340267b5fa10e0f1b9c681e726f2bf407cf2abadd8d0ffefe`.
The archive excludes large historical evidence and is not a reverse source of product changes.
The remaining WSL/Air entries now use fresh `source-2614676e` directories from the corrected product:
42,967,040-byte source archive, SHA256 `94cd4aff04573073ffc848ecf71d67933ed705d03160ba93f8ce1b3da64b1bcc`.
Both received archives were checked against that hash; no native math PASS follows from transport.
The [Air Apple partial receipt](air-apple-partial.json) signs actual source Direct/Graph consumers on
Mac16,12 Apple M4. Shared/static also passed and the serial batch exited 0. Its newer complete validation
source snapshot is c8874688, archive SHA256 `3770e2f2147120f527ff96a45af4a096ed846c791624e720582cb2eb4a88d4f5`;
product c96c7c6e and test/document-only follow-up identities remain distinct.
The [Air Vulkan receipt](air-vulkan-partial.json) covers six actual source/relocated shared/static consumers
after route-owned default loader discovery, without DYLD overrides. Its original loader failure remains
archived. Product a1c22ac3 and validation snapshot 55652ab2 are bound to the complete archive hash; only the
changed owner TU and links were rebuilt for the source correction.

The [Air CPU+Apple configuration failure](air-mixed-license-failure/receipt.json) occurred before
compilation: both routes registered the same DS4 source lock. The original configure log is retained;
this does not assign a mathematical consumer verdict. The bounded
[shared license registration repair](../../../docs/implementation/tasks/build/compute-mixed-shared-license-registration.md)
changes only which selected component owns the package license entry.
The [Air CPU+Apple receipt](air-cpu-apple-partial.json) now signs six actual ordinary Direct/Graph
consumers across source and relocated shared/static SDKs. The existing SDK loop completed with exit 0;
the original configuration failure remains separate. This validates both native routes within one
complete package, not H2 joint execution, overlap or engine-directed expert transfer.

[WSL native Vulkan device facts](wsl-vulkan-machine-facts.json) show only llvmpipe, physical device
type CPU. NVIDIA CUDA reports the RTX 4070 separately; this does not expose it to the Vulkan loader.
The ordinary Vulkan route disallows software devices. SDK compilation remains applicable; real GPU
math is unqualified on this machine, consistent with the declared platform limit. This device query
is not a shader, resource, operator or consumer PASS and does not increase the delivery matrix count.

The [Windows CPU+CUDA partial receipt](windows-cpu-cuda-partial.json) signs source Direct/Graph
real math PASS on the actual RTX 4070 and CPU. The original serial driver continues shared/static SDK
stages; source objects and results are retained. Combined delivery does not certify H1 joint execution.

**Current correction:** four prior Air CPU+Apple SDK claims are withdrawn: both actual SDK configs
contain only CPU. Their raw math PASS observations are retained, but do not qualify a mixed package.
The effective matrix is 34/90 PASS. Windows CPU+CUDA shared stopped on missing CUDA after the same
argument loss. [Original claims, metadata and failure](mixed-sdk-route-failure/receipt.json) are preserved;
the [bounded tool repair](../../../docs/implementation/tasks/build/compute-delivery-route-argv-repair.md)
must pass before the original SDK stages resume. Source mixed consumers remain valid.
The [archived SDK metadata audit](mixed-sdk-route-failure/archived-sdk-route-audit.json) checked all
twelve previously saved configs: ten single-route SDKs match their requests; the two already withdrawn
Air mixed configs do not. This inspection changes no math verdict and requires no native rebuild.
The corrected Windows CPU+CUDA shared SDK now passed both relocated ordinary consumers with exact
`cpu;cuda` producer/package metadata and DLLs adjacent to the EXEs. Its partial receipt preserves the
new tool identity separately from the unchanged product. Effective C3 count is now 36/90; static and
the four withdrawn Air mixed SDK entries remain unrun.
Windows CPU+CUDA static Direct/Graph also passed, and the original fixed SDK loop exited 0. All six
ordinary consumers for this group are now qualified; the effective C3 count is 38/90. Source, shared
and static results retain their distinct product/tool/linkage identities. The corrected Air mixed SDK
loop has resumed serially with exact `cpu;apple` arguments and producer metadata; its four entries
remain unqualified until the real consumers run.
The corrected Air CPU+Apple shared SDK now also passed both ordinary math consumers with exact
`cpu;apple` metadata. Its new files are under `air-cpu-apple-fixed`; the earlier CPU-only logs/configs
remain untouched. Effective C3 count is 40/90; the Air static pair is still unrun. The unchanged
Compute+Graph implementation union measures 62,508 code lines at immutable 5e94f4cd, self-check PASS.
Air corrected static Direct/Graph also passed and the SDK loop exited 0. CPU+Apple now has six
qualified ordinary consumers; the tool repair is closed. Effective C3 count is 42/90. All corrected
SDK artifacts keep exact `cpu;apple` metadata; the original CPU-only records remain separate.

The [Air CPU receipt](air-cpu-partial.json) signs two newly run source consumers and reuses four actual
CPU-only SDK observations. Fresh shared/static configuration matches their respective route, full
feature set and recipe exactly; product headers/source/locks/build inputs, machine and math contract
are unchanged. The old artifacts are not relabeled with a new source identity. This avoids rebuilding
the same SDKs and does not transfer a best-of WIN to a different domain. Effective C3 count is 48/90.
The [Air CPU+Vulkan partial receipt](air-cpu-vulkan-partial.json) now signs the two actual source
consumers with both native routes enabled. Shared/static remain unrun; the serial original batch
continues. Effective C3 count is 50/90, with no H3 joint/async/expert claim. Completed Windows mixed
SDK build objects were removed after archived hash and process checks, freeing 4.05 GiB while keeping
both relocated SDKs, source/consumer artifacts and dependency caches.
Air CPU+Vulkan shared also passed both relocated ordinary math consumers with exact `cpu;vulkan`
metadata. The partial receipt and new logs bind actual identities. Effective C3 count is 52/90; static
remains unrun, and H3 joint/async/expert execution is not certified by this delivery batch.
Air CPU+Vulkan static also passed both ordinary consumers with exact routes, and the original driver
exited 0. All five declared Air route groups now have source/shared/static qualification. Effective
C3 count is 54/90; Windows CPU+Vulkan and WSL groups remain unrun, and H1-H3 remain unimplemented.
The [Windows CPU+Vulkan partial receipt](windows-cpu-vulkan-partial.json) now signs its actual source
Direct/Graph consumers with both native routes enabled. The original serial driver continues
shared/static; effective C3 count is 56/90. No H3 joint/async/expert claim follows from this math batch.
Windows CPU+Vulkan shared also passed both relocated ordinary math consumers with exact routes and
DLLs adjacent to the EXEs. The partial receipt and logs bind the actual identities; effective C3 count
is 58/90. The original driver continues static; source/shared are not rerun.

Windows CPU+Vulkan static also passed both ordinary math consumers (Direct 1.15s, Graph 1.03s),
with exact routes and DLLs adjacent to the EXEs. The original full batch exited 0; archived static
logs/configuration complete this group. C3 now records 60/90 PASS, covering all five declared Windows
and Air route groups. WSL batches and H1-H3 joint/async/expert work remain unqualified.

The [WSL CPU source receipt](wsl-cpu-partial.json) signs Direct/Graph PASS after the bounded
target-local Linux native-program loader repair at `a2476b62`. Its verified source archive and original
failure/diagnostic remain separate. The original ordinary consumer succeeds without LD overrides;
temporarily missing the owned native MODULE fails before admission and restoring it succeeds.
C3 now records 62/90 PASS. WSL shared/static and other route batches, and H1-H3, remain unqualified.

[Installed Windows header receipt](windows-installed-headers/receipt.json) records the existing
bare-header checker on all installed Compute/Graph top-level, backend and Provider headers from the
CPU+Vulkan and CUDA shared SDKs: 63 headers in ten groups passed. It binds actual compiler and SDK
metadata; no SDK rebuild, static-macro, other-platform or mathematical qualification is inferred.

[Air installed-header evidence](air-installed-headers/receipt.json) passed 60 bare headers, and
[actual active-route checks](air-installed-headers/active-receipt.json) separately compiled each of
those headers with the installed `hyle::graph` usage requirements. Apple/Vulkan macros were confirmed
in the compile commands; MLX and Vulkan declarations were active. This does not sign math or P1.

[Windows active-route header evidence](windows-installed-headers/active-receipt.json) separately
compiled all63 installed headers with actual exported target usage requirements (CPU+Vulkan32,
CUDA31). Route defines were confirmed in compile commands. Both groups passed without rebuilding SDKs.
[Completed producer object cleanup](windows-cpu-vulkan/cleanup-build-objects.json) freed4,464,806,780
bytes while retaining SDKs, consumers, source artifacts, dependency cache and evidence.

The WSL CPU shared SDK's relocated Graph0.17s/Direct0.15s consumers also passed; updated
[per-entry receipt](wsl-cpu-partial.json) binds actual config and logs. Its shared runtime has sibling
`$ORIGIN`, the native CPU MODULE is in SDK/lib, and export hygiene passed17 libraries/1354 exports.
The original static loop continues; effective C3 count is64/90, with no hybrid/P1 claim.

[WSL CPU installed-header qualification](wsl-cpu-installed-headers/active-gcc14-receipt.json) passed30
isolated headers with the actual exported route definitions and GCC14.2.0. Existing bare-header checks
also passed30. An initial GCC13 consumer is retained as a diagnostic, explicitly below the declared
floor; only the small header consumer was corrected, without SDK/math/P1 repetition.

WSL CPU static Graph/Direct also passed and the original SDK loop exited0. Both consumer ELF lookup
paths include relocated SDK/lib, with the original CPU MODULE present and no LD override. Its source,
shared and static six-entry group is complete. Updated receipts retain original loader failures.
C3 is66/90; the next serial WSL CUDA batch has started. H1-H3 remain unimplemented.

[WSL CPU completed-build cleanup](wsl-cpu/cleanup-sdk-build-objects.json) removed741,239,406 bytes
of two owned completed producer object trees, preserving source objects, relocated SDKs, consumers,
cache and logs. [Actual CUDA source recipe](wsl-cuda/source-recipe-identity.json) records GCC14 C++,
the separate NVCC host compiler and absence of HYLE CPU math roots. CUDA math remains unrun.

[WSL CUDA source receipt](wsl-cuda-partial.json) now records actual ordinary FFN Graph1.07s/
Direct0.56s PASS on RTX4070/SM8.9, with immutable source/archive identity and original CTest logs.
The original driver continues shared/static. C3 is68/90; hybrid and full-model execution are not signed.
