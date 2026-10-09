# Vulkan original artifact batch

Owners/card: [audited artifact bindings](../../../docs/implementation/tasks/compute/vulkan-parity-v2-artifact-bindings.md).
Product candidate `9e00073bf`; qualification snapshots include only evidence/test
changes. Windows RTX4070, NVIDIA617.14, pinned original IREE compiler/runtime
`6ef606b6fa3e12bd2c12d36a07463069eb71185f`. DLLs are adjacent to the EXEs under
`C:/hv/vp1/bin`; real GPU runs enable the installed validation layer and suppress
Windows fault/missing-DLL UI. No performance sampling or new HYLE math kernel.

## Results and exact limits

| Domain | Evidence | Result |
|---|---|---|
| Original IREE owner | `native-156ab3c7-provider.log/.xml` | 7/7 PASS |
| U8/U32 storage, odd strides/tails, Copy/Fill/Select/View/Concat and unsigned arithmetic rejection | `native-156ab3c7-direct.log/.xml` | 2/2 affected storage IDs PASS |
| Existing FP8/checked Scan | Same Direct log and original Graph log | 5 Direct +2 Graph IDs PASS |
| NF4, nested NF4, AWQ and CT source/output F32/F16/BF16 | `native-1710f874-{direct,graph}-rounding.log/.xml` | Both batch IDs PASS: 48 format/scale/output/group combinations each, changed complete payloads, independent slots, replay and retained results |
| Nested partial-word scale codes and >256 groups | Same two batch IDs | PASS for groups9 and258 |
| Explicit cold decode→immutable typed weight→mature MatMul | `native-61d0585b-{direct,graph}-delta.log.gz/.xml.gz` | Both projection IDs PASS: four formats × three dtypes, changed activation and retained original decoded weight |
| Feature/domain rejection, actual OOM, cancellation/replay, close and quota release | Direct delta compressed raw | PASS |
| All 18 complete model geometries; Air, source/static/shared/relocation, final default/best-of | Not run in this batch | UNPROVEN; parent goal remains active |

Final source has **16 qualified decode entries**. The eight cross-dtype half
source cases use the original typed decode result followed by existing native
ggml Cast or IREE precision conversion. The intermediate storage, GPU copies,
conversion and completion are real and charged. No public schema/dtype/numeric
gate is weakened and no full-model F32 weight bank is hidden in the hot path.

## Failure causes and bounded corrections

Initial package configure rejected three new source licenses missing from its
install assets; the owning factor's runtime asset list was corrected. One test
fixture omitted explicit `Operation` construction; only that object was rebuilt.

The first numerical capture found high U8 nested scale indices lowered to
signed indices. Original and corrected SPIR-V are retained in `offline/` and
`offline-unsigned-fix/`; explicit unsigned indexing changes only the three
nested shaders. The other21 SPIR-V files were byte-identical. The128MiB fixture
also ran out of a quota that includes original64MiB Vulkan allocator slabs;
the existing FP8 fixture's256MiB budget was restored, while reserve-all still
tests true OOM and destruction verifies zero accounted bytes.

After that correction, all combinations became reachable and exposed original
compiler folding of intermediate half rounding. `61d0585b` keeps that FAIL.
Scalar and tensor `util.optimization_barrier` placements did not produce legal
offline artifacts; their logs are retained, without GPU or compiler-flag
roulette. Final composition uses the already qualified native conversion
fragments and removes the eight invalid one-stage exports.

Raw failed delta logs/XML totaled about106MiB. Their complete bytes are retained
in four lossless gzip files (about1.54MiB); the receipt records original and
compressed SHA-256/byte counts. Decompression was verified byte-for-byte before
removing only those four task-owned duplicate files. Original failures remain
failures; partial source samples do not become full catalogue or P1 PASS.
