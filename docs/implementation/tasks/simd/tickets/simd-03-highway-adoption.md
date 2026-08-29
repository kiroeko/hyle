# SIMD-03：Highway 直接承载可行性

- ticket state：`DONE`
- kind：architecture spike / product-non-mutating proof
- depends on：SIMD-02
- budget：1 工程日；性能预算 60 分钟

## 要回答的问题

这不是“Highway 能不能编译”的演示，而是决定本 epic 的主要删除边界：

1. ordinary HYLE pointer/span API 能否由 compiled Highway kernel 承载而不泄漏 vendor 类型；
2. expert source kernel 能否直接使用 Highway 或极薄 HYLE adapter，从而删除平行 D-tag operation surface；
3. Highway dynamic/static target 机制能否覆盖 HYLE runtime/exact/native/fat journey，Platform facts 如何避免
   冲突或重复；
4. source embedding、installed static/shared、offline dependency 与 Windows export/import 是否成立；
5. scalar/SSE4/AVX2/AVX-512/NEON/SVE/RVV 的支持与 HYLE 当前合同差异；
6. 同 TU wrapper 是否为零额外 hot-loop dispatch/copy/allocation，编译时间和 binary size 是否在预冻结预算。

## 隔离 proof

proof 放在 `C:\hyle-scratch` 或独立 fixture，不修改 HYLE 产品路径。至少实现同一 add/reduction/tail kernel 的：

- Highway direct expert source；
- HYLE ordinary façade 调 compiled Highway dynamic dispatch；
- fixed target/exact 入口；
- 当前 HYLE 同合同对照。

Windows 必须完成 source/static/shared 最小 journey；WSL/Mac 只做受支持编译/执行确认。P1 只用于证明 wrapper
没有 material 层级税，不签最终产品性能。

## 决策输出

逐 surface 只能给：

- `DIRECT`：消费者直接用 Highway，HYLE 不复制 API；
- `THIN_ADAPTER`：HYLE 保留小型 contract/target/dtype glue；
- `COMPILED_WRAPPER`：普通 HYLE batch API 隐藏 Highway implementation；
- `KEEP_HYLE`：必须有具名 semantic/ISA/delivery/complexity 缺口；
- `BLOCKED`：proof 无法在预算内证明，写 owner/resume condition。

验收后才能创建新的 `simd-v2 SCOPED` round。成功不是“再加一个 Highway backend”，而是冻结后续要删除
的 HYLE backend/ports/API 与保留的差异化代码。

## 隔离 proof 结果

proof 位于 canonical 产品树之外的 `C:\hyle-scratch\simd-highway-adoption`，只绑定 Highway
`f503c485cec6b4f2f596fa60d83484797c9eeb5c`，没有接受或修改 SIMD 产品代码。普通头只包含
`<cstddef>` 和 pointer/size 签名；实现 TU 使用 Highway。结果如下：

| journey | W1-WIN | W1-WSL | M1-MAC |
|---|---|---|---|
| ordinary source embedding | `PASS`，MSVC 19.50，target=AVX2 | `PASS`，GCC 13.3，target=AVX2 | `PASS`，AppleClang 21，target=NEON |
| installed-shape static/shared | `PASS/PASS` | `PASS/PASS` | `PASS/PASS` |
| expert direct Highway | `PASS` | `PASS` | `PASS` |
| fixed/exact artifact | `PASS`，`HWY_COMPILE_ONLY_STATIC + /arch:AVX2` | `PASS`，固定 AVX2 baseline | `PASS`，固定 NEON baseline |

Windows 五个 journey 为 5/5，WSL 为 5/5，Mac 为 5/5。Windows 动态 façade 的 static lib、shared DLL、
source executable 分别为 182,914、31,744、35,840 bytes；fixed static lib 为 47,956 bytes。Mac 对应
static/shared/source/fixed 分别为 8,024、55,528、57,824、1,296 bytes。这些只是 proof 的交付/规模上界，
不是最终 HYLE artifact budget。

反汇编确认 ordinary façade 被内联成 Highway dispatch body，没有 buffer copy、allocation 或 per-element
dispatch；fixed façade 是到固定 Highway target kernel 的 tail jump。向量只存在于 target TU 内，未跨普通
头、DLL 或函数 ABI。

## 有界性能与仪器结论

W1-WIN 使用现有 `bench/simd/op_parity_engine.h` 的 pin/order/replica 控制，只测 f32 add 的层级税；预算内
执行正序、反序和 fixed-vs-fixed control，没有形成产品或 SOTA 证书：

- 当前 HYLE fixed AVX2 与 Highway fixed AVX2 在 `n=4096/65536/1048576` 没有稳定 material 机制差异；
  `n=31` 的 scalar-tail 与 Highway partial-load 形状有小差异，后续只作为 tail-policy 输入；
- 每次直接调用 `HWY_DYNAMIC_DISPATCH` 在 `n=31` 约增加 3--4 ns/batch，在 L1 `n=4096` 行约有 5--10%
  batch-boundary tax，进入 streaming 带宽区后不可分辨；这不允许扩张为逐操作跑分；
- `HWY_DYNAMIC_POINTER` 若在 `ChosenTarget` 初始化前缓存，会永久缓存 slot-zero detection trampoline，实测
  可使 `n=4096` 慢十余倍。必须先显式初始化 target，再取指针；更合适的产品形状是由 HYLE Runtime 一次
  构造整张函数表，而不是每个 public op 自己持有 magic static；
- Runtime 仍有每批一次的函数间接调用。普通大 buffer 接受这项现有成本；微小热循环和融合 kernel 使用
  fixed/exact 或 expert source。后续产品票必须以“当前 HYLE runtime 对新 HYLE runtime”同合同复检，不能
  用 fixed 对 runtime 宣称回归。

## ISA 与保留缺口

| HYLE 当前合同 | Highway 冻结版本 | 裁决 |
|---|---|---|
| scalar、SSE4、AVX2 | scalar/EMU128、SSE2/SSSE3/SSE4、AVX2 | `DIRECT`，Highway 能覆盖且更细 |
| AVX-512 | AVX3/AVX3_DL/Zen4/SPR/AVX10.2 | GCC/Clang `DIRECT`；经典 cl.exe 被 Highway 官方 blocklist 禁用，进入 SIMD-04/05 的具名 gap，不保留整套 HYLE backend 作为默认答案 |
| NEON | NEON/NEON-BF16 | `DIRECT` |
| SVE | SVE/SVE2/SVE-256/SVE2-128 | `DIRECT`，无真硅只签后续 correctness |
| RVV | RVV | `DIRECT`，遵守 Highway 的 GCC>=13/Clang>=16 编译器边界 |
| BF16/VNNI/AVX10.2 specialty | Highway target 与 op 覆盖不完全等同 HYLE batch contract | `THIN_ADAPTER/KEEP_MINIMAL_GAP`，逐 capability 在 04/05/07 裁决 |

经典 MSVC AVX-512 是真实 compiler/delivery gap，但不推翻 vendor-first：后续优先考察 xsimd 的 MSVC
AVX-512 路径、clang-cl 目标产物或极小 specialty TU；只有三者不能满足已冻结合同才允许保留具名 HYLE
kernel，不允许据此保留 1.9 万行平行 register backend。

## Surface 决策

| surface | verdict | 后续删除/保留边界 |
|---|---|---|
| ordinary pointer/span/buffer API | `COMPILED_WRAPPER` | 保留自然 HYLE 合同；implementation 转为 vendor kernel |
| expert portable register kernel | `DIRECT / THIN_ADAPTER` | 文档优先 direct Highway；只保留 target/dtype/consumer glue，不复制完整 DSL |
| `native::*` pointer 型 source algorithm | `THIN_ADAPTER` | 可保留 HYLE 算法入口，但内部直接写 Highway；不得依赖平行 backend |
| Runtime/fat selection | `THIN_ADAPTER` | Platform 保留 CPU facts；HYLE 建一次函数表并发布实际 module tier，Highway 提供 target bodies |
| exact/fixed artifact | `COMPILED_WRAPPER` | 单 target 使用 compiler baseline + `HWY_COMPILE_ONLY_STATIC`，不携带无关动态 variants |
| `vec.h` + scalar/SSE/AVX/NEON 平行 backend | `DROP_OR_MINIMAL_ADAPTER` | 387-name 只作迁移 inventory；消费迁走后删除，缺口逐项举证 |
| checked-in Highway/xsimd math、bitpack、VQSort ports | `WRAP_OR_MINIMAL_PORT` | 04/10/11 以 accuracy/wire/order 合同决定；direct 等价即删 port |
| classic-MSVC AVX-512 与固定格式/小尾部差异 | `KEEP_MINIMAL_GAP` | 只保留具名差异代码，不形成第二套通用 vendor backend |

verdict：`DONE`。proof 证明 v2 应直接迁移和删除，而不是增加一个并行 Highway backend；同时冻结了
Runtime、MSVC AVX-512、固定格式与 tail 的有限 HYLE glue。SIMD-00 的 resume condition 已满足，可以创建
新的 `simd-v2 SCOPED` round。output commit：
`5becc37ff546e64351b00c56ac6eb39eee7fec3f`。
