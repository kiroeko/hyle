# SIMD v3 multi-provider routing closure

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前产品状态只以 [`module-status.json`](../../module-status.json) 为准。本页是 owner 在 2026-08-30
> 改变 implementation strategy 后的新一轮 `RESUME` 冻结卡。SIMD-00～11 与 v2 的实现/证据只作输入；
> [`SIMD-12`](tickets/simd-12-integration-closure.md) 已停止，不再授权删除旧 register/backend。

## Closure card（`AUDITED`）

- component / owner / round ID：`simd` / SIMD owner / `simd-v3`；task kind：`RESUME`。
- 前一 round / blocker / resume condition：`simd-v2` 从 `AUDITED` 停在
  `BLOCKED/SUPERSEDED_BY_OWNER_RESCOPE`。旧范围预设 Highway 为产品默认、xsimd 仅具名补位，并允许在
  少量 salvage 后删除 387-name register 面；current xsimd f32 `tanh` 复现 material advantage 后，owner
  要求 Highway、xsimd、HYLE 多实现并存和全量逐项裁决。恢复条件是先冻结新的路由架构、完整旧面 inventory
  与有界验证方法；本卡完成该重新定界，产品实现仍须等 SIMD-13 把有限 FIX 集审计为 `AUDITED`。
- baseline commit / tree：`77d22d11be31f3184e96f8379068fa9779ed33e0` /
  `c24327164348c3bc7da9bfca424593f84864126f`。该提交中的 `vec.h` 与七 backend 是 387-operation 冻结
  inventory 来源。当前工作树的 xsimd tanh benchmark/evidence、共享 Highway CMake 草稿和 Container
  consumer 草稿分别只归 SIMD-13/14/16 输入，不是 candidate。
- 进入原因 / demand IDs：
  - `SIMD-V3-COMPLETE-01`：只用 Highway 会遗漏 xsimd/HYLE 的已证实优势；只维护 HYLE 又无法持续取得
    vendor 的 ISA、编译器和新能力演进；
  - `SIMD-V3-ALL-OPS-01`：旧面机械提取为 387 个去重 operation，任何抽样 salvage 都可能删除未点名优势；
  - `SIMD-V3-ZERO-ROUTE-01`：多实现不能引入逐元素 dispatch、跨 vendor bridge copy/spill 或二次探测；
  - `SIMD-V3-UPGRADE-01`：依赖升级后只重签受影响 route group，同时允许更好的 HYLE 实现逐步接管，
    形成可维护的“忒修斯之船”演进，而不是再次整体重写。
- 真实 consumer / normal workload：未来 HYLE 量化推理引擎的 fixed/source-fat fused kernel 与普通
  continuous row/buffer API；`n=4096` 为 cache-scale，`n=1048576` 为适用 streaming，odd/tail 只签
  correctness。当前 Container/Compute 只验证受影响直接边。llama.cpp/ggml 是 north-star 差异输入，
  不冒充已有生产 consumer。
- 范围：把 Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c` 与 xsimd
  `3b3151ff665514a00a437cc985f47b7de99b772b` 都提升为锁定 header-only product candidates；建立不泄漏
  vendor 类型的 HYLE semantic façade、target-bound provider adapters 和编译期 route manifest；对 baseline
  的 387 个 operation 逐项完成 overload/source/target inventory、同合同比较与
  `KEEP_HYLE/ROUTE_HIGHWAY/ROUTE_XSIMD/REMOVE_REDUNDANT`；迁移普通 family、expert/source consumer 与
  交付后，才删除无 owner 的旧实现；完成三平台/三交付、文档、六问与 closure。
- 非目标：保留旧 API/头的兼容性，公开 vendor register ABI，运行时逐 operation 选择，完整复制任一 vendor
  spelling，`operation × overload × size × ISA` 无限笛卡尔积，GPU、tensor/layout/threading、GEMM/attention/
  quant format、per-ISA DLL、C/Native FFI、稳定跨工具链 C++ ABI、SLEEF 自动准入、中央 closure/performance
  仪器重写与 Formal certification。
- 依赖 / owner：SIMD 仍只依赖 HYLE Core、Platform；Highway 与 xsimd 是 build-only header product
  dependencies，不增加 runtime ABI。HYLE owns public semantics/route/delivery；vendor owns direct
  implementation；ported code 保留其 origin；Platform owns CPU/OS facts。Container 可共享中央 vendor
  dependency target，但不因此依赖 `hyle::simd`。
- tier / material / resource：`P1`，5% material margin。O(n) primitive 保持 O(n)/O(1)、无 hidden
  allocation/copy/lock/syscall；provider route 在 fixed/exact、ordinary 与 source-fat 的 target body 中都由
  模板/宏在编译期消失。unknown-CPU ordinary/source-fat 只允许既有的每次 whole-buffer 一次缓存 ISA-target
  间接调用，不能因 provider 再增加一层。不同 provider bridge 必须经 release codegen 证明零额外 instruction/spill/copy，
  否则只能选 provider-coherent kernel 或 native lowering。
- budgets：SIMD-13 4 工程时；SIMD-14 1 工程日 / P1 30 分钟；SIMD-15 2 工程日 / 串行机器 8 小时；
  SIMD-16 1 工程日 / P1 60 分钟；SIMD-17 1 工程日 / 串行机器 4 小时。每个 evidence group 最多一次
  qualification、一次反序复现、一次 codegen 机制检查、一个产品修复和一次重跑。预算到达必须形成
  `DONE/BLOCKED`，不得把漏项解释成继续研究。
- machine / instrument：W1-WIN、W1-WSL、M1-MAC；`closure-perf-v1`。Windows/MSVC 与 WSL/GCC 的 AVX2、
  macOS/Clang 的 NEON 作 current silicon route verdict；SVE/RVV 只签 correctness/codegen，待真硅前不得签
  winner。现有 `op_parity_engine.h` / `op-parity-report.py` 是统一仪器，不另造第五套计时器。
- strong peers：每项最多 Highway 与 xsimd 两个；HYLE 是 candidate。标准库/ggml/oneDNN 只在某个
  whole-range family 缺少同合同 vendor peer 时提供具名 consumer/算法对照，不同时扩成第三、第四 peer。
- correctness / delivery gates：语义、type/width、mask、NaN/Inf/signed-zero、rounding、alias、odd/tail、
  unsupported target fallback、property/differential、ASan/UBSan、public header self-containment、source
  embedding、installed static/shared、offline locked dependencies、exact/fat target 和受影响 consumer。
- performance gate：387/387 都必须有当前 disposition；314 个现有独立 parity operation 必须出现于当前
  report union 或引用完全相同 implementation/codegen group，另外 30 个 alias/synthetic/paired-info/algorithm
  与 43 个 non-timing helper/selector 必须有结构或 workload proof。逐项有账不等于逐项独立计时；任何
  operation、适用 target 或 evidence 引用缺失/`PENDING` 都使删除门失败。
- owner route policy（2026-08-30）：Highway 是同合同标准 primitive 的默认实现。最终 codegen 等价、
  合格结果持平、低于 5% material margin、运行顺序/编译器反转，或完成冻结的有限资格化后仍无法分胜负，
  均直接 `ROUTE_HIGHWAY`，不得继续扩建 benchmark。只有现有资格化证据在可表达的 type × ISA/platform
  scope 上证明 HYLE/xsimd material winner，才拆出有限例外；Highway 不具备或合同不同则保留 HYLE。
  此策略要求零 `PENDING`，但不声称默认 Highway 在所有未来 CPU/compiler 上绝对最快；具名 consumer、
  vendor 升级或新硬件 material 证据是后续重签触发器。
- 输出 / 停止：机器可检 inventory、route ledger、provider dependency/adapter、保留实现来源账、删除表、
  current evidence groups、三交付/三平台 receipt、六问和四项 LOC。所有适用 gate 通过才可关闭。有限性能
  批次无资格或没有 current silicon 时记录 `INCONCLUSIVE` 并按 owner policy 采用 Highway，不以旧矩阵冒充
  HYLE/xsimd winner；只有合同、正确性或交付因缺证据无法成立时才使产品 `BLOCKED`。

## 顺序 tickets

| ID | 状态 | 内容 | 删除权限 |
|---|---|---|---|
| [SIMD-13](tickets/simd-13-all-op-audit.md) | `DONE` | 冻结并机检 387/387 inventory；为每项指定比较种类、variants、适用 targets、candidate groups；完成双向能力审计并冻结有限 FIX | 无 |
| [SIMD-14](tickets/simd-14-provider-substrate.md) | `DONE` | Highway+xsimd product dependency、provider adapter、compile-time route manifest 与零开销 codegen proof | 只可删被等价替换的构建 seam |
| [SIMD-15](tickets/simd-15-route-qualification.md) | `DONE` | 按 implementation-equivalence groups 执行全部 387 项 current-peer/结构/workload 裁决，先提取所有赢家 | 无通用删除；可移动具名赢家 |
| [SIMD-16](tickets/simd-16-migration-cleanup.md) | `DONE` | ordinary/expert/Container/Ordering 消费迁移；仅删除 ledger 已终态且引用归零的旧实现 | 有限、逐项受 ledger 控制 |
| [SIMD-17](tickets/simd-17-delivery-closure.md) | `ACTIVE` | candidate、三平台/三交付、受影响 P1、文档/易用性、红队、六问、LOC 与 terminal receipt | 不再新增产品能力 |

SIMD-13 完成且 registry 进入 `AUDITED` 前不得修改产品 API、算法或 layout。其后也只能按冻结 ledger 的
有限 implementation groups 施工；新能力或新 benchmark 行不能临时扩大范围。

状态迁移使用两份明确快照验证：v2 `AUDITED` 前态位于
`C:\hyle-scratch\simd-v3\transitions\module-status-before-blocked.json`，SHA-256
`DE65C05A737848B2153C11F1F275868A5AFFEEFE3974196A3D1B5D7F41702763`；v2 terminal `BLOCKED` 快照位于
同目录 `module-status-v2-blocked.json`，SHA-256
`9932A9EC52CA9D98ADA4D676EEF391B2007D17D403E3FFD525A6D8CB96C1924A`。检查器已分别验证
`AUDITED -> BLOCKED` 与 `BLOCKED -> SCOPED`，没有跳级。

SIMD-13 完成后的 v3 `SCOPED` 快照位于同目录 `module-status-v3-scoped.json`，SHA-256
`7496331165384E3D87166FEEDD79119BDCE97200685E468FE9C3FDA2C2EDE1DA`；该快照用于机械验证
`SCOPED -> AUDITED`。进入 `AUDITED` 只授权下文冻结 FIX，387 个 route decision 仍全部为 `PENDING`。

## Withdrawn candidate and bounded return to `AUDITED`

- 首个 product commit/tree：`4a1c216cb546ef1df9ca041516ecca2b66d10859` /
  `5d7269b9342e99bbbd580f6f005359c172a4c134`，现已撤回，不再作为 terminal candidate；
- SVE QEMU 回放发现 `vqsort-inl.h` 的 scratch capacity 仍引用随旧 register API 删除的
  `kNativeVectorBytes`。这是产品可移植性缺陷，不是 benchmark 差异；
- 按 candidate 后产品变化规则，registry 由 `CANDIDATE` 退回 `AUDITED`。前态快照为
  `C:\hyle-scratch\simd-v3\transitions\module-status-v3-candidate-before-sve-fix.json`，SHA-256
  `51F1ECEA5006DC0D1A64880EECC7C42E23D0DFD8076930FAB55198E2147FE7AB`；
- 新增且只新增一个有限 FIX：scratch upper bound 使用当前锁定 Highway target authority
  `HWY_MAX_BYTES`；它在 SVE 下是 256-byte architectural maximum，保持 O(1) stack scratch、无 heap、
  无热路径 branch。修复后重跑 SVE/RVV QEMU、ordering correctness 与三平台适用回归，再创建新 candidate。

387-operation ledger、422 个 terminal scope、公开 pointer/count/POD 语义、provider route 和 P1 裁决均未
因此扩大或改变。后续新 candidate 仍必须遵守：只允许补验证、说明和 receipt；公开合同、算法、route 或
正常 workload 再改变时必须再次退回 `AUDITED`。

## SIMD-13 双向审计结果

自顶向下重新核对 inference/source-primary、ordinary buffer 与交付旅程；自底向上枚举 13 个 installed
ordinary headers、4 个 source-only root seam、七 backend、native subtree 和 387-operation inventory。
SIMD-00～11 已完成的语义能力继续作为当前源码事实，但其旧 mechanism/performance verdict 不继承到 v3：

| family / surface | 当前能力状态 | v3 finding / action |
|---|---|---|
| `dispatch.h`, `runtime.h` | ordinary 每 batch 一次选择；exact/fat 已有 source seam | 语义保持；provider identity 与 route metadata 改由 SIMD-14 的同一生成 manifest 拥有 |
| `types.h`, `dtype.h`, `dtype_ops.h` | 五 storage type 与 20 条异类型转换为 `COMPLETE` | 保持 HYLE rounding/layout；逐 target 在 vendor/HYLE candidates 中择优 |
| `ops.h`, `narrow_ops.h` | continuous row、mixed dot、narrow search/sum/total-order 为 `COMPLETE/ROUTED` | API 不因 387 个 register spelling 扩张；body 进入 provider route |
| `math.h` | f32/f64 real/complex、activation、softmax 合同为 `COMPLETE` | current xsimd f32 tanh material signal 是具名 FIX；全部 math op 同时按 Highway/xsimd/HYLE 资格化，不再预设 Highway |
| `search.h`, `scan.h` | 十 scalar type search/extrema/sorted 与 in/out scan 为 `COMPLETE` | 语义/复杂度不变；provider-coherent whole-buffer route，禁止循环内 bridge |
| `bitpack.h`, `random.h` | canonical bit layout 与 deterministic state lifecycle 为 `COMPLETE` | HYLE owns format/state；vendor 只能替换 target body |
| `sort.h` | no-hidden-heap full/partial/select 与 stable radix/argsort 为 `COMPLETE` | 保留已证 VQSort adaptation 为候选，不把 origin 误记自研；与当前 Highway 重新资格化 |
| `vec.h`, `native.h`, `backend/*`, `native/*` | 387 个 operation、1,797 个 definition site；source-only legacy；23-site parser 勘误见 SIMD-15 evidence | 不是自动保留 API，也不是可抽样删除垃圾；387/387 ledger 完成后逐项提取/路由/合并 |
| `target.h`, `fat_kernel.h` | current source expert 直接 Highway | `PARTIAL`：不能取得 xsimd/HYLE winner；SIMD-14 改为静态 route，同时保留显式 vendor pin 逃生口 |
| installed source/static/shared | ordinary pointer/POD 不泄漏 vendor type | 形状正确；加入 xsimd product candidate 后必须重签 offline/license/component/import 闭包 |

本轮没有从 ggml/llama source journey 或当前两家 vendor 发现新的未处置 `MUST` family；tensor layout、
threading、quant block/GEMM、attention/RoPE/SSM 仍明确归 Compute/推理引擎。xsimd 独有 scalar-like spelling
只作后续 capability 输入，不因 symbol union 扩大当前 API。新的有限 `FIX` 集恰为：

1. SIMD-14：双 product dependency、provider adapter、compile-time route manifest、零 bridge/codegen 合同；
2. SIMD-15：387 项 planned groups 与 whole-buffer math/family current-peer 资格化，先提取全部赢家；
3. SIMD-16：迁移 ordinary/source/Container/Ordering，只删除 ledger 已终态且引用归零的旧定义；
4. SIMD-17：三平台/三交付、易用性/文档、依赖/license、红队、六问和 terminal receipt。

除此之外没有产品修改授权；尤其不新增 benchmark-shaped API，不因单个 source hit 宣称 vendor 等价，也不
在 SIMD-15 前删除旧实现。
