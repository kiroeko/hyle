# SIMD vendor ownership

本文只描述长期 ownership 和升级原则，不承担操作覆盖率、性能排名或任务状态。当前 product state 见
[`module-status.json`](../implementation/module-status.json)，本轮版本与施工事实见
[`SIMD v4 closure card`](../implementation/tasks/simd/simd-v4-expert-sdk.md)。

## 两种依赖必须分开说

HYLE 的 ordinary SIMD 头不泄漏 Highway/xsimd 类型。当前目标把 Highway 与 xsimd 都作为中央锁定、
header-only 的产品候选：上游 project 不执行，HYLE target TU 只编译 route manifest 选中的实现，所以
静态/动态交付不增加第二个运行时 ABI。checked-in port、direct product dependency 与 benchmark-only peer
仍必须分开记账：

| origin | HYLE product paths | ownership |
|---|---|---|
| Highway direct | target-bound primitive 与 whole-buffer candidate、fixed-target/source-fat TU，以及使用同一锁定依赖的其他模块私有 kernel | 对应 HYLE 模块 owns public semantics/selection/delivery；锁定 Highway owns 其 register/math implementation；route ledger owns 当前采用范围 |
| Highway integrated source | `src/simd/sort/vqsort_hyle/**` 与 target TUs | Highway `83eff393...` owns VQSort algorithm；HYLE owns the retained minimal compiler/scalable/public-POD/static-entry adaptation and product contracts |
| xsimd direct | xsimd 支持的 target/type 上的 primitive 或完整 whole-buffer candidate | xsimd owns implementation；HYLE owns semantics、静态 route、交付与必要的零成本 adapter；只有 ledger 选中的头进入产品编译闭包，不向 ordinary API 泄漏 vendor type |
| HYLE implementation | 具名语义、复杂度、指令选择或 consumer 优势 body | HYLE owns implementation；每项必须有 current-peer 或结构证据，不能据此保留无关旧 backend |
| simdutf | container UTF-8 implementation，不属于 `hyle::simd` target | container owns semantics/delivery；vendored simdutf owns implementation mechanism |

中央 [`HyleDependencyLock.json`](../../cmake/HyleDependencyLock.json) 是 integrated origin、archive、内容 hash
和许可证投影的唯一机器账本。文件名或 namespace 已改为 HYLE 并不会抹掉第三方来源。

## 公共边界

- HYLE owns ordinary pointer/span buffer API、accuracy/layout/alias/error contract 与 Platform/build/package
  integration；这些用户合同不改成 vendor API。
- expert register/kernel surface 通过 target-bound seam 使用静态 route，或显式固定某个 vendor；极薄
  adapter 只提供 HYLE 需要的 semantic/target/dtype glue，不把完整 Highway/xsimd operation set 再复制一遍
  改名。`hyle::simd_expert`、`hyle_target_simd()` 与 `hyle_add_fat_kernel()` 在源码、installed static 和
  installed shared 三种交付中同形；安装包携带中央锁定的 vendor 开发头，但不导出 vendor register ABI。
- fixed/fat source kernel 由 route manifest、provider target namespace 与模板单态化承载；HYLE 的 CMake 只负责
  生成 per-ISA TU、施加准确 codegen flags、在 pointer/POD 边界选择一次。ordinary batch 的函数表是
  私有实现，不是用户可依赖的 layout。Platform 仍服务其他模块；SIMD 不复制已经正确的 register 实现。
- Highway 和 xsimd 是并存候选；同合同标准 primitive 以 Highway 为维护性默认，xsimd/HYLE 仅凭已证
  material 优势成为有限例外。选择必须细化到
  `operation/family × scalar type × target/compiler`。SLEEF 仍只在两者均不能满足已准入数学合同时另行准入。
- integrated port 是显式供应链债务：direct wrapper 可行即删除；不可行时只保留真实 contract delta，并记录
  upstream identity、local delta、许可证和内容 hash。ordering 已按有界 SIMD-11 试验保留最小 port：当前
  Highway 的 11 类型 × full/partial/select 直接实例化在 classic MSVC 下超出构建预算；保留项只承担
  compiler/scalable glue、HYLE public POD/static entry 和产品语义，不另建 public register DSL。
- HYLE 自实现只因真实 semantic/ISA/complexity/consumer gap 存在，不因 benchmark 某行获胜而复制。

## 实现选择与零额外开销路由

HYLE 不设“为了兼容而保留的旧 SIMD 实现”，但允许同一能力的多套候选长期共存，以便 vendor 升级和
HYLE 优化逐项替换。每个可执行 route 必须落入三类之一：

1. `DIRECT_VENDOR`：Highway 或 xsimd 已满足 HYLE 语义、资源和交付合同，target TU 直接实例化该实现；
2. `HYLE_PROVEN_IMPLEMENTATION`：成熟库缺少合同，或同合同 consumer 证据与机制共同证明 HYLE 算法有
   material 优势；实现归具体 capability owner，不借机维持一套通用平行 register 库；
3. `DELETE`：没有真实 consumer、语义 gap 或经资格化证明的实现优势。

exact/fixed route 由 HYLE 生成的 route manifest、provider 宏/target namespace 和模板在编译期确定，
生成代码没有 runtime selector 或间接函数调用。source-fat 与 ordinary runtime 在进程内只解析一次 CPU/ISA
target，随后每次 whole-buffer API 调用只允许一次已缓存的 target 间接函数调用；这里的“一次”按函数调用计，
不按某个 buffer 对象的生命周期计。provider 已在各 target body 编译时确定，因此不会增加第二层 dispatch。
vendor register 类型、adapter、callback 或二次 dispatch 不得进入
元素循环。不同 provider 的 native vector 不得在元素循环中无证据地来回转换；只有 codegen 证明没有额外
指令、spill、copy 或二次 dispatch 的 bridge 才能进入 route，否则使用完整 provider-coherent kernel，或把
获胜算法改写为当前 target 的 native lowering。宏能消除已知 target 的路由，不能在编译期决定未知客户 CPU；若采用 Highway dynamic dispatch，
它必须取代而不是叠加 HYLE selector。一个 HYLE 算法优于 vendor 只证明该算法值得保留，不自动证明其
历史 register abstraction、全部 operation 或公开 spelling 也值得保留。

build system 必须把 route 当成编译合同而不是运行时服务：selected provider 的头、target namespace、
compile definition 与 ISA options 在同一个 target 声明中产生；未选择的 candidate 不进入该实例化闭包。
新增 HYLE 优化时先增加独立 candidate body，再以 correctness、复杂度、最终 codegen 和 material consumer
证据重签有限 route；升级 Highway/xsimd 时也只重签受到源码或 target 变化影响的 group。这是可持续的
“忒修斯之船”维护入口，不授权重新复制完整 vendor surface。

机器链路为 `scripts/simd-op-routing.json` → 生成的 HYLE-internal
`hyle/simd/provider/operation_manifest.h` → `hyle/simd/provider/route.h`。清单机械包含 387/387 个旧
operation；`RouteFor<operation, lane, target, compiler, version>` 的 primary 永远是 `PENDING`，未资格化时
直接编译失败。终态 route 只生成模板特化，调用展开为 `Candidate<selected_provider>::run(...)`；不存在可被
outline 的路由函数。一个 provider candidate 是完整 body specialization，不能把共同 façade 误做成新的
register DSL。

route ledger 的 `candidate_audit` 只记录锁定源码中的候选形状，不能据此选择 provider。每个 route scope
另有 HYLE/Highway/xsimd 三方 `provider_assessments`：同名 token 仍是 `PENDING`，必须以语义、类型/target、
正确性和适用的结构/性能证据逐项落成终态。terminal verifier 会拒绝缺 provider、缺证据、缺 legacy site、
未知 target、重复 scope ID 或任一 `PENDING`；所以 387 项不是抽样报告中的一个总数，而是删除门的逐行
前置条件。

终态不要求为每个 primitive 证明一个跨机器的绝对性能赢家。Highway 对同合同标准 primitive 是默认 route；
HYLE/xsimd 只以已有、资格化且超过 5% 的 type × ISA/platform 证据建立有限例外。持平、最终 codegen 等价、
顺序或编译器反转、差异不 material，以及有界资格化后仍无赢家，全部终结为 Highway。compiler 只在产品
构建能够静态表达且差异可复现时成为例外维度，不能预先展开开放的 compiler 笛卡尔积。默认 route 的证据
证明的是合同、复杂度和选择政策，不冒充“所有未来硬件上 Highway 都最快”。

迁移账本以冻结 baseline 机械枚举全部旧 operation，并为每项记录 overload/source 分布、适用 target、
同合同 peers、evidence group 和 `KEEP_HYLE/ROUTE_HIGHWAY/ROUTE_XSIMD/REMOVE_REDUNDANT`。允许多个项目共享
一个相同实现/codegen 的证据组，但禁止隐式遗漏；账本存在缺项、未知映射或 `PENDING` 时，不得删除旧源或
形成 candidate。

## Dispatch 集成规则

- Platform 继续拥有全库 CPU/OS facts；Highway/xsimd 各自拥有其 target implementation 与 compiler
  blocklist。SIMD 选择平台事实、可编译 provider 与产品 route policy 的交集，并由
  `selected_target(family)` 报告实际采用的 module tier/features；
  `hyle::simd` 不再转发 Platform detector。
- ordinary batch route 在私有 runtime table 初始化时只解析一次 CPU/target 并构造函数表；每次调用只允许
  一个 batch-boundary indirect call。不得把“一次间接调用”误写成“每个 buffer 重新探测一次”，也不得在
  每个 op 中重复 CPU detection、target bit scan 或 allocation。
- 若使用 `HWY_DYNAMIC_POINTER`，必须先初始化 `ChosenTarget`；初始化前的 slot-zero pointer 是 detection
  trampoline，缓存它会让每次调用重复探测。也可以按 Highway 明确支持的方式由 HYLE selector 直接绑定
  `N_<TARGET>` symbol。
- fixed/exact artifact 使用 compiler baseline 与 `HWY_COMPILE_ONLY_STATIC`，由宏/模板直接绑定 target
  symbol；source-fat 每个生成 TU 使用
  `HWY_COMPILE_ONLY_SCALAR` 或 `HWY_COMPILE_ONLY_STATIC`，并编译期断言 `HWY_STATIC_TARGET` 与声明 tier
  一致。两条路径都在算法模板内直接使用所选 provider，普通 façade 不把 vector type 跨 TU/shared boundary。
  fixed/exact 的 runtime dispatch 成本为零；tiny fused hot loop 走 exact 或 expert target，不要求未知 CPU
  的 runtime route 冒充编译期零开销。
- HYLE canonical x86 tier 不承诺 AES/PCLMUL；direct Highway TU 使用上游支持的
  `HWY_DISABLE_PCLMUL_AES` 让 vendor target 与 Platform facts 对齐，不能为了满足 Highway target 名称偷偷
  提高部署最低要求。专用 crypto kernel 必须显式探测/绑定其扩展。
- Highway 在经典 cl.exe 下禁用 AVX3 family。fixed/exact AVX-512 fail-fast，unknown-CPU source-fat 明确
  capped at AVX2；需要 AVX-512 时使用 clang-cl，而不是把静默降档当作 AVX-512。

## 升级规则

1. 先冻结 Highway 与 xsimd 官方 commit，不追逐浮动 HEAD；同一 capability 的两个 strong peer 即这两者，
   HYLE 是被裁决 candidate，不再临时加入第三个 peer。
2. 先比较语义、accuracy、layout、alias、异常与 ISA，再比较性能。
3. vendor 升级只重签 route manifest 标记为受影响的 implementation groups：广 correctness、当前 peer
   parity 与少量正常 consumer sentinel；不重跑未受影响的全笛卡尔积，但每个 operation 的 ledger 引用必须
   仍可解析。
4. direct wrapper 可行时删除长期 port 与无优势平行 backend；若 port 必须存在，只移植与当前 origin family
   有关的 upstream 修复。
5. 任何不同 wire format、稳定性、NaN、dispatch 或资源合同的结果都标记 `NOT_COMPARABLE`。
6. 内容确定后由 CMake integrated-source hash 工具机械重算，禁止手写一个“看起来更新”的 hash。

## 明确负空间

Benchmark checkout 是 peer，不是安装产品依赖；Native FFI、稳定跨工具链 C++ ABI、per-ISA DLL、tensor/
model operator 和 GPU backend 不因 vendor 升级进入 SIMD。阶段性“哪个 op 曾赢”的记录只留在
implementation/evidence，不进入本架构页。
