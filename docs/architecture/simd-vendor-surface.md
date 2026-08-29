# SIMD vendor ownership

本文只描述长期 ownership 和升级原则，不承担操作覆盖率、性能排名或任务状态。当前 product state 见
[`module-status.json`](../implementation/module-status.json)，本轮版本与施工事实见
[`SIMD/Math epic`](../implementation/tasks/simd/simd-epic-v1.md)。

## 两种依赖必须分开说

HYLE 的 ordinary SIMD 头不泄漏 Highway/xsimd 类型。`simd-v2` 将 Highway 作为中央锁定、header-only
的直接产品机制：上游 project 不执行，HYLE 固定 target TU 直接编译所需实现，所以静态/动态交付不增加
第二个运行时 ABI。checked-in port 与 benchmark peer 仍必须和 direct product dependency 分开记账：

| origin | HYLE product paths | ownership |
|---|---|---|
| Highway direct | `src/simd/{ops,narrow,dtype,math,search,scan,bitpack,random}` 的 direct kernel body、fixed-target source seam 与 generated source-fat TU | HYLE owns public semantics/selection/delivery；锁定 Highway owns math/register implementation；HYLE 只保留 façade、canonical bitpack layout、RNG state 与具名 gap |
| Highway integrated source | `src/simd/sort/vqsort_hyle/**` 与 target TUs | HYLE owns adaptation/local delta；由 SIMD-11 裁决 direct/minimal-port/drop |
| xsimd | 当前只作独立数学 benchmark/accuracy peer，不进入产品或安装包 | 若未来出现 Highway 无法满足的具名 capability/accuracy gap，必须重新准入；历史 math port 已删除 |
| simdutf | container UTF-8 implementation，不属于 `hyle::simd` target | container owns semantics/delivery；vendored simdutf owns implementation mechanism |

中央 [`HyleDependencyLock.json`](../../cmake/HyleDependencyLock.json) 是 integrated origin、archive、内容 hash
和许可证投影的唯一机器账本。文件名或 namespace 已改为 HYLE 并不会抹掉第三方来源。

## 公共边界

- HYLE owns ordinary pointer/span buffer API、accuracy/layout/alias/error contract 与 Platform/build/package
  integration；这些用户合同不改成 vendor API。
- expert register/kernel surface 优先直接使用锁定 Highway，或由极薄 adapter 提供 HYLE 需要的 target/
  dtype glue；不把完整 Highway operation set 再复制一遍改名。`hyle_target_simd()` 是源码嵌入时的
  fixed-target seam；installed static/shared SDK 不导出 vendor register headers/ABI。
- fixed/fat source kernel 直接由 Highway 宏、target namespace 与模板单态化承载；HYLE 的 CMake 只负责
  生成 per-ISA TU、施加准确 codegen flags、在 pointer/POD 边界选择一次。ordinary batch 的函数表是
  私有实现，不是用户可依赖的 layout。Platform 仍服务其他模块；SIMD 不复制已经正确的 register 实现。
- xsimd/SLEEF 只补 Highway 无等价能力或不满足资格化 accuracy/resource contract 的数学 family。
- integrated port 是过渡债务：direct wrapper 可行即删除；不可行时只保留真实 contract delta，并记录
  upstream identity、local delta、许可证和内容 hash。
- HYLE 自实现只因真实 semantic/ISA/complexity/consumer gap 存在，不因 benchmark 某行获胜而复制。

## Dispatch 集成规则

- Platform 继续拥有全库 CPU/OS facts；Highway 拥有 target implementation 与 compiler blocklist。SIMD
  选择二者与产品 policy 的交集，并由 `selected_target(family)` 报告实际采用的 module tier/features；
  `hyle::simd` 不再转发 Platform detector。
- ordinary batch route 在私有 runtime table 初始化时一次构造函数表，每次调用只允许一个 batch-boundary indirect
  call；不得在每个 op 中重复 CPU detection、target bit scan 或 allocation。
- 若使用 `HWY_DYNAMIC_POINTER`，必须先初始化 `ChosenTarget`；初始化前的 slot-zero pointer 是 detection
  trampoline，缓存它会让每次调用重复探测。也可以按 Highway 明确支持的方式由 HYLE selector 直接绑定
  `N_<TARGET>` symbol。
- fixed/exact artifact 使用 compiler baseline 与 `HWY_COMPILE_ONLY_STATIC`；source-fat 每个生成 TU 使用
  `HWY_COMPILE_ONLY_SCALAR` 或 `HWY_COMPILE_ONLY_STATIC`，并编译期断言 `HWY_STATIC_TARGET` 与声明 tier
  一致。两条路径都在算法模板内直接使用 Highway，普通 façade 不把 vector type 跨 TU/shared boundary。
  tiny fused hot loop 走 exact 或 expert source，不要求 runtime route 冒充零开销。
- HYLE canonical x86 tier 不承诺 AES/PCLMUL；direct Highway TU 使用上游支持的
  `HWY_DISABLE_PCLMUL_AES` 让 vendor target 与 Platform facts 对齐，不能为了满足 Highway target 名称偷偷
  提高部署最低要求。专用 crypto kernel 必须显式探测/绑定其扩展。
- Highway 在经典 cl.exe 下禁用 AVX3 family。fixed/exact AVX-512 fail-fast，unknown-CPU source-fat 明确
  capped at AVX2；需要 AVX-512 时使用 clang-cl，而不是把静默降档当作 AVX-512。

## 升级规则

1. 先冻结官方 commit，不追逐浮动 HEAD；同一 capability 最多两个 strong peer。
2. 先比较语义、accuracy、layout、alias、异常与 ISA，再比较性能。
3. 只验证受影响 capability：广 correctness + 少量正常 consumer sentinel；不重跑全部操作笛卡尔积。
4. direct wrapper 可行时删除长期 port 与平行 backend；若 port 必须存在，只移植与当前 origin family
   有关的 upstream 修复。
5. 任何不同 wire format、稳定性、NaN、dispatch 或资源合同的结果都标记 `NOT_COMPARABLE`。
6. 内容确定后由 CMake integrated-source hash 工具机械重算，禁止手写一个“看起来更新”的 hash。

## 明确负空间

Benchmark checkout 是 peer，不是安装产品依赖；Native FFI、稳定跨工具链 C++ ABI、per-ISA DLL、tensor/
model operator 和 GPU backend 不因 vendor 升级进入 SIMD。阶段性“哪个 op 曾赢”的记录只留在
implementation/evidence，不进入本架构页。
