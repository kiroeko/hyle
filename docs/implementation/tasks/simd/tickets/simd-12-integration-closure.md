# SIMD-12：机制清算、集成、文档、六问与 closure

- ticket state：`SUPERSEDED`
- kind：product-mechanism liquidation / integration / verification
- depends on：SIMD-11
- budget：2026-08-30 owner 明确“不兼容旧实现”后重新冻结；此前消耗不重置，**本次计划复核后的剩余
  硬上限为 8 工程时，P1 60 分钟**
- frozen candidate：`TBD`。先前预冻结的 `c2432716` / `55cffd93` 在集成审计发现安装面泄漏与
  implementation owner 未清算后失效，不再是候选

> 2026-08-30 terminal note：owner 要求 Highway、xsimd 与 HYLE 多套候选并存，并对冻结 baseline
> 机械提取的全部 387 个 operation 逐项裁决。该事实击穿了本票“具名抽样后删除通用 register 面”的范围，
> 因此本票停止且不再授权任何删除；以下内容只保存当时的调查和计划事实。Container/Ordering 已完成的
> 合法改动与 f32 `tanh` current-peer 证据作为下一轮审计输入，不自动成为 route verdict。当前授权见
> [`simd-v3-provider-routing`](../simd-v3-provider-routing.md)。

## 为什么 SIMD-12 仍有产品修改

SIMD-00～11 已完成能力、语义、正确性和各票有限性能工作，但 SIMD-05 明确把旧 register mechanism 的
物理删除递延到“后续 family 与 Container 消费者归零”之后。当前 ordinary row/dtype/activation/search/
scan/bitpack/random 已经使用 direct Highway 或 HYLE-owned composition；遗留的 `vec.h`、`native/*` 与七套
`backend/*` 仍约 24,990 行，不能因它们不再安装就称为重构完成。

本票只完成原冻结 `SIMD-V2-VENDOR-01` / `SIMD-REG-01` 的最后清算，并在清算后冻结候选。它不重开
SIMD-00～11 的能力范围，不启动 Container 模块收口，也不新增 API、类型、ISA、benchmark shape 或中央
性能仪器。达到剩余预算仍不能删除通用 legacy mechanism 或形成候选时，SIMD 进入 `BLOCKED`；预算不得重置。

## 计划复核时的权威现状

| 项目 | 当前事实 | 结论 |
|---|---|---|
| SIMD 实现体量 | `include/hyle/simd` 约 26,496 行，`src/simd` 约 12,112 行 | 体量本身不是删除目标，但 24,990 行平行 register 机制必须有 owner |
| ordinary families | row/dtype/narrow/math/activation/search/scan/bitpack/random 已有 direct-Highway target body | 算法 body 不因机制清算重写；继承其票据证据，最终只做受影响回归 |
| Container 直接消费者 | string 与 DynamicBitSet 原先私用 `vec.h`；当前工作树已改为 direct Highway，Windows selected build 与 3 项 Container 回归通过 | 只是受影响 consumer 修复，不推进 Container 模块状态；尚缺旧 TU 删除、跨平台与有限 sentinel |
| Ordering | 约 4,639 行 Highway-derived VQSort port 是旧 register mechanism 的唯一剩余产品消费者 | 保留经证明的 ordering 算法，替换其通用 register 依赖 |
| 旧 abstraction 自测/bench | `vec_test`、旧 scalable tests、op-parity/gather/crypto/stream 等仍引用 `vec.h` | 只证明待删除 abstraction；不得反向成为产品 consumer |
| 性能优势 | SIMD-05 五类当前代表 register op 与 direct Highway 在 5% 内同档；SIMD-11 retained VQSort 对当前 Highway fallback 有 material WIN；旧矩阵另有少数高信号 WIN，但已标 `HISTORICAL/UNQUALIFIED` | 没有保留整套 register 层的证据；具名历史候选必须在删除前过一次 current-peer salvage gate |

上述工作树事实不是 candidate 或跨平台 `PASS`。Windows Container 回归只证明当前受影响路径可继续施工。

## 冻结机制裁决

| 产品面 | 终态裁决 | 性能含义 |
|---|---|---|
| ordinary row/dtype/narrow/math/search/scan/bitpack/random | `DIRECT_VENDOR` 或其上的最小 HYLE semantic composition | fixed target 热循环与 direct Highway 等价；ordinary 只多一次 batch-boundary call |
| Container string/bitset register lowering | `DIRECT_VENDOR`；NEON bitset count 保留 Container-owned CNT+UDOT specialty body | 不经 `hyle::simd`；specialty 只因既有真机机制/结果保留，不保留通用 DSL |
| VQSort ordering algorithm | `HYLE_PROVEN_IMPLEMENTATION`，继续登记 Highway `83eff393...` origin | 保住 SIMD-11 的复杂度、无分配合同和 material WIN；register lowering 改为 direct Highway 或 ordering-private 最小 adapter |
| `vec.h`、`native.h`、`native/*`、`backend/*` | `DELETE_AFTER_SALVAGE` | 没有兼容负担或保留整套 surface 的证据；当前 peer 仍证实 material WIN 的具名算法提取到 capability owner，不能连同通用 DSL 一起删除 |
| `target.h` / `fat_kernel.h` / CMake source seam | `KEEP_SOURCE_ONLY` | fixed/exact 由宏与模板直接绑定；source-fat 只在 whole-kernel 边界调用一次已缓存 pointer |
| private ordinary Runtime table | `KEEP_HYLE_SELECTION` | Platform facts 进程内解析一次；每次 ordinary whole-buffer API 调用一次间接调用；禁止再进 Highway dynamic dispatch |

“性能更优”不是要求 HYLE 包装同一 Highway 指令后凭空战胜 Highway。它要求路由到满足完整合同的最佳机制：
通用 primitive 直接获得 vendor codegen，具名 HYLE 算法优势继续保留，热循环没有 wrapper/二次 dispatch，
并且任何迁移都不相对冻结产品 baseline 产生 5% material regression。

## 剩余顺序工作包

同一时刻只执行一项。每项最多一次实现尝试、一次 codegen/机制检查、一次有限修复和一次重跑；工作包余额
不能转移为另一项的无限调查。

| ID | 状态 | 剩余工程预算 / P1 | 允许产物 | 完成或停止条件 |
|---|---|---:|---|---|
| `12-R1` Container consumer exit | `ACTIVE` | 1 h / 10 min | string/bitset direct-Highway TU、共享 locked dependency target、旧 wrapper TU 删除、当前设计事实 | Win/WSL/mac 适用 correctness；现有 string count 与 bitset count sentinel 无 material regression；失败只保留 Container-owned 具名 body，不恢复通用 DSL |
| `12-R2` Ordering mechanism isolation | `QUEUED` | 3 h / 25 min | VQSort 直接使用 `hwy::HWY_NAMESPACE`，或只位于 ordering 私有目录的最小 adapter | 全公开 ordering type/operation correctness；SIMD-11 三个既有 qualified rows 保持 5% non-inferiority；一次修复后仍失败则 `BLOCKED` |
| `12-R3` named salvage + legacy deletion | `QUEUED` | 1.5 h / 20 min | 对历史高信号项做 current-peer 有界复核；提取仍成立的最小算法；删除其余 vec/native/backend 与旧测试/bench/CMake seam | 每个候选 `EXTRACT/ROUTE/DELETE`；产品、样例、测试和 benchmark 无旧头引用；不能只从安装列表隐藏文件 |
| `12-R4` candidate and closure | `QUEUED` | 2.5 h / 5 min | 不可变 candidate、三交付/三平台、有限 P1 聚合、六问、四项 LOC 与 terminal receipt | 所有适用 gate 真实通过则 `CLOSED`；任一适用 gate 未裁决即 `BLOCKED/INCONCLUSIVE` |

### 12-R1：只修受影响 consumer，不提前收口 Container

Container 的公开类型、能力面、整体性能和文档仍留给模块顺序第 10 项。本工作包只允许：

1. 从同一 implementation header 生成 string 与 bitset 的 target TU，直接使用中央锁定 Highway；
2. Container 继续用 Platform facts 选择自己的 whole-kernel entry，不依赖 `hyle::simd` Runtime；
3. 保留已证明的 NEON CNT+UDOT count specialty，删除其余手写 suffix wrapper TU；
4. 使用现有 `string_ref_bench.cc` 的 4 KiB count 行与 `p1_bench.cc` 的 DynamicBitSet count N=4096 行，
   对冻结迁移前 HYLE baseline 做同构建/同机器重放；不新增 benchmark；
5. 只运行受影响 Container correctness/header journey。结果不能用于宣称 Container 已收口或整体 SOTA。

若 direct Highway 相对 baseline 出现超过 5% 的稳定回退，先查宽度、额外 dispatch、loop/codegen 和 compiler
blocklist；最多修一次。确有具名算法/编译器优势时将最小 body 归 Container，否则接受同档 direct vendor。

### 12-R2：保留 VQSort 算法，不保留整套寄存器库

SIMD-11 已证明当前 HYLE/Highway-derived ordering 路线相对锁定 Highway 的 classic-MSVC fallback：partial
为 `1.147x`，full f32 4096 为 `5.950x`，full f32 1M 为 `7.463x`。这证明 VQSort 算法 port 值得保留，
不证明 `vec.h` 的 387 个 spelling 或七 backend 值得保留。

本工作包保持 VQSort 算法、public POD/static entry、NaN/order、复杂度和无分配合同不变，只把
`vqsort_ops.h` 的 register binding 改为当前锁定 Highway target namespace。若 public KeyValue/UInt128 或
classic-MSVC/scalable delta 无法直接表达，只在 `src/simd/sort/vqsort_hyle/` 内保留实际被算法调用的最小
adapter；不得 include 通用 legacy header。性能只重放 SIMD-11 的上述三个 rows，并核对主循环没有新增
copy、allocation、HYLE wrapper 或第二次 dispatch。

### 12-R3：先救出真优势，再删除而不是隐藏

旧 op-parity 总账因 peer/仪器 freshness 已明确标为 `HISTORICAL/UNQUALIFIED`，其中 39/41/7 个平台 WIN
计数不能继承为当前结论。但下列具名项曾有超过 5% 且带机制解释的信号，删除前必须用当前锁定 Highway
做一次有界复核：AVX2 `LoadExpand i32`、`SetOnlyFirst i32`、`Expand i32`、NEON `DemoteTo` 与
`SumsOf8`、whole-range `copy_if`，以及 SIMD-04 曾观察到 current xsimd 在有限 valid-domain 中约快
4--9%、精度为 1 ULP 对 Highway 2 ULP 的 f32 `tanh`。`tanh` 先资格化完整 ordinary 数学合同，再同时
复核纯函数和 4096/1M tanh-GELU 真实 activation consumer；只有同一精度/特殊值/alias/tail/resource
合同下两层都稳定超过 5% 才允许提取最小算法或重新引入第二产品 vendor，孤立 raw vendor 行不能推翻
SIMD-04/08 的 direct-Highway 裁决。`ReverseBits`/`PopulationCount` 先作 current source/codegen 对照；
只有机制仍不同才进入同一 focused timing。Newton-refined reciprocal/rsqrt 属于不同 accuracy contract，
按 capability 组合裁决，不拿它和 raw estimate 冒充同合同竞速。

每个候选只有三种终态：

- 当前同合同结果仍 material WIN：把最小算法提取到 dtype/ordering/source-algorithm 等真实 owner，改成
  direct Highway 或局部 intrinsic body，并保留对应 focused property/sentinel；
- 当前结果同档，或优势只是旧 peer、placement、不同合同：路由 direct Highway；
- 没有自然公共/内部任务且只是旧 register spelling：删除。

这一步最多共享一次 current-peer build、双顺序测量和一次机制检查，不恢复 387-op matrix，也不因为一个
op 获胜保留其所在整套 backend。

删除 inventory 包括 `include/hyle/simd/vec.h`、`native.h`、`native/*`、`backend/*`。逐个处理其引用：

- 只验证 register abstraction 的测试/benchmark 删除；其历史结论保留在 implementation/evidence；
- 仍验证公开 family/scalable correctness 的测试改用 ordinary/direct target seam，而不是删除覆盖；
- installed-header filter、component-contract 例外和 `vec` build seam 随文件一起删除；
- `target.h`/`fat_kernel.h` 作为完整源码交付的 direct-Highway helper 保留，不安装；
- 终态搜索允许历史 task/evidence 提及旧路径，不允许产品、CMake、样例、当前 architecture/user/developer
  文档或活动测试/benchmark 引用。

### 12-R4：候选验证而非再开发

清算完成后才冻结 candidate，并执行：

1. Windows selected Release 的完整 public SIMD suite、header self-containment、exact/fat/source journey；
2. WSL GCC/ASan 与 macOS NEON 的 public/fixed suite；Ordering register body 变化时运行现有 SVE/RVV
   focused correctness，不用模拟器签性能；
3. 搬迁 shared SDK、搬迁 static SDK、source embedding、offline locked Highway、component/package/license
   graph，以及受影响 Container/Compute consumer；
4. unchanged family 聚合其 terminal evidence；只重放 Container 两个迁移 sentinel、Ordering 三行和一个
   ordinary Runtime sentinel。禁止重跑旧 387-op parity 或全 family/类型/size 笛卡尔积；
5. 六问、architecture/user/developer/implementation 权威分流、独立红队、四项 LOC，以及
   `check-docs.py`、`check-module-closure.py`、dependency/diff checker。

任一适用风险项未运行写 `BLOCKED`，由此无法裁决的聚合性能写 `INCONCLUSIVE`。没有资格化同合同 peer 的
family 只声明通过 consumer budget，不能外推“整个 SIMD 全面 SOTA”。

## 路由硬合同

- exact/source fixed：Highway 宏、target namespace 与模板在编译期直接绑定，runtime selector 和间接调用为零；
- source-fat / ordinary：CPU/target 在进程内解析一次，随后每次 whole-buffer API 调用只有一次缓存函数指针
  调用；这不是每个 buffer 重新探测，也不是每元素派发；
- 不允许 HYLE selector 后再次进入 `HWY_DYNAMIC_DISPATCH`，不允许逐 lane callback、整段 bridge copy、
  hidden allocation 或为了旧 spelling/layout 支付常态 metadata；
- classic cl.exe 的 Highway AVX3 blocklist 必须诚实封顶 AVX2；需要 direct AVX-512 时使用 clang-cl，
  不把 AVX2 产物命名成 AVX-512。

## 最终冻结验证清单

1. 聚合 SIMD-00～11 terminal record、MUST/ROUTED ledger、第三方 lock/license 与已知限制；确认没有被后票
   重新引入的 `PARTIAL/MISSING + KEEP`。
2. 六问固定回答：直接/间接第三方；推理 north-star 与 ggml/llama 消费完整度；同合同公平性能；Native
   FFI/C ABI 清洁负空间；source/static/shared 与 exact/fat 交付；普通/expert C++ 易用性与文档权威。
3. 独立终轮红队只攻击冻结候选的 ownership、alias/OOM、NaN/精度、ISA fallback、dispatch metadata、
   依赖升级和交付边界；新愿望进入 watchlist，不借红队重开已完成 ticket。
4. 形成不可变 candidate、verification receipt、reopen triggers 和 terminal `CLOSED/BLOCKED` 裁决。
