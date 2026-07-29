# HYLE 预发布就绪度与模块严格收口队列（2026-07-29）

## 结论

HYLE 的“实现进度”与“当前标准的收口证书”必须分开读：除 compute/graph 仍有产品能力尾，
其余模块的主体实现基本存在；当前持有严格证书的是
**core、error、platform、mem、time、build/package、util-core、obs、container**；core 因真实
SIMD/vqsort 消费者暴露的 compiler/cache primitive 重复实现已在 2026-07-29 完成三平台局部复证。mem 在 2026-07-26 成熟库红队中发现的
P0/API/分发性能缺口均已修复，并由 clean `c640072` 的三平台 shared same-link 与正确性/ABI/
sanitizer 证书重新冻结。早期标成 closure/frozen 的任务仍是有效施工证据，不自动等于新标准认证。

| 层次 | 当前判断 | 下一动作 |
|---|---|---|
| 严格认证基线 | core、error、platform、mem、time、build/package、util-core、obs、container | 只因明确触发器或依赖受影响项局部回验；core 与 StringInterner Apple cache-layout 边均已复证转绿 |
| 已实现、待新标准复审 | math、util-runtime、sched、io、simd | 按本文队列逐模块收口，不按旧 benchmark 形状默认通过 |
| 仍有能力尾 | compute、graph | 先用真实北极星补能力，再执行同一严格合同 |
| 整库发布面 | build/package 架构已重签；持续回归迭代中 | 随模块验收递增 focused consumer/sanitizer/feature 门；最终 ABI/FFI/bundle 总验收仍在 R6 |

build/package system 的模块 target 图、可组合子集与三种交付形状已经重签；在其他模块尚未验收、
仍会改变的前提下，2026-07-29 又启动持续回归任务，把 shared/static package、complete-source 与
AArch64 compile 从代表性组件提升为当前 all-component CPU 图，并明确 all-build 不等于模块验收。
其架构证书没有被重开，但验证集合将随每个模块收口递增。当前审查单元转为 **util-runtime**；
`util-core seam`、obs 与 container 的严格证书不因此作废。

## 底座五模块交叉引用复验（2026-07-25）

在继续收口新模块前，`core/error/platform/mem/time` 已按源码 include、链接 target、安装后
consumer 和二进制导出四层复验。最终二进制依赖图为：

```text
core      -> (none)
error     -> (none)
platform  -> error
time      -> error
mem       -> error + platform
```

合法但不增加二进制依赖的头文件边只有四类：time 的 timer queue 复用
`core/intrusive_list.h`；mem 复用 `core/cache.h`，fiber stack pool 复用 util-core 的
`util/spin_lock.h`；time 的 native clock 包含
`hyle/platform.h` 读取 `HYLE_OS_*` 编译环境宏。`hyle/platform.h` 是零链接的编译环境头，
不等同于 `hyle_platform` target。platform/mem/time 的 C shim 只通过 error 明确保留的
`src/error/status_access.h` 桥接 `hyle_status`；HYLE C++ 实现没有绕经 C ABI，也没有形成
私有实现反向依赖。

复验发现并修掉一个真实产品缺陷：public `SystemAllocator` 的 shared 消费者曾直接引用
`mi_*`，导致用户必须额外链接私有 `mimalloc_static`，测试中的手写依赖又遮住了这个问题。
现在 static/source 形状仍直接内联 mimalloc 热路径；shared 形状改为调用 `hyle_mem` 导出的
direct-import backend seam，mimalloc 保持 target-private。所有仅用于遮蔽传播错误的测试/
benchmark 直连依赖均已删除；真正把 raw mimalloc 当性能对手的 mem benchmark 仍可显式链接。

验收证据：Windows shared/static 五模块组均为 20/20；Linux GCC 14 shared 20/20，最终
static mem 闭包在 `-Werror` 下干净构建；AppleClang shared 20/20。三平台 shared/static
安装包的纯 C 与 C++ `SystemAllocator` consumer 均为 2/2，且 ELF/Mach-O shared
`hyle_mem` 的 `mi_`/`_mi_` 导出计数为 0。组件解析也逐根验证为
`core`、`error`、`error+platform`、`error+time`、`error+platform+mem`，没有环、反向边或
未声明的二进制依赖。

因此这五个底座模块的**相互引用可以关闭**；util-core 对 mem 的受影响边也已回验，下一审查
单元当时是 container；该单元和随后独立复审的 build/package 均已收口，当前到
util-runtime。性能边界另作
诚实限定：Windows 已量到 shared direct-import 相对同进程 raw mimalloc 约 4.5–6% 的 DSO
边界成本；三平台“同链接形状”的最终性能证书仍登记在 performance 文档，不能把本次依赖
正确性结论扩大成性能结论。该性能尾随后由 clean `c640072` 的 Win/WSL/M4 双 QoS same-link
n=15 证书关闭；默认 heap、SystemStd、system PMR 与 realloc 无 material LOSE。

这里关闭的是五模块之间的依赖方向，不是再次冻结每个模块的全部产品合同。后续成熟库
能力红队分别发现了 mem 的 alignment/并发统计/lifetime P0 与 time 的 integral duration/
时钟命名/证书 freshness P0；这些实现缺口与两模块三平台复证均已完成。过程与证据见
[`mem-mature-gap-audit.md`](./mem-mature-gap-audit.md) 与
[`time-mature-gap-audit.md`](./time-mature-gap-audit.md)。这些问题不推翻上面的依赖图。

## 严格收口合同

完整方法、审查表和证书/重开规则以
[`module-closure-methodology`](../architecture/module-closure-methodology.md) 为唯一稳定信源；
下面八项只是 readiness 状态表使用的压缩门。模块同时满足后，才标成“当前标准严格收口”：

1. **真实功能面**：列出外部用户与 HYLE 内部消费者的常用场景；同本地成熟库能力对账，缺项实现或写清负空间和理由。
2. **架构与类型形状**：ownership、lifetime、concurrency、failure、组合方式贴正常调用；删除只为 benchmark 方便而存在的 API/元数据/类型。
3. **复杂度与性能**：先争取同类 SOTA 算法复杂度，再在同合同正常 workload 对拍强 peer；微基准只能解释机制，不能决定公开形状。
4. **正确性与平台**：所有权、OOM/overflow、并发、失效、平台差异和 feature dispatch 红队；Windows/Linux/macOS，适用时 ASan/UBSan/TSan/特殊 ISA。
5. **C++ 易用性**：首选 source API 的默认/高级/组合三条用户旅程有可复制 recipe；RAII/typed 入口、默认值、失效规则和错误传播不要求用户理解内部状态，HYLE C++ 内部不得绕经 C ABI。
6. **C ABI 完整性与易用性**：覆盖全部语言无关能力；共享 `hyle_status`/error 语义，并提供所需 init/default/typed inline 便利层；纯 C 头、动态导出、所有权、callback/visitor 生命周期与 FFI 组合机械验证。
7. **依赖收束**：向下确认正确使用已冻结 error/platform/mem/time 等能力；向上扫描直接消费者是否自造重复机制或错误调用。只重测实际受影响边。
8. **文档归位**：architecture 管稳定设计，performance 管方法和点值，`guide/<module>.md` 管“什么情况选什么”的 C++/语言无关入口，`c-<module>.md` 只管精确 C 映射与坑，milestone 保留施工记录。

参考库不是 feature 数量竞赛。每个借鉴项必须能回答“哪个真实场景需要它”；每个性能结论
必须同合同、同所有权、同并发度、同错误语义。否则宁可登记未知，也不能用一个容易赢的
benchmark 宣称 SOTA。

## 施工顺序

排序依据是**接口爆炸半径 + 消费关系**，不是目录顺序，也不是谁的 benchmark 更多：

1. **core**（✅ 2026-07-29 局部复证完成）：旧 version/intrusive 证书保留；真实 SIMD/vqsort
   消费的 prefetch/assume/restrict/flatten/unroll 已补齐，并完成 Windows/WSL/Apple 受影响面复证。
2. **build/package system**（✅ 2026-07-29）：模块归属拆分后，组件/cache/终态依赖、三种交付、双测试形状、三平台 package consumer 与 configure/incremental 性能重签完成。
3. **util-core seam**（✅ 2026-07-25）：低层 header-only 能力及 include/link 边已收口；mem 的 SpinLock 消费已局部回验，Registry 未被提前冻结。
4. **obs**（✅ 2026-07-29）：runtime 已按新方法复证 exact-byte record、RAII/quiescent sink、compile floor 与 24-symbol C surface；trace gate 及 mem/sched/io/compute hooks 的既有证书保留。
5. **container**（✅ 2026-07-27）：功能负空间、复杂度、争用、allocator/error/platform、C/C++ 形状、三平台/Sanitizer 与性能证书已收口。
6. **util-runtime / 完整 util**：container 稳定后再冻结 Registry、type-erasure 与完整 C ABI，合并 util-core 的证书。
7. **sched**：汇合 platform/mem/time/container；按真实任务、同步、取消、关闭和 blocking workload 审算法及 API。
8. **io**：在 sched 合同稳定后审 proactor、网络/文件/TLS/HTTP 的普通路径、错误、deadline/cancel 与跨后端性能。
9. **math↔simd 边界审查，再收 simd**：先裁决 scalar/std/SIMD/消费域归属，再冻结 SIMD dispatch、dtype/math/sort/search。
10. **compute**：补完真实 CPU/GPU/ggml/Vulkan 产品能力后收口；复用已冻结 simd/sched/mem/platform。
11. **graph**：最后汇合 sched/compute/simd，完成 DAG 与数据并行北极星能力后收口。
12. **发布总验收**：install/export、跨编译器 ABI、纯 C/TS/Python 最小 FFI 和发布 bundle。

util 的两步是其真实产品分层：前一步只冻结低层自包含头，后一步才冻结依赖 container 的
`hyle_util` runtime。两步必须使用同一任务页累计证据，禁止演变为两轮全模块复审。

math 的处理相反：它当前不是经过独立需求设计的产品，而是迁移残余。严格复审必须与 simd
的标量/向量边界一起做；可能结论包括保留清晰 scalar helper、迁移到 std/其他模块或撤销
独立 target。先决定边界，再决定 guide/C ABI，不能用文档和导出把历史偶然固化下来。

## 当前模块判定

| 模块 | 实现状态 | 当前严格证书 |
|---|---|---|
| error | 功能/API/native detail/24-symbol C ABI 完成 | ✅ 2026-07-29 消费者合同再收口；C 首错聚合直接可达；Win/WSL/M4 n=15 同能力 Abseil 无 LOSE，较弱 std 下界单列；全库 adoption 随各消费者复审继续 |
| platform | 功能主体完成 | ✅ 消费边已复关；FileInfo 的 signed ns-precision Unix timestamp 已通过 Win/WSL/M4 与 WSL ASan/UBSan，NUMA 主体和既有性能证书未受影响 |
| mem | 功能/API/P0/分发性能已完成 | ✅ 彻底收口；clean `c640072` 保留 allocator 三平台 shared same-link n=15 与 WSL TSan；当前 109/109 C surface，HugeRegion selected-node/chunk/weighted C++/C 已补齐，Linux 极细条带使用固定批次 placement |
| time | 功能主体完成 | ✅ 严格收口；signed Unix/OS/chrono 互操作、42/42 C surface 与 platform/io 消费边已通过 Win/WSL/M4、WSL ASan/UBSan；timer SOTA 证书继续有效 |
| core | 功能与 compiler/cache primitive 完成 | ✅ `71dabf7` 的 IntrusiveList 证书保留；新增 prefetch/assume/restrict/flatten/unroll 已经 Win/WSL/M4 codegen、正确性和真实消费者复证后重新冻结 |
| build/package system | 组织与三种交付形状完成 | ✅ 架构严格收口；🚧 持续门升级：all-component package/source/AArch64 与动态 exported-target consumer 已落代码，待本轮三平台及 GitHub 额度恢复后复证；focused API consumers 随各模块验收递增 |
| util-core | 低层公开头能力完成 | ✅ 严格收口；实际使用形状、依赖边、三平台正确性/TSan/性能已签证 |
| obs | 功能主体完成 | ✅ 2026-07-29 `4d31bcf`/`06cec5a` 严格复证；exact-byte structured producer、RAII/quiescent sink、compile-floor log + trace seam；24/24 C surface，三平台 Release/package/n=15 与 WSL sanitizer 已签证 |
| container | 功能/API/依赖/主体性能证书完成 | ✅ 主体严格证书保留；core 语义消费者扫描将 StringInterner Apple shard spacing 64→128B，M4 correctness 与 8t normal workload 受影响项已复证 |
| util-runtime、math、sched、io | 功能主体基本完成 | 待按新合同认证；旧 closure 是输入证据，不是自动通行证 |
| simd | 功能主体基本完成 | 🟡 strict Round 1 完成；f16/bf16 lanes、共享 error、scalable 真硅性能与最终发布面仍为 P0 |
| compute、graph | 主干存在、不是 stub | 仍有能力尾；完成能力后按同合同认证 |

## 停止与重开规则

严格收口不是声称“永远没有更快写法”，而是停止无证据的重复全审。通过后只有以下事件可重开：

- correctness/security/ABI 缺陷；
- 新平台、ISA、后端或编译器使原合同不成立；
- 同合同强 peer 证明复杂度档落后或可复现的实质性能差距；
- 真实消费者暴露缺失能力、难用接口或错误依赖；
- 发布工程要求改变模块边界。

亚纳秒点值波动、没有 workload 的猜测 API、新库出现、命名偏好、为了增加 op/LOC，都不能
单独触发重开。下游模块改变只回验它真正触碰的依赖边。

## 证据信源

- 构建拓扑和慢变顺序：[`../architecture/roadmap.md`](../architecture/roadmap.md)；参考库角色：[`../architecture/references.md`](../architecture/references.md)。
- 当前任务与 LOC：[`README.md`](./README.md)；每个旧 closure 文件只是该轮施工证据。
- 稳定设计：[`../architecture/`](../architecture/README.md)；性能：[`../performance/`](../performance/README.md)；用户入口：[`../guide/`](../guide/README.md)。
- 已认证或当前复开审计的模块/系统：[`error-module-reclosure.md`](./error-module-reclosure.md)、[`platform-remaining.md`](./platform-remaining.md)、[`mem-mature-gap-audit.md`](./mem-mature-gap-audit.md)、[`time-mature-gap-audit.md`](./time-mature-gap-audit.md)、[`core-module-closure.md`](./core-module-closure.md)、[`build-package-system-closure.md`](./build-package-system-closure.md)、[`util-module-closure.md`](./util-module-closure.md)、[`obs-module-closure.md`](./obs-module-closure.md)、[`container-module-closure.md`](./container-module-closure.md)。绿色表示当前最终形状已认证，黄色表示旧证据仍是输入但已有新触发器。
