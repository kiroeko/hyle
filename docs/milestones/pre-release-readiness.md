# HYLE 预发布就绪度与模块严格收口队列（2026-08-25）

> **2026-08-25 dependency-triggered / red-team 重开：**本轮重新打开
> **core、error、platform、mem、time、build/package、util、obs、container** 九个模块；
> 下文 2026-07-27--30 的 ✅ 是上一次签署基线，不再自动等于当前发布绿证。当前只可逐项
> 使用本轮已经复验的受影响边，直到九模块的 Windows/WSL/macOS、shared/static/package、
> sanitizer 和受影响性能门由本轮总验收重新签署。mem 因 mimalloc 2.3.2→3.5.0，container
> 因 simdutf→9.1.0 与默认 hash/string API 修正，obs 因 Tracy→0.14.1 与 C sink/长度合同
> 修正而明确保持打开；util 自身算法未变，但其 mem/container 下游受影响边目前只完成
> Windows 与 GCC 14 WSL focused 复验。其余六模块的最终细项由本轮交叉审计汇总，不能沿用
> 本页旧“当前全绿”措辞。当前 working tree 仍在修复本轮红队发现，任何较早的 aggregate/package
> 通过都会被后续受影响改动失效；只有最终提交上的重跑才能恢复绿证。

## 结论

HYLE 的“实现进度”与“当前标准的收口证书”必须分开读：除 compute/graph 仍有产品能力尾，
其余模块的主体实现基本存在；**九个底座模块当前全部处于本轮重开状态**。它们在
2026-07-27--30 持有的严格证书仍是有效历史输入，却不是 2026-08-25 的自动发布通行证。
core 因真实 SIMD/vqsort 消费者暴露的 compiler/cache primitive 重复实现已在 2026-07-29 完成三平台局部复证。mem 在 2026-07-26 成熟库红队中发现的
P0/API/分发性能缺口均已修复；2026-07-29 又以 `8c85331` 关闭 PMR 零字节、显式 runtime
adapter 与 installed mem consumer 合同，并由三平台 shared same-link 与正确性/ABI/sanitizer
局部证书重新冻结。早期标成 closure/frozen 的任务仍是有效施工证据，不自动等于新标准认证。

| 层次 | 当前判断 | 下一动作 |
|---|---|---|
| 上次严格认证基线 | core、error、platform、mem、time、build/package、util、obs、container | 仅作历史输入；本轮九模块均已由依赖升级/红队发现触发重开 |
| 本轮已过局部边 | mem/container/obs/util 的 Windows Release 与 GCC 14 WSL focused；container 的 Windows/GCC 14、obs 的 Windows installed shared C/C++ consumer；obs exact Tracy 0.14.1 trace-on | 不扩大为三平台、static、sanitizer 或完整性能证书 |
| 已实现、待新标准复审 | math、sched、io、simd | 按本文队列逐模块收口，不按旧 benchmark 形状默认通过 |
| 仍有能力尾 | compute、graph | 先用真实北极星补能力，再执行同一严格合同 |
| 整库发布面 | build/package 的 2026-07 架构证书只作历史输入；本轮同样重开 | 随模块验收递增 focused consumer/sanitizer/feature 门；最终 ABI/FFI/bundle 总验收仍在 R6 |

build/package system 的模块 target 图、可组合子集与三种交付形状曾在 2026-07-29 重签；当时又
启动持续回归任务，把 shared/static package、complete-source 与 AArch64 compile 从代表性组件
提升为 all-component CPU 图，并明确 all-build 不等于模块验收。2026-08-25 的依赖升级和红队
发现满足明确触发器，因此 build/package 与其余八模块一起重新进入验收。每个旧证书仍可作为
未受影响能力的输入，但必须由
本轮总验收明确裁决哪些边可保留，不能因“此前通过”自动恢复 util、obs 或 container 绿证。

## 底座五模块交叉引用复验（历史证据，2026-07-25）

> 本节原样保留 2026-07 的依赖审查结论与数字，用于解释设计来路；它不声明 2026-08-25
> working tree 已通过同一矩阵。现行结果只写入本页顶部状态表与“当前模块判定”。

在继续收口新模块前，`core/error/platform/mem/time` 已按源码 include、链接 target、安装后
consumer 和二进制导出四层复验。最终 runtime artifact 依赖图为：

```text
core      -> (none)
error     -> (none)
platform  -> error
time      -> error
mem       -> error + platform
```

component/target 解析还必须表达无 binary artifact 的低层 header seam，因此 mem 的真实
component closure 是 `mem -> error + platform + util_core`；这条边已经同时存在于中央声明和
`hyle_mem` target。不能把“util-core 不产生 DLL/so”误写成“mem 不依赖 util-core”。

“不增加独立 runtime artifact”不是固定五条边的封闭清单，必须按源码与 target 图机械审计。
当前至少包括四种形状：① target 传播的 header-only component，例如 mem 复用 util-core 的
`util/spin_lock.h`；② public value/compile-environment 头，例如 time→`core/intrusive_list.h`、
mem→`core/cache.h`、platform→time 的 C-compatible `UnixTimestamp` 和 time→零链接
`hyle/platform.h` 宏；③ private implementation header seam，例如 container 的
`str_search_impl.h` / `bitset_ops_impl.h` 复用 `hyle/simd/vec.h`，但不调用 `hyle_simd` binary；
④ compile-time cross-cut hook，例如 mem public inline 与 compiled backend 都包含
`hyle/obs/trace.h`，trace OFF 时消失，trace ON 时由中央单一 backend target 组装。另有
platform/mem/time C shim 经 error 明确保留的 private `src/error/status_access.h` 桥接
`hyle_status`。每条边都须分别检查 include 可见性、component closure、真实 link artifact 与
安装后 consumer，不能因“不新增 DLL/so”就从依赖账本隐藏。

复验发现并修掉一个真实产品缺陷：public `SystemAllocator` 的 shared 消费者曾直接引用
`mi_*`，导致用户必须额外链接私有 `mimalloc_static`，测试中的手写依赖又遮住了这个问题。
现在 static/source 形状仍直接内联 mimalloc 热路径；shared 形状改为调用 `hyle_mem` 导出的
direct-import backend seam，mimalloc 保持 target-private。所有仅用于遮蔽传播错误的测试/
benchmark 直连依赖均已删除；真正把 raw mimalloc 当性能对手的 mem benchmark 仍可显式链接。

验收证据：Windows shared/static 五模块组均为 20/20；Linux GCC 14 shared 20/20，最终
static mem 闭包在 `-Werror` 下干净构建；AppleClang shared 20/20。三平台 shared/static
安装包的纯 C 与 C++ `SystemAllocator` consumer 均为 2/2，且 ELF/Mach-O shared
`hyle_mem` 的 `mi_`/`_mi_` 导出计数为 0。组件解析也逐根验证为
`core`、`error`、`error+platform`、`error+time`、`error+platform+util_core+mem`，没有环、反向边或
未声明的二进制依赖。

因此这五个底座模块的**相互引用可以关闭**；util-core 对 mem 的受影响边也已回验，下一审查
单元当时是 container；该单元、随后独立复审的 build/package 与 util-runtime 均已收口，后续
未认证队列从 sched 继续。性能边界另作
诚实限定：Windows 已量到 shared direct-import 相对同进程 raw mimalloc 约 4.5–6% 的 DSO
边界成本；三平台“同链接形状”的最终性能证书仍登记在 performance 文档，不能把本次依赖
正确性结论扩大成性能结论。该性能尾随后由 clean `c640072` 的 Win/WSL/M4 双 QoS same-link
n=15 证书关闭；默认 heap、SystemStd、system PMR 与 realloc 无 material LOSE。

### 七组件历史组合复证（2026-07-29）

在 core、error、obs-runtime、platform、time、util-core、mem 各自严格证书完成后，又以
`b4f08f4` 为同一源码基线组装七组件闭包，而不是把独立通过机械相加：Windows MSVC 19.50
28/28、WSL GCC 14 28/28、M4 AppleClang 21 27/27 全绿；M4 唯一少项是按设计不登记的 x86 TSC
校准测试。覆盖 C++ 行为、纯 C ABI、C symbol surface、header self-contained、TSC/NUMA 与断言
失败路径。WSL 的 mem death-test 107.59s 是该环境已知的子进程启动成本，其余组合测试约 15s，
不把它误判为产品热路性能。

同一基线另生成只含七组件闭包的 Windows shared/static 安装包；13 个 source-independent focused
C/C++ consumer 在两种链接形状下共 26/26 configure/build/run 通过。中央 manifest 已在 configure
时同时投影 component closure 与真实 PUBLIC/PRIVATE target links，并由终态 property audit 拒绝
后加的未声明边；本轮 include 扫描补记了 `platform/fs -> time/value` 的 header-only 真相，没有
为它虚构 time DLL 依赖。

GitHub Actions run `30486529929` 的四个 job 均在零 step、约 3s 时失败；check annotation 明确为
组织近期付款失败或 spending limit，而不是 workflow/code failure。仓库内三平台与分发证据已绿，
远端 CI 仍须账户 Billing 恢复后重跑；不得通过缩小或停用流水线把这个外部阻塞伪装成通过。

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

1. **core**（🟡 2026-08-25 重开）：2026-07 version/intrusive 与 compiler/cache primitive
   证书仅作历史输入；本轮总验收前不恢复绿证。
2. **build/package system**（🟡 2026-08-25 重开）：2026-07 的组件/cache/终态依赖、三种交付、
   双测试形状和三平台 package 结果仅作历史输入；本轮依赖升级后的完整交付图待重签。
3. **util-core seam**（🟡 2026-08-25 重开；历史基线 `3c32be6`）：低层 header 能力由真实
   `hyle::util_core` INTERFACE component/target 承载；Ref/ScopeGuard 易用形状、mem/graph 消费、
   三平台 installed consumer/TSan/n=15 与 M4 双 QoS 是历史证据，本轮总验收前不自动恢复。
4. **obs**（🟡 2026-08-25 重开）：上次 runtime 证书是输入；C surface 已扩为 29 symbol，
   Windows/GCC 14 WSL trace-off 与 exact Tracy 0.14.1 Windows trace-on 受影响边通过，
   Windows installed shared C/C++ consumer 通过，WSL/macOS package、static/sanitizer 待总验收。
5. **container**（🟡 2026-08-25 重开）：默认 hash 等价类、公开 string helper、C btree OOM
   与 simdutf 9.1 受影响面已在 Windows/GCC 14 WSL 通过；两平台 installed shared C/C++
   consumer 也通过，macOS/static/sanitizer/正式性能仍待重新签署。
6. **util-runtime / 完整 util**（🟡 2026-08-25 重开；历史基线 `1b148ef`/`70fa4e8`）：Registry
   mutable→frozen 形状、共享 Status/allocator、完整 C ABI、TSan/package 与三平台性能结果只作
   历史输入；mem/container 下游变化与本轮总验收未完成。
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
| error | 功能/API/native detail/30-symbol C ABI 主体完成；本轮重开 | 🟡 2026-07-30 adoption/n=15 是历史输入；dependency-triggered/red-team 总验收前不恢复绿证，细项由本轮交叉审计汇总 |
| platform | 功能/API/120-symbol C ABI 主体完成；本轮重开 | 🟡 `a073931` 的三平台/package/sanitizer/性能是历史输入；本轮交叉审计与总验收未签 |
| mem | 功能/API/P0 主体存在；mimalloc 3.5.0 受影响边重开 | 🟡 Windows 与 GCC 14 WSL focused 已通过 runtime `mi_version()==30500`、usable/good/stats/collect、真实跨线程 free+owner collect、1GiB taxonomy，以及 v3 小对象弱对齐回归；shared installed/static/source、macOS、sanitizer、NUMA/大页硬件和正式 allocator 性能待本轮重签 |
| time | 功能/API/交付主体完成；本轮重开 | 🟡 2026-07-29 profiling/C parity/三平台/sanitizer/timer 性能是历史输入；本轮交叉审计与总验收未签 |
| core | 功能与 compiler/cache primitive 主体完成；本轮重开 | 🟡 `71dabf7` 与后续三平台 codegen/消费者复证是历史输入；本轮总验收前不恢复绿证 |
| build/package system | 组织与三种交付形状主体完成；本轮重开 | 🟡 历史架构证书只作输入；all-component package/source/AArch64、动态 exported-target consumer 与升级后依赖图待本轮三平台复证 |
| util-core | 低层公开头能力主体完成；本轮重开 | 🟡 `5dfb74a`/`3c32be6` 的三平台/package/TSan/性能是历史输入；mem 等真实消费者变化与本轮总验收未签 |
| obs | 功能主体完成；Tracy/C sink/长度/fiber parity 边重开 | 🟡 Windows 与 GCC 14 WSL Release trace-off、29/29 C surface、exact Tracy 0.14.1 Windows trace-on，以及 Windows installed shared C/C++ consumer 已通过；WSL/macOS package、static、sanitizer 与 feature consumer 待本轮重签；旧 runtime n=15 只覆盖未改热路 |
| container | 功能主体完成；hash/string/simdutf 边重开 | 🟡 Windows/GCC 14 WSL 各 308/308 typed/纯 C/surface/header 通过，Windows 与 GCC 14 installed shared C/C++ consumer 通过（C 含 `-fshort-enums`）；64 个 SipHash KAT、Eq/hash 拒绝负例，以及 simdutf 9.1 的单遍 validate 与 validate+count invalid/tail/ASCII/non-ASCII 回归已覆盖；float/double/string/SipHash/UTF-8 均只有 Windows 单点（UTF-8 当前点未编同合同 peer），macOS/sanitizer/static 与正式性能待重签 |
| util-runtime | Registry/demangle/C adapter 主体完成；九模块总复审中 | 🟡 自身 API/算法未改，Windows/GCC 14 WSL 受影响 mem/container 下游测试通过；其余本轮细项与三平台集成由总验收裁决，旧证书不自动恢复 |
| math、sched、io | 功能主体基本完成 | 待按新合同认证；旧 closure 是输入证据，不是自动通行证 |
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
- 历史认证记录或当前复开审计输入：[`error-module-reclosure.md`](./error-module-reclosure.md)、[`platform-remaining.md`](./platform-remaining.md)、[`mem-mature-gap-audit.md`](./mem-mature-gap-audit.md)、[`time-mature-gap-audit.md`](./time-mature-gap-audit.md)、[`core-module-closure.md`](./core-module-closure.md)、[`build-package-system-closure.md`](./build-package-system-closure.md)、[`util-module-closure.md`](./util-module-closure.md)、[`obs-module-closure.md`](./obs-module-closure.md)、[`container-module-closure.md`](./container-module-closure.md)。这些任务页中的绿色绑定各自日期/commit；现行最终形状是否认证只看本页状态表。
