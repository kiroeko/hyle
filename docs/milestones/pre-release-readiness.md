# HYLE 预发布就绪度与模块严格收口队列（2026-08-25）

> **2026-08-25 Native FFI 口径更正：**HYLE 没有 native-C 产品客户；`include/hyle/c/*.h`
> 是 Java、.NET、Rust、Python 等运行时直接加载 shared library 所用的 C-syntax ABI schema。
> 过去的 pure-C include/compile/link、`static inline`/宏 recipe 和 C 测试覆盖只能证明 schema 或
> native-C helper 可用，**不等于 direct Native FFI 收口**。core 的运行时行为基线为 `9459d3f`，
> source/public-contract、正负 bootstrap harness 与持久化证据绑定 `c0b9a84`；除 core 外的所有模块当前统一标黄，等待逐模块
> 先裁决真实 foreign-runtime 任务。没有任务可以明确 `FFI=N/A`；只有保留 direct FFI 的模块才
> 重审真实 exports、opaque/POD/callback/buffer ownership、managed lifetime、batch 形状和
> Java/.NET direct binding。旧证书继续保留为历史施工/回归输入，但已被本口径 superseded。

## 结论

HYLE 的“历史任务完成”与“当前严格收口”必须分开读。2026-07-27--30 的证书保存设计来路、
已发现缺陷和可复用回归，但不能自动成为现在的证书，尤其不能以 native-C fixture 代替真实
managed-runtime 绑定。core 的 C++ source/性能主体沿用此前严格复审，并在 `c0b9a84` 重新裁定
Native FFI 边界：编译期 compiler/cache/prefetch/list 能力合理不导出，direct FFI 只保留一个
ABI bootstrap 和两个 product identity symbols。其余模块须各自建立 review 记录、冻结 candidate，
任何 direct-binding、所有权或 lifetime 未知都保持黄色。

| 层次 | 当前判断 | 下一动作 |
|---|---|---|
| 当前严格证书 | core | 运行时行为基线 `9459d3f`，source/public-contract/验证 harness `c0b9a84`；Native FFI 候选 manifest 为 3 个符号，Windows Java/.NET exact installed-DLL 正向调用与 incompatible-major 早拒绝、WSL/macOS shared CTest/fixture 单导出已通过；无已发布 binding，故没有兼容义务 |
| 旧 C/API 施工与证书 | error、platform、mem、time、obs、util、build/package、container、simd、sched、io、math | pure-C/API 数字只作设计/算法/回归输入；无论当时是否签署，当前全部黄色并按 managed direct-binding 标准逐个重审 |
| 本轮已有中间快照 | mem/container/obs/util 的 Windows Release 与 GCC 14 WSL focused；container 的 Windows/GCC 14、obs 的 Windows installed shared C/C++ consumer；obs exact Tracy 0.14.1 trace-on | 后续改动可使其失效；不扩大为最终 clean、三平台、static、sanitizer、完整性能证书或再次收口 |
| 尚未整体收口 | container、math、sched、io、simd | 🟡 本轮只验证依赖升级触及项；功能与 Native FFI 均留给后续独立任务 |
| 仍有能力尾 | compute、graph | 🟡 先用真实北极星补能力，再执行同一严格合同与 Native FFI 裁决 |
| 整库发布面 | build/package 的 2026-07 架构证书是历史输入 | 🟡 在各模块 direct FFI 边界稳定后，复审 shared artifact、exact-path managed smoke、依赖图、离线/source bundle 与 all-component CPU 图 |

build/package system 的模块 target 图、可组合子集与三种交付形状曾在 2026-07-29 重签；当时又
启动持续回归任务，把 shared/static package、complete-source 与 AArch64 compile 从代表性组件
提升为 all-component CPU 图，并明确 all-build 不等于模块验收。本轮仍保留这些设计成果，但会在
前七模块边界稳定后按当前标准完整复审，不能用一次全图 build 替代 package usability、组件隔离、
许可证/offline provenance 或三种交付物证书。

### 依赖升级实际触达矩阵（相对 `6dcd038`，仅作复审输入）

下表解释为什么旧证书失效，不再界定完整复审的上限，也不声明当前 dirty working tree 已通过。
详细依赖身份与证据门见
[`dependency-upgrade-2026-08`](./dependency-upgrade-2026-08.md)。

> 下表沿用的 “C ABI/纯 C/symbol surface” 是当时的变更定位词，不是当前 FFI 证书。
> 其中任何一项即使再次通过，也只能成为 schema/helper 证据；direct FFI 仍须按本页新门单独验收。

| 历史收口范围 | 本批次实际改动 | 完整复审的优先证据入口 |
|---|---|---|
| core | 公开 C/C++ prefetch、Clang unroll 条件、intrusive-list callback/container-of 合同 | 优先复证头消费者、compiler/ISA codegen、sanitizer 与点值；功能/API/依赖仍按全量 closure 重新裁决 |
| error | 公开 C status code 固定宽 ABI 与 C adapter/ABI 检查 | 优先复证纯 C/C++、surface、`-fshort-enums`/跨编译器 ABI 与 nanobench；完整功能/采用扫描仍须重做 |
| platform | 公开 thread/CPU/FS/NUMA C/C++ 合同及实现，含 versioned/FFI record、所有权和错误路径 | 优先复证 API/ABI、跨平台正确性、package consumer 及改动热路；完整 OS 能力/绕行扫描仍须重做 |
| mem | C NUMA 类型复用、region options/flag ABI，以及 mimalloc-v3 allocation/alignment/realloc 与 NUMA/huge-page adapter | 优先复证 v3 正确性/所有权、C/C++/三种交付、sanitizer 与 allocator 性能；全功能审查仍须重做 |
| time | `Instant` 溢出合同、C timer callback/reentrancy/cancel 语义与 clock/timer adapter | 优先复证极值/回调生命周期、C/C++/package、跨平台时钟与性能；完整 clock/timer 对账仍须重做 |
| obs | C log record/level、caller-storage sink、fiber trace、Tracy text/plot adapter 与 C 实现 | 优先复证 29-symbol/FFI/sink lifetime、trace OFF/ON static/shared、package 与热路；完整功能/API 审查仍须重做 |
| util | Registry priority ABI、frozen-name overflow 与 demangle portability | 优先复证固定宽 C ABI、OOM/overflow、C++ consumer、package/sanitizer 与 registry 性能；util-core/runtime 同轮完整复审 |
| build/package | lock/provenance、component graph、FetchContent/offline/source bundle、CI/package 合同 | 优先复证 shared/static/complete-source、模块化 installed consumer、离线输入与 all-component CPU 图；设计/可组合性仍全审 |

`container/simd/sched/io` 不进入上表的“已收口范围”：它们本轮分别只做
simdutf/hash/string、Highway/xsimd、Boost.Context fcontext、nghttp2 HTTP/2 的定向诊断和
正确性/交付回归；性能输赢是后续 closure 的输入，不是本升级任务的模块验收结论。
`compute/graph` 仍只做现有 adapter 的机械 build/test/package 兼容。

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
单元当时是 container。container 完成过多轮施工任务和局部证据，但按当前口径未取得与八个底座
相同强度的整体严格证书；build/package 与 util-runtime 则已收口。性能边界另作
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
5. **C++ 易用性**：首选 source API 的默认/高级/组合三条用户旅程有可复制 recipe；RAII/typed 入口、默认值、失效规则和错误传播不要求用户理解内部状态，HYLE C++ 内部不得绕经 Native FFI ABI。
6. **Native FFI 完整性与易用性**：先从真实外语 runtime 任务决定本模块是否需要 direct FFI；
   没有任务可明确 `N/A`，不得制造 parity export。存在任务时，C-syntax schema 只描述真实
   shared-library exports；无原生 C 产品旅程。逐项裁决 exported symbol、opaque handle、固定宽/
   evolvable POD、callback+context、buffer ownership、managed lifetime 与 batch-first 形状；
   `static inline`/宏/native-C recipe 不计 direct FFI。Java 与 .NET 必须对 exact artifact 无 shim
   直绑并先检查待调用 artifact 自己的 ABI bootstrap，WSL/macOS 至少验证真实 shared export
   和最小调用；一个 core 查询不能替其他 DLL/so/dylib 背书。编译期能力可有理由不导出。
7. **依赖收束**：向下确认正确使用当前已签署的地基（目前只有 core；随队列增加）；向上扫描
   直接消费者是否自造重复机制或错误调用。只重测实际受影响边。
8. **文档归位**：architecture 管稳定设计，performance 管方法和点值，`guide/<module>.md` 管 C++ source 选择，`c-<module>.md` 管 Native FFI schema、真实导出、managed ownership/lifetime 与 binding 映射；native-C helper 必须另列且不得冒充 FFI，milestone 保留施工记录。

参考库不是 feature 数量竞赛。每个借鉴项必须能回答“哪个真实场景需要它”；每个性能结论
必须同合同、同所有权、同并发度、同错误语义。否则宁可登记未知，也不能用一个容易赢的
benchmark 宣称 SOTA。

## 施工顺序

排序依据是**接口爆炸半径 + 消费关系**，不是目录顺序，也不是谁的 benchmark 更多：

1. **core** ✅：运行时行为基线 `9459d3f`、source/public-contract/正负 harness `c0b9a84`；C++ source 边界、3-symbol Native FFI 候选、
   Windows Java/.NET exact-path direct smoke 与 WSL/macOS shared minimal call 已裁决，当前 `SIGNED`。
2. **error**：在 core 签署后复审共享失败模型、context 复杂度、Native FFI ownership、全库 adoption 和
   两类真实错误路径性能。
3. **platform**：在 error 稳定后复审 OS/CPU/thread/fs/VM/NUMA 能力、C++ source 与 Native FFI 形状，向上扫描
   syscall/CPU probe 绕行。
4. **mem**：在 error/platform 稳定后复审 allocator/pool/region/NUMA/PMR 的功能、ownership、
   footprint、managed lifetime/buffer ABI 和同合同 allocator 性能。
5. **time**：复审 clock/duration/timer queue、溢出/callback/reentrancy、Native FFI 和真实 timer workload；
   同时承接 core IntrusiveList 的直接消费者门。
6. **obs**：复审 log/trace/metric/sink、disabled/normal hot path、caller storage、Native FFI callback/lifetime 与 Tracy seam。
7. **util-core seam**：只裁决不依赖 container runtime 的低层自包含头；完整 util 不能越过它的
   container 直接依赖先签。
8. **build/package system（已签子集）**：待前七单元 public/target/FFI 边稳定后，完整复审
   component graph、shared Native FFI artifact、三种 C++ 交付、offline/source bundle、许可证和
   持续 CI；后续模块只做受影响边回验。
9. **container → util-runtime**：container 签署后才能审 Registry/runtime；随后回验新增的
   build/package 组件闭包。再继续 sched → io → math↔simd → compute → graph。
10. **发布总验收**：install/export、Java/.NET exact-path direct binding、Windows/WSL/macOS 真实
    shared artifact 和发布 bundle；正式承诺 Rust/Python binding 时追加其 package consumer。
    native-C fixture 不可代替这些门。

util 的两步是其真实产品分层：前一步只冻结低层自包含头，后一步才冻结依赖 container 的
`hyle_util` runtime。两步使用同一能力账本累计证据，但分别签署审查单元；第二步只回验第一步
被真实影响的边，禁止演变为两轮无边界全审。

math 的处理相反：它当前不是经过独立需求设计的产品，而是迁移残余。严格复审必须与 simd
的标量/向量边界一起做；可能结论包括保留清晰 scalar helper、迁移到 std/其他模块或撤销
独立 target。先决定边界，再决定 guide/Native FFI，不能用文档和导出把历史偶然固化下来。

## 当前模块判定

| 模块 | 实现状态 | 当前严格证书 |
|---|---|---|
| core | C++ source 边界与编译期能力保持；Native FFI 候选收为 artifact bootstrap + product identity | 🟢 `SIGNED`：运行时行为基线 `9459d3f`、source/public-contract/正负 harness `c0b9a84`；FFI exports 3/3，Windows Java/.NET strict bootstrap-order exact installed-DLL 正向调用和 incompatible-major 早拒绝、WSL ELF 与 macOS Mach-O core 4/4/fixture 单导出、WSL ASan/UBSan、三种 C++ 交付均通过；无发布兼容承诺，旧 pure-C parity 仅作历史证据 |
| error | 当前完整复审已进入实现/验证；旧 raw-word 30-symbol parity 已撤，C++ carrier/context/OS UTF-8 与 6-symbol opaque managed receiver 已形成候选 | 🟡 `CANDIDATE` 前施工：待 Java/.NET exact-artifact、三平台/三交付、sanitizer、当前 peer error-only raw performance 与最终 adoption 红队；见 [`error-module-review-2026-08`](./error-module-review-2026-08.md) |
| platform | 2026-07 pure-C/ABI 历史证书存在 | 🟡 待按 opaque OS resource、callback/buffer/lifetime、Java/.NET direct binding、三平台与性能重审 |
| mem | 2026-07 pure-C/ABI 历史证书存在；mimalloc/NUMA/options 已改变输入 | 🟡 待按 allocator/pool/region/NUMA、opaque ownership、managed lifetime/buffer、三平台交付与性能重审 |
| time | 2026-07 pure-C/ABI 历史证书存在；`Instant`/timer 合同已改 | 🟡 待按 clock/timer、callback+context/reentrancy、managed lifetime/batch、Java/.NET 与五态性能重审 |
| obs | 2026-07 pure-C/ABI 历史证书存在；Tracy/record/sink/fiber 已改 | 🟡 待按真实 exports、sink/callback lifetime、disabled/normal hot path、Java/.NET 与三平台性能重审；旧 29-symbol 数字不构成现行证书 |
| util-core | 2026-07 历史证书存在；与完整 util 同轮复审 | 🟡 待重审 C++ source 边界；编译期能力可合理不导出，不能为凑 FFI parity 制造 DLL hop |
| util-runtime | 2026-07 pure-C/ABI 历史证书存在 | 🟡 待按 Registry opaque/callback、buffer ownership、managed lifetime、Java/.NET、OOM 与性能重审 |
| build/package system | 2026-07 历史证书存在；依赖与交付图直接变化 | 🟡 待各模块 FFI 边稳定后重审 shared artifacts、exact-path managed smoke、component graph、offline/license/CI |
| container | 尚未严格收口；旧 pure-C callable/export 数字存在 | 🟡 待重审 batch-first whole-buffer API、opaque ownership、managed lifetime、Java/.NET direct binding、功能与同合同性能 |
| math、sched、io | 功能主体基本完成、尚未严格收口 | 🟡 待功能边界与 Native FFI 重新裁决；既有 native-C recipe/局部测试不作 closure 结论 |
| simd | 功能主体基本完成、旧 244-symbol C 面存在 | 🟡 待以 managed whole-buffer/batch 场景重审真实 exports、buffer ownership、Java/.NET direct binding、ISA correctness 与性能；旧 pure-C 247/244 数字不构成 FFI 证书 |
| compute、graph | 半成品 | 🟡 先补真实能力，再按同一 C++ source/Native FFI 合同认证 |

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
- 当前 core 证书与施工账本：[`core-module-closure.md`](./core-module-closure.md)；当前 error 复审：[`error-module-review-2026-08.md`](./error-module-review-2026-08.md)；其余历史认证/施工输入：[`error-module-reclosure.md`](./error-module-reclosure.md)、[`platform-remaining.md`](./platform-remaining.md)、[`mem-mature-gap-audit.md`](./mem-mature-gap-audit.md)、[`time-mature-gap-audit.md`](./time-mature-gap-audit.md)、[`build-package-system-closure.md`](./build-package-system-closure.md)、[`util-module-closure.md`](./util-module-closure.md)、[`obs-module-closure.md`](./obs-module-closure.md)、[`container-module-closure.md`](./container-module-closure.md)。除 core 外，这些旧任务页中的绿色只绑定各自日期/commit；container 等未达到当前严格口径的模块不能因旧任务名带 closure 就视为已收口。
