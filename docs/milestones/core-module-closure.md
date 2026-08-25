# core 模块严格复审记录（2026-07-25—2026-08-25）

> **当前证书（2026-08-25）：`SIGNED`。**这表示 core 按当前模块收口方法完成验收，**不表示
> 未发布的 FFI 候选已经冻结或需要兼容**。运行时行为基线为 `9459d3f`（tree
> `925b516d00226a20767361ec59b3fee057ae1b21`）；最终 source/public-contract、严格 bootstrap-order、
> incompatible-major harness 和证据输入绑定 `c0b9a84`（tree
> `795fc79e925df13c6df9750129afea7fc2029c3d`）。后续提交只改注释、证据工具/探针和 fixture/sample
> 分类，不改变 core 的运行时行为或 claim-bearing 热路径。当前队列状态仍以
> [`pre-release-readiness`](./pre-release-readiness.md) 为准。

> 本页只记录施工与验收。稳定设计见
> [`core-design`](../architecture/core-design.md)，使用方式见 [`guide/core`](../guide/core.md) /
> [`guide/c-core`](../guide/c-core.md)，性能数字只见
> [`benchmarks-core`](../performance/benchmarks-core.md)。

## 2026-08-25 Native FFI 口径更正与当前证书

本次更正不新增一个“原生 C 用户面”。HYLE 没有 native-C 产品客户；
`include/hyle/c/core.h` 使用 C17 可描述语法，是 Java、.NET、Rust、Python 等运行时直接绑定
shared library 的 **Native FFI ABI schema**。C 编译器 fixture 只验证 schema 和 calling
convention，不代表 HYLE 承诺 C source API。

| 身份 | 当前结论 |
|---|---|
| review state | 🟢 `SIGNED`；core 的 C++ source、linked identity 与当前 direct Native FFI 候选已按新口径闭合；无已发布 binding，故没有 ABI 兼容义务 |
| product implementation | `9459d3fa91e75b95d261067284451c45fe96f54a` / tree `925b516d00226a20767361ec59b3fee057ae1b21` |
| source / public contract / validation harness | `c0b9a84d81ad6df1fb81fcec6235425a8f3448ab` / tree `795fc79e925df13c6df9750129afea7fc2029c3d` |
| Native FFI export manifest | **3/3**：`hyle_ffi_abi_version`、`hyle_version`、`hyle_version_string` |
| Windows shared/direct binding | **PASS**：MSVC 19.50.35728 Release `components=core/shared` 4/4（含产品 export 3/3 与 header self-contained）；.NET SDK 10/`net8.0` 与 Java 25 从命令行给定的 exact **installed DLL** 路径加载，先解析并校验该 core artifact 的 bootstrap，再解析/调用两个产品版本符号；两者加载只导出 ABI major 2、无产品符号的 test-only DLL 时均先拒绝 generation，再确认产品符号不存在 |
| C++ delivery shapes | **PASS**：`c0b9a84` 下 Windows installed headers+shared 与 headers+static 的独立 C++ consumer 各 1/1；complete-source `add_subdirectory` static C++ consumer 1/1。三者都运行 version/cache/compiler/intrusive 正常旅程 |
| WSL shared boundary | **PASS**：detached `c0b9a84`、GCC 14.2 Release core 4/4；test-only incompatible fixture 的 ELF 动态导出只有 `hyle_ffi_abi_version` |
| macOS shared boundary | **PASS**：detached `c0b9a84`、AppleClang 21 Release core 4/4；test-only incompatible dylib 的外部定义只有 `_hyle_ffi_abi_version` |
| sanitizer | **PASS**：detached `c0b9a84`、GCC 14.2 RelWithDebInfo 的 ASan 与 UBSan core 各 4/4；core 不拥有并发 runtime，TSan 不是本模块当前适用门 |
| required CI | **WAIVED / `[skip ci]`**：本月 Actions 额度接近上限，本轮未启动新 workflow；以上 Windows/WSL/macOS fresh 真机证据承担模块签署，且不冒充持续 CI 回归已新增。build/package 后续复审再决定长期 managed-binding job |
| performance binding | **保留**：`a951f0a`→`9459d3f` 未改变 IntrusiveList、prefetch 或 compiler contract 的 claim-bearing 热实现；`9459d3f`→`c0b9a84` 也只改变合同文字、test/evidence/probe/sample 分类。因此既有三平台同合同 SOTA gate 仍绑定同一热代码，没有伪造新性能点值 |
| 全库代码 | **301,385**（tracked，含 docs/in-tree third_party） |
| 自实现（含测试） | **141,773** |
| 自实现（不含测试） | **95,984** |
| core 实现 | **689**（11 个 native code-bearing files；不含 tests/bench/docs/samples/CMake） |

可复核证据已进入版本库，不再依赖无法审计的 PowerShell transcript 或 `.scratch` 文件名：

- [`c0b9a84` 证据 manifest](../evidence/core/manifest-c0b9a84.md) 绑定完整 commit/tree、三台机器的
  实际 checkout、工具链/配置、源码/探针/产物哈希，并解释沿用旧标签的 build-cache 目录；
- [managed FFI structured evidence](../evidence/core/managed-ffi-windows-c0b9a84.json) 保存 Java/.NET
  四条正负路径的命令、stdout/stderr、return code、工具版本和输入 SHA-256；
- 同目录保存 Windows/WSL/macOS core 4/4、WSL ASan/UBSan、三种 C++ 交付 CTest 日志及
  [FFI export transcript](../evidence/core/ffi-exports-c0b9a84.log)。

当前能力必须按三层读取：

| 层 | core 当前能力 | 是否属于 direct FFI |
|---|---|---|
| C++ source / linked API | compile/runtime product identity、compiler contract、target-family cache layout、prefetch intent、`IntrusiveList` | 只有 linked runtime identity 的三个 C-schema symbols 属于 |
| native-C helper | 当前 core 不提供受支持的 C source 产品旅程；C17 只承担 ABI schema 的机械可编译性 | 否 |
| direct Native FFI | 从 exact core artifact 调用 bootstrap，取得全局 ABI generation；通过后再读 packed/string product identity | 是，恰好 3 个真实导出 |

compiler attribute、branch/assume/restrict/unroll/flatten、cache layout 和 prefetch 都在调用方 TU
决定代码形状，导成逐次 DLL 调用会改变语义或把热路变慢；`IntrusiveList` 则要求节点地址稳定、
依赖 C++ 生命周期和批量 O(1) 重链。它们合理保留为 C++ source 能力，不因未出现在 managed
binding 中而构成 FFI 缺口。runtime CPU/cache observation 仍归 `platform`，不下沉到 core。

旧 pure-C list 曾使用自指 sentinel：**任何已初始化的 list（即使为空）都不能 move/copy、
`memcpy` 或按值传递**，因为 sentinel link 指向对象自身；已链接节点地址也必须稳定，且 node/list
任一 storage 失效前必须 detach。该 helper 已从当前 core schema 删除；这条历史陷阱只解释为何
不能把旧 C fixture 当成可管理语言直接绑定的 list ABI。

下面的 `a951f0a`/`2707583` 证书、pure-C parity、两个动态 version symbols 和旧 LOC 都保留为
施工来路。它们已被本节的 `c0b9a84` source/验证 identity 和 Native FFI 判据 **superseded**，不得继续作为
当前 C/FFI 收口结论。

## 2026-08-25 旧 source/C 口径证书（历史，已 superseded）

> 本节完整保留当时的功能、性能和三平台证据。其 “C17 source API 完整”、pure-C consumer、
> “动态 ABI 只有两个 version symbols”及当时的 candidate/LOC 都不是当前 Native FFI 证书。

| 身份 | 结论 |
|---|---|
| 当时 review state | 当时为 `SIGNED`；功能、设计、复杂度、C++/C、依赖、三平台、交付、性能和文档门闭合 |
| product candidate | `a951f0a436336366df9729e663c2fccfd357cff2` / tree `714d660299bb67d2c855452e8cbdb14cd89ec841` |
| evidence harness | `2707583f2fdd0d38f77f3302480be662dfade2f6` / tree `ac85153f7d3241472ace9fc8556afdb88311082a` |
| public surface identity | SHA-256 `58038784ef0231b6ad73b7dc5e1b6622b3f8d79aa9a22c186090d1e268baba62` |
| strong peer | Boost.Intrusive clean `df8eeb90846648464e3c8e378e312f0cec772e83` |
| runner | SHA-256 `22b285cb469479522f719158e2b9c1757aefb604d4a781e801ebbd9d76dca26d` |
| 当时 verdict | core 结束关注；只有 methodology 的重开触发器才重新转黄 |

### 功能与边界裁决

core 只承担四组零上游基础合同：① compile-time 与 linked-runtime 版本 identity；②
force/no-inline、branch、assume/restrict/unroll/flatten、fatal/source-location 等 compiler contract；
③ target-family cache layout 与三种 prefetch intent；④ stable-address、caller-owned、零分配的
IntrusiveList。runtime CPU/topology 属于 platform，owning/general containers 属于 container，
时钟/timer 语义属于 time，错误值属于 error；本轮没有为了 benchmark 把这些上层状态吸进 core。

IntrusiveList 对照 Boost/Folly 后补齐双向 iterator、positional insert/erase、单节点/整表 splice、
append/swap/move、clear/consume/prefix move 与 auto-unlink。节点和整表操作维持 O(1)，只有显式
遍历、clear、consume 和 prefix scan 为 O(n)；默认 hook 仍只有两个指针，不增加 size/owner/
统计元数据。cache 推荐按编译目标确定：x86-64/ARM64EC 64/64B，native Apple ARM64
128/128B，未认证 ARM64/RV64 fallback 64/128B；dense 结构使用 target 值，只有稀疏持续争用
control state 才 opt-in 128B max。

### API 与交付裁决

- C++23 source API 按单头选择，IntrusiveList 提供 range/RAII auto-unlink 与 checked convenience；
  cache/compiler seam 均 header-inline，不制造 DLL hop。
- C17 source API 覆盖全部语言无关能力：transparent hook/list、静态初始化、const traversal、
  checked insert、pop/unlink/clear/splice/append，以及同一 cache/compiler contract。C 没有析构，
  所以 owner 必须在 storage 失效前显式 unlink/clear；对非空 list 直接 re-init 会留下悬空 hook。
- 动态 C ABI 有意只有 `hyle_version`、`hyle_version_string`；其余是 C source/编译期能力，不适合
  变成逐操作 ABI 调用。runtime version string 为模块拥有的 borrowed storage，module unload 前复制。
- installed shared/static 与 complete-source `add_subdirectory` 三种形状均以只请求 core 的 C/C++
  consumer 验证；target 传播 C++23/C17、native MSVC conforming preprocessor 与 Windows runtime
  staging。cache-layout 是预发布 source/C++ layout ABI break，所有 TU/plugin/object 必须 clean rebuild。

### 正确性、平台与消费者

- Windows MSVC 19.50 Release/Debug 的 core 受影响闭包各 20/20，clang-cl 20.1.8 core-only 5/5；
  shared/static package C/C++、source C/C++ 均通过，DLL 没有导出 header-inline version storage。
- Linux GCC 14 native-ext4 的 Release 31/31、Debug shipping 30/30、ASan/UBSan 各 9/9；
  a951 的 TSan affected closure 133/133 由产品零差异绑定。pure C/C++、ELF surface 与
  shared/static/source consumer 通过；Debug strict Werror 另暴露 sched coroutine allocation 的
  `-Wmismatched-new-delete`，属于 sched 债务，不改变 core 结果。
- Apple M4/AppleClang native Release/Debug 各 19/19，ASan/UBSan focused seam、TSan queue 4/4
  与 SharedPool/cache 5/5、shared/static/source 通过；Rosetta
  x86-64 验证 64/64B profile、core C/C++、platform cache 与 SharedPool dense layout。CXX-only
  source parent、generic deployment target 和 Rosetta all-container ISA 选择问题登记给 build/package。
- target-family footprint、runtime probe coherency、queue/map/interner/SharedPool/scheduler/time 直接
  消费边均复验；a951→270 产品树零差异使较早的 package/sanitizer 结果仍严格绑定同一产品字节。

### 性能裁决

正式产品 claim 只覆盖与 Boost 同 ownership/metadata/工作量的 N=4096 FIFO、arbitrary remove、
read-only traversal 和 auto-detach；预声明 margin 3%，每进程输出原始 paired JSON，Windows/
Linux/macOS 均跑 default 与 peer-first。Windows default remove 在 n=31 为 `UNRESOLVED`，按事前
规则保留原日志并以同 margin 扩到 n=63 后转 `PASS`。最终四条正常路径三平台（M4 另含
interactive/utility）全部 non-inferiority `PASS`；精确关系仍逐格保留
`WIN/EQUIVALENT/INCONCLUSIVE`，没有把单边 gate 改写成“平”。完整结果和 hash 见性能页；
最终提交的全库 CI 由 GitHub Actions 留档。

N=32 的方向会随等价 code shape/placement 改变，故预先降为机制诊断；whole-list append 只计
两次重链，记 `NOT_COMPARABLE/N/A`。time 的 N=8 consumer 复验又显示固定执行序会改变 peer
速度，而且 HYLE 同时维护 `size/next-deadline/owner` 强语义；它是 time 的
`NOT_COMPARABLE`/语义成本债务，不是 core list 同合同 LOSE。该发现已反写方法论：消费者门若
混入上层额外语义，必须先隔离 primitive 再按实际收费模块归因。

### 最终四项代码量

`python scripts/loc-report.py --module core --self-check` 在最终文档工作树上的固定口径：

| 口径 | code |
|---|---:|
| 全库代码（tracked，含 docs/in-tree third_party） | 300,127 |
| 自实现（含测试） | 142,185 |
| 自实现（不含测试） | 96,254 |
| core 实现 | 959 |

core 边界为 `include/hyle/core + include/hyle/c/core.h + include/hyle/c/compiler.h +
include/hyle/macros.h + include/hyle/visibility.h + src/core`；不含 tests/bench/docs/samples/CMake。

## 2026-08-25 审查期记录（当时黄色，已由上方证书取代）

> 本节中的“当前/待完成”固定描述签署前的审查现场，不代表现在仍黄色。

本轮不是给 2026-07 证书做一次局部补丁，而是按当前
[`module-closure-methodology`](../architecture/module-closure-methodology.md) 对 core 的功能、
设计、复杂度、C++/C 易用性、依赖、性能资格、平台和文档重新逐门裁决。旧勾选项只作查漏与
回归输入；没有证据要求重写已有实现，也不能把历史绿色直接迁移成当前绿色。

| 项 | 当前状态 |
|---|---|
| review state | `AUDITED`；产品/cache-layout 修改与验证仍在进行，面向用户为黄色 |
| 起点 | `a8db371` dependency checkpoint；本轮 compiler/list 施工点为 `6b21140`、`87587dd` |
| package/sample 增量 | `40c11b1` 修正 aggregate hook 默认初始化；C target 的 C17 uplift 与 native MSVC conforming preprocessor 纳入验收 |
| benchmark 起点 | `217c8be` 提供五类 workload；它不是最终性能证书 identity |
| candidate commit/tree | **none：尚未冻结**；最终由 review owner 填写 |
| current verdict | 黄色；任何旧 `TIE/WIN` 都不是当前五态 verdict |

### 本轮已作出的产品裁决

- version、compiler contract、prefetch、cache layout 与 IntrusiveList 仍属于零 HYLE 上游的 core；
  没有把 error/platform/mem/time 或 owning container 吸入根模块。
- C++ IntrusiveList 采用 stable-address、caller-owned member hook；无 size/owner 常态元数据，单节点
  与整批操作保持成熟 intrusive container 的 O(1) 档。C 以 transparent hook/list、inline pointer
  primitive 和显式 lifetime 表达同一语言无关能力，不伪造 C++ RAII/callback 动态 ABI。
- cache layout 改为 target-family recommendation：x86-64 默认 64B/64B constructive/
  destructive，native Apple ARM64 默认 128B/128B；ARM64EC 按 x64 ABI 为 64B/64B；未认证
  native Windows ARM64（`_M_ARM64`）/generic ARM64 使用 conservative 64B/128B。dense arrays
  使用 target 值，稀疏且持续跨线程写的 control atomics 可显式选择 128B max。best-effort
  runtime observation 继续归 platform，不再让所有 x86 cell 为跨家族
  maximum 永久付双倍 footprint。
- 这是有意的预发布 source/C++ layout ABI break：x86 static destructive 值从 128B 变为 64B，
  用户 C `_Alignas` 类型、header-defined C++ consumers 及其 objects/plugins 必须 clean rebuild；
  动态 C 导出仍只有两个 version symbols，但同为 0.1.0 的 runtime 数字不会检测旧/新 object 混用。
- C++ source baseline 为 C++23，pure-C 为 C17；安装/source CMake target 传播语言与 native MSVC
  preprocessor 合同。安装 SDK 的统一 public header tree 与 component target/link closure 是两件事。

### 当时的性能资格

- 当前 harness 覆盖 FIFO、refill+cancel、read-only traversal、whole-batch handoff 和 non-owning
  teardown；四类有 N=32/4096，teardown 为 N=4096。
- N=32 cancel 在 compile-time/runtime bound 的合同等价改写下方向翻转，固定为
  `INCONCLUSIVE/placement-sensitive`，不得挑有利版本。该短 aggregate 降为非阻塞机制
  诊断；小批 cancel 产品风险由 time 真实 arm/cancel/expire workload 承重。
- whole-batch splice 只计两次 O(1) 重链，是机制诊断；产品结果由 time/scheduler 的真实
  arm/cancel/expire/handoff/drain workload 承重。
- 其余 claim-bearing 行须在 sampling 前登记 equivalence margin，并完成默认序、peer-first、短
  aggregate placement 变体、三平台和真实 consumer gates；单顺序 runner 输出不是模块证书。

### 当时的剩余验收门（历史快照，非当前 TODO）

以下条目逐字保留签署前曾经缺失的证据类别；其关闭结果已经归档在上方旧口径证书，当前判据则由
本页顶部的 Native FFI 证书取代。

- 当时待冻结最终 candidate commit/tree 和 public surface identity；
- 当时待完成 Windows/MSVC、Linux/GCC、macOS/AppleClang 的 Release/Debug、适用 sanitizer 与 codegen
  绑定同一 candidate；
- 当时待完成 pure C/C++ header、nodiscard、C17/C++23、shared/static/source package consumer 全矩阵；
- 当时待完成 target-family cache layout 的 runtime coherency、对象 footprint 与直接消费者正确性/性能回归；
- 当时待完成五类 core workload 按五态标准归档，所有 claim-bearing `INCONCLUSIVE/LOSE` 有裁决；
  N=32 cancel 作为非阻塞机制诊断保留未知，但必须有 time 真实消费者门；
- 当时待完成 time/scheduler list consumer 与 container/mem/cache consumer 的实际受影响边复证；
- 当时待完成 architecture/guide/C guide/performance/readiness 最终一致性审计；
- 当时待填写四项 LOC、最终 SHA、三平台日志索引、push 和 required CI。

只有上列全部关闭，状态才能从黄色转为 `SIGNED`。以下内容均为 2026-07 历史施工记录。

## 2026-07-28 真实消费者触发的局部重开

旧证书完成了 version/cache layout/IntrusiveList，但“扫描全部直接消费者”的方法看不见没有
include core、而是在本地复制 primitive 的代码。SIMD/vqsort 复审找到私有 `Prefetch`、
`HWY_ASSUME/HWY_NOINLINE`，以及被降为空宏的 `HWY_RESTRICT/HWY_UNROLL`；本地 Abseil/Highway
证明 local/NTA/for-write prefetch、assume、restrict 和 loop hint 是成熟基础形状，并且已有
真实 kernel 消费，不是 benchmark 反推功能。

- [x] C++ 与 pure-C header-inline 补 local、non-temporal、for-write 三种 prefetch；
- [x] C++ 与 pure-C source compiler seam 补 assume/restrict/flatten/unroll，共用 backend；vqsort 删除对应
  本地 compiler branches；
- [x] Windows MSVC Release/Debug core C/C++、surface 与 header self-contained 4/4；
- [x] Windows SIMD/vqsort 正确性；KV32 partial-sort n=15 对 Highway 为 WIN；
- [x] WSL GCC14 core 4/4、SIMD/vqsort 1/1；同一 KV32 workload 对 Highway 为 TIE；
- [x] MSVC ARM64 C/C++ 交叉编译，三种 hint 反汇编为 PLDL1KEEP/PLDL1STRM/PSTL1KEEP；
- [x] C++ 任务入口与 C 能力映射复审；纯 C 补齐 hook/init、空/NULL、非法 splice、back/clear
  生命周期机械覆盖，batch/遍历组合写入 guide；
- [x] 全库硬编码 cache-layout 语义扫描：`StringInterner::Shard` 从 `alignas(64)` 迁移到 core
  destructive-interference 常量；SIMD/entry scratch 的其余 64B 是向量/最大对象对齐，不误改；
- [x] Apple M4 AppleClang/ARM64 最终复证：core C/C++/codegen、NEON vqsort normal workload，
  以及采用 128B core spacing 后的 StringInterner correctness + concurrent normal workload；
- [x] 最终文档与四项 LOC 工作树快照；
- [x] Apple 门完成后绑定实现 commit、提交并推送。

当时将 `71dabf7` 的 IntrusiveList 性能证书继续视为有效；2026-07-29 Apple 门完成后，
core 的功能、正确性、C/C++ source surface、三平台 codegen 与真实消费者复证在当时重新转绿。

## 完成定义

- [x] 清点 `core/`、根级 macros/visibility、公开 C 头及全部直接消费者；
- [x] 按真实 plugin、cache layout、stable-node batch 场景裁定能力，不从 benchmark 反推 API；
- [x] 对照 oneTBB version、Boost/Folly intrusive 与成熟 cache interference 形状；
- [x] C++ 生命周期/复杂度、纯 C 等价能力、动态符号与 header 自包含有机械测试；
- [x] Windows/WSL/Apple 正确性与三平台同形性能全部归档；
- [x] 四项 LOC、文档索引、提交与推送完成。

## 本轮变更

1. 删除纯 scaffolding `scaffolding_ok()` 与 `HYLE_PAD` 测试形状；宏只保留真实消费者。
2. 版本定义收为一份 C/C++ 共用 source，区分 header compile-time 与 linked runtime；CMake
   project version 用 `static_assert` 防漂移，字符串返回不再构造 `std::string`。
3. 缓存布局从硬编码 64B 宏升级为类型化 core 契约；Apple Silicon 使用 128B。best-effort
   runtime observation 仍归 platform。队列、SharedPool、SpinLock、scheduler 与 Swiss prefetch 已迁移。
4. IntrusiveList 补 Folly/Boost-shaped auto-unlink hook、list clear/destructor、O(1) move 和
   launder；不加 size/owner 元数据。time delivery-list 析构语义同步回验。
5. C 头补零分配 intrusive list inline 能力、静态缓存布局与完整 pure-C smoke；动态 ABI 仍仅
   两个 version 符号，并新增 manifest/surface gate。
6. IntrusiveList 的测试与 benchmark 从 container 归回 core；新增节点/list 析构、hook copy/move、
   batch splice/consume 和错误 membership 测试。
7. 移除 `macros.h` 的无关标准头后，修复 `mem/system_inline.h` 对 `SIZE_MAX` 的隐式 include
   依赖；这是下游 header 自包含修复，不回退为根头兜底。
8. 补独立 `hyle/c/compiler.h`：C kernel 取得 assume/restrict/flatten/unroll/default-unroll，同 C++ 共用
   language-neutral backend；保持零导出符号，并补 pure-C 有效路径与 Debug 违约门。
9. 对齐成熟 Highway kernel 的递归内联合同，补 C/C++ `FLATTEN`；vqsort 使用统一 core seam，
   不再在 port 中保留另一套编译器分支。

## 2026-07-29 Apple 最终门

- Apple M4 / AppleClang 21：core Release 4/4、Debug 4/4；core+SIMD+container 受影响组
  Release 11/11。
- pure-C Release object 中 compiler-contract loop 保持本地向量化/展开；三种 prefetch 分别
  生成 `prfm pldl1keep`、`prfm pldl1strm`、`prfm pstl1keep`，无 runtime helper/DLL hop。
- M4 真机捕获 NEON partial `ReduceMin/ReduceMax` 把 inactive physical zero lanes 纳入结果的
  正确性缺陷；同族审计后 SSE4/AVX2/AVX512 的部分归约也改为只折叠 active lanes，并扩展
  sum/mul/min/max/and/or 回归。Windows SIMD 5/5、WSL core+SIMD 9/9、M4 受影响组 11/11。
- KV32 partial-sort 的默认/对手优先双顺序结果不一致且 spread 较高，严格按 order-swap 规则
  判整体统计 TIE，不把单顺序的 sign 结果写成稳定胜负。Apple 128B spacing 后 StringInterner
  8-thread normal workload 为 14.678 ns/op，对同合同 sharded Abseil+mutex 19.679 ns/op，
  15/15、1.341x WIN。

## 2026-07 历史验收证据

- Windows MSVC Release：core/time/mem/container/util/sched 相关 18/18；core C surface 与
  C/C++ header self-contained 通过；最终 API delta 8/8。
- WSL GCC14 Release：core/time/mem/container/util/sched 相关 18/18；该组总耗时由既有
  `util_test` death-test 子进程链主导（116.98s）。此前并行超载的四个 cache-layout 队列
  conservation 压测也已隔离 4/4；最终 API delta 8/8。
- Apple M4 AppleClang：同一相关组 18/18，最终 API delta 8/8。
- Windows/WSL Boost 同形 n=15：FIFO/remove/batch-discard 三场景当时均记为统计 TIE；
  Apple 默认与 rival-first 双顺序当时均记为 FIFO HYLE win、remove/discard TIE。
  “三平台无 LOSE”只是旧统计规则下的历史结论，不映射当前五态；数字只在
  performance 页。
- 最终实现绑定 `71dabf7`；本任务页与性能证书的归档提交紧随其后。

## 2026-07 历史四项代码量

| 固定口径 | code |
|---|---:|
| 全库（含 docs/in-tree third_party） | 246,177 |
| 自实现（含测试） | 116,282 |
| 自实现（不含测试） | 79,955 |
| core 实现 | 490 |

core 口径为 `include/hyle/core + include/hyle/c/core.h + include/hyle/c/compiler.h +
include/hyle/macros.h + include/hyle/visibility.h + src/core`，不含测试；根级 macros/visibility
与独立 C compiler 头都是 core 的逻辑产品面，不能因为不在 `core/` 目录而漏计。

## 2026-07-29 严格回审增量：纯 C compiler parity 与安装态消费

按 `module-closure-methodology` 重新从真实 C kernel 用户反查，发现旧结论把“C 有
assume/restrict/flatten/unroll”误写成“C 对全部语言无关能力都有路径”。C++ 已有的强/禁内联、
branch hint、unreachable 与 release-surviving fatal assert 同样是纯 C 内核/扩展的自然源码
能力，而不是动态 ABI；缺失会迫使用户复制工具链分支。

- [x] `compiler_defs.h` 成为 C/C++ force-inline、noinline、likely/unlikely、unreachable 以及既有
  assume/restrict/flatten/unroll 的单一 backend；两种公开面只保留语言命名与诊断差异；
- [x] C 新增 `HYLE_C_FORCE_INLINE/NOINLINE/LIKELY/UNLIKELY/UNREACHABLE/FATAL_ASSERT`；未知
  compiler 安全退化，UB 与 fail-fast 边界写入 C guide；
- [x] MSVC Release core C++/C/surface/header 4/4；新宏由纯 C TU 与 C++ death test 覆盖；
- [x] 新增安装后 `samples/package/core/{c,cpp}`，shared/static 各自只请求并链接 `hyle::core`，
  两种语言 4/4；此前只有源码树测试、没有真实 installed component consumer 的证据缺口关闭。

本增量没有添加 runtime 状态、导出符号或 benchmark-only 类型。宏自身没有可独立测量的算法
workload，性能仍由真实 SIMD/kernel 消费者承担；本轮只要求三平台 codegen/编译合同不产生
helper/DLL hop。Windows MSVC 19.50、WSL GCC 14.2、AppleClang 21 的 Release core/error
定向门均为 8/8；GCC/M4 的 force-inline helper 不产生符号，MSVC 仅保留 TU-local static
section，三者 noinline 入口均无 DLL/helper 引用。WSL ASan/UBSan/TSan 又各 8/8。

Windows、Linux、M4 的 shared/static 安装包都由独立 core C/C++ consumer 复证（每平台 4/4）；
实现固定为 `f31e8d0`。因此当时判定 core 自身的功能面、复杂度、C/C++ source API、
安装组件和文档重新转绿，并采用“只因新消费者/依赖变化重开受影响边”的口径。
该口径已被本页顶部 2026-08 完整复审覆盖，不是当前证书。

本轮最终四项 LOC：全库 278,051；自实现含测试 135,523；自实现不含测试 93,027；core
实现 653。全库数包含两个安装 probe 与本轮最终文档。

2026-07-29 Apple 门通过后的最终工作树快照如下；它同时包含工作树中此前未提交的其他模块/
构建系统改动，因此前三项不是“本轮 core 净增”。提交只改变 tracked 状态，不改变这些代码量。

| 固定口径 | code |
|---|---:|
| 全库（tracked + untracked，含 docs/in-tree third_party） | 274,580 |
| 自实现（含测试） | 134,129 |
| 自实现（不含测试） | 92,124 |
| core 实现 | 646 |

## 2026-07 历史功能与负空间结论

当时的证书判断 core 覆盖基础库需要的 version/cache/compiler contract/intrusive primitive，
且 C 对所有语言无关能力都有路径。runtime CPU 探测归 platform，owning container 归 container，Status
归 error；不把这些吸入零依赖根。IntrusiveList 的 push/remove 采用成熟库的 valid-membership
前置条件，time 有 exact-owner 真实消费者；可选 membership 分别走 `try_push_*` 与幂等 hook
unlink。没有 benchmark-only seam；O(1) size/iterator 生态也无需求证据，故不增长常态元数据。

## 2026-07 证书当时登记的后续触发器

只有 header/linked version 协商需求、新 cache-line 平台、真实消费者需要新的 intrusive
原语、同合同强 peer 证明复杂度/性能落档，或构建/安装 ABI 改变时重开。下一审查单元是构建
与组装系统，然后才进入 util-core seam。
