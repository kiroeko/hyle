# core 模块严格收口任务（2026-07-25）

> 本页只记录施工与验收。稳定设计见
> [`core-design`](../architecture/core-design.md)，使用方式见 [`guide/core`](../guide/core.md) /
> [`guide/c-core`](../guide/c-core.md)，性能数字只见
> [`benchmarks-core`](../performance/benchmarks-core.md)。

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

`71dabf7` 的 IntrusiveList 性能证书继续有效；2026-07-29 Apple 门完成后，core 的功能、
正确性、C/C++ source surface、三平台 codegen 与真实消费者复证重新转绿。

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
3. 缓存布局从硬编码 64B 宏升级为类型化 core 契约；Apple Silicon 使用 128B。runtime 精确
   查询仍归 platform。队列、SharedPool、SpinLock、scheduler 与 Swiss prefetch 已迁移。
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

## 当前验收证据

- Windows MSVC Release：core/time/mem/container/util/sched 相关 18/18；core C surface 与
  C/C++ header self-contained 通过；最终 API delta 8/8。
- WSL GCC14 Release：core/time/mem/container/util/sched 相关 18/18；该组总耗时由既有
  `util_test` death-test 子进程链主导（116.98s）。此前并行超载的四个 cache-layout 队列
  conservation 压测也已隔离 4/4；最终 API delta 8/8。
- Apple M4 AppleClang：同一相关组 18/18，最终 API delta 8/8。
- Windows/WSL Boost 同形 n=15：FIFO/remove/batch-discard 三场景均为统计 TIE；Apple 默认与
  rival-first 双顺序均为 FIFO HYLE win、remove/discard TIE。三平台无 LOSE，数字只在
  performance 页。
- 最终实现绑定 `71dabf7`；本任务页与性能证书的归档提交紧随其后。

## 四项代码量

| 固定口径 | code |
|---|---:|
| 全库（含 docs/in-tree third_party） | 246,177 |
| 自实现（含测试） | 116,282 |
| 自实现（不含测试） | 79,955 |
| core 实现 | 490 |

core 口径为 `include/hyle/core + include/hyle/c/core.h + include/hyle/c/compiler.h +
include/hyle/macros.h + include/hyle/visibility.h + src/core`，不含测试；根级 macros/visibility
与独立 C compiler 头都是 core 的逻辑产品面，不能因为不在 `core/` 目录而漏计。

2026-07-29 Apple 门通过后的最终工作树快照如下；它同时包含工作树中此前未提交的其他模块/
构建系统改动，因此前三项不是“本轮 core 净增”。提交只改变 tracked 状态，不改变这些代码量。

| 固定口径 | code |
|---|---:|
| 全库（tracked + untracked，含 docs/in-tree third_party） | 274,580 |
| 自实现（含测试） | 134,129 |
| 自实现（不含测试） | 92,124 |
| core 实现 | 646 |

## 功能与负空间结论

core 当前覆盖基础库需要的 version/cache/compiler contract/intrusive primitive，且 C 对所有
语言无关能力都有路径。runtime CPU 探测归 platform，owning container 归 container，Status
归 error；不把这些吸入零依赖根。IntrusiveList 的 push/remove 采用成熟库的 valid-membership
前置条件，time 有 exact-owner 真实消费者；可选 membership 分别走 `try_push_*` 与幂等 hook
unlink。没有 benchmark-only seam；O(1) size/iterator 生态也无需求证据，故不增长常态元数据。

## 后续触发器

只有 header/linked version 协商需求、新 cache-line 平台、真实消费者需要新的 intrusive
原语、同合同强 peer 证明复杂度/性能落档，或构建/安装 ABI 改变时重开。下一审查单元是构建
与组装系统，然后才进入 util-core seam。
