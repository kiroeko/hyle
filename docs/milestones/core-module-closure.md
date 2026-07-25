# core 模块严格收口任务（2026-07-25）

> 本页只记录施工与验收。稳定设计见
> [`core-design`](../architecture/core-design.md)，使用方式见 [`guide/core`](../guide/core.md) /
> [`guide/c-core`](../guide/c-core.md)，性能数字只见
> [`benchmarks-core`](../performance/benchmarks-core.md)。

## 完成定义

- [x] 清点 `core/`、根级 macros/visibility、公开 C 头及全部直接消费者；
- [x] 按真实 plugin、cache layout、stable-node batch 场景裁定能力，不从 benchmark 反推 API；
- [x] 对照 oneTBB version、Boost/Folly intrusive 与成熟 cache interference 形状；
- [x] C++ 生命周期/复杂度、纯 C 等价能力、动态符号与 header 自包含有机械测试；
- [ ] Windows/WSL/Apple 正确性与三平台同形性能全部归档；
- [ ] 四项 LOC、文档索引、提交与推送完成。

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

## 当前验收证据

- Windows MSVC Release：core/time/mem/container/util/sched 相关 18/18；core C surface 与
  C/C++ header self-contained 通过。
- WSL GCC14 Release：core/time/mem/container/util/sched 相关 18/18；该组总耗时由既有
  `util_test` death-test 子进程链主导（116.98s）。此前并行超载的四个 cache-layout 队列
  conservation 压测也已隔离 4/4。
- Windows Boost 同形 n=15：FIFO/remove/batch-discard 三场景 paired sign 均为 HYLE win；
  数字只在 performance 页。
- WSL Boost 同形 n=15：FIFO/remove/batch-discard 均为 HYLE win；数字只在 performance 页。
- Apple 待本任务提交同步后补齐。

## 功能与负空间结论

core 当前覆盖基础库需要的 version/cache/compiler contract/intrusive primitive，且 C 对所有
语言无关能力都有路径。runtime CPU 探测归 platform，owning container 归 container，Status
归 error；不把这些吸入零依赖根。IntrusiveList 的 unchecked seam 有 time exact-owner 真实
消费者，不是 benchmark 接口；O(1) size/iterator 生态没有需求证据，故不增长元数据和面积。

## 后续触发器

只有 header/linked version 协商需求、新 cache-line 平台、真实消费者需要新的 intrusive
原语、同合同强 peer 证明复杂度/性能落档，或构建/安装 ABI 改变时重开。下一审查单元是构建
与组装系统，然后才进入 util-core seam。
