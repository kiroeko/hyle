# error-03：error 模块实际使用导向重收口

> 本文是 Jira 式任务记录。稳定设计见
> [`../architecture/error-design.md`](../architecture/error-design.md)，用户选型见
> [`../guide/error.md`](../guide/error.md)，C/FFI 映射见
> [`../guide/c-error.md`](../guide/c-error.md)，性能数字只见
> [`../performance/benchmarks-error.md`](../performance/benchmarks-error.md)。

## 目标

把最早按 clay 和测试形状形成的 error 模块，重构成基础加速库真实调用需要的失败值：
8B `Status`、`std::expected` Result、显式语义 context、完整 C owning handle，以及
OS/CUDA/Vulkan 叶子诊断。未发布，不保留未经审查的旧 ABI。

## 本轮实施

- Status 所有自身操作 `noexcept`；非法枚举归一为 `Unknown`；OOM 保留 code，context
  OOM 保留原消息。
- 新增一次分配、扁平文本的 `WithContext`；OK/空 context 快路径在消费方头内联。
- 传播宏收成 `HYLE_RETURN_IF_ERROR` / `HYLE_ASSIGN_OR_RETURN`，并迁入 compute/graph。
- `make_unexpected(Status)` 收短常见 Result 错误返回；`HYLE_RETURN_IF_ERROR` 覆盖真实公开
  API 已使用的 `Result<void>`；`Status::Update`/`IgnoreError` 覆盖首错聚合和显式忽略。
- C++ owning string/stream 格式化拆到 opt-in `error/format.h`；底层 buffer-fill 继续作为
  C ABI/日志基础，并改为块复制长消息。
- heap Status retain 采用成熟 carrier 的 header-inline relaxed 原子路径，最后 release/free
  仍留 DLL；合法 tagged word 的 heap 判断收成单比较，关闭 Linux/M4 codegen 翻转。
- C ABI 保留 Status 的 create/clone/move/free/query/format/context 全能力；删除无人使用的
  error handler、record、soft assert 和第二套日志 C API。
- CUDA、CUDA driver、NVRTC、Vulkan、Thread OS 边界保留 operation、原生名称、数值与
  可得的供应商说明；compute 语义边界增加 context。
- 当前 architecture、guide、performance 分别承接设计、recipe 和数字；error-01/02
  只保留为历史任务。

## 验收状态（2026-07-24，完成）

- Windows MSVC Release：全新构建目录中的 error C++/纯 C、platform C++/纯 C、
  compute、5 个 Vulkan 变体与 graph 共 11 项测试通过；当前接口 benchmark n=15 已完成。
- Windows ASan error C++/纯 C 2 项通过；C ABI 14 个 Status 符号（含 context）及 C++
  context 符号已核对，旧 handler/report 符号不存在；diff 静态检查通过。
- 当前 `3539169` 已在 Windows MSVC 19.50、WSL GCC 14.2、Apple M4 AppleClang 21
  分别通过 error 43/43 C++ + 纯 C；两套 benchmark 各平台 n=15。当前功能面所有同形
  核心/正常路径均为 WIN/TIE，无未解释 LOSE，数字只见 performance。
- header-inline retain 增量另通过 Windows ASan 与 WSL TSan 的 43/43 C++ + 纯 C，跨线程
  shared-message 测试覆盖实际 retain/release 竞争。

## 易用性/正常路径复审增量

本轮不是“参考库有就照搬”：新增项均由 HYLE 自身调用证实（公开 `Result<void>`、大量
Result 错误构造、graph 首错聚合、测试/日志的人读格式化）。payload、原生码公共 detail、
StatusBuilder、自动日志和 `CO_*` 宏继续拒绝进入 error 核心。benchmark 覆盖正常格式化、
`Result<void>` 与聚合形状；核心与正常路径拆成两个 executable，避免 whole-TU inline
budget 污染核心证据。

## 全库 adoption audit：mem 消费者切片（2026-07-24）

- roadmap 明确 error 是 P0 零上游根；mem 显式 PUBLIC 依赖 error，不存在依赖环。
- mem 的句柄工厂、external init、HugeRegion、leak copy 与 1GiB page reserve 等冷路径
  统一为 `hyle_status + out`；进入时清空 out，HugeRegion 保留原始 VM/NUMA Status。
- 删除 mem 私有 `hyle_mem_error`、`hyle_mem_last_error*` TLS 通道，消除 stale-error 与
  “NULL 后另查线程状态”的两步协议；C 用户直接使用 error 的 code/message/传播宏。
- alloc/realloc、region bump、pool pop/acquire 等热路径仍返回指针/NULL，未增加 Status
  构造、TLS 写或 DLL 调用；成功冷工厂返回 8B 零值 Status，零分配。
- 回审原生 C++ 后，将唯一仍为 bool 的进程级冷策略 `reserve_huge_1g_pages` 改为 Status；
  C ABI 直接转移原始 Status，不再把失败二次猜成笼统 Unavailable。pool/region 的
  `valid()/NULL` 保留为原地不可移动 owning object 与 admission 热路的值合同，不引入
  `Result<unique_ptr>` 控制分配。
- 回审 instrumentation 后，`StatsAllocator`/`LeakTracker` 冷构造改为 `Result`，C ABI 直接转移
  同一 Status；被测 backing 与诊断 metadata 显式分域，构造任一点 OOM 都原域回滚。由此
  Stats 可以包住 fixed-block/strict-quota allocator，而不让 control/hash 偷占或违反其容量。
- HugeRegion rollback 与 FiberStackPool 析构式归还中的非主错误，用 `IgnoreError()` 明确标注
  “保留原错误/无可报告调用者”，不再用 C cast 丢弃共享 error。
- Windows Release 的 mem C++/纯 C 定向测试通过。该工作属于 error adoption，不重开
  已冻结的 mem 能力面或性能设计。

## 完成条件

以上条件已经满足，error 自身功能面冻结。后续只按 architecture 的真实触发器重开；
HYLE 全库是否仍有私有错误系统、漏 context 或错误码滥用，作为下一阶段 adoption audit，
不反向扩大 error 核心 API。
