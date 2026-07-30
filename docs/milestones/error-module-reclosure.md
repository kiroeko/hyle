# error-03：error 模块实际使用导向重收口

> 本文是 Jira 式任务记录。稳定设计见
> [`../architecture/error-design.md`](../architecture/error-design.md)，用户选型见
> [`../guide/error.md`](../guide/error.md)，C/FFI 映射见
> [`../guide/c-error.md`](../guide/c-error.md)，性能数字只见
> [`../performance/benchmarks-error.md`](../performance/benchmarks-error.md)。

## 目标

把最早按 clay 和测试形状形成的 error 模块，重构成基础加速库真实调用需要的失败值：
8B `Status`、`std::expected` Result、显式语义 context、完整 C owning handle，以及
OS/CUDA/Vulkan 叶子诊断及定长机器可读 native detail。未发布，不保留未经审查的旧 ABI。

## 本轮实施

- Status 所有自身操作 `noexcept`；非法枚举归一为 `Unknown`；OOM 保留 code，context
  OOM 保留原消息。
- 新增扁平连续文本的 `WithContext`；OK/空 context 快路径在消费方头内联。首层精确分配，
  当前增量进一步让连续 rvalue 链按几何 prefix storage 渐进增长。
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

## 第一轮验收状态（2026-07-24，历史证书）

- Windows MSVC Release：全新构建目录中的 error C++/纯 C、platform C++/纯 C、
  compute、5 个 Vulkan 变体与 graph 共 11 项测试通过；当前接口 benchmark n=15 已完成。
- Windows ASan error C++/纯 C 2 项通过；C ABI 14 个 Status 符号（含 context）及 C++
  context 符号已核对，旧 handler/report 符号不存在；diff 静态检查通过。
- 当前 `3539169` 已在 Windows MSVC 19.50、WSL GCC 14.2、Apple M4 AppleClang 21
  分别通过 error 43/43 C++ + 纯 C；两套 benchmark 各平台 n=15。当前功能面所有同形
  核心/正常路径均为 WIN/TIE，无未解释 LOSE，数字只见 performance。
- header-inline retain 增量另通过 Windows ASan 与 WSL TSan 的 43/43 C++ + 纯 C，跨线程
  shared-message 测试覆盖实际 retain/release 竞争。
- `b6f2470` 将 raw rep 所有权转移彻底移出公开 C++ API，只保留 `src/error` 内部 friend
  bridge；C handle 新增 `HYLE_STATUS_INIT`，并用编译期测试锁住不再公开 adopt/release。
- 最终增量在 Windows/WSL/M4 的 error、mem、platform、io C bridge 定向测试均为 5/5；
  Windows ASan error C++/纯 C 为 2/2。三层真实语义 context 的 n=15 配对结果分别为
  **WIN 5.114 / 6.290 / 6.724x**，复杂度边界与外推限制已写入 architecture/performance。

## 易用性/正常路径复审增量

本轮不是“参考库有就照搬”：新增项均由 HYLE 自身调用证实（公开 `Result<void>`、大量
Result 错误构造、graph 首错聚合、测试/日志的人读格式化）。任意 payload、StatusBuilder、
自动日志和 `CO_*` 宏继续拒绝进入 error 核心。固定 native domain/code 后续由真实 Vulkan
恢复需求触发，见本文末的 2026-07-29 再开审计。benchmark 覆盖正常格式化、
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

以上条件在当时已经满足。error 后续确因“message 无法安全支撑 Vulkan 恢复分支”的真实
触发器重开并于 2026-07-29 再次收口；
HYLE 全库是否仍有私有错误系统、漏 context 或错误码滥用，作为下一阶段 adoption audit，
不反向扩大 error 核心 API。

## 成熟库反向审计增量（2026-07-25）

- 对照 Abseil `Status/ErrnoToStatus`、Folly `Expected/exception_wrapper`、Boost.Outcome/
  `system_code` 与 oneTBB 异常边界。接受真正重复出现的 errno canonical adapter；C++ 与 C
  均覆盖，error C surface 从 14 个增至 16 个并新增机械导出/独立头门禁。
- platform 的 POSIX fs、MappedFile、VM、NUMA translator 改为复用统一 errno taxonomy，仅
  保留操作语义明确的 override；thread 的 EAGAIN 资源耗尽映射正是这种 override。
- 没有引入任意 payload、StatusBuilder、自动 source location/backtrace、exception_ptr 或
  隐式日志：当前没有程序化消费者，加入会扩大每个失败值的形状和依赖。
- errno 工厂为 O(1) 分类、一次最终 Status 分配；三平台对 Abseil 同合同 n=15 为
  2.858x / 1.011x / 2.940x WIN。Linux 小回退通过去掉无意义的整块 scratch 清零修复，未引入
  全局错误字符串 cache 或 benchmark-only 状态。

## 2026-07-29 再开审计：结构化原生失败与消费者合同证书

### 为什么重开

Vulkan 的 initialization failed、incompatible driver 与 device lost 都可合理归入
`Unavailable`，但恢复动作不同。旧形状只保留 canonical code + message，消费者若要重建
device 只能解析人读文本。这是产品缺陷，不是 benchmark 缺项。

成熟库对照给出两种方向：Abseil 的任意 payload，以及 Boost.System 风格 category/value。
HYLE 的真实 OS/CUDA/Vulkan 消费者只需要一组稳定 domain 和一个 `int64_t`，因此新增窄
`NativeError`，没有把多 payload map、type URL、Cord 或动态注册表强加给每个 Status。
`Status` 仍为 one-word handle；只有实际携带 detail 的错误才分配并多存一个 8B code。

### 产品/API 结果

- C++ 增加 `Status::FromNative` / `native_error()`；errno、Win32、Vulkan、CUDA runtime、
  CUDA driver 与 NVRTC translator 保留同一个 machine-readable 原生值。
- C ABI 增加 native create/query 及全部 length-aware `*_n` 入口，支持嵌入 NUL 且避免必须
  `strlen`；首错聚合补齐消费式 `hyle_status_update` 与表达式宏，当前 error C surface 为
  24 个导出，纯 C 独立头和机械 symbol test 锁定。
- `Result<T>` 继续是 `std::expected<T, Status>`，保留标准 monadic/API 生态；受控构造和传播
  会 fail-fast 拒绝 `Status::Ok()` 进入 error arm。宏对命名 lvalue 复制、对临时值移动。
- 非法 Result 的诊断被移入冷 `noinline` helper，正常传播不为 abort 文本扩大每个栈帧；
  没有新增 Status 字段或 benchmark-only API。
- 移除 `operator<<`，owning 文本仍由 opt-in `error/format.h` 提供；日志继续属于 obs。

### 性能证书为什么重做

最终 benchmark 只保留两种真实消费者信息合同：

1. 无附加诊断：OK、code-only、真实四调用帧传播与 caller 消费；
2. 带诊断：owning message、语义 context、machine-readable native value、errno、格式化和聚合。

GCC 的一次约 18% 假 LOSE 来自 std/Abseil 外层包装被内联而 HYLE 没有；M4 的一次约 8%
假 LOSE 来自把“三个连续 leaf 调用”误写成“四层传播”，导致多个临时 Result 同时存活。
两处都通过对称函数边界和真实 nested call chain 修复测量，产品类型没有为分数改变。

最终 `6d42add` 在 Windows MSVC 19.50、WSL GCC 14.2、Apple M4 AppleClang 21 上 n=15：
同能力 Abseil 的无诊断项全部 WIN/TIE，带诊断项全部 WIN。能力更弱的
`std::error_code/expected` 单独作为下界，Windows code-only construct 与 Result success
分别快 1.46x/1.15x，诚实保留为选型边界，不冒充同能力比较。完整点值只见
[`benchmarks-error.md`](../performance/benchmarks-error.md)。

### 正确性与最终状态

Windows error C++/纯 C/surface/header 4/4 通过；WSL 与 M4 定向 error test 通过。另补测试
确认用户直接用标准 `std::unexpected(Status::Ok())` 绕过工厂后，传播入口仍 fail-fast。
architecture、C++/语言无关 guide、C guide、performance 与本任务记录已按各自职责更新。

error 自身的功能面、复杂度、C++ API、24-symbol C ABI、消费者用法和三平台性能证书至此
再次收口。未来仍只按 architecture 记录的真实消费者触发器重开，不因参考库 API 数量或
某个微基准可赢而扩大产品面。

## 2026-07-29 增量：渐进 context 表示

实现固定于 `8dace9d`。主动对照 Abseil StatusBuilder、anyhow/error-stack 的链式 frame 与
Boost.LEAF 动态错误槽后，
确认旧扁平 `WithContext` 的连续 rvalue 链存在 O(k²) 总复制。没有等待深链先成为事故，也没有
新增 benchmark-only builder/rope：普通 Status 和连续 message/C ABI 合同保持不变；第一层
context 精确分配，第二层才升级为几何 prefix storage，唯一 rvalue 原地 prepend、共享值 COW，
总复制降为 O(最终消息字节)。

确定性测试覆盖 128 层扩容次数、消息顺序、共享 capacity rep、context alias、native detail、
OOM 无分配进展及纯 C 64 层路径。Windows/WSL/Apple M4 的 C++ 59/59 与纯 C 均通过；发布形状
（`HYLE_TEST_INSTRUMENTATION=OFF`）n=15 中，一层与旧证书持平，三层从
120.361/56.547/62.786 ns 降到 78.994/43.255/42.714 ns；对同信息 Abseil 仍为三平台 WIN。
数字只由 performance 文档承接，本节只记录任务与验收结论。

## 2026-07-29 严格回审增量：Win32 OS adapter 归一

语义重复扫描发现 `NativeErrorDomain::Win32` 虽已公开，platform 的 fs、DynLib、Thread、NUMA、
MappedFile 与 VirtualMemory 仍各自复制 `FormatMessage + canonical switch + FromNative`。这不是
SDK-specific translator，而是与 errno 同层的通用 OS 边界；六套近似表已对未知值使用
`Internal/Unavailable` 等不同分类，旧 architecture 把它留给叶模块的裁决因此重开。

- [x] C++ 新增跨 host 的 O(1) `status_code_from_win32(uint32_t)` 与
  `status_from_win32(code, context)`；Windows 错误路径用 `FormatMessage`，非 Windows 工具保留
  数字描述，成功路径仍零分配；
- [x] common code 覆盖文件/路径、权限、资源、timeout/cancel、precondition、range、unsupported、
  conflict、auth 和 availability；未知值诚实为 `Unknown`，原始 Win32 detail 始终保留；
- [x] C ABI 补 `hyle_status_code_from_win32`、`hyle_status_from_win32[_n]`，surface 从历史 24
  增至 27 个导出；length-aware context、OOM 保级和跨 host 分类有机械测试；
- [x] shared/static 安装后 error C/C++ probe 都直接消费新入口；Windows error C++/C/surface/
  header 4/4，core+error 安装 probe 两种 linkage 8/8。

CUDA/Vulkan 仍依赖 SDK 类型和 SDK-specific recovery，继续归 compute 叶模块；Win32 公共头不
include `windows.h`。platform 的六套旧 translator 将随紧接的 platform 回审迁移到公共入口，
以真实消费者证明该能力并删除重复表，而不是制造高错误率 benchmark。

本轮最终四项 LOC：全库 278,051；自实现含测试 135,523；自实现不含测试 93,027；error
实现 1,563。实现固定为 `f31e8d0`：Windows MSVC 19.50、WSL GCC 14.2、AppleClang 21 的
Release core/error 定向门各 8/8；WSL ASan/UBSan/TSan 各 8/8。三平台 shared/static 的
core/error C/C++ installed probes 每平台 8/8，证明 27-symbol ABI、header-inline 便利与组件
target 在安装态一致。

Windows 对既有两种信息合同复跑 n=15：无诊断对 Abseil 为 1 项 TIE、4 项 WIN；带诊断
10 项全部 WIN。新 Win32 入口不为跑分制造高错误率 workload；固定 switch 的 O(1) 分类、错误
路径单次 Status 分配，以及下一轮 platform 删除六套重复 translator 是它的算法/消费证据。
error 自身产品面和性能门至此重新转绿。紧接的 platform 回审已在 `a073931` 把 POSIX/Win32
generic adapter 全部迁到公共入口并删除七套近似 translator，`error -> platform` adoption edge
随三平台 Release/sanitizer 与 shared/static installed probes 转绿；CUDA/Vulkan 的 SDK-specific
translator 继续由 compute owner 持有，不属于漏迁。

## 2026-07-30 七底座 adoption 回审：HRESULT 与遗漏的 native detail

core/error/obs/platform/time/mem/util-core/util-runtime 的语义调用扫描发现三个遗漏：Windows
`_wputenv_s` 的 errno、`SetThreadDescription` 的 HRESULT、mimalloc 1 GiB huge-page 预留的
errno 曾只进入手写 message，机器可读 native detail 丢失。本轮新增跨 host、无 `windows.h`
的 `status_code_from_hresult` / `status_from_hresult` 及 C 对等入口；非负 HRESULT 为成功，
`HRESULT_FROM_WIN32` 复用 Win32 taxonomy，detail 保留独立 `HResult` domain。platform 两条边与
mem 冷控制面现直接消费 error translator，C surface 27→30。它没有改变 Status carrier、正常
成功路径或已有 errno/Win32 性能形状。Windows 受影响门 18/18、WSL GCC14 `-Werror` 七底座
闭包 36/36、Apple M4 闭包 35/35、WSL ASan error/platform/mem 12/12；Windows shared/static
安装包的纯 C/C++ error consumer 共 4/4，并直接调用新 HRESULT 入口。

本轮四项 LOC：全库 281,337；自实现含测试 136,563；自实现不含测试 93,653；当前
core/error/obs/platform/time/mem/完整 util 七底座实现 20,216。
