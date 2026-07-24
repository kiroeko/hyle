# error / platform / mem 功能与 C ABI 复审

> **历史快照，已被后续复审替代：**mem 固定块池变化见
> [`mem-close-plan`](./mem-close-plan.md)；error 的诊断 handler 已删除，当前产品形状见
> [`error-module-reclosure`](./error-module-reclosure.md)。后文旧称、入口计数和 handler 描述只记录
> 2026-07-19 当时的审计，不能当作当前 API 参考。

> 2026-07-19。依据
> [`c-abi-design.md`](../architecture/c-abi-design.md)、
> [`c-abi-authoring-guide.md`](../architecture/c-abi-authoring-guide.md) 与
> [`close-01-c-abi-shim.md`](./close-01-c-abi-shim.md)，对三个基础模块做
> 功能面、下游利用、红队实现、测试/性能、C ABI/guide 五维复审。

## 结论

三个模块的核心功能与 C ABI 已收口；platform 不再只是早期 VM/thread
薄面，而是覆盖全部八个公开子系统。机械核对 C 头中的调用入口：

| 模块 | C 入口 | 纯 C test 直接出现 | guide 直接出现 | 结论 |
|---|---:|---:|---:|---|
| error | 52 | 52 | 52 | 当时快照：含后来删除的诊断 handler |
| platform | 114 | 114 | 114 | wait/thread/dynlib/VM/NUMA/fs/map/CPU 全覆盖 |
| mem | 51 | 51 | 51 | allocator/arena/pool/hugepage 策略全覆盖 |

这里统计函数与 header-inline 入口，不把 enum 常量虚增成“操作数”。机械
出现不是正确性证明，所以又保留了生命周期、OOM、非法参数、并发、death、
长路径和真实数据 round-trip 测试。

## 功能与下游利用

### error

- `Status` 是单字 tagged carrier；OK/无消息错误零分配，canonical 17 码、
  `Result<T>` 与传播宏齐全。
- 当时曾提供诊断 handler；2026-07-23 reclosure 已删除，日志转由 `obs`、合同失败转由
  fail-fast。这是历史实施记录，不是当前能力。
- 下游并非只使用 `Internal`：compute 使用
  `InvalidArgument/ResourceExhausted/Unavailable/Unimplemented`，sched 使用
  `NotFound/Unavailable`；本轮把 platform thread 的 OS 错误从一律
  `Internal` 细分为资源耗尽、权限、参数与内部错误。
- **后续纠正：**这里把接口成本误写成了依赖约束。roadmap 的真实层级是
  error(P0) → platform(P1) → mem(P2)，error 自身只用 CRT `malloc/free`，因此
  mem 可以且现在显式依赖 error，不存在 mem→error→mem 环。allocator 热路继续走
  NULL 是返回形状与成本裁决，不能作为拒绝依赖 error 的理由。

### platform

- C++ 与 C 能力均覆盖 wait/Parker、Thread、DynLib、VM/VmRegion、
  hugepage/NUMA、fs/env/fd、MappedFile、CPU/cache/topology。
- scheduler 已使用 Thread 的 worker 命名与 affinity；SIMD dispatcher 使用
  CPU/ISA 探测；mem HugeRegion 使用 VM/NUMA 原语。没有“实现了却完全绕开”
  的主干能力。
- fs/path 返回 buffer-fill 或 visitor，不把 `std::string`/容器所有权跨 ABI。
  非页对齐 mapped-file offset 是真实支持能力，旧文档的“必须页对齐”已纠正。

### mem

- system allocator、MemoryArena、StackAllocator、Arena、GrowablePool、
  HugeRegion 与进程级 hugepage/NUMA 策略齐全。
- ObjectPool/GrowablePool 的模板构造语法只留 C++，但裸槽能力已经按
  size/align 擦除到 C；这不是能力缺口。
- StatsAllocator/LeakTracker/ErrorRecord 类似，是 debug 内省对象而非通用
  FFI 产品面；保持 C++-only。若未来消费者需要稳定遥测，应新增只读 snapshot
  POD，而不是导出 C++ 类布局。

## 本轮红队修复

| 风险 | 修复 |
|---|---|
| scoped handler 乱序恢复可重新发布已释放 vtable | C/C++ 共用 compare-and-restore 严格 LIFO 门；乱序 death test |
| 非法 C severity 可穿透为未定义 C++ enum 值 | 非法值归一为 Error；Fatal 只认显式常量 |
| C page align 在 `SIZE_MAX` 附近 wrap | 两个 inline align helper 加溢出 guard |
| thread OS 失败全部报 Internal | Windows/POSIX errno 映射到 canonical Status |
| thread wrapper 在 native thread 启动后 OOM | 先分配 wrapper，再 spawn，不再被迫 join 任意用户任务 |
| thread opts 无演进/越界读取护栏 | 首字段 `struct_size` + 初始化 helper；零/过小尺寸拒绝并有边界测试 |
| fs/Parker C 指针/布尔失败无可读原因 | 新增 `hyle_platform_last_error`，成功清 stale |
| fs 非法 open flags 被静默接受 | unknown bit、无 WRITE 的 create/truncate/append 均拒绝 |
| C/C++ page/string 边界 | NULL size-query、UTF-16 转换长度、touch close/stamp、长 HOME/TEMP/CWD 缓冲修正 |
| mem 成功后遗留旧 last-error | factories/acquire 成功清空；grow OOM 与 NULL pool 写原因 |
| HugeRegion 未知/冲突 flags 被吞 | C 边界拒绝并给出原因 |
| C/C++ POD/enum 两面可能漂移 | layout/trivial/offset 与关键 enum `static_assert` |

文件系统递归删除保证“不跟随当下可见的 symlink/junction”，已有跨平台测试。
它不是针对攻击者并发 rename/reparse 的 sandbox primitive：和常规 path API
一样，调用期间外部恶意改树仍有 TOCTOU。安全边界消费者应使用 OS
descriptor/handle-relative sandbox；不要把通用递归删除用于不可信并发目录。

## 测试与性能判断

- 正确性：三个纯 C consumer 对每个入口至少直接调用一次；C++ 测试继续覆盖
  Status refcount/OOM、handler 并发/重入/death、Parker/Thread、VM guard、
  symlink 数据安全、UTF-8/长路径、MappedFile 非对齐、allocator OOM/并发等。
- 性能：error benchmark 覆盖 OK construct/copy、code-only/message error、
  refcount copy、成功/失败传播，并对 std/Abseil；platform 覆盖 Parker 快慢路、
  ping-pong、thread、directory visitor、VM reserve/commit/protect、raw wait、
  timeout 与 mapped-file；mem 覆盖 mimalloc/pmr 对拍、realloc、多线程、
  bump/rewind、固定/增长/共享池。
- 不给每个 C shim 再造一份 benchmark：这些入口是薄转发，算法成本已由同一
  实现的 benchmark 覆盖。诊断、拓扑查询、权限修改等冷路径也不以纳秒吞吐
  为产品指标。若桥接层将来加入复制/分配，再单列 ABI-overhead benchmark。

本轮真机验收：

- Windows MSVC Release：全库 40/40 CTest 通过，覆盖 error/platform/mem
  的直接与下游消费者。
- WSL GCC（源码同步到 `/root/hyle` 原生 ext4 构建，不在 `/mnt/c` 编译）：
  三模块 C/C++ 六套 6/6 通过。
- Apple Silicon / Apple Clang：三模块 C/C++ 六套 6/6 通过。额外全库增量
  构建停在本次 diff 之外的 SIMD benchmark 配置：
  ARM64 `hyle_dtype_bench` 被传入 `-mavx2/-mfma/...`；不影响上述六个核心
  target 的编译与运行，但属于 SIMD CMake 的独立待修项，不能把它误写成
  “macOS 全库构建通过”。

## ABI/易用性约束

- 纯自由函数导出；无模块 api table、`api_get` 或版本宏。
- RAII 用 opaque handle，模板用 size/align 擦除，字符串用 buffer-fill，
  迭代用 callback+ctx。
- Thread 失败用 `hyle_status`；VM/NUMA/map、DynLib、Parker/fs 分别有明确
  last-error 通道。
- `hyle_thread_spawn_opts` 给不想拼 has-flags 的 FFI 用户；具名
  priority/protect/map/NUMA/open/perms enum 提升 bindgen 可发现性。
- 发布前仍需 close-01 的跨编译器链接与真实语言 FFI smoke；这不是三个模块
  内部功能缺口，而是整个发行物的最终验收门。
