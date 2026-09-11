# Process v1：受管子进程生命周期

<!-- MODULE CLOSURE RECORD: AUDITED -->

## Closure card

- component / owner / round：`process` / Process maintainers / `process-v1-managed-child`；kind=`NEW`；
  canonical checkout=`C:\public\project-hyle\hyle`；baseline=
  `c13c1d0af21a85521b237f71548437c4a5afba83`。component identity 已登记为 `QUEUED`，尚无 target、
  product 或公开头。
- trigger 与证据：[基础能力扩展决策](../../../architecture/foundation-capability-expansion.md)准入 G03。
  llama.cpp 的 server/router 与 multimodal helper、vLLM/SGLang 的 worker/server supervisor 以及 ds4 的工具
  lifecycle 都重复组合 argv/env/cwd、stdio、wait、timeout、tree termination 和 reap；HYLE 现有 Platform、
  Sched 与 IO 只有分散原子，不能自然完成该任务。
- 正常 consumer 任务：不经过 shell 地启动一个模型 worker/helper，选择继承、null、已打开文件或异步 pipe
  stdio；读 readiness/log、写控制输入；同步或从 HYLE fiber 等待退出；必要时请求终止或强制终止整个受管
  process tree；任何 owner abandonment 最终都可靠 reap。
- 主产品纵切：推理 server 启动长期 worker，捕获 stdout/stderr，deadline/cancel 只取消等待而不偷偷杀进程，
  shutdown 明确关闭 pipe、终止 tree 并 drain。正交纵切：渲染/资产工具启动短命 compiler/converter，以文件
  stdio、替换环境和 cwd 运行，取得 typed exit。
- scope：逐项 argv、exact/PATH executable、cwd、环境继承加 set/unset 或 replacement、stdin/stdout/stderr
  inherit/null/open-file/pipe（含 stderr→stdout merge）；move-only child owner；try-wait、同步/fiber-native
  wait、deadline/cancel；POSIX process group 与 Windows Job Object tree ownership；request/force termination、
  explicit release-to-reaper，以及 module runtime stop-admission/terminate/drain。
- non-goals：shell 字符串/pipeline、PTY/terminal emulation、sandbox/container/user/token、resource limits、daemon
  policy、restart/backoff/readiness probe、log rotation、JSON/RPC schema、remote process、Native FFI、Compute/Graph。
- 候选依赖待 AUDITED 裁决：优先复用 `core,error,time,platform,platform_thread,platform_fs,mem,util_core,
  container_intrusive,sched,io_core`；只为匿名 child pipe 所需的窄 adopt seam 定向重开 io_core，不复制第二
  event loop、timer、cancel、allocator、error 或 pipe runtime。
- 性能 tier：`P0`。spawn/exit 是控制路径，先审算法复杂度、alloc/handle/thread/syscall、wait wake 与常驻
  metadata；只有找到完整 argv/env/stdio/tree/reap 同合同 peer 才运行一次 P1，不以 raw `CreateProcess`/
  `posix_spawn` 对完整 façade 排名。
- 具名源码：Boost.Process v2 `696b785c77352491ebe0e7c4a8dc348bbc9d5d53`、libuv
  `3e606bc30a842701f8d12007d3ade662d258b76b`；consumer 为 llama.cpp `e71b80510c848c`、ggml
  `e91ded11bdcd`、ds4 `f62ca29a3087`、vLLM `51da0ca66c80`、SGLang `dcebe8c4733a`。
- 调查与纠正预算：一次 consumer/peer/API/backend 源码审计；AUDITED 前冻结一个公开 API、一个 runtime/
  reaper ownership 和三 OS carrier；至多四个产品批次（public/core、POSIX、Windows、integration/tests）。
  correctness 固定失败先看源码/handle ownership，不重复运行寻找 PASS。
- 预声明验证：empty/NUL/invalid argv/env/cwd、PATH/exact、Unicode/quoting、全部 stdio 组合、large bidirectional
  pipe/backpressure/EOF、spawn failure、exit code/signal、try/wait、多次 wait、cancel/deadline race、request/
  force tree termination、owner destruction、explicit detach、runtime shutdown、concurrent spawn/exit、OOM、外部
  child 不被 reap；Windows/Linux/macOS、ASan/UBSan/适用 TSan、逐头、source/static/shared、红队、P0。
- stop / blocker：若不能在不引入第二 runtime、不抢占外部 child、不泄漏/zombie、也不让每个 child 永久
  占一个 HYLE blocking worker 的前提下给三平台一致 lifecycle，则 `BLOCKED`；不能降低成 benchmark launcher
  或把不同 termination 语义伪装成统一 graceful stop。

## SCOPED 问题的审计裁决

1. 公开 owner 固定为显式 `ProcessRuntime` 与 move-only `ChildProcess`。runtime 是 admission、native exit
   carrier 和最终 reap owner；child 是操作与 pipe owner。child 被放弃时立即关闭 parent pipe、请求强制终止
   tree，再把 OS 生命周期交还 runtime，析构不等待。`release_to_reaper()` 是明确的“继续运行但仍由 runtime
   reap”，不是泄漏 native handle 的 detach。
2. 三平台 carrier 固定为 Linux `pidfd + epoll`、Darwin `EVFILT_PROC + kqueue`、Windows registered wait
   向单一 IO completion carrier 投递。每个 runtime 至多一个 HYLE carrier thread；没有轮询、`waitpid(-1)`、
   SIGCHLD process-global handler、每 child blocking worker 或第二 task runtime。Linux kernel 不支持 pidfd 时
   `create/spawn` 返回 typed `Unimplemented`，不静默退化为轮询。
3. pipe 复用 io_core `LocalStream` 的 proactor、deadline/cancel 和 backpressure。io_core 需要一次窄重开，加入
   无名字、只可 adopt、不可 connect/listen 的 anonymous stream endpoint；Process 不公开 native fd/HANDLE，
   也不伪造 filesystem/named-pipe 地址。
4. 公共名字固定为 `request_termination()` 与 `force_kill_tree()`。前者仅在 POSIX 映射进程组 `SIGTERM`；
   Windows 返回 `Unimplemented`，应用通过 stdin/protocol 合作退出。后者映射 POSIX process-group `SIGKILL`
   与 Windows Job Object termination；typed exit 明确区分 normal、signal 和 HYLE requested force。
5. runtime/state 使用注入的 `AllocatorRef` 和双层引用：registry 在 reap 前持一份 process state，公开 child
   持一份；process state 反向保活 runtime state。`shutdown()` 先拒绝新 spawn，等待正在 spawn 的调用完成，
   关闭仍存 parent pipe、force tree、等全部 native completion/reap，停止并 join 唯一 carrier。重复 shutdown
   幂等；runtime 析构执行同一闭合路径，不要求用户记住隐藏 cleanup 顺序。

## 能力、消费者与 peer 源码审计（`AUDITED`）

| 能力族 | 真实消费 / 成熟来源 | 现状 | 裁决 |
|---|---|---|---|
| argv / executable | llama.cpp tool/MCP helper；Boost.Process、libuv | HYLE 无 owner | `FIX`：逐项 UTF-8 argv，exact/PATH 明示，永不默认 shell |
| cwd / environment | llama.cpp `common_subproc`；Boost process_environment | 只有 process-global env 原子 | `FIX`：inherit+set/unset 或 replacement，所有 view 仅借到 spawn 返回 |
| stdio | llama.cpp readiness/log/control；libuv stdio container | 只有 file/local stream 原子 | `FIX`：inherit/null/file/pipe/merge，pipe 直接成为 HYLE async stream |
| exit / bounded wait | vLLM/SGLang worker supervisor；Boost async_wait | 无 child state | `FIX`：O(1) cached try/wait，thread/fiber 共用，取消只取消等待 |
| tree lifetime / reap | llama.cpp grandchild pipe regression；Windows libuv job | 无组合 owner | `FIX`：POSIX group、Windows job，abandon 与 runtime shutdown 均 reap |
| shell / PTY / policy | application/service 层 | 不属于原子 owner | `REJECT`：不因 benchmark launcher 扩张产品面 |

源码锚点如下：

- llama.cpp `e71b80510c84` 的 `common/subproc.cpp:19-100` 已组合 argv/env/cwd、三 stdio、terminate、join，
  `tools/server/server-tools.cpp:185-289` 又为 timeout 建线程；MCP grandchild fixture 明确记录只杀 direct child
  会令继承 pipe 永不 EOF。这证明缺的是 tree/reap owner，而不是一个薄 `system()` wrapper。
- Boost.Process v2 `696b785c7735` 的 `process.hpp:241-393`、`stdio.hpp:302-350` 与
  `environment.hpp:1695-1815` 证明 move-only process、stdio、环境和 async wait 是成熟形状，但其 completion
  绑定 Asio，不能成为 HYLE 的第二 runtime。
- libuv `3e606bc30a84` 的 `include/uv.h:1047-1196` 与 `src/win/process.c:74-130,445-1050` 证明 stdio
  inheritance、Windows quoting/CreateProcess 和 job ownership 的机制；其 loop/type universe 不进入 HYLE API。
- vLLM `51da0ca66c80` 的 multiprocess executor / DP supervisor 与 SGLang `dcebe8c4733a` 的 process-tree
  helpers 重复 stop、kill、join；ds4 没有导出另一套更低层 MUST。ggml 本体也没有改变 process owner 边界。

## 冻结的公开 C++ 形状

```cpp
enum class ExecutableSearch { kExact, kPath };
enum class EnvironmentMode { kInherit, kReplace };
struct EnvironmentChange { string_view name; optional<string_view> value; };
struct ProcessStdio { inherit(); null(); pipe(); file(const FileHandle&); merge_stdout(); };
struct ProcessStartOptions {
  string_view executable;
  span<const string_view> arguments; // 不含 argv[0]
  ExecutableSearch executable_search;
  optional<string_view> working_directory;
  EnvironmentMode environment_mode;
  span<const EnvironmentChange> environment_changes;
  ProcessStdio stdin_stream, stdout_stream, stderr_stream;
  io::IoContext* pipe_io;
  unsigned pipe_owner;
};
struct ProcessWaitOptions { optional<MonotonicTimePoint> deadline; const CancelToken* cancel; };
class ProcessRuntime { static Result<ProcessRuntime> create(AllocatorRef); spawn(...); begin_shutdown(); shutdown(); };
class ChildProcess { try_wait(); wait(...); request_termination(); force_kill_tree(); release_to_reaper(); /* pipes */ };
```

`arguments` 不含 argv[0]，实现始终把 executable 作为 argv[0]；这覆盖普通 direct-exec，不为 POSIX-only
custom argv0 污染 portable API。`ProcessStdio::file` 借用 `FileHandle` 到同步 spawn 返回并在 child 侧复制/
dup；不转移 caller owner。任一 `pipe()` 要求有效 `pipe_io/pipe_owner`，成功后分别通过
`stdin_pipe()`、`stdout_pipe()`、`stderr_pipe()` 返回 child-owned `LocalStream*`；merge 不制造第二条 pipe。
所有状态查询与 wait 可并发，pipe 自身仍遵守 IO 的每方向单 in-flight 合同。

`wait()` 返回 `Result<ProcessExit>`：进程完成是值，wait 的 cancel/deadline/owner misuse 是 Status；取消或
超时绝不终止 child，调用者可再次等待。`try_wait()` 只读已经由 carrier 发布的状态，不执行 OS wait。
Process ID 是诊断值，不是可关闭/可等待的 native handle。没有 Native FFI 或旧 C API 兼容面。

## backend、资源与复杂度账单

| 路线 | spawn / exit carrier | 常驻与热路径成本 | 复杂度 / 风险封口 |
|---|---|---|---|
| Linux | `posix_spawn[p]` + process group；每 child 一个 pidfd，runtime 一个 epoll+eventfd+thread | O(child) state/fd；无 polling、无 SIGCHLD handler；普通 Sched/IO 对象零新增字段 | spawn O(argv+env)，exit O(1) event + exact `waitpid(pid)` |
| Darwin | `posix_spawn[p]` + group；runtime 一个 kqueue/user wake/thread，per-child EVFILT_PROC | O(child) kernel filter/state；不抢收外部 child | spawn O(argv+env)，exit O(1) event + exact waitpid |
| Windows | `CreateProcessW` suspended + STARTUPINFOEX handle list；先 assign kill-on-close Job 再 resume；registered wait 投到 runtime carrier | O(child) process/job/wait handles；不继承无关 handles；无 busy poll | quoting/env O(total bytes log entries for required env sort)，exit O(1) |
| wait | cached state + `sched::ConditionVariable`；CancelCallback 只唤醒 waiter | 无每次 wait allocation、无 OS syscall；裸线程 park，fiber suspend | try O(1)，wait O(1) wake；多 waiter notify-all |
| stdio pipe | POSIX socketpair、Windows overlapped named-pipe pair，parent 交给 LocalStream | 每 pipe 一对 handle；payload 走既有 proactor，无额外 adapter copy | IO 原有 backpressure/cancel/deadline合同 |

process 是 spawn/control 路径，性能等级保持 `P0`。Boost/libuv 没有同时满足 HYLE allocator、fiber wait、
LocalStream、tree ownership、abandon/reap 和 runtime shutdown 的同合同入口，因此不预造 P1 排名；raw
`posix_spawn`/`CreateProcessW` 与完整 façade 是明确禁止的苹果对橘子比较。

## 冻结有限实现集

| 批次 | 唯一允许改动 | 验收 |
|---|---|---|
| `G03-C01` | 公开值类型、ProcessRuntime/ChildProcess owner、状态/ref/wait core 与 component/product build seam | 逐头、invalid/OOM、try/repeated/cancel/deadline、move/abandon/shutdown |
| `G03-C02` | POSIX spawn/env/stdio/process-group、Linux pidfd/epoll 与 Darwin kqueue carrier | exact/PATH、Unicode bytes、全部 stdio、外部 child 不被 reap、tree/EOF |
| `G03-C03` | Windows UTF-8/UTF-16、CRT quoting、sorted env、restricted handle list、named pipe、Job/registered wait | empty/quote/backslash/Unicode、handle leak、grandchild tree、concurrent spawn |
| `G03-C04` | io_core anonymous adopt 的合法定向重开、组合测试、三平台/三交付、架构与用户文档 | source/static/shared、sanitizer、red-team、P0、依赖图/registry 一致 |

公开 API、OS carrier 和资源上限至此为 **PASS**，允许进入上述四批实现。correctness、平台、交付和红队尚未
运行，不得提前写为通过；失败先检查 ownership/source，不增加第五批或反复采样寻找偶然 PASS。
