# Process v1：受管子进程生命周期

<!-- MODULE CLOSURE RECORD: SCOPED -->

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

## SCOPED 待裁决问题

1. 公开 owner 是单一 `ChildProcess` 还是额外显式 runtime；谁在 child handle 丢弃后继续持有 reap 责任？
2. POSIX 优先 `posix_spawn[p]`、Windows `CreateProcessW + STARTUPINFOEX handle list`；异步退出分别应使用
   Linux pidfd、Darwin kqueue、Windows registered wait，还是可证明等价且不轮询的更小 carrier？
3. pipe 是否可复用 io_core 的 `LocalStream` + anonymous adopt，而不把 OS native handle 或假的 endpoint
   暴露给用户？
4. `request_terminate` 与 `force_kill` 的跨平台差异如何命名，确保 Windows 不冒充 POSIX SIGTERM？
5. runtime shutdown、Child 析构与 explicit detach 的引用/allocator 生命周期如何在零 zombie 与非阻塞析构
   之间闭合？

本卡只冻结问题、范围和预算；registry 为 `SCOPED` 后才能继续能力/API/backend 审计，当前不得写产品代码。
