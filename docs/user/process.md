# Process 使用指南

Process 用于从推理 server、资产管线或本地工具中启动并可靠管理 helper/worker。它只执行逐项 argv，永远
不解释 shell 字符串。源码、static SDK 与 shared SDK 使用相同的 C++23 API：选择 `process` 产品并链接
`hyle::process`。

## 启动并等待

```cpp
#include <array>
#include <utility>
#include <hyle/process/process.h>

auto made = hyle::process::ProcessRuntime::create();
if (!made) return made.error();
auto runtime = *std::move(made);

std::array<std::string_view, 2> args{"--model", "model.gguf"};
hyle::process::ProcessStartOptions start;
start.executable = "model-worker";
start.executable_search = hyle::process::ExecutableSearch::kPath;
start.arguments = args;                       // 不含 argv[0]
start.stdin_stream = hyle::process::ProcessStdio::null();
start.stdout_stream = hyle::process::ProcessStdio::null();
start.stderr_stream = hyle::process::ProcessStdio::null();

auto spawned = runtime.spawn(start);
if (!spawned) return spawned.error();
auto child = *std::move(spawned);
auto exited = child.wait();                    // thread park / fiber suspend
if (!exited) return exited.error();
runtime.shutdown().IgnoreError();
```

`kExact` 把名称直接交给 OS；`kPath` 明确请求 host PATH search。两者都不调用 shell。参数、cwd、环境 name/
value 等 view 只需活到同步 `spawn()` 返回。一个 runtime 可并发管理多个 child；不要为每个 child 创建一个
runtime。

## 环境与 stdio 怎么选

- 默认 `EnvironmentMode::kInherit`；`EnvironmentChange{name, value}` 覆盖/增加，`value == nullopt` 删除。
- `kReplace` 只使用给出的 name/value，不能带 unset 项。重复 name、NUL 或含 `=` 的 name 会失败。
- `inherit()` 适合直接共享终端/文件；`null()` 明确丢弃输出或提供 EOF 输入。
- `file(handle)` 把已经打开的同步文件绑定到 child。native file 必须保持打开到 `spawn()` 返回，之后 caller
  与 child 的副本独立。
- `pipe()` 返回 parent-side async `LocalStream*`；至少一个 pipe 时必须填写有效 `pipe_io` 和
  `pipe_owner`。读写要在该 IO worker 上执行，并遵守每方向最多一个 in-flight 操作。
- `merge_stdout()` 只允许用于 stderr，且不会再创建一条 stderr pipe。

大输出必须持续异步 drain，不能先等待 child 再读满 pipe；否则 child 可能因 OS pipe 背压停止。输入写完后
调用 `close_stdin()`，让 child 观察 EOF。返回的 pipe 指针由 `ChildProcess` 拥有，不能越过 child 的 move、
析构、release 或 runtime shutdown 使用。

## 等待、停止和关闭

`try_wait()` 只读取 carrier 已缓存的结果，不执行 OS wait。`wait()` 可被多个线程/fiber 重复调用；返回值
区分正常退出、signal 和 HYLE 请求的 force kill。deadline 或 `CancelToken` 只结束当前 wait，child 继续
运行，之后仍可再次等待。

应用应优先通过 stdin/本地协议请求合作关闭。POSIX 可用 `request_termination()` 向整个 process group 发
`SIGTERM`；Windows 会诚实返回 `Unimplemented`，不要把强杀冒充 graceful stop。最后手段是
`force_kill_tree()`。`release_to_reaper()` 让 child 继续运行但放弃操作 owner；runtime 仍会 reap，并在
shutdown 时终止它。

推荐关闭顺序：停止新的应用 admission，结束所有 pipe I/O，关闭 stdin，合作等待一个 deadline，必要时
强杀，然后调用 `ProcessRuntime::shutdown()`。runtime 使用的 allocator 必须活得比 runtime 和它产生的全部
child 更久。

## 常见边界

- HYLE 不解析引号、重定向、变量展开或 pipeline；需要 shell 时由应用显式启动 shell，并承担注入风险。
- `kPath` 使用 host 的 PATH search mechanism，不承诺由本次 replacement environment 中的 PATH 改写查找。
- POSIX 环境继承从 process-global `environ` 取快照；应用不能在另一个线程并发修改全局环境。
- Process group/Job 只管理本次 child tree；不会 `waitpid(-1)` 或抢收应用自行创建的进程。
- Process 是生命周期原子，不提供 restart、readiness、health check 或 worker pool policy；这些由推理引擎、
  服务 supervisor 或将来的组合层负责。
