# HYLE 生命周期与失败边界

本文规定跨模块的失败、取消、背压、关闭和资源隔离语义。具体类型与实现由所属模块在各自 review
中签署；本页不记录某次任务是否完成。

## 1. 失败分类

| 类别 | 含义 | 表达 |
|---|---|---|
| 程序不变量 | HYLE 或调用方违反不能恢复的合同 | fail-fast + 诊断，不能 catch 后继续 |
| 运营失败 | I/O、OOM、设备、协议或外部输入的预期失败 | `Status/Result<T>` 值，调用方显式处理 |
| 未就绪/容量状态 | poll empty、try-lock、bounded queue full 等正常控制状态 | 紧凑值/枚举，避免制造重错误对象 |
| 取消 | 调用方不再需要结果 | 协作式取消，通常映射 canonical cancelled 状态 |
| 资源压力 | quota、backpressure、admission 被拒 | 有界、可恢复、可观测的结果 |
| 故障隔离 | 进程崩溃、driver/device loss 等不能在对象内恢复的边界 | 进程/backend 隔离与上层重启策略 |

日志不是失败通道。error 保存机器可处理语义与必要诊断；obs 决定何时记录。模块不得在构造或
传播 `Status` 时自动打印，也不得以 TLS last-error、`false/-1` 或私有 error enum 绕过 error。

## 2. 异常边界

HYLE 公共操作默认不抛异常：运营失败返回值，程序 bug 立即暴露。这个规则不是为了隐藏失败，
而是让热路径、异步 completion 和跨模块组合拥有单一、显式的失败通道。

第三方 C++ API 可能抛异常。adapter 可以在最窄边界捕获外来异常并转换为 `Status`，但不能捕获
HYLE 自身不变量后继续。是否以 `-fno-exceptions` 编译是独立的 build/code-size/codegen 决策，
必须测量并确保所有依赖兼容；它不改变上述语义。

## 3. 结构化取消

取消具有 owner 和作用域。父任务结束或取消时，仍属于它的子操作必须收到信号；已经脱离作用域
的工作需要显式 owner，不得依赖栈上 token 幸存。

取消是协作式的：

1. 请求方设置取消状态；
2. 排队、等待和适当的计算边界检查；
3. 新工作停止进入；
4. 已提交到 OS/GPU/第三方 backend 的工作尝试 native cancel；
5. 即使 native cancel 不可用或失败，晚到 completion 仍被安全接收和丢弃/归还；
6. owner 在所有可能触碰资源的 completion quiescent 前保持资源存活。

“返回 cancelled”不等于底层操作已经停止。API 必须区分 request accepted、operation prevented、
completion drained 和结果 ignored，不能让用户在后台仍可能写 buffer 时释放或复用它。

## 4. Cancel-safe ownership

异步提交前就要决定谁拥有 buffer、descriptor、callback context 和 native handle：

- kernel/OS 持有期间，调用方不能移动、释放或改变会破坏操作的状态；
- 取消若不能保证底层不再访问，资源转给 operation state，直到 completion/drain；
- partial progress（例如部分读写）与取消/错误同时发生时，结果要保留已完成数量；
- callback unregister 需要定义是否同步、是否允许一个 callback 已在途以及怎样等待 quiescence；
- timeout 是由 time 触发的取消政策，不应在每个 backend 发明另一套生命周期。

## 5. 背压与容量

无界排队会把短暂过载变成 OOM 和 tail collapse。每个可积压资源都要有 owner、容量和满时策略：

| 资源 | owner | 满时允许的策略 |
|---|---|---|
| scheduler/global injection | sched | reject、等待容量、调用方执行或显式降级 |
| file/socket submissions | io/backend | suspend 到可提交、返回 backpressure、取消 |
| memory pool/arena/quota | mem/consumer | recoverable OOM、释放/回收、显式 fallback |
| device queue/staging | compute/backend | 有界等待、分批、上层 admission |
| logs/traces | obs | 按产品政策 drop/sample/block；默认不能意外阻塞热路 |

策略属于资源 owner 或上层 consumer；底层 primitive 不应悄悄无限增长。容量、等待公平性、drop、
partial progress 和统计必须在 API 附近可发现。

## 6. 优雅关闭

可重入、可验证的关闭通常按以下状态推进：

```text
RUNNING -> STOP_ACCEPTING -> CANCEL/FINISH -> DRAIN -> RELEASED
```

- `STOP_ACCEPTING` 后新任务立即得到明确结果；
- 已接收任务按合同完成或取消，不静默遗失；
- `DRAIN` 等待所有 callback/completion、借用与跨线程释放结束；
- worker、backend、allocator 和 obs sink 按依赖逆序释放；
- 重复 shutdown 的行为明确，析构不能在未知线程无限等待；
- 关闭期间的错误仍可观测，但不能重新启动已关闭的下层。

模块关闭协议必须能在 source/static/shared 三种 C++ 交付下用相同用户 journey 验证。

## 7. 跨模块责任

- error：canonical failure、context 与 OS/backend translator；
- time：deadline 与 timer mechanism，不拥有请求策略；
- sched：task/fiber 生命周期、结构化子任务、waker 和 drain；
- io/compute：native cancel、operation state、late completion 和 buffer ownership；
- mem：resource lifetime、quota/OOM 和跨线程释放合同；
- obs：不改变控制流的诊断与统计；
- 推理引擎/应用：SLO、重试、admission、batch、公平与进程重启策略。

下层不得为了“方便”持有上层 scheduler/request/model 对象；需要通知时使用所属层定义的窄 seam，
并明确其 owner 和 quiescence。

## 8. 验收问题

每个异步或资源型 API 至少回答并测试：

1. 谁创建、谁销毁、move 后怎样；
2. 何时可以安全释放/复用 buffer；
3. 取消前、提交中、完成竞态和完成后分别发生什么；
4. native cancel 不支持时怎样 drain；
5. 队列/内存满时是否有界、可恢复并可观测；
6. shutdown 与 callback 重入是否死锁；
7. OOM、backend loss、partial progress 和 late completion 是否保真；
8. sanitizer、fault injection、stress 与真实 backend 是否覆盖高风险窗口。

不清楚这些答案的 API 不能因 happy-path benchmark 很快而签署。
