# HYLE 生命周期与失败边界设计

> 之前的设计挖透了 happy path（fiber 怎么跑/切换/唤醒），但**失败边界**（取消/背压/关闭/隔离/跨边界错误）几乎空白——而真实产品(北极星:推理引擎 serving,及其消费者并发 agent)恰恰活在这些边界上:请求超时要取消、连接洪峰要背压、停机要优雅关闭。本文补齐。（下文以并发 agent 为主要举例,但这些边界对推理引擎 serving 同样成立。）
>
> 统领原则（用户定，源自 clay 经验）：**失败就暴露，绝不 try-catch。**

## 0. 失败哲学：四类失败，泾渭分明

"绝不 try-catch" 不是"不处理失败"，而是**按失败的性质分类处理，bug 类绝不吞**。四类：

| 类 | 是什么 | 处理 | 例 |
|---|---|---|---|
| **① bug / 不变量违反** | 代码错了：use-after-free、锁序违反、不可能的状态 | **`HYLE_FATAL_ASSERT` → abort + 诊断，响亮地崩，绝不 catch** | 双重 free、枚举越界、空指针解引用 |
| **② 运营失败**（可恢复）| 外部世界的预期失败 | **`Status`/`expected` 值，显式处理**，不 throw 不崩 | IO 错误、超时、分配失败、解析失败、连接重置 |
| **③ 取消** | 不是失败，是控制信号 | **`Status::Cancelled` 值，协作式传播**（见 §1）| agent 超时、父取消子 |
| **④ 跨进程故障隔离** | 一个 agent 崩了不该拖垮别人 | **进程边界**（联邦模型），非进程内 catch（见 §4）| 失控 agent 段错误 |

**核心纪律（铁，与编译开关无关）——bug 绝不 catch**：
- **bug 被 catch = 把崩溃推迟到更难查的地方**。use-after-free 当场 abort + 栈，胜过吞掉后在三个模块外随机炸。**这条不动摇:自家 bug 一律 fail-fast,绝不 catch。**
- **异常跨 C-ABI 边界是 UB**，HYLE 对外纯 C 头（charter §6），异常根本过不去。

**throw 可以用,但谨慎——用数据和后果说话(非教条禁止)**：
- **默认 no-throw**(bug→fail-fast,运营失败→Status)。要破例用 throw,**必须拿测量数据 + 后果分析撑**,不靠"感觉该用"。
- **判据**:throw 只在 ①不影响热路径性能(zero-cost 异常不抛时运行时开销≈0,这点对热路径成立) ②或为隔壁稳定(隔离外来异常)且 ③catch 的是**外来异常不是自家 bug** ——三条都满足才考虑。
- **`-fno-exceptions` 旗标本身 = 实测决定,不预先拍死**:happy-path 运行时它不买性能(zero-cost),但买**更小二进制 + 更强优化**(编译器可假设无 throw)。到底值多少,**量了再定**(套"硬问题用数据说话"纪律)。

**三层模型(纪律铁 / 旗标待测 / adapter 可开但封死)**：

| 区 | 异常 | 说明 |
|---|---|---|
| **核心区**(热路径、近 C-ABI)| no-throw 纪律铁定;是否加 `-fno-exceptions` 旗标 **= 实测决定** | happy-path 零运行时成本;旗标买二进制/优化,量了再定 |
| **adapter 区**(包会抛的 C++ 库,如 JSON)| **可开异常,但 catch 必转 Status、绝不外泄** | 隔壁稳定的正解;外来异常不得跨进核心区/C-ABI。优先用 nothrow API(simdjson 有)绕开 |
| **任何区** | **自家 bug 绝不 catch** | fail-fast,违反=藏 bug |

> **clay 实际怎么做（查证,作对照）**：clay **没有 `-fno-exceptions`**,异常默认开着;自有代码仅 6 处 `throw`（registry 冲突、string 转数字）;14 处 catch 集中在边界——worker 执行用户 task 周围 catch-all、`graph_io` 在 nlohmann JSON 解析周围 catch。即"异常开着但极少用 + 边界兜底"。HYLE 比 clay 更有条件收紧（自带 allocator 返 null 不 throw + 自带容器,绕开 std 抛异常主源），但**收到多紧由数据定**,不教条。

**待调研**：①`-fno-exceptions` 给 HYLE 代码买到的二进制/优化收益（实测）②借入 C++ 库的隔离 TU 机制（分文件编译 + 边界转 Status）。compute/serialize 开工时定,不在此锁死。

**关键澄清——fail-fast vs 故障隔离不矛盾**：
- HYLE infra **自身**的 bug（①）→ fail-fast，abort 整进程。infra bug 不可恢复，崩了对。
- agent **级**的隔离（④）→ **进程边界**：真实部署是**多个 agent 实例、每实例若干 fiber**（不是单进程百万 fiber）。隔离天然落在**每实例一进程（或有界组）**边界——一个实例的 infra 级 bug fail-fast 崩掉的**只是那一个 agent 实例**，不波及整群。这让 fail-fast 与隔离**完全自洽**：进程内 fiber 同命（数量有界、可接受），跨实例靠进程边界 + io/shmem 联邦。
- agent **业务**失败（LLM 返错、工具失败）是 ②，是 `Status` 值、agent 逻辑处理，**根本不是崩溃**。

## 1. 取消（cancellation）——serving/agent 的命门

推理请求超时要取消在途 io、agent 超时要取消 LLM 调用、父任务挂了要级联取消子 fiber。这是高并发 serving 与并发 agent 的核心操作(对应 io-06 带超时 io + sched-07 结构化取消)。

### 模型：结构化并发 + 取消令牌树

- **`CancelToken`**：共享原子标志 + waiter 链。
- **令牌成树**：子令牌派生自父；**取消父 → 级联取消所有子**。这正好映射"父 agent 取消子 agent"。spawn 子 fiber 时传子令牌。
- **协作式**：fiber 不被强行"杀"，而是在**取消点**检查令牌、提前返回。

### await 点即取消点（与零异常天然契合）

C++ 无异常**不能强制展开栈**（不像 Rust drop future）。HYLE 的解法优雅且零异常：

- **每个 await 点是隐式取消点**：fiber `co_await` 一个操作时，若令牌已取消，awaitable **返回 `Status::Cancelled`** 而非结果。
- **`HYLE_ASSIGN_OR_RAISE` 自动向上传播**：取消值像普通错误一样早返回，**正常 return 路径展开栈、RAII/`HYLE_DEFER` 逐层触发清理**。
- 净效果：**用 `co_await` + `ASSIGN_OR_RAISE` 写的代码自动可取消、自动清理，无需异常、无需强制 unwind、无需 try-catch**。深层不查令牌的纯 CPU 循环不可取消（协作式的固有代价）——需要时手动插 `HYLE_CHECK_CANCEL()`。

### 难点：取消正阻塞在 io/GPU 上的 fiber

| 后端 | 取消机制 |
|---|---|
| **io_uring** | 提交 `IORING_OP_ASYNC_CANCEL`（按 user_data 取消在飞 SQE）；原 op 以 `-ECANCELED` 完成 → Waker 触发 → fiber 续跑见 Cancelled |
| **IOCP** | `CancelIoEx` |
| **epoll/kqueue**（reactor）| 移除 fd 兴趣 + 唤醒 fiber |
| **GPU** | **kernel 内不可抢占**（物理事实，CUDA 证实）。"如何更细地中断"是**待调研开放问题**（见下），现在不锁方案 |

> **GPU 取消：事实 + 待调研（不写处方，避免半成品带偏实现）**
> **已知物理事实**：GPU 无 kernel 内抢占，已 launch 的 kernel 算到底。
> **参考实证（仅作素材，非定论）**：ggml CPU 在 node 间查 `cplan.abort_callback`（`ggml-cpu.c:3067`，粒度=一个 op）；Metal 在命令缓冲段间查（`ggml-metal-context.m:599`）；CUDA 连 op 边界都不查（`ggml-cuda.cu:4468`），"可中断感"来自上层 token 循环（`llama_decode` 是最小取消单位）。
> **方向假设（待验证）**："协作式 op/批边界查标志 + 丢未提交批"可能是比"等整图"更细的伪抢占，取消延迟上界≈单 op/批时长。但**这需要更专业的调查 + 实验才能定**（不同后端粒度差异大、切图开销、与 ggml_bridge 的交互），**现在不写死方案**。
> **行动项**：留待 compute 模块开工前专项调研（连同 sched-07 取消的 GPU 侧落地一并定）。

### cancel-safe：缓冲所有权（io_uring 的经典坑）

取消 io_uring read 后，**内核可能仍写你的缓冲，直到 cancel 真正完成**。所以：**操作持有缓冲直到 CQE 真正到达（成功 OR -ECANCELED 完成），不是直到请求取消那一刻**。HYLE awaitable 的所有权约定：**buffer 生命周期 = 到完成 CQE，取消时也不例外**。这是"completion 而非 readiness"的所有权模型，必须写进 io awaitable 契约。

## 2. 背压 / 流控——大量 fiber 不能一拥而上打爆有界资源

有界资源满了怎么办，必须有策略，否则大量并发 fiber 直接压垮：

| 满的资源 | 策略 |
|---|---|
| **run-queue（有界 ring）** | 满 → 溢出到 global（已设计，sched 2.2）|
| **io_uring SQ** | 满 → 提交方 fiber **让出等 SQ 有位**（背压传导到发起者），不丢、不无界缓冲 |
| **channel（fiber 间）** | **有界 channel**：满则发送 fiber 阻塞让出（CSP 语义），天然背压 |
| **新 fiber spawn** | 受**资源限额**门控（§4）：内存/数量超限则 spawn 返回 `Status::ResourceExhausted`，调用方决定退避 |

**原则**：背压**传导到源头**（让发起者等/被拒），绝不**无界缓冲**（那是把 OOM 推迟）。有界 + 阻塞让出（fiber 让出代价极低）是百万 agent 的正解。

## 3. 优雅关闭（graceful shutdown）

runtime 怎么停，清理顺序怎么定：

- **两阶段**：①stop accepting（拒新任务/新连接）②drain（等在飞 fiber 完成或超时）→ 超时则取消剩余（§1）。
- **清理顺序**（逆依赖）：先停 `io`/`compute`（completion 源）→ 排空 `sched`（fiber 跑完）→ 释放 `mem`。
- **析构顺序灾难**（clay 前车之鉴）：全局/单例用显式 init/shutdown，**不依赖静态析构顺序**；Meyers singleton 仅用于无依赖的；有依赖的走显式 `runtime.shutdown()`。

## 4. 资源限额 / 故障隔离

- **per-agent 资源预算**：内存配额（fiber 栈 + 分配，接 mem/ 的**计量 Allocator wrapper**——包一层记字节数的 `mem::Allocator`）、fiber 数上限。超限 → `Status::ResourceExhausted`，不崩。
- **故障隔离 = 进程边界**（非 try-catch）：真实模型是多 agent 实例、每实例若干 fiber——隔离落在**每实例一进程（或有界组）**，实例崩只崩自己。进程内 fiber 同命（数量有界，非百万），是 fail-fast 的合理代价。
- **失控检测**：sysmon（已在 sched 提）兼测长期占核 fiber / 内存暴涨 agent。

## 5. 跨 C-ABI 边界的错误约定

内部 `std::expected` **过不了 C 头**。FFI 错误约定（charter §6 定了纯 C 头、没定错误怎么传）：

- C 函数返回 **错误码 enum**（`hyle_status_t`），结果走 **out 参数**。
- 详细错误信息走 **`hyle_last_error_message(handle)`** 取线程局部错误串（errno 式，但线程局部）。
- 不透明 handle 的 create 失败返回 NULL + 设错误码。
- **取消跨边界**：C 侧传 `hyle_cancel_token_t` handle，FFI 调用可被它取消。

## 影响哪些里程碑（回填 milestones）

本设计**改动既有里程碑的接口**，幸亏在写码前补：

- **sched-04（Waker）**：awaitable 接口须含取消语义（返回 Cancelled）；新增 `CancelToken` 树。
- **sched-05（fiber-sync）**：channel 必须有界（背压）；CV/mutex 的等待要可取消。
- **io-01（proactor 抽象）**：提交接口须支持取消 + cancel-safe 缓冲所有权契约；SQ 满的背压。
- **新增里程碑**：`sched-07 取消(CancelToken树+await取消点)`、`close-01 C-ABI 错误约定`扩充、`runtime 关闭`(归 sched 或新 lifecycle 里程碑)。
- **mem**：per-agent 内存计量挂在**计量 Allocator wrapper**(包一层 `mem::Allocator` 记字节)。

## 一句话

HYLE 的失败哲学：**bug 响亮地崩（fail-fast，绝不 try-catch）、运营失败显式返回（Status/expected）、取消是协作式控制信号（await 点 + 正常返回展开 + RAII 清理，仍不 try-catch）、隔离靠进程边界（非进程内 catch）。** 取消、背压、有界、隔离——这四样让百万 agent 不至于一崩全崩、一拥而上打爆。

## 参考

- `engineering-charter.md` §2（零异常 + expected）、§4（覆盖/sanitizer）。
- `sched-design.md`（fiber/Waker/进程联邦/sysmon）、`io-design.md`（proactor/取消接口待定项）。
- io_uring `IORING_OP_ASYNC_CANCEL`；结构化并发（Rust tokio CancellationToken、Kotlin/Swift 结构化并发）的取消树思路。
