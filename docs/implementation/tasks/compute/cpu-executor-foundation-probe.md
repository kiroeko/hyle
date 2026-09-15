# CPU foundation：真实持久 executor 的公开组合

- owner：总计划 F2/工作包 4、Graph v10/G107；baseline `405fd289`。这是具名组合验证，不新增 factor，
  不重开已 CLOSED Sched，也不扩大 Native/oneDNN 算子目录。
- 反例来源：Native/oneDNN 的 executor 分片单测主要用串行 callback probe；两个独立 Invocation 的
  并发不能证明一个 Invocation 内实际 fan-out、嵌套与固定 scratch 的正确性。
- consumer：长期持有 CPU worker runtime 的引擎，把同一 executor 借给 Native 和 oneDNN；一次同步
  MatMul/MLP 完成后可立即读取结果。反偏置是已在该 runtime callback 中运行的嵌套调用。
- API 固定：优先组合 `sched::Scheduler`/`bulk_execute` 与现有 `ParallelExecutorRef`。不为样例添加
  product API；executor infallible/synchronous/nested-progress 合同不能通过忽略 Status 或失败后重跑
  整批工作伪造。caller 管理 runtime 的创建/close，不把全局线程池塞入 Compute。
- 有限调查：读 public Sched bulk 的 failure/close/嵌套语义与实现一次，选择一个不漏任务、不重复执行的
  adapter；若不能自然满足 executor 合同，记录精确阻塞机制，再按新 card 裁决 `compute_sched`，不
  用测试专属 friend/私有 include 绕过。
- 允许实现：一个 public C++ 组合样例或现有 consumer 的小型扩展；1/2 worker，真实 task 执行、
  caller/worker nested 两种入口、两个独立 Invocation、输入变化和不同 scratch，结果与独立 oracle 比较。
  通过实际 worker identity/任务完成证明并行使用；不强求 OS 每次同时调度，不用墙钟差判 correctness。
- 生命周期：先创建 executor，再 Context/artifact/invocation；所有 Completion 终态和 consumer 销毁后
  才关闭 runtime。关闭期间不得默默丢任务；借用超期本来是 caller 违约，不将其做成非法内存测试。
- 成本：P0 分开核对 HYLE plan、provider、caller executor 的 allocation/fan-out/wait，不能将 adapter
  成本排除后宣称完整路线零分配或 SOTA。本 probe 不计时；对应公平并行性能在统一候选 P1 中单列，
  不与既有串行 wrapper-tax 行混算。
- 预声明验证：先 Windows Release、最多 2 build jobs、串行 bounded CTest，只构建此 consumer 的
  selected Graph/Sched/oneDNN 闭包；完整实现后同一 consumer 验证 source/static/shared。Linux 与两台
  Apple 的基础资格按总 gate；当前 x64-only oneDNN 在 Apple Silicon N/A，但 Native/executor 仍适用。
  docs/closure/diff 与 change-impact plan 必跑；不跑全库、CI 或 P1。
- 预算/stop：一个 adapter 方案加一次 fixture 纠正；若发现 executor 运营失败会漏算/重复、嵌套不进展、
  常态成本不符合公开承诺，停止样例资格并记录 owner，不能改更宽产品来迎合一次 probe。

## 实现前调查结论

`src/sched/bulk.cc`：无 cancel token 且 body 恒成功时，部分 runner admission 失败只降低并发，已接纳的
runner 排空同一 cursor；零接纳才返回 admission error。worker-stack 嵌套直接串行；正常批量状态在
caller stack，raw runner 不按 item 分配。于是一个**有效且保持开放直到计算完成**的 Scheduler 可自然
承载现有 infallible executor。关闭应发生在所有 Compute consumer 结束之后；这是借用 executor 的
使用前提，不是把关闭失败当成功。样例检查 bulk Status，前提违约时停止，不在失败后重跑或吞掉错误。

有限实现确定为 `samples/package/graph/cpu_executor`，只依赖公开 Graph/Compute/Sched/Mem；worker
identity 和分片计数是样例观察器，不进入产品或性能结论。Native 与可选 oneDNN 运行相同 RMS+gated-MLP，
一个 executor 同时借给各 Provider；输入/独立 double oracle 在调度前准备，worker callback 只执行已
准备计算。先 source consumer，后安装 static/shared；每项实际结果单列，不由编译成功推断运行成功。

## 有限 fixture 复核

首版 `5c1846d6` 的 Windows source/static/shared 均执行通过。复核发现 task 计数是整次 worker 生命周期
的聚合，混入了前置队列检查，不能单独证明每条 Compute 路线使用 executor。因此使用预声明的一次 fixture
纠正，将 replay 计数按 Native/oneDNN 分开、排除 compile/prepare/队列检查，要求两 worker Native 与
oneDNN 真正调用 executor，且 oneDNN 实际进入 nested bulk。Native 单任务/已在 parallel 环境时源码明确
inline，不能为测试强制它调度。只重编译这个 consumer 并执行三种交付，不改/重建产品或重跑 P1。

## Windows 检查点

最终样例候选 `f6c45efd75e63c96aa406fc5ad29209f73e149b0`，Windows/MSVC Release 的 source、installed
static、installed shared 全部实际通过；[原始记录](../../../../evidence/correctness/compute-cpu-executor-foundation-v1/README.md)
保留具体命令、计数、交付身份和限制。两 worker Native 实际 15 次 executor 调用/30 个 task，oneDNN
12 次/24 个 task；oneDNN nested bulk 每种 worker 设置均实际执行 2 次。每条路线均通过独立 oracle、
变化输入、并发独立 Invocation、嵌套、allocator 回收与 prepared metadata 不增长检查；最后验证 runtime
关闭拒绝和 worker 回收。
Native 单 worker/已在 parallel 时 inline，是既有避免多余调度的正常路径，不冒称 worker fan-out。

P0 维持 O(1) caller bulk 状态、O(worker-count) raw runners；没有逐 item callable 分配、新全局 pool 或
Graph→Sched 产品依赖。executor 保持开放的使用前提成立时，现有公开组合已能承载本旅程，因此不新增
`compute_sched`、不重开 Sched。此结论只覆盖已运行的 Windows 三交付；Linux/两台 Apple、sanitizer、
最终并行 P1 和整体基础 gate 仍待执行，不把本记录签为 Compute/Graph CLOSED 或完整性能通过。

维护验证通过：closure 39/schema 4、docs 309/1122 links、diff check。selector 对样例变化选 Graph delivery，
本卡执行的是上述具名 consumer 三交付，不冒称整套 generic package smoke；结果文档只选 docs/closure、
跳过 94 项。registry 与产品树未变。
