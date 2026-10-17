# G10 CPU 线程预算与实际消费

状态：**AUDITED，2026-10-17；四项有限 FIX 与 API/消费/风险冻结，尚未实现验收**。

本轮由当前 G07–G10 goal 明确授权，承接[原 G10 候选](../project/inference-single-machine-foundation-audit-2026-10.md)
和[消费适配](../project/inference-consumer-adaptation-plan-2026-10.md)。
重构 baseline 为 `0f852b231`；原 owner 为 `compute_route_cpu` 和 `compute_provider_native`。
G07/G08 已接受，G09 的产品与合格 CUDA P1 保留，三组 A/A 资格阻塞由
[独立工具卡](../tooling/inference-g09-native-workload-aa.md)负责。G10 不降低或代签 G09 的性能门。

## 真实触发与产品边界

单机 CPU 及 CPU+单 GPU 的引擎消费者需要协调 prefill/decode 与两个并发 invocation 的线程预算，
能分清 caller、借用 executor、原 ggml workers 和 Provider 私有池的控制范围。
当前 [CPU options](../../../../include/hyle/compute/cpu.h)明确 `native_threads` 仅约束 ggml，
executor concurrency 不隐式创建该池；[ATen runtime](../../../../src/compute_provider_native/aten_cpu_provider.cc)
明确 `uses_private_threads=true`。这两个现有合同并不能证明全进程单线程或总并行度上界。
本轮作为新的控制/可观测能力需求打开两 owner，不把原已声明的边界追认成旧产品缺陷。

复用原 Context/Program/Invocation、executor、Sched 与 Platform 的 affinity/NUMA 能力。
不新增引擎 scheduler、模型/request/token 身份、全局线程池框架或多 GPU/多机政策；
不以临时 process-global setter 冒充独立 context 的预算，不把未知私有线程写成 0。

## AUDITED 前的有限交付

1. 核现有 public option、runtime properties、实际原生线程路径和其进程/线程/context/invocation
   作用域；为可约束与只能报告的部分逐项给出证据，特别检查 ATen、ggml、oneDNN 及平台差异。
2. 沿已冻结的 SGLang `7224fce1`、vLLM `98f72111`、llama.cpp `cbe28bb3`，每族至多两处主要
   源机制加一个反偏置；不刷新依赖，不扩充完整引擎调研。
3. 固定一个 CPU-only 消费与一个 CPU+GPU 消费的公开调用草图，覆盖两个 phase contexts 与两个
   并发 invocation。先确认真实数学消费及预算语义，再决定是否需要极短反证；不从计时挑 API。
4. 固定 public 形状、owner/lifetime/失败/并发及常态成本账和有限 FIX；其他能力借用或明确拒绝。
   无兼容包袱，避免重复短文件；优先在现有自然 owner 中形成最小完整合同。

SCOPED 仅做以上有限审计与源事实收集。实现前必须完成 AUDITED、绑定受影响 owner、预声明风险与
selected gates；普通增量构建最多 4 jobs/4 logical、BelowNormal，所有平台重型工作串行。
仅对真实改变的 owner/直接消费者补证；原数学、包及 SDK 资格按身份复用，不做全局构建。
时间较长的同配置验证沿现有授权追加，不重置失败。新增 benchmark/仪器问题另卡，不在产品模块扩建。

## 有限审计与固定 FIX

[源身份及审计原件](../../../../evidence/architecture/inference-g10-thread-budget-audit/receipt.json)
和[独立 peer/作用域审查](../../../../evidence/architecture/inference-g10-thread-budget-audit/peer-review.json)
核对三组固定 peer 及实际 Torch2.8 build 源。参考 checkout 的当前 HEAD 不冒充固定 ref，
无 `.git` 的历史 archive 不作为可执行 Git checkout。没有通过计时选形状。

| 族 | 当前事实/消费 | 裁决 |
|---|---|---|
| caller/借用 executor | `ParallelExecutorRef` 同步、不得失败、需嵌套进展；Native/oneDNN 已借用 | COMPLETE/KEEP；池总并发仍由 backing 定义 |
| ggml native team | 每 invocation 独立 pool，预算含提交线程，显式N>1才有workers；close释放并等待原实际退休 | COMPLETE/KEEP，双slot不等于共享一组N线程 |
| ATen普通程序与Native低精度 | 两者均可进入同一Torch运行时；后者prepare有cold `call_once` native warmup | PARTIAL/FIX，不能只覆盖显式ATen provider |
| 共享运行时控制与报告 | OpenMP setter写共享atomic并设置caller TLS/MKL local；Native为singleton池、晚改可能只WARN不生效 | MISSING/FIX：startup-only set/get核实、当前caller limits报告与作用域说明 |
| 引擎phase/跨请求总预算 | llama按实际batched工作选pool，不按phase名字猜；vLLM/SGLang启动及CPU绑核策略各异 | APPLICATION/ROUTED；不用模型标签决定HYLE线程数 |

| ID | 固定改动 | 验收边界 |
|---|---|---|
| T-F1 | `cpu.h`提供`configure_runtime_threads(uint32_t)`与`runtime_thread_info()`；在既有native实现/helper承载，由CPU facade转换公开值 | 显式startup共享intra-op控制；backend、requested及当前caller intra/inter-op limits可分辨；disabled如实Unimplemented/unknown |
| T-F2 | 一个native代码映像的cold mutex串行配置与首次使用；普通ATen Context、Native Context及其低精度prepare初始化入口冻结配置窗口 | 相同显式值仅getter仍匹配才幂等；不同/late拒绝；setter/getter不符或异常后窗口保守关闭，未成功requested保持unknown，不回滚；不持锁执行数学 |
| T-F3 | 原cold CPU默认规则资格检查拒绝新的显式非1或失败/partial配置 | 既有未配置行为、规则表、数学和旧P1不改；不把新线程合同沿用为旧recipe资格 |
| T-F4 | 具名API/并发/Native旁路/disabled消费者、交付与guide | 普通CPU及CPU+GPU消费；默认options与原数学复用；不代签G09 |

## API 与生命周期冻结

公开数据放在已有`cpu.h`，不新增短文件或公共函数表。`RuntimeThreadBackend`区分unavailable、
ATen OpenMP、ATen Native；`RuntimeThreadInfo`保存backend、optional requested/intra-op/inter-op
limits、configuration_open与usage_started。limits不是live pthread数，interop只报告，不调度Torch inter-op。

open且未配置时查询只返回编译可用性和unknown limits，不调用ATen getter，避免先初始化caller TLS。
应用在任何相关HYLE/foreign ATen用户启动前显式配置一次，再构造phase专用contexts/queues/invocations；
phase的executor和ggml native_threads继续显式传入原options。`0`或`>INT_MAX`拒绝，featureOFF报告缺席。
配置实际调用setter后须getter相等才发布requested；同值幂等不重设，foreign漂移仅报告/拒绝。
合法首次Context创建尝试及原Native低精度初始化也关闭窗口，OOM/失败不重开；Context全部销毁后不还原。

guard属于当前`compute_provider_native`实际代码映像，不是跨DSO registry。两个STATIC副本可有两份guard
却共享Torch DLL；其他HYLE副本和foreign库由应用统一startup并保持设置。不能检测/回滚任意foreign既有
初始化，不承诺所有已存在OMP caller TLS同步。两个并发调用可能各有OMP team或共享Native池加caller，
本API不是全进程OS/BLAS/驱动线程总上界；额外vendor私有成本继续unknown，不记0。

普通消费为startup配置共享limit后，分别创建native_threads=1与N的phase contexts，独立invocations
共用只读源、各自workspace/输出，checked completion后释放再drain。CPU+GPU沿原公开placement/transfer/
native completion组合，GPU vendor/service线程另有其合同。正交消费为两个并发CPU批处理流，仍只共享明确
声明的executor/runtime；不引入token/model/request身份。

常态成本：仅cold API/创建/prepare增加一个固定native代码映像状态与短锁，不给全部Prepared/Invocation
增加字段、不新增OS线程池、不在数学/replay路径set/restore或持锁。SDK自身startup初始化与池存储上界
无法完整证明时保留unknown；不伪装为由Context allocator精确扣费。

## 预声明验证与停止

- P0：新能力是cold配置/报告与资格边界，无新数学或调度算法，不作P1/token/s比较；审固定状态大小、
  互斥发布、无hot字段/锁/新池、原Native/ATen数学不变及vendor未知费用。
- 独立进程的startup/same/conflict/invalid、late-context与Native低精度旁路、查询不提前初始化、
  configure-vs-first-use并发；实际Native后端晚setter不生效及OpenMP foreign漂移按适用平台取证。
- 普通CPU两phase/two-invocation数值与生命周期，原oneDNN/ggml executor/native预算相关消费者；
  Graph CPU与一个Windows CPU+CUDA公开直接消费，不重跑原算子全域或G09 P1。
- Windows/WSL/Air CPU实际source/static/shared共9交付：新API的独立公开头、安装/迁址/导出/installed-only
  consumers；另一个featureOFF入口验证无Torch依赖与诚实unsupported。未变GPU/backend/数学资格复用。
- 新状态的ASan/UBSan及TSan/并发风险按实际宿主与foreign边界取证；不以featureOFF空实现代签配置状态。
  非产品改动先跑selector，再补对应factor与直接consumer；closure/docs每轮必跑。

达到有限FIX/验证停止点即裁决；实际外部阻塞保留原失败与精确恢复条件，不为通过而扩改公共仪器。
G09三组资格阻塞与本轮无因果结论，仍由其独立工具卡按恢复条件处理。
