# G10 CPU 线程预算与实际消费

状态：**CLOSED，2026-10-17；T-F1–T-F4接受，九项交付、CPU+CUDA实际消费、新状态适用sanitizer、P0及有限终审通过**。

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
普通CPU facade首次尝试在外层metadata分配前关闭窗口，OOM/失败不重开；expert Context以相关provider
callback入场为界，不把尚未进入provider的common参数/metadata拒绝称为运行时使用。原Native低精度
初始化同样关闭窗口；Context全部销毁后不还原。

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

## 实现与首轮 Windows 检查点

T-F1–T-F3及T-F4消费者/guide已写入，未增加数学算法、每次replay字段或新线程池。
有限代码审查发现并修复：Native Context缓存资格在检测foreign漂移后仍用于新prepare；
普通CPU工厂首层metadata OOM未关闭窗口；测试第二线程创建失败时双参与者barrier可能死锁。
分别在原cold propose重查允许状态、facade最早入场关闭窗口、测试采用异常安全的单向start gate。
OpenMP漂移反例不套用到Native后端，后者保留真实两线程数学后晚setter拒绝的独立场景。

Windows/MSVC19.50.35728静态CPU+CUDA树实际定向编译通过。9项fresh-process检查全部通过：
startup/same/conflict/invalid与两个phase并发数值、query-first、late-Native含原低精度数学、
late-ATen、configure-vs-use、conflicting-config race、外部漂移、已有Context新prepare及metadata OOM。
实际4 logical/4 jobs/BelowNormal；未编译CUDA kernel，原146对象的242个派生依赖与5个canonical输入
逐字节及原命令/依赖新鲜度核对后复用。测试模式引起的原SIMD内部测试定义重编译已如实留账，
不将它称为G10数学改动，也不据此签shipping SDK。

额外的`qualified-drift-required`前置检查失败并保留：该CPU+CUDA树绑定另一ggml peer，
原`HYLE_NATIVE_DEFAULT_PEER_F4`构建条件不成立，CPU2 placement也不会使其获得旧recipe。
9项功能PASS不因此改写。随后有限独立核查确认，当前CPU-only锁的patch身份同样为
`e74fc9bb8964d04f`，不等于旧F4的`bdae1e43311ee486`；正常当前交付域均不能进入该正向分支。
因此仅这个正向缓存运行反例记`NOT_APPLICABLE`，保留原required前置FAIL，不记cached runtime PASS；
原cold guard保留源审与当前真实拒绝/foreign验证。不能为制造覆盖而强开宏、回退vendor或修改peer条件，
依据与精确源公式见[适用性裁决](../../../../evidence/architecture/inference-g10-thread-runtime-implementation/applicability.json)。

本检查点当时尚待原executor/native预算、Graph/CPU+CUDA直接消费者、九项交付、
featureOFF适用性和新状态sanitizer；后续实际结果见下方。G09三组A/A资格阻塞独立保留。
[检查点原始输入、代码审查及全部尝试](../../../../evidence/architecture/inference-g10-thread-runtime-implementation/receipt.json)
绑定这次实际构建、9项PASS、保留的拒绝与累计629.67秒；不是完整交付/风险验收。

## 不可变候选与完整消费

候选为`0022dd6adc3b2d903f91e9a1db0e73517dd142ba`，tree由Git唯一导出。
之后只补文档、状态及证据，没有再改public API、对象字段、数学或常态成本。
[九项交付原件](../../../../evidence/architecture/inference-g10-runtime-delivery/receipt.json)
绑定同一完整tracked源码cohort：Windows、WSL、MacBook Air各自source、迁址static、迁址shared。
每格独立编译CPU/Graph公开头并执行七个fresh-process状态场景和原Graph FFN/executor消费者，
后者覆盖worker=1/2、独立数值oracle、双调用/嵌套进展及平衡回收。shipping producer为tests、bench、
fault instrumentation和sanitizer OFF；安装消费者仅使用迁址前缀。九格实际通过，累计755.96秒。
Windows source首轮收集器误找private fixture目录，原构建已完成；核对原manifest及全部源码/fixture后，
直接执行既有产物完成检查，保留首次失败，没有重编或重置费用。

[联合消费及风险原件](../../../../evidence/architecture/inference-g10-runtime-final-validation/receipt.json)
包含RTX4070上的原CPU+CUDA样例：共享limit=2后，CPU norm/casts与CUDA gated FFN、两个真实slot、
weight fragments/DMA依赖、leases、replay和close/drain均通过。首轮包装丢失原相对头路径的编译失败保留；
修正包装为包含原样例及其自然相对头布局，产品不变。原146个CUDA对象按字节、命令和依赖证明复用，
没有编译kernel；该tests-on树不冒充安装SDK或P1。

新状态实际feature ON的WSL ASan、WSL UBSan、Air TSan均通过同一七状态场景、Graph/executor数学与
foreign消费。每格保留四个changed core TU和三个app的实际instrumented命令。
TSan只复用原RefCount final acquire-fence模型，精确hash及去注解后canonical字节一致；新mutex/state
没有HB注解或suppression。不签vendor私有池、LSan或GPU driver风险。
Windows ASan旧Vulkan-only树的618-edge冷依赖预检被拒绝，未build；按独立源审选择已有WSL CPU树，
不签Windows ASan。Air首轮父工程漏带原macOS14.4 deployment配置的拒绝同样保留。
联合/风险全部尝试累计480.90秒，包含失败；不是仅成功耗时。

### 适用性与逐能力证据

| 能力/消费 | 正常与边界 | 失败/生命周期 | 并发与实际消费 | 平台/交付 |
|---|---|---|---|---|
| startup set/report | PASS：invalid、same、conflict、query-first、真实getter | PASS：late拒绝、成功requested保留、foreign漂移不修复 | PASS：configure-vs-use/configure竞态；新state三sanitizer | PASS：九交付实际OpenMP |
| first-use与Native旁路 | PASS：普通ATen、expert Native及原低精度数学 | PASS：普通facade外层metadata OOM也冻结；close不重开 | PASS：入口与配置竞态，数学完成后checked释放 | PASS：九交付及风险消费者 |
| phase/executor/ggml | KEEP/PASS：原options不变，native_threads=1/2 | PASS：独立workspace/output、close/drain、平衡回收 | PASS：双phase并发及原Graph FFN/executor worker1/2与嵌套消费 | PASS：九交付；未重跑原算子全域 |
| CPU+GPU组合 | PASS：原placement、native DMA与CPU norm/CUDA FFN | PASS：两slot、lease、完成与teardown | PASS：显式共享启动limit后消费 | PASS：Windows CPU+CUDA；其余GPU数学/原交付按未改身份复用 |
| vendor setter/getter异常 | SOURCE_REVIEW：close-before-set、核实后publish | 源审保守关闭/无回滚；实际bad_alloc/c10/std::exception未诱发 | 不把metadata OOM或正常漂移冒充这个分支的runtime PASS | 未运行分支如实保留；未制造vendor替身 |

当前普通CPU路线由唯一route集合派生Torch ON；旧`HYLE_COMPUTE_TORCH_CPU`开关被明确拒绝，
GPU-only又不交付CPU facade。因此本批三平台的公开CPU+ATen-OFF消费语义上`NOT_APPLICABLE`；
两次OFF配置拒绝保留，不新增开关、不强宏、不签OFF runtime PASS。
当前三平台实际线程backend都是OpenMP，ATen Native singleton晚setter反例同样不适用；
它与已经实际覆盖的HYLE Native provider/低精度旁路是两个不同维度。
旧F4正向cached-recipe条件按前述peer身份不适用，仍不签正向运行覆盖。

### 有限审查与七门

[独立R3](../../../../evidence/architecture/inference-g10-runtime-final-validation/review.json)
核对九交付、源码适用性及风险入口，产品和private runner的open MUST均为空；
R1/R2已修问题由首轮原件保留。R3不代签随后才实际执行的sanitizer或CPU+CUDA，后者分别绑定自身原件。
原生setter异常仅源审的限制保留，没有扩展fault seam或新仪器。

| Gate | 判据与当前裁决 |
|---|---|
| scope | PASS：固定T-F1–T-F4；不加引擎政策或G09修复假说 |
| architecture | PASS：原CPU public头/Native owner承载；原executor、ggml与native完成语义复用 |
| correctness | PASS：具名状态、OOM、竞态、真实数学/生命周期及新状态适用sanitizer；不适用/仅源审格明确区分 |
| delivery | PASS：同候选九项source/static/shared公开头与消费者；原GPU资格不混签 |
| usability_docs | PASS：guide说明启动期共享作用域、foreign协调、非总线程上界及当前CPU路线强制依赖；docs合同已检查 |
| red_team | PASS：有限独立审查无open MUST，全部前置失败/费用与未运行限制保留 |
| engineering_performance | PASS/P0：仅cold常数状态与锁，数学/replay无新增字段、setter或锁；不新增P1 |

[P0账与实际符号](../../../../evidence/architecture/inference-g10-runtime-final-validation/p0.json)
记录一个native代码映像的固定state；Linux/GCC14 x86_64实际shipping对象的该BSS符号为48 bytes，
不是所有平台统一sizeof。原`DefaultContextOptions`逐字节源块相同，没有新per-Context/Invocation字段或
HYLE线程池。vendor/platform启动内存、池及锁竞争OS费用继续unknown，不记0。
两个owner的统一LOC工具`--commit ... --self-check`通过；当前conventional边界只统计src，
分别为CPU252行/4文件、Native5343行/17文件，不冒充含split public头的总量。

本轮产品停止变化。只有当前合同的可复现缺陷、真实消费者新语义或SDK/threading backend变化等具名事实
可以触发新轮；普通复看或未资格化benchmark不重开。G09三组A/A/P1不因G10获得豁免。

## Terminal

两个owner依次由AUDITED进入CANDIDATE，再以同一不可变产品候选进入CLOSED；
registry维护门使用包含前态的明确Git revision。候选后仅文档、证据与派生报告末尾空行规范化，
原始报告及首次staged whitespace拒绝的说明保留，未重跑数学或sanitizer。
[最终维护裁决](../../../../evidence/architecture/inference-g10-runtime-final-validation/acceptance.json)
保存selector的docs/closure选择、146项skip/reuse、带前态的状态检查、docs和diff结果。
验收只覆盖本卡冻结的线程预算语义；G09原有三组资格阻塞不变。本goal已接受G07/G08/G10三项，
剩G09三组A/A资格及其后续P1/终审；不把功能完成写成四项全部验收。
