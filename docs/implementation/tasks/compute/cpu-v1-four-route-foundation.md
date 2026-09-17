# R3：CPU 默认计算与 Graph 完整纵切

<!-- PROJECT UNIT REVIEW: CLOSED -->

R3 已完成实现和适用验证；用户当时接受的 CPU Graph 小片段是接口成本预算取舍，
其[比较服务不等价](cpu-r3-comparison-boundary-clarification.md)，不应解释为算法性能退化。
当前终局记录为[具名验收卡](cpu-r3-acceptance-exception.md)；下文保留实施时序，不再启动 CPU P1。

## Closure card

- baseline：`032b1dc7`；R1/R2 已完成，Compute/Graph 的产品共同语义候选为 `4e9f74a0`。
- owner：CPU 默认 Direct/Graph 组合、Native、oneDNN 与 oneDNN region。新默认组合有不同依赖闭包，
  分别登记 compute_route_cpu、graph_route_cpu；普通用户仍只选择 Compute/Graph 与 CPU feature。
- 真实缺口：compute/cpu.h 只装 Native，用户必须手排 oneDNN；普通 Graph 缺少自动接入成熟 region 的
  CPU 入口。cpu.h 又归 Native factor，不能直接塞入未声明的 oneDNN 或反向 Graph 依赖。
- 主纵切：变化输入的 F32 RMSNorm + gated FFN、解码/批处理；正交纵切：普通非模型 Direct 运算、
  有界动态输入/独立 Invocation、caller worker 上的嵌套调用与资源退出。
- 复用：现有 CPU Device/Context/Queue、PreparedOperation/Executable、Native 19-family、oneDNN
  BRGEMM/region、公开 Sched bulk。各自真实 owner/生命周期保留，不另造万能 executor 或线程池。
- 非目标：CPU Q4 kernel、其它算子目录、模型 CPU+GPU、ggml CPU blocker 解除、Sched 产品重审、
  GPU runtime 新实现或公共性能仪器改造。
- 预算：定向审计 45 分钟；有限 FIX 后实现/普通验证最多六小时；API/规则固定后一次适用 P1，
  最多一次有源码机制依据的性能纠正。只处理实际反例，不追加整体架构研究或无界探针。

## Owner 和交付范围

新增两个 factor 先登记 QUEUED，再 SCOPED/AUDITED；当前不创建 target/option 或声称能力存在。
Native/oneDNN/Graph-oneDNN 的旧 R1 AUDITED 只保留旧范围，在本卡完成有限审计后重新绑定 AUDITED。

新默认入口改变 Compute/Graph 产品聚合闭包。现有 selector 把变更前后的产品 roots 都作为 owning
contract，因此 Compute、Graph、CPU/CUDA backend 需要合法 SCOPED→AUDITED 的**交付边界**绑定。
这些 owner 本轮仅处理聚合/头归属与直接消费，R2 的机制结案不撤销；不因此重审 CPU/CUDA backend
算法、重跑 CUDA P1 或提前施工 R4。公共治理脚本保持不变。

## 已限定的工作面，审计后固定签名和规则

| ID | 动作 | 完成判据 |
|---|---|---|
| C41 | 固定 CPU Direct 和 Graph 普通入口及真实 factor 边，迁移 cpu.h 归属 | 普通用户不排列 Provider；没有 Compute→Graph 回边，CPU OFF 不泄漏新头/库 |
| C42 | 接入既有 Native/oneDNN 单算子与完整 region 的开发期默认规则 | 准备只匹配/绑定；不试跑用户输入、不搜 timing profile；缺席/拒绝有合法基线或明确错误 |
| C43 | 按需求盘点既有 19-family 并保留有效算法；真实 FFN/批处理组合 | 不以目录数代替实际数值、packing/alias/预算与失败验证；Q4 schema 不冒充 CPU kernel |
| C44 | 将公开 Sched 的既有真实消费接到普通默认入口 | 1/2 worker、变化输入、嵌套/并发独立 scratch、关闭与真实 owner 回收；不漏算/重复执行 |
| C45 | 相同普通 C++ 消费、适用平台与三交付，迁移指南/样例/受影响测试 | 源码/静态/动态入口与实际选择一致；Windows/WSL/两 Mac 按真实支持域取证 |
| C46 | 固定 API/算法后的默认入口 P1 与终局 | 同实现 direct 包装税及最强适用完整片段；同数值/格式/线程/资源/完成，方向与 5% 工程门分开 |

## 定向证据入口

- 四路线计划的 R3 与 Compute/Graph/compute-routes 长期合同，不重读全历史队列。
- cpu.h、Native build_plan、oneDNN select_route、Graph oneDNN match_pattern、中央 factor/product 图。
- [已登记 public Sched 消费](cpu-executor-foundation-probe.md)：有效且保持 open 的 runtime、body 恒成功、
  无 cancel 的同步 bulk；部分 admission 减少并发而排空 cursor，零 admission 才返回错误。样例检查
  Status，不能吞掉失败或重跑已接纳前缀。借用期限到所有 Compute consumer 完成后。
- bench/compute_route_cpu/route_p1_bench.cc 与 bench/graph_route_cpu/region_p1_bench.cc 的现有同合同
  direct/完整片段机制；旧强制 Provider 结果只作研发证据，不能签新默认入口。

## 预声明验证

先 selector 选择 owner 与真实直接消费者；新 component/public header/聚合 seam 才扩展相应交付。
Windows、WSL 的 Native+oneDNN；Apple Silicon 的 Native/Sched，当前 x64-only oneDNN 不伪称可用。
普通默认入口覆盖合格规则的边界/留出 shape、oneDNN OFF、非法/不支持输入、冷 OOM、并发独立资源、
Context/runtime 关闭、packing 持有与重复执行。GPU backend 只回放本次交付边界真正受影响的门。

P0 先列默认组合、冷查询/JIT/packing、每槽 scratch、executor/fan-out/wait 与 metadata；不把 prepare
后的 allocator/转换/完成成本排除。已有公共性能 instrument 负责 A/A、双顺序、placement、n=15 和
5% material margin。先完成相同合同与极短 preflight，未通过就按有限源码原因裁决，不换姿势补 PASS。
API/规则未固定前不启动正式 P1。各 native 路线的性能不留给 R7 首次接线。

## 身份登记阶段验证

本次中央 manifest 仅追加两个 QUEUED 声明；没有边、split header、product root、target、源码或编译
选项变化。CMake 的 QUEUED 合同本身拒绝任何 target metadata；selector 已成功解析新旧图，closure
通过 41 个 component 的 registry/状态迁移，docs/diff 通过。

selector 对中央 manifest 保守生成全量计划（外部 r3-register-plan.json）。该调度不是产品风险 PASS：
本步骤没有产品输入或可构建组件变化，ordinary build/header/delivery/P1 对身份登记不适用，未执行。
这遵循 change-impact 文档中“适用风险及最终裁决仍以当前 closure card 为准”的边界；不改 selector，
也不把全量 selected 写成已通过。实际 C41–C46 的依赖/源码落地后，再执行本卡声明的适用 CPU/Graph
及交付门。没有以 nominal identity 启动全库重审。

## 已冻结的普通调用与 owner

Direct 保留自然名称 `compute::cpu::create_context(DefaultContextOptions = {}) -> Result<Context>`，
迁为 compute_route_cpu 的实际 compiled factory。options 仍只含 borrowed metadata allocator/executor；
默认 caller-thread，不隐建 worker。准备、Invocation/Queue、Completion 复用既有 CPU 机制。

Graph 新增 `graph::cpu::create_context(compute::cpu::DefaultContextOptions = {}) -> Result<Context>`，
由 graph_route_cpu 组装 CPU region 与 leaf；用户随后照常 compile/create_invocation/submit/wait。
两种入口不收 Provider 列表或 timing。Graph 的普通头为 hyle/graph/cpu.h；不为方便增加重复 compile
wrapper 或第二套程序/资源 owner。显式专家 Context/provider 工厂仍用于确有选择约束的消费者。

compute_route_cpu 的公开边为 Compute/CPU backend，私有实现边为 Native/可选 oneDNN；
graph_route_cpu 的公开边为 Compute CPU route/Graph，私有实现边为 Native/oneDNN/oneDNN region。
公开 cpu.h 不再隐式 include Native provider 头。两条冷 factory 按各自粒度组装固定短数组，Context
复制并拥有 Provider 状态；不建立全局 registry 或新的热分派层。CPU OFF 时从产品 roots 去除两个
新因子；仍只有原 HYLE_COMPUTE_CPU/ONEDNN 选择，没有新 route enum 或重复 option。

## 有限能力盘点

下列已有 family 全部 KEEP 在现有已声明 dtype/layout/数值域；超域仍明确拒绝，不能隐藏转换。
本轮默认接线不改 kernel、数据格式、alias 或输出规则；对应现有 Native 差分/边界测试继续有效。

| family | 正常旅程与主要边界 | owner 路径 |
|---|---|---|
| View | reshape/slice/permute/broadcast，精确 alias、零 kernel | Native transform |
| Copy | plain dense/strided → dense materialize | Native transform |
| Fill | 已登记 scalar dtype 精确 bit pattern | Native transform |
| Cast | f32/f16/bf16/E4M3/E5M2，显式 materialize/舍入，out-of-place | Native cast + SIMD |
| Unary | f32 激活/数学函数、plain stride | Native elementwise + SIMD |
| Binary | f32 算术/broadcast、fmin/fmax | Native elementwise + SIMD |
| Select | Bool condition，plain 1/2/4/8-byte 分支 bits | Native elementwise |
| Reduce | f32 Sum/Mean/Min/Max、axis/keep-dim/stride | Native reduction + SIMD |
| RmsNorm | dense f32、可选 weight、精确原地、已声明 shape family | Native + SIMD |
| LayerNorm | dense f32、scale/bias、精确原地 | Native normalization + SIMD |
| MatMul | f32 batch broadcast、row-major/转置权重、零维 | Native；合格二维域可用 oneDNN |
| GatedActivation | f32 packed/separate，SiLU/GELU/GELU-tanh | Native + SIMD；region 可吸收 |
| Gather | plain scalar、i32/i64 indices、dense output | Native indexing |
| Scatter | scalar assign、f32 add、明确原地规则 | Native indexing |
| Concat | variadic plain inputs、axis、dense output | Native indexing |
| RoPE | f32 stride、整数 positions、显式 frequency/tail | Native transformer/fat kernel |
| Softmax | f32 axis/scale/log/Bool或additive mask | Native transformer + SIMD |
| TopK | f32 axis、stable/sorted、双输出、caller workspace | Native transformer |
| ScaledDotProductAttention | f32 BSHD/BHSD、mask/causal、online softmax、caller workspace | Native attention |

R2 grouped Q4 为 schema；不添加 CPU Q4 carrier。普通非模型 Unary/Reduce/View/Copy 与上述 FFN 共用
同一资源/错误合同，不用模型形状收窄已有入口。

## AUDITED：有限 FIX 与验证冻结

C41–C46 是本轮完整 FIX 集；前述 API、依赖方向及 19-family 保留表已经固定。没有发现需要
修改 CPU backend、Native kernel、共同 Graph planner 或 Sched 产品的基础缺陷。

默认规则在开发期固定：Direct 先匹配 oneDNN 已有二维 dense f32 MatMul 域，串行 rows < 16
继续由 Native 执行；转置权重/其它不适用域由 Native 的明确能力回答。调用者提供 executor 时沿用
已有 N tile 与实际并发度条件。Graph 先匹配 oneDNN 已有四节点常量权重 gated FFN region，保留
其融合、packing 和独立 Invocation scratch；其它节点由 leaf 规则处理。region 不机械套用 leaf
行数阈值。准备中只查询/绑定，没有输入试跑或计时；正式 P1 若给出完整片段反例，只允许一次
有源码成本依据的规则纠正，不能反复测到 PASS。

P0：每次 Context 创建只组装至多三个 Provider；Provider 状态由 Context 拥有，allocator/executor
按已公开期限借用。prepare/compile 冷阶段允许既有 JIT、packing、资源规划；热阶段继续使用固定
PreparedOperation/Executable、每 Invocation scratch 和真实 Completion。默认入口不添加热层、
全局状态、线程或逐次分配。专家 Native 的差分/预算测试改用明确 Native fixture，防止普通默认
变更偷偷改变测试基准。新入口自身检查实际规则、错误、冷 OOM、变化输入与独立重放。

性能采用 W1-WIN 的两个新版本合同：compute-cpu-route-p1-v3 与 graph-cpu-route-p1-v1。
复用既有 nanobench runner、A/A、双顺序、placement、n=15 与 5% 门；bench 归入两个 route owner，
不修改公共统计/调度器。Direct 对比默认与明确 Native，Graph 对比同实现 direct 包装税、完整
oneDNN 序列及明确 Native 完整序列，保留正常 shape 和留出 shape 的数值检查。公开 Sched 的
1/2-worker 完整消费另作同 executor 条件的真实成本验证；不以串行吞吐冒充 worker 收益。
本轮 WSL/两 Mac 签正确性与交付；不签未经运行的跨硬件 SOTA 性能。

普通验证限定为新 CPU routes、Native/oneDNN/Graph-oneDNN 和直接受影响的 Compute/Graph
测试/公共头，以及 Compute/Graph 的 source/static/shared、CPU OFF 头与闭包隔离。Windows、
WSL 与两 Mac 使用相同普通入口；x64 oneDNN ON/OFF、Apple Native 实际支持域分别取证。
WSL 定向 ASan/UBSan/TSan 仅覆盖新组合与真实 consumer，R2 未变化共同机制复用原身份证据。
所有重型构建与运行串行，Ninja parallel 2。文档/状态提交仅 closure/docs/diff。

本卡绑定 compute、graph、compute_backend_cpu、compute_backend_cuda 的交付边界，以及
compute_provider_native、compute_provider_onednn、graph_provider_onednn、compute_route_cpu、
graph_route_cpu 的上述有限实现/默认接入范围。旧三 Provider 的 R1 AUDITED 本次明确重新定界；
compute_provider_ggml 的 BLOCKED 与 compute_sched 的 QUEUED 均不解除。

标准 CPU 发行的既有 HYLE_COMPUTE_ONEDNN 默认值改为 AUTO，x64 自动接入，Apple Silicon 使用
Native，OFF 是显式裁剪。compute_route_cpu/graph_route_cpu 公开头直接使用 core visibility，因此
各自声明 PUBLIC core；Direct 头直接 include Compute device.h，保留真实公开类型依赖。

### 首轮实现验证（尚非候选终审）

Windows MSVC 19.50/Ninja shared、oneDNN OFF：17/17 定向 Compute/Graph/CPU route/Native
正确性和公共头检查通过。新默认 MatMul 检查 M=1/15/16/17、129x65 与转置权重、三个变化输入、
独立 double oracle、无热 metadata 分配、关闭拒绝和冷限额失败回收。源码编译问题只修复了新头
明确需要的 PUBLIC core/Compute 声明。AUTO 版本与真实 Sched consumer、其它平台/三交付/P1
仍待执行，不计 PASS。原始日志位于本轮外部 scratch，候选固定后归档。

Windows AUTO/oneDNN 初轮：9 项通过、2 项因定向 build 未包含可执行体而未运行。补构建发现
oneDNN composition 测试还依赖旧 Native PUBLIC CPU backend 边，已改为该 consumer 的显式
backend 依赖；这两个测试随后 2/2 通过。默认 Graph 的 Sched consumer 实际输出 region=1；
1/2 worker 的默认路线分别执行 12/24 个 worker task、各 2 次 nested bulk，独立 oracle/并发/释放
全部通过。原失败/未运行日志保留，不写成首次 11/11。

P1 的 Graph 合同同时覆盖串行与同一真实两 worker Sched executor；使用公共 instrument 已有
n-worker-physical-cores 模式，不改 instrument。实际 worker 在 cold start 固定/验证 CPU identity；
完整 default/direct-fused/direct-sequence/Native 工作均计入 bulk admission、执行和等待。1-worker
correctness 由普通 consumer 覆盖。benchmark 已通过 MSVC 定向语法检查；尚未启动 P1。

## 不可变候选与验收入口

- candidate：`79e96cebeb176c48d66f6b6a156fa9d910b89d82`，API、算法和 benchmark 代码固定。
- Windows detached 验证副本与 Native comparator 均指向该提交。正式合同在后续 metadata 提交中
  绑定这个已存在的 peer commit，避免自引用 hash；合同更新不改候选产品/benchmark 源码。
- Native/oneDNN 双入口与 C44 实消费已有 Windows 初轮结果；C45 四环境三交付、定向 sanitizer、
  C46 正式 P1 仍待完成。没有把 syntax check、研究 probe 或历史 P1 记为新候选 PASS。
- `compute-cpu-route-p1-v3`、`graph-cpu-route-p1-v1` 只使用既有 instrument。固定 n=15→31 是工具
  同一候选的既定精度计划，不允许修改实现后无限重测；最多一次有源码成本依据的产品规则纠正。

### 候选验证中的配置纠正

Direct 首次完成采样的调用误加 `--formal`。合同 claim 和本卡要求的是工程 P1；依
instrument-v1 的现行规定，未预登记 Formal 八项工具链身份的 profile 只能走 engineering。
该次四个 workload 的 5% 非劣门与全部 controls 均通过，但 build-contract 资格失败，聚合性能
为 INCONCLUSIVE；原包保留为 DIAGNOSTIC，不能改写成 PASS。下一次使用合同要求的普通
`qualify-platform`，候选、合同、统计门槛与仪器均不改变。Graph 尚未采样，直接使用正确模式。
这次纠正不涉及产品规则，也不把缺失的 SDK identity 补写进旧 manifest。

Unix 三交付外部 recipe 首次误把 tests-on 构建配置为可安装 SDK，CMake 正确拒绝。
修正为 source/test 构建 INSTALL=OFF；SDK 必须 tests-off 并实际重新构建后再安装、搬移和消费。
原 configure 失败日志保留；产品源码、候选 archive 不变。

WSL GCC 14.2 的该候选已完成 23/23 定向正确性/公共头检查，包括同一普通 Direct、Graph、真实
Sched consumer；oneDNN AUTO 实际启用。随后关闭测试、重新构建并安装/搬移 SDK，shared 与
static 的普通 Direct/Graph/Sched 三个消费者分别 3/3 通过。Sanitizer 仍待执行；各模式由 CMake
重编译受影响 HYLE 目标，不重复编译配置未变的外部 oneDNN。每种配置均保留自己的命令、
build/test log，不以 source 结果代替安装交付验收。

Graph 首次工程 P1 已达 n=31 上限：资格与五项 controls 全部 PASS，12 项中 11 项非劣门 PASS。
串行 17×129×257 的 Native 完整图稳定快约 4%～5%，两层区间跨过 5% 门，因此总性能为
INCONCLUSIVE；不能把它写成 parity/PASS。停止旧候选采样。Air source 19/19 与 relocated
shared 3/3 通过，static 则发现新 route 的缺席 oneDNN 私有依赖泄漏到安装接口。

三个直接相关 owner 已按 [有限纠正卡](cpu-v1-candidate-corrections.md) 退回 AUDITED，固定
C45a/C46a；其余 owner 仍保留本卡候选身份。修正只处理这两个具体反例，不改公开接口、默认
规则、公共构建系统或性能仪器。旧失败/未决包已提交，纠正结果尚待完整验收。

修正候选 `d9712e68` 的 Graph 实际 workload A/A 触发固定停止线，两个 Graph owner 已在
[纠正卡](cpu-v1-candidate-corrections.md) 裁决 BLOCKED；没有开展产品采样或放宽门。
compute_route_cpu 的 C45a 固定后回归本卡、candidate=d9712e68，继续尚未完成的 Direct P1 与
交付/sanitizer。R3 总体验收仍 INCONCLUSIVE，不因其余 owner 的实现可用而计作整包完成。

用户要求核对拓扑与性能模式后的实际诊断见
[独立 caller 验收环境纠正](cpu-graph-external-caller-qualification.md)。该轮明确重定实验资源为
三个独立 P 核，依法从 BLOCKED 进入 SCOPED；产品投影保持 d9712e68，原 v1 失败不改写。

## 普通验收已完成，性能仍待裁决

当前产品 d9712e68：Windows/WSL source 各23/23、Air/Mini source各19/19；四环境 relocated
shared/static 的普通 Direct/Graph/Sched consumer 均各3/3 PASS。WSL 新 routes/Sched consumer
的 ASan/UBSan/TSan 各3/3 PASS，外部 oneDNN JIT 未插桩不作 sanitizer 声明。Windows CPU OFF /
CUDA OFF source/installed 各2/2，裁剪头/exports 和矛盾 oneDNN ON 拒绝也通过。

Windows 单测的初始 DLL loader 错误保留；按用户要求把 HYLE 与 GoogleTest DLL 复制到每个 exe
目录并通过加载预检后，才获得23/23结果。之后所有 Windows 验证禁止错误弹窗并设置超时。
没有因验证 recipe 修改产品源码或公共构建系统。原始包、哈希与定向源码复核均在
[本轮证据](../../../../evidence/correctness/compute-cpu-r3-foundation/README.md)。

普通验证到此停止重复运行。尚缺 Direct 的正确 engineering P1 和 Graph v2 的一次 A/A/P1
裁决，R3 聚合继续 INCONCLUSIVE；不能把三交付或工具修复当作性能门完成。

## Graph v2 的实际裁决

Graph 资格及全部 controls PASS，12 项中 11 项非劣门 PASS；双 worker 同实现 direct fused
在 n=31 仍有两个 CI 跨 5% 边界，聚合 INCONCLUSIVE。三个 Graph owner 进入其当前
[停止记录](cpu-graph-external-caller-qualification.md)。用户随后明确要求专门调研此成本阻塞；
另立有限源码审计，不重复原样采样。Direct 正确工程模式正在独立完成，R3 尚未关闭。

## Direct 工程 P1 已通过

候选 d9712e68 / compute-cpu-route-p1-v3 的 engineering qualify-platform 已实际完成：
qualification、所有 controls、4/4 非劣门与 deep verify 全部 PASS，n=15 即停止。严格方向为
M=32 1.047825–1.063461 WIN，其余三个 Native 域 INCONCLUSIVE；不把它们标作 parity。
[完整证据](../../../../evidence/performance/compute-cpu-route-p1-v3-d9712e68-w1-win/README.md)
已保留。旧误用 Formal 的 DIAGNOSTIC 不改写；本轮不再测 Direct。Graph 按用户要求进入
[定向成本修正](cpu-graph-replay-cost-audit.md)，其产品/普通验证与 P1 尚未完成，R3 仍未 CLOSED。

## Graph 定向成本修正的停止裁决

[Graph replay 成本任务](cpu-graph-replay-cost-audit.md) 的 RC1/RC2、四环境三交付和三项
sanitizer 已完成。新 Graph v3 P1 的资格 PASS，旧双 worker 非劣项已通过；约 12 微秒的
单线程 decode-fused 仍有一个 CI 越过 5% 门，11/12 工程门 PASS、聚合 INCONCLUSIVE。
样本已到 n=31 上限，三个 Graph owner BLOCKED，等待明确取舍或新有限任务。Direct 4/4 PASS
继续有效，R3 总包及四路线基础架构均未完成；不通过追加原样测量解除阻塞。
