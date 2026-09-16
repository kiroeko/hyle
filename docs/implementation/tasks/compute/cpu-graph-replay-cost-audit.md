# R3 Graph：同实现重复执行成本的定向审计

<!-- PROJECT UNIT REVIEW: BLOCKED -->

## Closure card

- owner：graph、graph_route_cpu、graph_provider_onednn；baseline `5e9eb06a`。
- trigger：用户在 Graph v2 n=31 裁决后明确要求专门调研并解决阻塞；
  [前轮停止卡](cpu-graph-external-caller-qualification.md) 与全部失败/未决事实保留。
- 事实：qualification 与 workload A/A、公共 controls PASS；同实现双 worker FFN 的 Graph/direct
  median ratio 为 0.970361–0.982654，两个 CI 下界仍越过 1/1.05。其余 11 个工程门 PASS。
- 正常 consumer：同一个已编译的 CPU Graph，固定 shape/资源/队列，输入值变化，反复 submit/wait；
  两个独立 Invocation 仍必须可并发，关闭与失败仍必须按真实终结点回收。
- 对照：相同 oneDNN BRGEMM pair/down 与 HYLE SIMD SwiGLU 的直接融合调用；相同两 worker、
  三个物理 P 核、输入/精度/输出和完成边界。现有未融合与 Native 完整片段作为反偏置对照。
- 范围：沿 benchmark → Graph prepared replay → region enqueue → CPU Queue/completion 调用链
  建立逐阶段成本账单，检查冷准备已经证明的事实是否在热路径被重复计算。只按具体调用读依赖。
- 非目标：变更数学内核/默认 Provider 规则、扩充算子、重开 Sched 算法、放宽统计门或重新调核/idle；
  不新增公共 profile、通用执行器、用户手动排序或无需求的双轨 API。
- 初始 SCOPED 阶段只作源码审计，不授权尚未定界的产品改动。确有多余工作后固定有限 FIX 并进入 AUDITED；
  如缺口属于 Compute 等其他 owner，先单独绑定精确依赖，不借本卡越权修改。
- 预算：源码/生成代码成本审计最多 40 分钟；只保留一个最小机制纠正方向。确需动态归因时，在现行
  Direct 验收结束后预先固定至多 8 个短 diagnostic 进程，使用真实片段，不以 probe 签性能 PASS。
  不允许重新跑原候选完整 P1 来寻找不同结果。若没有可证明的可消除成本，给出具体取舍并停止。
- 预声明验证：审计/文档只跑 selector、docs/closure/diff；进入产品 FIX 前再按其实际影响冻结
  correctness、生命周期/并发/失败、三交付与平台回放范围。API/算法稳定后，最多一份新候选的既有
  graph-cpu-route-p1-v2 工程 P1，原条件和 n=15→31 上限不变；适用未通过不能 CLOSED。

## 审计问题与不变量

1. 两侧各执行两次真实 bulk（gate/up 与 down），相同数学与 host mapping 是否确实等价？
2. Graph 的队列身份、lane/segment/step、依赖和 completion 选择，哪些随每次提交变化，哪些已经固定？
3. 可否在冷准备固定程序结构，同时保留队列关闭/重复接纳、Invocation 排他、provider 失败和资源保活？
4. 单片段、普通多算子、多个独立 Invocation 与异步/多段路径是否保持各自合同？
5. 修复必须来自真实重复执行成本，不为本次 M=17 shape 特判，也不移除正确性/生命周期保护来换数字。

## 初始源码锚点

- `bench/graph_route_cpu/region_p1_bench.cc`：GraphRoute::run 与 DirectFusedRoute::run；
  两侧完成后才返回，直接调用仍计 host mapping/close。
- `src/graph/executable.cc`：Invocation::submit、enqueue_segment_steps、完成组合与终结回调。
- `src/graph_provider_onednn/onednn_region_provider.cc`：region host 映射、BRGEMM pair/down 与激活。
- 具体热路径依赖按上述函数调用追踪，不重新审计整个 Compute、Mem 或 Sched。

## AUDITED：固定一个机制纠正

源码确认：`prepare_invocation_bindings` 已缓存合法的 region views，物理资源规划已经调用
`ContextRef::validate_resource`，外部 bind 又强制资源身份与编译结果相同；重绑定会撤销 replay。
单算子使用 `OperationInvocationAccess::enqueue_prevalidated`，region 却调用
`ProviderExtensionAccess::enqueue_admitted`，每次对四输入/一输出重新调用 backend resource
validator，随后经 Episode trampoline 再调用既定 region callback。这个差异不是数学或 Sched 工作。

不能直接删除所有检查：调用者可以把同一地址的 Queue 对象 move-assign 成另一个 Context 的 Queue。
现有 pointer 相等检查不能证明当前 owner 相同。保留每次接纳时的 Context/Queue admission；在 Graph
提交的 lane 检查中用既有 `ContextRef::owns` 核对当前 owner，再由私有 region 包装直接调用已绑定
callback。不修改 Compute/oneDNN/Sched 产品，不添加字段、公共 API、形状特判或新的程序类型。

| FIX | 范围 | 验收 |
|---|---|---|
| RC1 | Graph 私有 shared-region 重放采用冷准备已经证明的 views/resources；省去重复逐端口资源验证和 Episode 转发；每次先检查 Queue 当前 Context | 数学、host mapping/coherence、Queue admission、Completion/失败/保活原样保持；不改变 delegated 路径 |
| RC2 | 定向防回归与受影响证据 | 计数证明资源验证留在冷阶段；变化输入正确；错资源 bind、原地址换 Context Queue、close 均仍拒绝；真实 CPU FFN/Sched 消费与交付继续实际验证 |

### 成本账单与保留项

- 冷：资源规划/描述、精确资源身份、完整 view、alias、region runtime/vtable、队列 owner 已验证。
- 热保留：Invocation 排他与 output-ready；每 lane 一次 owner 比较；Queue/Context 关闭接纳；
  一个 region callback；五个 host map/close、必要 invalidate/flush；两次真实 bulk；Completion 终结。
- 热移除：每 region 的逐输入/输出 backend resource validator；只用于第二次转发的栈 Episode 和回调。
- 没有 hot allocation/copy 新增，没有常态 metadata/layout 变化；独立 Invocation 不共享新的可变态。
- 不把五次 validator 的消除换算成未经测量的加速比；新候选是否过门仍须实际 P1 裁决。

### 预声明验证冻结

产品仅 `src/graph/executable.cc`、`src/graph/plan_state.h`；测试仅既有 region suite。Windows 定向
Graph/oneDNN-region/default CPU Graph/真实 Sched consumer、对应公共头与 relocated Graph shared/static
消费；WSL/Air/Mini 回放同一 Graph 直接消费者及 Graph 三交付。WSL Graph/region/Sched consumer 的
ASan/UBSan/TSan 继续覆盖新路径；外部 oneDNN JIT 仍不宣称被插桩。原 Direct/Compute/Native/oneDNN
产品不变，复用其独立验收，不重跑 Direct P1。Windows 每次启动前相邻部署实际 DLL，并禁止错误弹窗。

除上述适用检查，运行 selector 与 docs/closure/diff；红队重点核对 cold proof、Queue owner 被替换、
关闭/接纳前缀/异步完成保活。现行 Direct P1 完成前不运行编译、测试或动态探针。
一轮实现及必要 correctness 纠正后冻结产品；下一版 Graph 合同仅更新 candidate/peer provenance
identity，完整继承 v2 的 12 workload、资源/affinity/idle、A/A 门、顺序、placement、统计和 n=15→31。
只运行一次匹配的新候选工程 P1；原 v2 未决证据保留。无法通过时按具体结果裁决，不追加产品方向或样本。

七问：第三方锁/许可证和数学实现不变；正常反复提交与普通多算子是既有真实消费；不重开底层；
成本来自源码中已证明的重复验证；无 FFI/公开 API 变化；三交付相同实现；普通用户调用不变。
有限 FIX 至此冻结。原产品性能门仍未通过，AUDITED 仅表示允许上述定向实现。

## 实现与首轮正确性

RC1 已在两个 Graph 私有文件落地：region 改用显式命名的 enqueue_prevalidated；每 lane 在提交前
检查当前 Queue 的 Context，原始 Queue/Context admission、Completion、coherence 与数学调用未改。
RC2 新增两个回归检查：异步 region 在变化输入下重放不再调用 backend resource validator，错误资源
仍在 bind 被拒绝；相同 Queue 对象换成不同 Context 后在接纳前拒绝，恢复合法 Queue 后可再次执行，
Context close 后仍拒绝。没有 Compute、Provider 数学实现或 Sched 产品改动。

Windows / MSVC / HYLE_WERROR=ON 的七项 source 检查已通过，包含完整 Graph/Executable、真实
oneDNN region、默认 CPU Graph、public Sched consumer 与两个普通 source 消费者。实际 DLL 均先
放到 exe 同目录，加载预检后运行。selector 只选择 Graph 三个相关 factor、configure/docs/closure，
其余 94 项 skipped；未因这次内部修正重复 Direct P1。

后续 Graph 平台/交付/sanitizer 与新候选 P1 尚待执行；当前不得 CLOSED。已完成的独立 Direct
工程 P1 4/4 PASS，原始包另行保留。这里的测试计数不是性能方向或整个 R3 的完成证明。

## 不可变候选与独立源码复核

产品冻结为 `1a3a8d0e40a2a17250119369331c712718a4d880`。Windows 已验的三个 source hash
与该提交逐项一致。统一 loc-report.py self-check PASS：Compute + Graph 全部实现 31,021 行，
不含 tests/bench；旧实现仍在计数内，该数字不代表新基础架构完成量。

候选后单独复核：bind 检查描述、精确 resource 和 alias 后撤销缓存；prepare_replay 校验 active lane、
Context owner、完整 views 及 completion capacity；submit 重新核对 Queue 的当前 Context，再进入
QueueSequence 的真实 admission。region extension 继续保活 Provider，Invocation/plan 引用和
accepted-prefix/failure/异步 completion 回收路径未改；delegated 与已录制 capture 的重放不经过此次
删去的 shared-region adapter。capture 录制仍在冷阶段验证 Queue/resources 后提交步骤。
没有新增 hot allocation、共享可变态、公开字段或 API。
新增测试分别覆盖 cold proof 与同地址替换 Queue，既有 async/失败/生命周期 suite 已实际通过。
本遍源码复核没有发现需要退回产品的缺陷；平台与 sanitizer verdict 仍依实际运行填写。

[Graph P1 v3 合同](../../../../bench/contracts/graph-cpu-route-p1-v3.json) 只更新合同 id 和
Native peer 的不可变来源到该产品提交；12 workload、两个 placement、执行/分配正反、实际 A/A、
同一三 P 核拓扑/low-latency、5% 非劣门和 n=15→31 均逐字段继承 v2。合同提交不改变产品投影，
实际 runner source revision 由新的 manifest 保留；不覆盖、不重跑原 v2。

## 平台与交付检查点

Windows source 7/7、搬迁后 shared Graph/Sched 消费 2/2 PASS；WSL source 7/7、Air/Mini 各
6/6，三台 Unix 的搬迁后 shared/static Graph/Sched 均各 2/2 PASS。每台验证副本的 891 个
product/build/test 文件都逐项核对 Windows 候选 hash。原始记录见
[本轮证据](../../../../evidence/correctness/graph-cpu-replay-cost/README.md)。Windows static 与
WSL sanitizer、新 Graph P1 仍待实际完成；没有把普通验证写成性能 PASS。

全库性能总账只作当前状态同步：移出已重开的旧 Compute/CUDA/Graph claim，保留历史证据；
补记已 CLOSED 的 compute_ops schema 与 Mem reservation P0。Mem logical-range 的源码和合同
未变，明确标作 inherited unchanged route；不重新采样。施工中的 Direct 全部四项和旧 Graph 全部
十二项方向继续单列，包含固定 LOSS 与未决项，不冒充当前 CLOSED 覆盖。

WSL ASan/UBSan/TSan 后续实际完成，各 5/5 PASS；Graph/Executable、oneDNN region、默认
Graph 与真实 Sched consumer 全部回放，外部 oneDNN JIT 不作插桩声明。linux.tar.gz 已保留全部
原始记录。普通验证到此仅余 Windows static 消费；为避免重编同一依赖，将直接使用新 P1 的冻结
静态产品产物安装/搬迁。Graph P1 已以 source `7fa6dc0a8b3b857a1835525de0d0e7be5f8db5e9`、
Native peer `1a3a8d0e40a2a17250119369331c712718a4d880` 启动；两者产品投影相同，尚无性能裁决。

## Terminal：修正已完成，工程性能仍未决

RC1/RC2 与全部预声明普通验证完成。Windows source 7/7、WSL source 7/7、Air/Mini source 各
6/6；四环境搬迁后 shared/static Graph/Sched 消费均各 2/2 PASS。WSL ASan/UBSan/TSan 各
5/5 PASS。Windows static 直接安装已冻结 P1 产物，未重复编译；运行前 DLL 相邻部署。

[Graph v3 完整工程证据](../../../../evidence/performance/graph-cpu-route-p1-v3-7fa6dc0a-w1-win/README.md)
已保留。qualification、实际 workload A/A、全部五项 controls、artifact/build validators 和 deep
verify 的身份/路径检查均 PASS；n=15 后按原合同进入 n=31，此后停止。工程门 11/12 PASS，聚合
INCONCLUSIVE。严格方向为 4 LOSS / 2 WIN / 6 INCONCLUSIVE，未把非劣改写为 parity。

原双 worker 同实现项已过非劣门；新候选的 median ratio 0.972652–0.977621，全部 CI 在 1 下方，
所以严格方向仍是 LOSS。这个跨候选变化不能当作同轮配对的修复加速比。

本轮唯一未决项是 `decode-fused-boundary-1x256x512`，即 Graph 与相同 oneDNN BRGEMM + HYLE
SIMD 的直接融合调用。Graph 单片段约 11.79–12.29 微秒，paired median ratio 0.966131–0.974263，
对应约 2.6%–3.5% 额外耗时。`placement-peer-first-candidate-allocation-first` 的 98.9326% CI 为
[0.9517370300377059, 0.9726927370864213]，低于非劣下界 1/1.05=0.9523809523809523；其退化
上界约 5.07%，因此不能签 5% 非劣 PASS。这是未决，不是已经证明 material FAIL；稳定小幅落后
本身保留为 LOSS。绝对时间范围不用于重算 paired ratio。

| Gate | Verdict |
|---|---|
| scope | PASS：只做 RC1/RC2，原比较条件和样本预算不变 |
| architecture | PASS：cold proof 复用，保留 Queue 当前 owner/admission/真实 completion |
| correctness | PASS：四环境 Graph/真实 FFN/Sched、变化输入、关闭/异步/失败与三项 sanitizer |
| delivery | PASS：四环境 source + relocated static/shared 普通消费 |
| usability_docs | PASS：普通入口不变，成本/适用域与证据限制明确 |
| red_team | PASS：候选后独立源码复核及受影响风险回放，没有新产品缺陷 |
| engineering_performance | INCONCLUSIVE：资格通过，但一个已运行的适用 P1 项仍未能通过非劣门 |

三个 Graph owner 因此 BLOCKED，不能 CLOSED；原 R3 其它 owner 和整体四路线 goal 不算完成。
本卡的一轮机制修正和唯一新候选采样预算已使用完毕，不再追加本候选采样、核/idle/顺序变体，
也不把约 0.07 个百分点的 CI 越界自动豁免。用户需要裁决是否接受这个已量化的未决限制，或授权
另一个有明确预算的成本优化轮；批准前保持上述事实和状态，不按文档修订重建。

resume_condition：用户明确裁决该小片段验收取舍，或授权新的有限成本优化轮；依新边界进入
SCOPED/AUDITED。不能恢复旧 FIX 队列。统一实现统计仍为 Compute + Graph 31,021 行，不含测试，
包含保留的既有代码，不代表新基础架构完成量。
