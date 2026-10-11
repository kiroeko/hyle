# F2：准备批量搬运最终收口

状态：CLOSED，2026-10-12；原施工卡为[F2七FIX](foundation-f2-batched-transfers.md)。
启动baseline为7f53bf532b044803ea36f792196e446c66b58ac4；终审前registry ref为338438419。
本记录仅关闭原F2范围，F1保持既有CLOSED，F3尚未实施，不表示完整推理引擎完成。

## 固定owner与候选

| owner | 完整candidate | 本轮边界 |
|---|---|---|
| compute | c26261e3958c7eb46ecde2a4ba7c5654a03f1b93 | byte/phase/control、CPU/CUDA公开程序及两个cold prepare分类修复 |
| compute_backend_cuda | 5b3efe2cce3fcab2bc55cf749af2d862a786ef25 | 原native batch/fallback及真实晚失败终态释放边界 |
| compute_route_apple | 03162add05d3e26fe3314c6bcdaafa5a277a3a76 | 原Runtime/Job/MLX stream上的独立native copy程序 |
| compute_route_vulkan | 03162add05d3e26fe3314c6bcdaafa5a277a3a76 | 原Run/Device/pool/fence上的独立native copy程序 |

最终全源码snapshot为c26261e39/tree185e3c211a65754f5487e7e1d19f8252e68f29d9。
后续只维护测试、文档与证据；其他三个owner在该snapshot内的product scope与各自候选相同。
[最终收据](../../../../evidence/correctness/foundation-f2-final/receipt.json)绑定源收据、原日志SHA核对与统一LOC，
[当前矩阵](foundation-f2-verification-matrix.md)给出每个平台/风险实证。

## 原七项FIX与完整旅程

| FIX | 验收结果 |
|---|---|
| F2-01 | immutable prepared program、独立fixed invocation/submission、资源/字节/metadata容量；两槽三轮replay、旧句柄/满容量/空批/OOM实际通过 |
| F2-02 | 真实physical identity、checked origin/capacity、排序alias扫描、显式phase闭包；重复source、冲突、自复制、padding与500组独立oracle通过 |
| F2-03 | CPU原同步byte copy；CUDA固定PFN_v13000与冷proc探测，缺席沿原same-stream memcpy，STREAM源排序；actual bytes/calls、producer、prefix失败通过 |
| F2-04 | Apple原Runtime/Job与MLX command stream、单phase blit encoder；共享资源排序、原command buffer完成、取消/晚失败/旧ready与实际回收通过 |
| F2-05 | Vulkan原Run/pools/fences、buffer-pair regions、cache/family handoff；usage/对齐/owned view/unknown external拒绝、真实unknown观察与退休通过 |
| F2-06 | F1 guard与producer交接，last-write逻辑覆盖后ready，partial acceptance、未知/cancel/abandon、旧结果与实际最后使用回收；四路线证据通过 |
| F2-07 | 普通hyle::graph/Compute入口，两组两槽，原BF16/GGUF Q8或Apple affine Q4搬运后原Direct/Graph数学；source/static/shared、迁址/新头/缺席/运行库及P0/指南通过 |

主纵切为量化权重/状态范围，正交纵切为分组资源搬运/分块数值消费；不引入模型/layer/token策略。
producer与native执行owner保持分治，不增加数学/copy kernel、万能Queue、第二runtime、用户内部
factor/provider开关或FFI。运行期GPU生成索引、peer/multi-GPU、文件管线与完整模型均不计入F2。

## 语义风险矩阵

| 能力/旅程 | 正常/边界 | failure/OOM/overflow | ownership/teardown | cancel/concurrency | 消费/交付 |
|---|---|---|---|---|---|
| 范围/phase/alias | PASS，空/自复制/重复source/坏ID/显式hazard | PASS，checked容量/地址/byte sums与冷OOM | PASS，physical backing非wrapper identity | PASS，阶段独立性不靠数组顺序 | 三编译器新头与独立oraclePASS |
| 固定program/slot/replay | PASS，两槽三轮 | PASS，失败binding不消费/满容量拒绝 | PASS，旧submission/reader阻止复用 | PASS，thread-compatible槽与原owner序列 | 三平台CPU及各native样例PASS |
| native copy/dependency | PASS，upload/D2D/readback与合法CPU变体 | PASS，冷缺席、接纳prefix后的失败 | PASS，descriptor/index/producer保活 | PASS，unknown不当终态；原cancel能力如实 | Windows CUDA/Vulkan、Air Apple/Vulkan、WSL CUDA PASS |
| group ready与数学 | PASS，logical覆盖/实际last-write退休 | PASS，失败组不发布，旧成功组可读 | PASS，readonly后继/旧结果保活 | PASS，较早独立组可观察，未承诺物理重叠 | BF16/原编码Direct+Graph oracle PASS |
| 普通cold prepare | PASS，纯resource mismatch显式decline | PASS，hard Unimplemented/OOM不重试，partial artifact回收 | PASS，三个resource predicate及原error用例 | 正常热submit/布局未改 | Win58/90/Graph853/CPU833，WSL58/91/15及三编译器ordinary数学PASS |
| SDK/API边界 | PASS，source与迁址static/shared | PASS，未构建route/头明确缺席 | PASS，scope匹配的旧证据与新cold消费者区分 | header-only不造binary journey | Windows DLL、Air metallib/dyld、WSL ldd实证PASS |

WSL Vulkan实际只有llvmpipe，因此其GPU执行格为NOT_RUN_NO_QUALIFIED_GPU，适用source/安装/
头/absence格已通过；不冒充该平台GPU PASS。真正GPU数学/同步分别由Windows RTX4070和Air
M4/MoltenVK证据签发。非适用设备/平台不是本轮新增承诺；没有适用而未完成的F2风险格。

## 七门裁决

| gate | 证据与结论 | verdict |
|---|---|---|
| scope | 原审计、固定七FIX、两个真实counterexample依法定界；按用户明确要求排除未受影响全局回放 | PASS |
| architecture | common只拥有geometry/control；原QueueSequence、Runtime/Job、Run/Device分治，中央private依赖/组件合同实证 | PASS |
| correctness | 上述风险矩阵与[22份源收据审核](../../../../evidence/correctness/foundation-f2-final/evidence-audit.json)，372个显式引用SHA匹配，失败/skip恢复保留 | PASS |
| delivery | 当前矩阵的source/static/shared与scope匹配证据；cold body更新后Win三交付、WSL及Air实际数学通过，未改变新头/包/原生接缝不重跑 | PASS |
| usability_docs | 公共类型附近的借用/固定槽/ready合同、[用户指南](../../../user/compute.md)、普通样例与docs/closure检查 | PASS |
| red_team | [候选终审](foundation-f2-final-review.md)，真实产品反例已修并复验；fixture/范围问题有独立处置与原FAIL，不用exit0掩盖skip | PASS |
| engineering_performance | P0结构/实际成本账：固定metadata、排序/有界phase算法、原native calls/完成/host wait；无timed赢家或vendor零成本承诺 | PASS |

## P0与限制

成本详见矩阵：prepare/create冷分配固定N/R/G数组，replay为O(N log N+R log R+NG+RG+G²)，
G≤64；submit为O(R+E+G)，E为显式依赖边。报告分别计logical/physical/staging/metadata、
native entry calls、native dependencies、阶段完成与host waits。CPU保持同步；CUDA缺席fallback
明确O(R) native calls；Apple原worker每native phase等待实际MCB；Vulkan每buffer pair一次copy。
冷数学prepare/Graph compile独立计费，不混入copy replay稳定allocation证明。

vendor native control/allocator成本保留UNKNOWN，严格预算不能猜0。Resource/allocator以及借用
Queue地址须活到全部actual use退休；pending析构放弃观察不等于回收。CUDA sticky terminal后
native物理free仍UNKNOWN，逻辑owner/guards归零不是GPU residency归零，按原native合同重启
进程；HYLE不reset共享primary context。上述限制在指南和收据明确，不通过重放/忽略错误隐藏。

## 预算、历史与停止

原调研/纠正预算未重置。CUDA first-use测试TIMEOUT、实际释放fatal、ordinary prepare硬错误
FAIL、locale/cache/目标选择/精确账单fixture FAIL、合并测试TIMEOUT与7项fixture skip全部保留。
实际产品修复均在相应owner退回AUDITED后提交；fixture维护没有重开CLOSED产品。
2026-10-12用户要求未涉及其他模块不做全局验证；原广泛selector调度是历史事实，未受影响
factor/其他facade标NOT_SELECTED_UNCHANGED_BOUNDARY，不写PASS，也不是F2未决缺口。

统一LOC使用338438419不可变tree，产品与c26261e39相同：全库2,412,351；自实现不含测试
168,556；本轮四owner中央归属union实现26,930（141文件），self-check PASS。
指标是完整owner当前规模，不是新增行数或完成度，不手工相加各历史快照。

收口后停止F2常态施工。合法reopen trigger仅为新的可复现copy/alias/ready/lifetime/错误或交付
缺陷、真实consumer证明的新常用能力、相关采用依赖升级或合格material成本回归。
不得用历史pending/旧task或继续研究恢复本轮。下一步只按当前用户goal依法启动原F3八FIX。
