# Foundation A 四路线 Graph 语义接缝收口

状态：CLOSED，2026-10-12。只关闭A-01..04及两个已冻结owner接缝，不关闭F4/F5或当前总goal。
七owner：graph、graph_route_apple/vulkan、graph_route_cpu_cuda/apple/vulkan、compute_route_vulkan。
baseline与有限预算沿[主卡](foundation-a-route-symmetry.md)、
[CPU+CUDA卡](foundation-a-cpu-cuda-variants.md)、[Vulkan接缝卡](../compute/foundation-a-vulkan-control-seam.md)。
原F1–F3与其他CLOSED范围不重开；本轮不扩算子目录、万能runtime、模型策略或全引擎。

## 不可变候选与有限FIX

product546b0d2468e5de29031c34668f4afa4bc31b9a17；snapshot9d0aa797a仅加状态/卡，产品等价。
最终[收据](../../../../evidence/correctness/foundation-a-final/receipt.json)保存完整tree、18个installed
格的job身份/逐stage日志SHA校验、原失败和不适用边界。候选后只改测试/说明/证据/registry。

| FIX | 验收结果 | Verdict |
|---|---|---|
| A-01 | 原shape program/guards/ViewSpec/输出推导与Graph合法性复用；exact快照保留原ValueId/NodeId/effect身份，各native artifact/owner保持 | PASS |
| A-02 | 两精确点全表预检/重复与表外拒绝、独立slot/旧结果；三hybrid与native冷typed factory；CPUCUDA拒绝逐点重置family-wide有限compiler配额 | PASS |
| A-03 | 数据+控制可达性/活前驱、冷端点分片/物化，原native完成/barrier排序；checked前驱接受后失败与好index成功、跨fragment、禁止hostwait拒绝 | PASS |
| A-04 | 显式qualified CPU effects原端口/footprints、全图alias预检/绑定校验、effect-only、失败前缀/取消/close；未知GPU carrier/mandatory-alias和opaque alias未证明者明确拒绝 | PASS |
| A-CUDA-01 | 新有限表入口沿原共同compiler/Executable/Invocation；实际RTX4070 Win/WSL三个交付形式、真实host→CUDA→host/旧结果及quota拒绝 | PASS |
| A-VK-01 | PlanStep只加两个冷bool并抬原last_uses/分片；原commands/managed completion排序，原无control图默认形状保持；实际Win/Air与WSL包格 | PASS |

主consumer为有限batch/状态pure initial-final，正交consumer为变长资产/矩阵独立分支排序及
sidecar集合更新。CPU custom body与资源由普通consumer声明，HYLE不解释模型或业务事务。

## 逐风险与实际平台交付

| 能力/风险 | 证据/裁决 |
|---|---|
| 共同形状/表/身份/guard/View、全表失败回收、OOM | [共同原子](../../../../evidence/correctness/foundation-a-shared-exact/README.md)与[最终候选前41项](../../../../evidence/correctness/foundation-a-redteam-first/README.md)；原四fixture失败保留；ASan/UBSan候选各42项补全提取后的effect预检 |
| 控制前驱及native错误 | [Win native/hybrid](../../../../evidence/correctness/foundation-a-vulkan-source/README.md)；Air Vulkan当前SDK实际checked Gather；Air Apple native-only static/shared当前反例：无data consumer坏index返回OutOfRange，好index独立后继正确 |
| qualified CPU effects/全图footprints | [Win](../../../../evidence/correctness/foundation-a-effects-first/README.md)、[Air最终source](../../../../evidence/correctness/foundation-a-apple-candidate/README.md)与各hybrid SDK；正常write→GPU→read、missing/duplicate/alias/constants/input、OOM与accepted DataLoss前缀 |
| 权限/ownership/close | CPU artifact原权限，fixed footprint borrowed；native必须有真实readonly/alias证明；两独立native slots/旧output、旧result阻止reset、effect-only及close后Context再用 |
| cancel/concurrency | Win/Air public consumer真实已开始CPU prefix取消，后继不开始、原完成退休；table immutable/无新增全局共享runtime，TSan NOT_APPLICABLE；不借此代签GPU producer未知完成，后者无产品机制变更 |
| sanitizer | [ASan/UBSan](../../../../evidence/correctness/foundation-a-sanitizers/README.md)候选各42项；原selector/configure FAIL保留，测试专用effect overlay不改candidate |
| CPUCUDA source | [WSL真实source](../../../../evidence/correctness/foundation-a-wsl-cuda-source/README.md)及Win原source/SDK，RTX4070；旧LibTorch CUDA12/nativeCUDA13 loader警告保持，未改锁 |

source原公开target/artifact编译与普通consumer旅程、installed static/shared使用同entry。
18个SDK格均实际install→rename迁址→only-prefix public consumer/独立头TU→loader→缺路线
诊断。矩阵中适用数学实际运行；软件WSL Vulkan只签构建/链接/包，不冒充硬件。

| SKU / 系统 | Static | Shared | 数学 / 说明 |
|---|---|---|---|
| CPU+CUDA Windows | [PASS](../../../../evidence/correctness/foundation-a-sdk-cpu-cuda-static/README.md) | [PASS](../../../../evidence/correctness/foundation-a-sdk-cpu-cuda-shared/README.md) | actual RTX4070；有限形状/混合/旧结果/quota |
| CPU+CUDA WSL | [PASS](../../../../evidence/correctness/foundation-a-sdk-wsl-cuda-static/README.md) | [PASS](../../../../evidence/correctness/foundation-a-sdk-wsl-cuda-shared/README.md) | actual RTX4070；ldd真实迁址 |
| CPU+Vulkan Windows | [PASS](../../../../evidence/correctness/foundation-a-sdk-vulkan-static/README.md) | [PASS](../../../../evidence/correctness/foundation-a-sdk-vulkan-shared/README.md) | finite/control/完整public effects/cancel |
| CPU+Apple Air | [PASS](../../../../evidence/correctness/foundation-a-sdk-apple-static/README.md) | [PASS](../../../../evidence/correctness/foundation-a-sdk-apple-shared/README.md) | actual M4；finite/control/effects/cancel；首配置身份FAIL保留 |
| CPU+Vulkan Air | [PASS](../../../../evidence/correctness/foundation-a-sdk-air-vulkan-static/README.md) | [PASS](../../../../evidence/correctness/foundation-a-sdk-air-vulkan-shared/README.md) | actual M4/MoltenVK；有限形状/control错误/effects/cancel |
| native Apple Air | [PASS](../../../../evidence/correctness/foundation-a-native-sdk-air-apple-static/README.md) | [PASS](../../../../evidence/correctness/foundation-a-native-sdk-air-apple-shared/README.md) | 无CPU；坏/好checked control前驱、两槽/旧结果；缺CPU明确诊断 |
| native Vulkan Windows | [PASS](../../../../evidence/correctness/foundation-a-native-sdk-windows-vulkan-static/README.md) | [PASS](../../../../evidence/correctness/foundation-a-native-sdk-windows-vulkan-shared/README.md) | 无CPU；真实finite/control错误与独立结果 |
| native Vulkan Air | [PASS](../../../../evidence/correctness/foundation-a-native-sdk-air-vulkan-static/README.md) | [PASS](../../../../evidence/correctness/foundation-a-native-sdk-air-vulkan-shared/README.md) | 无CPU；真实finite/control错误与独立结果 |
| native Vulkan WSL | [PASS包](../../../../evidence/correctness/foundation-a-native-sdk-wsl-vulkan-static/README.md) | [PASS包](../../../../evidence/correctness/foundation-a-native-sdk-wsl-vulkan-shared/README.md) | 当前loader仅llvmpipe type4，GPU math NOT_APPLICABLE；source/新头/ldd/缺CPU/迁址均适用并PASS |

所有Air格核对Mac16,12/M4/16GiB/macOS27.0.1/Clang21，仅Air，没有Mini代签。库按实时可用内存
j4≥12GiB/j2≥8GiB/j1≥4GiB，test1；重型门跨机器串行，原源码cache复用但binary不跨配置。
仅选择实际owner/必要直接consumer与package seams，没有全库算子回放、P1或修改公共构建工具。

## 七门、红队与成本

| Gate | 事实与边界 | Verdict |
|---|---|---|
| scope | 原四FIX及两个有限seam、consumer/baseline/预算冻结；无F4/F5产品混入 | PASS |
| architecture | 共享Graph静态规则/身份，native执行/退休分立，CPU effects原Provider，无新runtime/kernel | PASS |
| correctness | 上述逐能力矩阵/原子/OOM/ASan/UBSan/actual controlled errors/qualified effects/取消/旧结果 | PASS |
| delivery | 18实际installed格与普通source；纯native缺CPU独立取证；只按当前平台资格签数学 | PASS |
| usability_docs | [Graph指南](../../../user/graph.md)解释finite-only/Context/allocator/footprint寿命、hostwait/private costs和拒绝域；普通public C++ consumer | PASS |
| red_team | 独立于实现的复核阶段发现mandatory-alias漏洞并于候选前修复；候选后二审查出缺native-only/Apple错误证据，现全部补齐；原失败保存 | PASS |
| engineering_performance | [P0/统一LOC](../../../../evidence/architecture/foundation-a-p0/README.md)，冷表/分析/footprint账与原warm artifact，无timed parity/SOTA声明 | PASS P0 |

第三方版本/许可证沿原锁和[固定peer研究](../project/foundation-f45-reference-research-2026-10.md)，
不升级MLX/ggml/FlashInfer或引入新运行依赖；功能由两个纵切与成熟owner证据准入。
复用原合法性、Provider及完成owner；C++ typed API无新C/Native FFI；source/SDK实际资格与指南
完成七问。whole engine、所有general/capture/原生GPUmutable ports、paged与性能差异不属A证书。

二审（root，分离审查阶段，未启子agent）依次反证全表先验、factory失败前缀/allocator、
default/foreign handles、控制祖先/mandatoryalias、全图/跨fragment footprint、CPUCUDA总配额、
native-only安装/缺CPU及guide误导。未发现新产品缺陷；证据缺格全部有实际终局结果。
已修产品缺陷为候选前一次mandatory-alias，另修纯control诊断文字；候选后Apple只扩测试。

冷V点准备O(V(N+E+D)+V²D)、冷选择O(D+VD)、native分析O(N+E+C)；submit无shape搜索/JIT/
cache增长/DAG遍历，沿原artifact。控制输出物化、managed waits和边界copy显式计费；
native私有bytes仍UNKNOWN。CPU effects fixed backing由caller保活/跨invocation同步，accepted
failure没有rollback。旧全图collector仍SmallVector0，无新增默认Tensor字段、线程/锁/worker。
统一LOC：全库2,471,964行/6,021files，self非tests170,581/695，七owner去重27,286/83；
snapshot9d为产品等价，self-check通过，不以LOC或测试数量算完成率。

原fixture输出spec/Context工厂/链接依赖FAIL、ASan selector冲突FAIL、Air配置身份漂移FAIL与
恢复累计账均保留；换日志/证据目录没有重置活动budget。未运行项未写PASS，WSL软件GPU与
新增并发机制的NA理由已明确。文档/closure/diff例行检查不构建。

合法reopen只由新的实际shape/control/effect权限、寿命/交付反例、准入consumer/锁升级或
合格成本回归触发；原历史TODO与F4/F5下一步不授权日常重写A。总goal继续F4/F5。
用户新明确的单机高性能单/多请求目标与goal后代码量/剩余seam见
[条件评估](../project/foundation-post-goal-engine-effort-2026-10.md)：A的P0不是整引擎性能证书。
