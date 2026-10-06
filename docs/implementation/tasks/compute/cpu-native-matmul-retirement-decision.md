# CPU Native MatMul：源级清退与独有资源能力的取舍

<!-- MODULE REVIEW: AUDITED -->

owner `compute_provider_native`；用户已明确选择源级删除并撤回具名严格能力。
这是[ND3-D比较](cpu-native-domain-performance-verdict-v1.md)之后的有限实施子批。
Windows baseline `e8fec259ec6e687acce81affd09deec39c0e302b`。

## 已经知道的事实

- 两个M1布局的合格P1都没有可靠优势；M32/转置RHS的Native吞吐约为原ggml的0.522–0.535，
  row-major M32 A/A未资格。一次预算已结束，不再优化或重测寻找WIN。
- ordinary CPU 的F32数学已由oneDNN、优化ggml及ATen实际资格化覆盖；更高rank、byte stride等
  由原ATen承载。不能因为某个adapter或资源上界未知，把MatMul数学写成唯一自研实现。
- Native目前仍有可观察的**资源/执行合同差异**：caller workspace、额外private invocation为0；
  更高rank也能prepared。成熟库未全部给出同一严格资源/执行资格；删除会让对应的Native-only或
  严格unknown-reject/prepared-only请求明确失败。它们不是普通模型所需算子缺口。
- 当前资源审计包含冷准入证据，不证明目标模型必须使用这项严格能力；不能把它扩成全目录KEEP。
  另一方面，当前closure card明确要求保留有证据的独有资源域，不能不说明就撤回。

## 推荐的实际删除方案

1. 删除Native普通F32 MatMul的row-major、transposed-weight、batch/broadcast与K0专用执行/准备
   分支和只服务它们的helpers；Native expert冷prepare/propose明确Unimplemented。
2. 删除 `src/compute/matmul_kernel_impl.h` 与 `matmul_kernels.h`，删除Native CMake的七个fat
   MatMul生成路线。引用扫描只有Native的include/CMake，未发现其它产品使用；不是SIMD通用库。
3. 保留MatMul公开schema、全部成熟调用、原Torch/ds4 lowp数学，以及SIMD的dot/FMA/codec等基础。
4. 将旧Native-only矩阵fixture迁至已接成熟factory/普通入口，并保留Native明确拒绝回归；普通
   Direct/Graph、更高rank/布局/empty、必要两平台三交付一次批量验收。
5. 文档明确撤回上面具名严格资源域；未知mature上界继续未知，不能改成0或悄悄放宽请求。

实际修改前绑定Native以及两份历史common-path私有文件的Compute owner，SCOPED/AUDITED固定
有限FIX；不重审Compute runtime、不改公共路由/性能仪器或GPU。

## 另一种选择

保留MatMul仅作为具名严格资源/高rank prepared能力，不写性能WIN，不作为普通默认优选依据。
这会保留其实际数学代码；不能将它报告为“已删除”，也不能用1000 fallback优先级当去留证据。

其余14个落后场景多数与小形状已赢的实现共享函数，按条件裁决，不整族删除。已验RMSNorm/Copy
whole-domain能力和SIMD基础保持。默认WIN规则仍需按本机/版本/精确shape/layout绑定，不能拆掉
成熟完整region。

## 用户决定与有限 FIX（已审计）

2026-10-06 用户明确选择“删除自研 MatMul，明确撤回这些严格能力”。本轮无需再次确认。
baseline为上方不可变e8fec259；仅普通Native F32 MatMul退役，原Torch/ds4借用不变。

| ID | 有限修改 | 验收 |
|---|---|---|
| MM1 | native_provider.cc删MatMul专用执行/helpers/准备与fan-out，冷阶段明确Unimplemented；Native CMake删compute_matmul fat路线 | 源引用0，Native propose/prepare拒绝F32及其它accumulation/empty/high-rank；SIMD与borrowed源identity不变 |
| MM2 | 迁移matmul_gated_test与operation_test的原Native矩阵fixture到ATen；超大empty的Native-only成功预期转拒绝；新增明确资源撤回/普通替代消费 | 保留独立数值参考、row/transposed/broadcast/K0/empty与changed replay/close、ordinary Direct/Graph；不把拒绝算执行PASS |
| MM3 | 更新CPU guide的Native目录/并行/资源限制，记录已删除而非1000后备 | source/static/shared用法一致，严格请求明确失败不放宽 |
| MM4 | 直接消费扫描发现旧CPU/Graph MatMul比较targets与cpu_executor样例仍用退役Native矩阵/FFN。移除两个仅比较旧自研的benchmark目录；样例只用ordinary CPU、不强制已撤回FFN region或prepared-only | 实际executor样例在Win/Linux执行；保留原数值、并发、nested completion、owner平衡，不因特定provider身份制造失败；原P1合同/证据历史保留 |

Compute owner单独绑定两份历史common-path私有kernel删除及operation_test迁移，见
[相邻小卡](cpu-native-matmul-private-kernel-cleanup.md)。没有公共API或Compute runtime算法变化。
预算：一次批量修改/构建，必要失败纠正最多一批；算法优化0，P1不重跑。P0检查删除后无新增热成本，
原一次合格P1是退役依据，不伪称新候选的性能PASS。其它14LOSS域/187优势规则仍独立待处理。
预声明：Win/Linux owning Native factor、迁移矩阵/实际普通CPU与Graph消费者、native header；
同普通consumer source shared/static与installed shared/static、Native-only和严格请求拒绝。
未变成熟实现/原quant与其它D5数值/资源证据按input identity复用，不重跑全目录。
Windows产品真相；j2构建/j1测试/重型串行/DLL邻exe/隐藏子进程。新SDK使用独立小前缀，不覆盖
本轮775b2bed静态SDK性能输入；只清本批已完成的consumer临时目录。
closure/docs/selector/统一29路径LOC适用；实际全部通过前不关闭本子批或整个goal。
