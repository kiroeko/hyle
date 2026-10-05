# CPU成熟接线：浮点Cast与稳定泛型路由

<!-- MODULE REVIEW: SCOPED -->

- owner：`compute_provider_native`、`compute_provider_ggml`；baseline `b881d2788e0848404c536342b806d6642f3aefa7`，Windows canonical。
  CPU/Graph现有assembler不变；只改原成熟factory的精确准入/数值程序。不修改其已关闭公共规划器。
- trigger：全族审计仍有FP8 numeric Cast成熟候选未接；新增普通Direct与whole Graph实际反例：
  gate=-90/value=1e38，ggml返回0、稳定参考约-7.37461；value=Inf返回NaN而参考-Inf。
  generic Sigmoid(-90)返回0而F32参考8.19400869e-40。见本轮[证据](../../../../evidence/correctness/cpu-aten-cast-stable-route/README.md)。
- 主消费：五种推理浮点storage转换（F32/F16/BF16/E4M3FN/E5M2）和稳定门控→MatMul；
  正交消费：Cast all-bits/rounding/empty/odd-byte及Sigmoid尾部→Multiply大值。
- scope：已有Cast，exact/rank0..8/合法byte stride/inferred dense output/out-of-place，同dtype原位存储bits
  materialization；不同dtype按公共RNE、FP8有限saturation、Inf和NaN mapping合同，不重新定义dtype。
  已锁ATen原Tensor转换/Byte packing与where/clamp/bit-view组合候选；没有自研payload循环/数学核。
  ATen增加stable Sigmoid原Double→F32→storage候选。泛型ggml Sigmoid/SiLU只有承载完整同合同后
  才可准入；原模型program保留各自独立已验收数值合同，不能以本轮generic反例直接推翻。
- 路由：只撤回已证明不合格的泛型候选，普通prepared/Graph匹配现有成熟ATen；用户不排序、不提供profile。
  若原ggml的F32程序不能保证stable-tail，则不能把错误整region保留为性能基线；此时必要边界由HYLE
  规划，ATen managed leaf和后续mature MatMul组合。其它ggml/oneDNN原region保留；不添万能执行器。
- 资源：既有冷private native packing槽、checked host mappings、真实Completion与异常/close；managed
  original Tensor metadata/scratch、private threads、unknown内存，不承诺capture或零热分配。
  FP8是实际原数值转换，不用上一轮Byte bit-copy冒充；Cast全部alias拒绝。
- 依赖：既有LibTorch2.8.0 CPU，已锁SDK仅Windows/Linux x86_64；不升级ggml/SDK、不改factor/package。
  不写Native自研math、不计时/淘汰SIMD算子；SIMD基础保留，其它路线暂停。
- 预算：一次原Cast程序与Sigmoid准入witness，固定TC1–4后批量产品修改；Win/Linux完整受影响
  Native/ggml/CPU/Graph factor与同一ordinary consumer三交付，最多一次具体产品纠正；重型串行j2。
  不逐dtype逐算子重建SDK，不重复旧ISA/loader、未改完整模型、P1或公共治理系统。
- 验证：独立FP8枚举/nearest-even参考、全部源storage bits及tie/边界/±Inf/NaN/±0、全25种dtype对、
  rank0/8/empty/zero与odd byte stride、out-of-place拒绝、重放/close/OOM/资源；稳定Sigmoid三dtype
  尾部及原大值gate下游；成熟identity与原有效其它region回归。模型fixture缺失如实跳过不签PASS。
  Windows DLL邻exe/隐藏启动；selector/closure/docs/diff、候选Compute+Graph统一实现LOC。
- tier：P0，包含所有packing/cast/native Tensor mask/临时数据及必要图边界/融合损失；不签SOTA/WIN。
- stop：原库程序不能匹配具体域即保存具名差异，不放宽数值、不写自研math或判UNIQUE；达到预算
  按协议裁决，未运行适用验证不签PASS。此轮完成不是CPU整体完成。

## 有限审计对象（尚未产品FIX）

| ID | 对象与拟接缝 | 必需证据 |
|---|---|---|
| TC1 | 原版五dtype Cast，RNE/有限sat/NaN mapping/同dtypebits、对齐Byte packing | 枚举与稳定独立参考；不把原to溢出NaN误当公共sat |
| TC2 | 原版Double Sigmoid；原ggml两个generic域资格 | 具名反例、stable-tail oracle；同模型program与generic合同分开 |
| TC3 | private invocation/managed成本/失败/close/alias边界 | 一批资源与普通重放资格，native物理OOM与实际冷OOM分开 |
| TC4 | ordinary Direct/Graph原最终结果、三交付与剩余ledger | 不用Native fallback代签接入；记录必要region边界，不以未资格核计时 |

当前SCOPED，Cast原程序尚未审计完成，产品/三交付/工程门UNPROVEN；完成原witness后固定有限FIX才写产品。
