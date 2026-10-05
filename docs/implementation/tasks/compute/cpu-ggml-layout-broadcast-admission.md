# CPU ggml：布局、广播与宽标量批量接线

<!-- MODULE REVIEW: SCOPED -->

- owner `compute_provider_ggml`，baseline `786a3ec36e4a9b19b6eed572f1e9c96dc93396f5`，Windows canonical。
- trigger：全族审计的 MatMul row-major batch/左小右大、Softmax 非末轴、SDPA BHSD/value depth、
  Concat 宽 scalar 是现存公开 CPU 旅程；当前 adapter 拒绝，原 API/源码已提供相应布局与 byte-copy。
- 主纵切：多 batch/head 广播线性层、两种 attention layout、不同 K/V depth；正交纵切：普通 tensor
  非末轴概率归一化和 mixed-width storage 拼接，准备一次、变化输入、Direct 与完整 Graph。
- scope：下表四组成熟能力批量准入及一批资格；无新数学、API/默认排序/ISA/ABI/pin/构建仪器，
  Native、SIMD/GPU 数学保持冻结。rank>4 MatMul、Bool/log-softmax、mask attention、I64 索引、
  duplicate Scatter、alias/bounded/不兼容布局不借此签完整支持或唯一。
- 资源：既有 arena/pool/slot/checked mappings；原 transpose/cont/repeat 的 payload 冷分配、热复用，
  包含在实际成本。外部 alias 拒绝；索引 math 不能因 widening 存储类型而误准入。
- P0：layout 变换 O(n)、广播必要展开与 RHS repack 的完整字节，attention 原完整程序；
  Softmax 复用上一包安全特殊行，不拆成 HYLE 数学。此批不启动 Native P1，不签 WIN。
- 预算：一次当前原 API/实现审计，LB1–5 有限 FIX，一批 Windows/Linux 受影响测试和相同消费者的
  source/static/shared；最多一轮具体失败纠正。不逐算子重复 SDK、不重复模型/loader/GPU/P1。
- 验证：provider/普通 CPU/Graph；F32 row-major/transposed rank2..4、shared/双向leading广播、变化输入、
  K0非空输出与独立double oracle；三dtype Softmax axis0/1/−2、mask广播/特殊行/恢复；
  BSHD/BHSD、三dtype、causal/noncausal、GQA与不同Dk/Dv；Copy/Concat 全16种plain scalar bits，
  大整数、NaN payload/负零、空部件、strided inputs与变化输入；拒绝未接宽 Gather/index/numeric Cast。
  生命周期/OOM/close/alias复跑现有受影响provider；selector/closure/docs/diff、统一LOC。
- stop：原 kernel 对具体 dtype/layout 不能完整承载则记录具名限制与证据，不增加自研算法或签UNIQUE。

## 待冻结有限 FIX

| ID | 允许改动 | 必需验收 |
|---|---|---|
| LB1 | F32 plain RHS dense row-major 的既有原transpose/cont准入；lhs leading broadcast用原repeat；K0用原零identity/repeat | 双向NumPy广播、实际值、成本、K0下游 |
| LB2 | Softmax任意axis用原reshape/permute/cont移动到row轴、复用安全Softmax后逆变换 | 三dtype、mask/特殊行传播、下游/重放、原末轴回归 |
| LB3 | SDPA BSHD/BHSD布局转换与Dk/Dv独立；仅按因果建mask | 三dtype/因果/GQA/独立oracle/两种布局 |
| LB4 | plain scalar native representation供原Copy/Concat使用；同时收紧未提供宽Gather/Cast数学谓词 | 全storage bits、strides、原被接受域不回归、未接算术明确拒绝 |
| LB5 | 一批普通Direct/Graph、三交付、文档/实际成本与剩余ledger | 普通入口自动固定成熟程序；未运行不PASS |

当前只SCOPED；固定原接口、数值、资源和边界后进入AUDITED再写产品。
