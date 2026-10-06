# CPU 剩余数值语义：重复 ScatterAdd 与整数 RoPE

<!-- MODULE REVIEW: AUDITED -->

- owners `compute_provider_native`、`compute_provider_ggml`，baseline
  `e80cf22ccf285934693f40da9cc558b457283d7b`，Windows canonical。ggml以本卡重开SCOPED；Native保持
  父卡AUDITED的旧FIX，RN定界/审计完成后再重绑定本卡，不制造非法AUDITED→SCOPED跳转。
- trigger：ND-R1 已验收。剩余域进入 ND3 之前出现
  [真实数值反例](../../../../evidence/correctness/cpu-residual-scatter-rope/README.md)：Native F16/BF16
  ScatterAdd 逐更新 storage rounding；Native 与 ggml 通用 I32 RoPE 在三角函数前丢整数位置。
  这是 CLOSED ggml 的合法重开事实，不是因为再审一遍或 timing 输赢。
- 消费与范围：已有三 floating 公共重复索引聚合、I32/I64 positions 与显式 F32 frequency 的 RoPE；
  两 layout、prefix/尾部原 bits、axis/byte stride、empty、alias、越界无写/恢复、变化输入、真实 wait。
  当前 CPU math；SIMD、原库、GPU、模型加载/整模型和其它算子不改，不新增自研数学。
- 同合同：generic RoPE 沿 TM2 的原 Double phase 保留整数位置，不因 dtype 或选择 Provider
  偷换成先截断位置的 F32 phase；原模型专用数值程序独立，不能拿两种中间精度签性能胜负。
- 方案：成熟 ATen ScatterAdd 已接；ATen RoPE 原程序可对低精度 data 原生提升 Double、clone 与
  prefix copy，但 guard 尚只允许 F32，需先按原库程序接入/资格，再撤不合格的普通准入。
  不用 strict/prepared 政策维持错误数学；没有满足该政策的程序明确拒绝，不放宽预算。
- 资源/架构：沿原 Program、native 输入 bytes packing、mappings/checked outputs、managed replay、
  私有内存 unknown 和 Completion/OOM/close。不改公共 API、组装器、Pin、线程政策或通用性能核心。
- 有界：一批诊断已经完成；最多一次有限 FIX，批量写完后一次 owning factors/普通 Direct+Graph/
  header/Windows+Linux 三交付与源码身份复用；仅修具体失败，j2、重型串行、DLL 邻 exe/隐藏启动。
  一轮数学资格/冷拒绝 fixture；最多一次具名 fixture 纠正。不为每种 dtype 单独新建验证工程。
- P0：包括原 Double conversions、Tensor 分配、packing、prefix/tail clone、索引完整检查与真终态。
  不计时、不优化，错误域退出 ND3；后续合格普通域仍按父卡一份有限 P1，不恢复旧施工队列。
- scope/failure/fix 必须先固定为 AUDITED 才改产品。适用门未跑不签 PASS；达到预算依法裁决。
  关闭此子批后继续 [父卡 ND3](cpu-native-final-domain-verdict.md)，不签整个 CPU 或四路线完成。

## 待固定的有限输出

1. 原 ATen F16/BF16 RoPE 接入及 I32/I64、两 layout/axis/strides、尾部 raw bits/并发/close/OOM资格。
2. Native half ScatterAdd、Native 通用 RoPE，以及 ggml 通用三 floating/I32 RoPE 的准入与死分支去留。
3. 旧错误数学预期迁移；专家冷拒绝与普通成熟 Direct/Graph 的两平台三交付。

上方为SCOPED定界过程。源码与实际反例现已完成审计，两owner重绑定本卡AUDITED，仅授权下列FIX。
原库、SIMD 本体和 GPU 专用/原模型程序保留；不能误删有效借用。

## 固定审计/FIX

| ID | 有限产品改动 | 必需验收 |
|---|---|---|
| RN1 | Native F16/BF16 ScatterAdd明确冷拒绝；保留F32 Add及各storage Assign，原专用half Add不再执行 | propose/prepare拒绝；成熟重复/越界无写/alias/empty；旧ordered-rounding预期迁移 |
| RN2 | Native通用RoPE明确拒绝，删除本factor专用位置读取/选择/执行/准入和无其它用途的private plan fields | 两index widths专家冷拒绝；其它Native Transformer执行不变 |
| RN3 | ggml CPU通用三floating/I32 RoPE准入与对应lowering退役；其它原程序/模型扩展不改 | expert拒绝，不冒充ggml kernel失败；其它ggml族实跑与region消费 |
| RN4 | 已接ATen原Double phase RoPE开放F16/BF16 data，frequency仍F32；原Tensor数学/clone-tail与资源模型不改，更新implementation版本 | 三floating、I32/I64、两layout、axis/byte-stride、sNaN tail bits、empty、alias、并发/close/cold OOM；普通Direct/Graph两平台三交付 |

不为不同数值语义扩张性能对比；本轮不签WIN。RoPE旧F32-phase计算预期不能代签整数端口资格。
RN2删除仅限Native拥有的死代码；仍在其它owner下的旧CPU专用kernel/helper先登记末尾定向清理，
不借本卡改Compute/runtime CLOSED源码或SIMD基础，不用“暂存”保留不合格准入。

七问：原ATen/ggml版本、许可证/SDK沿已锁闭包；三dtype能力先成熟接入再撤错误路径；复用已关闭
runtime/packing/Completion；P0完整Tensor转换/分配/同步计成本，无新的数学核；无C/FFI或公共API；
三交付使用同一ordinary consumer及邻置DLL；guide明确整数位置/managed/专家拒绝。
最终七门、红队及适用结果仍须实际验证；当前AUDITED不是完成声明。
