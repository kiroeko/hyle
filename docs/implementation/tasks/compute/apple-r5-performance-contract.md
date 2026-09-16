# R5 Apple 性能：完整 region 与同生命周期比较

本记录从属 [R5 主卡](apple-v1-four-route-foundation.md) A5/A6。当前为 AUDITED；在计时前固定比较口径，
不是新的产品 owner；源码和合同已接线，尚未开始正式 P1，不能以编译或一次点值声明通过。

## 比较与成本

- 默认为锁定 MLX v0.32.2 的完整编译 region，无 Provider 排序或运行时计时选择。
- Direct 候选：普通 Context + typed lower 组成完整 affine Q4 gate/up → SiLU → F32 down。
- Graph 候选：公开 typed DAG 冷降成相同完整 MLX region，重复执行不再读 Graph。
- 同合同参考：独立 C++ callable 直接表达 MLX quantized_matmul/multiply/sigmoid/matmul，
  经同一 Context 提供真实 admission、固定 native owner、失败传播、Completion 和结果 lease。
  不消费 typed lower、Graph lowering 或候选代码来生成数学链。
- 上述相同生命周期比较检查 frontend/融合接入非劣；成熟实现本身相同，不能记为 HYLE 算法胜利。
- 裸 MLX caller-thread compile/eval/checked synchronize 另作 NATURAL_ROUTE 成本诊断。
  它不提供任意 caller 的 owning async 交付/close/cancel 服务，不与 HYLE 同合同排名。
  原生对照保留同步、同样的 math、输入轮换与原数组输出，不添加人为等待。

两侧共享不可变已 ready 压缩权重；每侧独立 Context/stream、两个运行槽和两个 resident 输入。
每次从 bind 开始，到实际成功完成和本次结果 lease 释放为止计入，不能只测排队。
原生 managed array/metadata、cache hit、内部 temp、线程唤醒和 synchronization 全部在边界内。
常量生成、冷编译/暖机、输入首次驻留不在热态，两侧同阶段执行并单独记录时间与原生内存。
不宣称零分配 replay；MLX 内存计数是 native allocator 统计，不等价于整个进程/Metal driver 峰值。

## 有界 consumer 与数值

沿已登记 vLLM `51918e252af4dfff381001f42b7ebe6a0454501b` 的 Llama-2-7b TP1 FFN
维度 K4096、H11008、O4096；M1 decode 与 M16 short-prefill 为有限 PROVISIONAL_ARCHETYPE。
仅采用形状来源，不声称同 CUDA 的格式或 FP16 数值相同，更不声称模型吞吐或实际调用频率。
Apple 两侧一致采用 unsigned affine Q4/group64、F32 scales/biases/activation/down/output。
权重 q*scale+bias，打包低 nibble 优先；不从未量化源权重取得错误 oracle。
完整 output 先对独立 CPU double oracle；再对两侧结果。两组 resident 输入必须产生不同结果。
正式采样前固定 F32 容差为绝对 `1e-5 + 1e-4*abs(expected)`，不因性能结果更改。
冷数据使用固定整数混合生成，避免短周期矩阵抵消；每组 oracle 至少半数元素必须能拒绝全零。

预检审阅发现，最初周期数据的 M1 输出最大绝对值仅约 0.000323，原 0.003 绝对容差可放过
全零输出。该 fixture 在任何计时前被拒绝，改为上述数据和 F32 数值门；不是产品算法失败，
不消耗产品性能纠正名额，也不把旧预检算作最终 oracle 通过。

## 资格、预算、停止

- 只在 MacBook Air Mac16,12 的 M1-MAC engineering profile 正式采样；Mini 仅 portability。
  记录实际 M4/architecture、OS build、SDK、Metal compiler、MLX pin 与工具链，而非沿用旧环境文字。
- macOS 无公共硬 P/E-core pin；采用既有 interactive QoS、AC、低电量模式关闭和独占采样锁。
  输出 caller 与 cold native owner 的实际 QoS；未观测到的配置不签 PASS。
- 候选/参考分配顺序、执行正反序、两种可验证链接/函数 placement 分层，两个 workload 都保留。
- 公共统一仪器不改；模块有限 A/A validator 在每种实现/顺序/placement 先证明自身比较合格。
  5% A/A factor-median bias、10% gross process instability，失败先查源码/责任边界/成本。
- Direct/Graph 各一次 API/算法固定后的不可变候选 P1；5% NI 与方向分开报告；n15→最多31
  仅用于资格成立而区间未决；最多一次具名源码机制纠正；不调宽门或无限换测量方式。
- 原生诊断每种入口四顺序层×15进程，完整 oracle 先通过；只记录自然成本，不套 NI/算法排名。
- 需要 product fix 时仍归 A1–A6，并回到 AUDITED；没有证据的定时优化不改接口或 metadata。
