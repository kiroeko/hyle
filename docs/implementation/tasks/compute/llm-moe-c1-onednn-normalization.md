# C1-AA：oneDNN 归一化及原生 post-op 接入

## Closure card

- owner：`/root`；`compute_provider_onednn`、`graph_provider_onednn`。
- baseline：`4a7685523d139a54aab9043bdf3d5b767a2bed20`，工作树干净。
- trigger：用户要求先实际绑定成熟 Provider，再比较、补缺和删除冗余实现。C1 已准入的
  F03/F04 及本地 Qwen3.5-2B 的归一化/缩放/残差片段仍主要走 Native；锁定 oneDNN 已有
  RMSNorm、LayerNorm、mixed parameter dtype 和 fused binary post-ops，现有构建却仅启用
  MATMUL。该新增真实 consumer 域满足前轮新增能力/公开合同 reopen trigger；不是旧卡续跑。
- 主 consumer：普通 CPU Direct/Graph 的 RMSNorm/LayerNorm，及 norm→Mul/Add 片段。
  正交 consumer：多维批量归一化、无参数/带参数、两个 invocation、重绑及原地输出。
- scope：复用 oneDNN 3.13.2 的 normalization primitive 与原生 post-ops；冷绑定、调用方
  workspace、借用 executor、失败回报及普通路线；对应构建/三交付、数值/生命周期与公平性能。
- non-goals：不自写归一化算法，不接全部 oneDNN 算子，不改 SIMD、Sched、公共治理/性能仪器，
  不改既有 BRGEMM/gated-MLP 算法，不自动解除 ggml-CPU BLOCKED，不把局部接入记为 C1 完成。
- 依赖：现有 Compute/Graph/Mem、锁定 oneDNN；真实直接消费者为 CPU/Graph CPU 默认工厂。
  私有 CMake dependency options 属本接入范围；不重写构建系统。
- 性能：P1，完整同合同 Direct/Graph 对比现有 Native；oneDNN 内部分配、转换、同步、Graph
  边界和融合损益全部计费。未完成数值/资源资格前不定赢家；不靠省略检查赢。
- 预算：两条具名机制（原版 Graph partition、primitive/post-ops），源码选择最多60分钟；
  产品按有限 AA01–AA04 实施，最多两轮具名性能纠正；一个稳定不可变候选执行适用 P1。
- 预声明验证：selector 所选 gates；W1 static/shared、WSL GCC14，x64 oneDNN 实际运行；
  独立 FP64 oracle、F32/F16/BF16、2048及不规则宽度、decode/prefill、特殊值、大均值小方差、
  精度/布局拒绝、OOM、两个 invocation/并发/重入/资源失败、Graph 可观测中间输出/分叉边界。
  CMake seam 变化执行适用 source/static/shared installed consumer 与依赖隔离；不代签 ARM。
  DLL 邻置、PATH unset，parallel2、重型验证串行，复用固定构建目录。
- 停止：本轮适用验收齐全才 CLOSED；未运行项如实保留，实质 blocker 具名裁决。C1 其他操作、
  平台/模型验证继续归主卡；不以本轮关闭替代全部范围。

## 有限待审项

| ID | 缺口 | 允许边界 |
|---|---|---|
| AA01 | 构建只启用 MATMUL/SEQ，成熟 norm 未进入产品 | 固定 pin 不变，增加 LAYERNORM、借用 HYLE executor 的 THREADPOOL runtime |
| AA02 | Direct 未绑定上游 norm | 共享私有 immutable primitive plan、每 invocation stream/memory handles、caller scratchpad；F32 accumulation，dense F32/F16/BF16，明确不支持域 |
| AA03 | Graph 未保留原生 norm post-op 融合 | 原有 region 扩展识别 norm→Add/Mul，不吞掉 published/多消费者中间结果，不降低半精度中间舍入要求 |
| AA04 | 没有该实际路线的资格与默认规则 | 正确性/所有权/交付先通过；固定同合同 full-cost 规则及文档，再裁决 Native 重复域是否可移除 |

本卡当前为 SCOPED。审计及 API/成本冻结提交后才修改产品。
