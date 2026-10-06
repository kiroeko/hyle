# CPU RoPE：中间精度与同合同审计

<!-- MODULE REVIEW: CANDIDATE -->

- owner `compute_provider_ggml`；baseline `e80cf22ccf285934693f40da9cc558b457283d7b`。
- RN诊断曾把保留完整整数的Double phase oracle用于当前Float32模型程序，需核对真实consumer；
  重开仅授权本次有限审计，不恢复旧施工队列或引入新的数学/接口。
- 原始差异保留在[诊断](../../../../evidence/correctness/cpu-residual-scatter-rope/README.md)。它证明
  F32/Double中间相位会产生不同结果，没有证明当前模型的原Float32程序错误。
- [官方Qwen3.5 text rotary](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
  明确将position/frequency用于Float32相位计算；本地原模型config保留BF16 data、F32相关中间值。
  不用外部Double oracle替换模型的明确中间精度；完整模型其它舍入阶段仍独立资格，不由此签bitwise。
- 当前ggml explicit_rope uses original cast/mul/cos/sin/concat节点并产生F32 phase；Native I32为同级
  对照，ATen Double phase是另一计算程序。后续相位程序不同的比较NOT_COMPARABLE，不签SOTA。
- 审计输出KEEP/BORROW，不需要产品FIX。ggml source/headers/runtime、pin、build及普通组装器
  与先前资格身份相同；RN尝试改动已逐字撤回，仅保存未提交diff作工作记录，没有测试或P1据此签署。
- P0/no-product-change：不重建、不采样，复用现有TC数学/资源与两平台三交付；验收仅source identity、
  selector、closure/docs与语义边界。无API/layout/资源/执行/安装行为变化，不制造新的binary journey。
- scope、架构、成本、依赖/许可证、C++交付、易用文档和红队均只裁决此精度比较命题；不声明CPU
  剩余自研已裁决，也不重新认证所有模型。

无FIX候选沿baseline `e80cf22ccf285934693f40da9cc558b457283d7b`，19个ggml/runtime/公开factory输入
逐字SHA256一致，见[源码身份](../../../../evidence/correctness/cpu-residual-scatter-rope/ggml-no-change.json)。
本轮只裁决同合同边界，产品能力、安装、执行和资源没有变化；复用已有TC资格，不触发新构建/P1。
