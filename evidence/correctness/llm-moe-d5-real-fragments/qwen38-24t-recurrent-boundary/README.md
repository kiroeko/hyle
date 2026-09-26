# Qwen3.8-2.4T：官方投影与连通 recurrent 边界

候选：`59e7fd87e2564bcae893d6e159b57dd96aa80d9f`。产品源码相对 `9e7c57928` 未改变；
新增内容是共享 recipe 与四路线 Direct/Graph 测试消费者。

## 运行结论

| 路线 | Direct | Graph | 范围 |
|---|---|---|---|
| CPU | 2/2 PASS | 2/2 PASS | 官方九类 BF16 投影；官方及两种 GGUF 连通 R05；原 GGUF attention/MTP |
| Apple | 2/2 PASS | 2/2 PASS | 同上，MacBook Air M4，Release WERROR |
| Vulkan | 2/2 PASS | 2/2 PASS | 同上，RTX 4070，Release WERROR |
| CUDA | 2/2 FAIL | 2/2 FAIL | 数值观察通过；稳定 Softplus 命中 `hyle-cuda`，成熟 Provider 断言失败 |

每个 GoogleTest 内按权重角色/profile 批量执行，以上数字不是算子数。使用既有隔离构建树增量编译，
仅运行新共享批次。此前完整 owner 证据见
[语义与 checked ID 批次](../qwen38-24t-semantic-and-checked-id-batch/README.md)；
本目录不宣称在新候选上完整重跑所有 owner，不宣称全量 selector、三交付或 P1 PASS。
选门计划与 skipped reasons 原样保存在 [selected-plan.json](selected-plan.json)。
机器、XML、日志摘要与散列见 [receipt.json](receipt.json)。

## 真实语义边界

官方 BF16 projection 保留九类真实角色与内宽，BF16 输入/权重/输出，最后 Cast F32 观察。
recurrent recipe 连通 conv、SiLU、Q/K/V 拆分、Q/K L2 norm、raw a/b 门控、dt_bias、decay、
GDN 更新和输出 RMSNorm/gate；两次变化输入携带状态，保留 H=8192、K16/V128、D128。
CPU 初跑发现普通 SiLU 命中冻结实现，改用现有成熟 GatedActivation：conv 配显式 ones，
输出 norm/gate 自然组合；ones 的存储/带宽必须计入后续完整成本，不声称普通 SiLU 已成熟接入。
默认 packed-halves 输入方式导致的一次测试异常已显式改为 separate ports，初跑日志仍保留。

官方 Q/K 按每组八个 V head 展开；GGUF 的转换程序已经重排 V、conv、gate 与输出投影，
其运行时应保留 cyclic 广播，不能再重复展开。来源：
[固定 Transformers Qwen3.5 MoE forward](https://github.com/huggingface/transformers/blob/c1b80cf3ee183f94d8b21dfa7e312d1eea4942ad/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)、
[固定 llama.cpp Qwen converter](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/conversion/qwen.py)。
GGUF `ssm_a` 已存 `-exp(A_log)`；官方仍存 A_log，本 fixture 分别表达。

## CUDA 精确缺口与有界候选验证

现有 ggml CUDA adapter 明确拒绝普通 Softplus；原版 `op_softplus` 使用阈值20与
`logf(1+expf(x))`，小负值发生相消。冻结 `hyle-cuda` 能算出稳定结果，但不计成熟覆盖。
没有修改 adapter、上游 kernel 或通用数值要求。

本机 cuDNN 9.24 原生 `CUDNN_POINTWISE_SOFTPLUS_FWD` 探针使用 CUDA 13.3、RTX4070 SM89。
原 `[1,1,128,1]` 布局返回零个可执行配置；仅改 descriptor 为无复制的 `[1,128,1,1]` 后，
MODE_A 的四个可用配置均运行成功、workspace=0，但每个在两次输入、共256个数值观察中失败25个。
`x=-20` 输出0（参考约2.06e-9），`x=-15` 输出约4.32e-7（参考约3.06e-7，绝对差1.26e-7）。
这不是模型状态误差结论，也不是所有 cuDNN engine 的穷举。
探针、编译日志和结果在 [cudnn-probe](cudnn-probe/)；它使用相邻 DLL，无提权或独立窗口。
SDK DLL 副本验证后清除；再次运行脚本会在仓库 `.scratch` 内重新准备。
接口依据：[NVIDIA cuDNN graph API](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html)。

固定 SGLang `06992c92ad44` 的
[Triton GDN gating](https://github.com/sgl-project/sglang/blob/06992c92ad4494fe9f9636e75be3e35040512447/python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py)
与 [CuTe DSL GDN](https://github.com/sgl-project/sglang/blob/06992c92ad4494fe9f9636e75be3e35040512447/python/sglang/kernels/ops/attention/cutedsl_gdn.py)
也采用 `log(1+exp)`；原始文件 SHA-256 在 receipt，源码只作审计，未接入产品。
FlashInfer 固定版 Mamba helper 同类相消；hash TopK 虽有稳定 helper，其完整程序不是 Softplus，
不能抽出 helper 再造 kernel 计作成熟实现。

## 资格纠正与下一决策

旧 Qwen 六画像的48格 R05 向 GDN 输入主机预计算 gate/decay，只证明 conv/GDN primitive。
当前实际连通门控后发现，这些旧结果不足以签完整 R05；ledger 退回 **0 RUN_PASS / 176 未完整资格**，
原非 R05 region 收据作为 partial receipt 保留。没有删除已有能力或重跑所有通过项。

[最小待决卡](../../../../docs/implementation/tasks/compute/llm-moe-d5-recurrent-gate-numeric-decision.md)
建议保留普通稳定 Softplus 合同，单独资格化成熟模型 gate 的完整片段误差；未选择合同前停止扩张。
已知绝对小误差不能直接当作整模型不合格，也不能没有完整状态验证就忽略。
三交付、P1 和聚合验收仍未通过。统一脚本统计见 [loc.json](loc.json)：**58,117实现行 / 228文件**，不含测试。
