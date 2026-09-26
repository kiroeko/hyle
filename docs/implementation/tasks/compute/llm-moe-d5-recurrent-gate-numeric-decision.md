# D5：recurrent gate 数值合同待决

<!-- MODULE REVIEW: BLOCKED -->

**已满足恢复条件：** 用户明确选择“完整片段误差资格化”。后续当前施工见
[模型完整门控片段卡](llm-moe-d5-model-gate-qualification.md)；本页保留原阻塞事实，不再是当前 owner record。

## 当前结论

- owner：`compute_ops`；baseline / 测试候选：`59e7fd87e2564bcae893d6e159b57dd96aa80d9f`。
- 本轮只补共享测试消费者、原生库探针与证据，没有修改产品代码或新增 HYLE 数学。
- Qwen3.8-2.4T 官方九类 BF16 投影和官方/两种 GGUF 的连通 recurrent 边界，已在
  CPU、Apple、Vulkan Direct/Graph 通过。CUDA 数值检查通过，但 Softplus 命中冻结的
  `hyle-cuda`；因此成熟 Provider 断言失败，不能签成熟覆盖。
- [批次收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-recurrent-boundary/README.md)
  保存初跑失败、最终六入口通过、CUDA 两入口失败和 cuDNN 探针。
- 实现规模仍为 **58,117 行 / 228 文件**，不含测试；本轮 `include/`、`src/` 相对
  `9e7c579282030db876dac98ca20b6e20e189ce70` 没有变化。

## 具体缺口及已完成的有界调查

| 成熟候选 | 已核对事实 | 本次裁决 |
|---|---|---|
| 当前 ggml CUDA | `op_softplus` 使用阈值 20 和 `logf(1 + expf(x))`；小负数尾值会发生相消；HYLE adapter 有意不接纳普通 Softplus | 不改变通用算子的稳定精度合同，不直接放开准入 |
| cuDNN 9.24 / CUDA 13.3 / SM89 | 原布局 `[1,1,128,1]` 无可用 engine；无复制地解释成 `[1,128,1,1]` 后，MODE_A 返回四个可用配置，均可执行、workspace=0；每个配置在 256 个数值观察中失败 25 个 | 实测不满足本次普通稳定 Softplus 探针；不新增 cuDNN Provider |
| 固定 SGLang `06992c92ad44` | `fused_gdn_gating.py` 和 `cutedsl_gdn.py` 的原版 gate 也使用 `log(1+exp)`；前者 Triton，后者 CuTe DSL | 是成熟模型计算采用此近似的证据，不是满足现有通用稳定精度或已接入 C++ 的证据 |
| 固定 FlashInfer `69ff11fc4954` | Mamba softplus 为 `__logf(1+__expf(x))`；hash TopK 内含稳定 softplus 辅助函数，但完整程序是另一个操作 | 不抽出 helper 再写 HYLE kernel 冒充完整原版 Provider |

cuDNN 观察示例：`x=-20` 时输出 0，F32 独立参考为 `2.06115369e-9`；`x=-15` 时输出
`4.32019988e-7`，参考为 `3.05902262e-7`，相对差约 41%，**绝对差约 `1.26e-7`**。
这证明该候选不能满足逐元素小尾数相对精度；**不证明模型状态或推理结果会产生同等比例误差**。
尚未执行“原版近似 gate 对完整 recurrent 状态”的资格验证，不能宣称它满足或不满足模型级误差。

## 旧 48 格的资格纠正

本次实际连通 raw a/b、dt_bias、decay、conv/SiLU、Q/K 归一化、GDN state 和输出 gate 后，
发现旧 Qwen3.5-2B、Qwen3.8-27B 六画像的 R05 只分别验证 conv / GDN primitive，并向 GDN
输入主机预计算的 log-decay / beta。具体入口包括
`tests/compute_route_cuda/d5_qwen35_execution_test.cc` 的
`D5Qwen35OfficialCausalConvAndGdnUsePreparedCalls`，以及共用
`tests/compute_models/d5_qwen38_gdn_fixture.h` 和各路线的对应消费者。
因此旧 R05 收据不能证明运行时门控边界，更不能证明其成熟 Provider 可达。

将 48 个整格退回 `VALIDATION_NOT_RUN`，保留各自非 R05 的既有 partial receipt 和原整格历史。
旧 R05 primitive 运行事实保留在原始 evidence，但不继续列为完整 R05 `RUN_PASS`。
当前为 **0/176 完整格、176/176 尚无完整资格**；这不表示已经通过的投影、词表、MoE、attention、
conv 和 GDN primitive 被删除或需要无差别重跑。补足门控后按相同身份复用原证据。

## 待决方案

推荐**保留普通 Softplus 的稳定精度，单独资格化原版成熟模型 gate 的完整片段误差**：

1. 先用已固定 ggml 原版 gate/region 做一个有限数值资格批次，保留真实 head/state 宽度、
   官方与 GGUF 的 head 排列、变化输入、负数尾部和连续状态；对照独立稳定参考。
2. 冻结完整片段的绝对/相对误差和状态观察边界后再运行，不能看到结果后放宽阈值。
   不把靠近零的 gate 相对误差直接当成整个模型的误差，也不遗漏必要的状态/完成检查。
3. 通过后，只把这个已限定的模型 gate/region 放入成熟路由；公共 `Unary::kSoftplus` 的精确域
   保持原要求。具体公开语义与 provider owner 在该有限审计里固定，不能偷偷给同一个通用请求
   降低精度。没有合格的模型专用表达就拒绝，不重构整个路由系统。
4. 不通过则保留具名成熟缺口，停止该域，不新增/优化 HYLE 数学，不继续无预算找库。
5. 用现有共享 recipe 同批补 Qwen 2B / 27B / 2.4T 九个画像的 R05；接线批末统一验证，
   复用本轮六入口与此前其它 region 的证据；三交付与 P1 仍在稳定后执行。

替代方案：继续要求模型 gate 本身满足普通稳定 Softplus 的逐元素尾数精度，将 CUDA 该域明确记为
成熟能力缺口，保留冻结实现但不计成熟覆盖，继续其它可验收域。本次调查不声称所有可能库均缺失。

## 停止与恢复条件

当前 D5 主卡明确规定：“若精确缺口需要新数学或 Provider，记录最小后续卡并停止，不在 D5 扩张。”
本问题继续接线需要先选择数值合同，或接入本轮之外的新成熟 carrier；原授权没有明确选择这两种
不同的产品保证。因此 `compute_ops` 从 `AUDITED` 进入 `BLOCKED`，不扩张原 FIX。

用户选择上面的模型专用数值资格方案，或明确接受精确成熟缺口后，按 `BLOCKED -> SCOPED -> AUDITED`
绑定该有限后续工作再恢复。其它 owner 的已完成 FIX 和 CLOSED 记录不重开。D5-A5 三交付、
D5-A6 P1 仍未运行，聚合验收为 `INCONCLUSIVE`。本记录不宣称 Compute+Graph 或当前 goal 完成。
