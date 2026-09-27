# D5-F47：CPU 官方混合精度的普通入口拒绝

**8 个 CPU 格记为 `BACKEND_REJECTED`，新增 `RUN_PASS` 为零。**
候选 `2ce385555f140865ef7857682c792c774b85d8df`；Windows i9-13900K，oneDNN
`71094df`、ggml `456172ec`。产品代码未改，当前默认 CPU 工厂仍组装 oneDNN、ggml CPU
和残余 native；没有用户手动排序、替代量化格式或恢复自研数学。

| 官方画像 | 完整 MoE 节点 | Direct 首个精确拒绝 | Graph |
|---|---:|---|---|
| GLM-5.3 | 30 | 准备成功 11 个执行节点后，`GroupedFloat8Linear` 返回 Unimplemented | 完整图冷编译拒绝 |
| GLM-5.3-Flash | 48 | 同上 | 完整图冷编译拒绝 |
| DeepSeek V4 Pro | 51 | 准备成功 14 个执行节点后，`GroupedFloat4Linear` 返回 Unimplemented | 完整图冷编译拒绝 |
| DeepSeek V4 Flash | 51 | 同上 | 完整图冷编译拒绝 |

使用原始 `glm_moe` / `moe_chain`，保留真实内宽、top-k、官方编码、动态激活量化和共享专家，
只使用原先已准入的 resident expert 缩容。Graph 实际绑定常量，Direct 逐节点调用普通 prepare。
拒绝发生在执行前，**没有声称这几个片段执行、状态反馈或性能通过**。R11 是完整画像的必要
region，因此整画像当前不能完成；没有把未检查的其它 region 全部写成拒绝或通过。

详见 [结果与二进制身份](summary.json)、[Direct 原始日志](final/compute.log)、
[Graph 原始日志](final/graph.log)、[固定源码事实](sources.json)。`BACKEND_REJECTED` 在这里
限定于当前机器、固定版本和 HYLE 已接入的默认 CPU 路线，**不是所有成熟 CPU 库不可行**。
日志末尾的 `hyle-native` 是默认搜索耗尽后返回的诊断，不表示调用了自研数学。

## 两处测试/构建问题已分开保留

- [首轮](initial/receipt.json)：GLM 两项拒绝明确；DeepSeek 配方写死 CUDA 融合路由门，
  先在该门失败，不能据此签 FP4 缺口。测试改为已经在 CPU 消费者验证的成熟路由组合，
  原 BF16 router 与量化专家合同不变，CUDA 默认仍保持原融合程序。
- [旧对象误用](stale-object-attempt/receipt.json)：Ninja 期待英文 showIncludes，驱动实际用
  中文，两个测试对象依赖表为空。修改头后没有重编译；该次不是纠正候选的有效验证。
  固定 `VSLANG=1033`，一次失效 179 个 HYLE/测试对象，保留第三方构建缓存。
  最终两份 [Direct](final/compute-header-dependencies.log) /
  [Graph](final/graph-header-dependencies.log) 依赖表均包含实际修改的头。
- [最终批次](final/receipt.json)：两组测试各覆盖四画像，零失败/跳过；这是拒绝边界检查
  PASS，不是数值支持 PASS。13 项输入工具检查和 ledger 结构检查通过。

selector 的 [选中与跳过原因](final/selection-plan.json)保留。使用本卡声明的两个完整配方
边界测试完成 CPU 直接消费者定向检查；不声称重跑了两路线全套测试。共享测试 helper 的
默认参数仍为 true，原 CUDA 配方的门、token 类型和其余节点不变，未触发其它路线重验。
本批无产品/API/依赖变化，不触发三交付或 P1；它们仍属于 D5 的未完成验收。

## 有限裁决

当前 oneDNN reference 的 F16 中间舍入、SGLang 的 AMX/W8A16 路线，以及 vLLM 的具名
W8A16/截断编码程序，都不能直接代替所需合同。这只是排除了这些具体接法。
后续成熟 CPU 承载至少需要直接 RNE E4M3、动态 group128、K-block scale 累加、DeepSeek
A8W4 与 FP4 group32 floor；见 [F47 后续边界](../../../../docs/implementation/tasks/compute/llm-moe-d5-cpu-official-boundary.md)。
本批停止重复试拼和原生调查，不能把新 Provider/runtime 接入暗中扩入 D5。
其它路线及可用 CPU 画像的交付、性能继续；本结果不宣称整个 Compute+Graph 完成。

当前 ledger：**152 RUN_PASS / 8 BACKEND_REJECTED / 16 VALIDATION_NOT_RUN**。
统一实现统计 **60,373 行 / 241 文件**，不含测试和第三方，见 [LOC](loc.json)。
