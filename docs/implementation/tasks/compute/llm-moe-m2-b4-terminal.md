# M2-B4 成熟 Provider MoE 与模型片段终局

<!-- MODULE REVIEW: CLOSED -->

施工边界见[B4 closure card](llm-moe-m2-b4-moe-model-fragments.md)。

2026-09-24，B4 在产品候选
`d40aa5329ac9e6783556b1595118b28742cde7c7` 上关闭。公共层新增自然布局的 `GroupedMatMul` 与
DeepSeek V4 `HyperConnectionCombine/Pre/Post` typed descriptor；CPU、CUDA、Apple Metal 和
Vulkan 都 lower 到锁定 ggml 的 `mul_mat_id`、`dsv4_hc_comb/pre/post` 原版节点。B1/B2 已有的
TopK、Gather/Scatter、普通 MatMul、activation 和 residual 继续组成 routing、dispatch 与 combine。
本批没有新增 HYLE 数学 kernel、CUDA `.cu`、Metal shader、SPIR-V 或 host expert loop。

## 实际资格与验证

| 路线 | 已绑定域 | 实际验证 |
|---|---|---|
| CPU | dense F32 grouped weights/activations、I32 expert ids；dense F32、`hc=4` 的 combine/pre/post | Windows shared：四枚 op 的 Direct 与 Graph 表驱动独立 oracle 全过，Graph 固定到 ggml CPU retained recipe |
| CUDA | 同一自然 schema，native device buffer/stream，原版 backend 最终裁决 | Windows RTX 4070：四枚 op 的 Direct 与 Graph 全过；I32 ids 只在 grouped-matmul 第三端口准入，无 CPU fallback |
| Apple | 同一 schema 经既有 whole-region seam 到原版 ggml Metal；普通组合仍由 MLX 承载 | MacBook Air shared 与 Mac mini static 的 provider/Direct/Graph 全过；实机编译并执行四枚 Metal pipeline |
| Vulkan | 同一 schema 经 ggml Vulkan managed region；原版 `supports_op` 最终裁决 | Windows RTX 4070：provider/Direct/Graph 全过，结果由 device 下载后与公共 oracle 比较 |

同一 fixture 覆盖重复/偏斜 expert ids、mHC combine 的稳定归一、pre 与 post 组合；公共 schema 还拒绝
错误 dtype、rank、shape 与 `hc`。准备结果保留各路线原生图、buffer 和完成对象，热执行不搜索或
计时。Apple seam 泛化时发现 `noexcept` 资格检查使用动态容器会在静态故障注入下终止；已改为固定
两输出栈存储，原故障门在 Air shared 与 Mini static 均通过。P1 对本批 `NOT_APPLICABLE`：没有冻结
的完整 MoE/MTP 模型片段候选，P0 已确认无逐 token host dispatch 和无热路径 Provider 裁决。

## 有限残差

- 当前 grouped weight 第一批只签 dense F32；精确量化 grouped weight 必须由 B6 的 storage-format
  资格逐项决定，不能用相似位宽代签。
- 完整 checkpoint 的 routing→dispatch→expert→combine、hash routing、shared/latent/SiTU、跨 block
  residual 与完整 MTP carry 尚未作为模型片段运行，保持 `UNPROVEN`，归 M3-04。
- 服务端专家调度、跨设备通信、采样与 speculative engine 属于推理引擎，不进入 Compute/Graph。

这些残差不要求 M2 自研算法；M3 先用真实消费图汇总成熟 Provider 的覆盖，只有具名缺口才能进入
M4。持续承载 M2 的组件 owner 保持 `AUDITED`；B5 必须另行 SCOPED→AUDITED 后才能修改产品代码。

统一 `loc-report.py --self-check` 在不可变候选上报告 Compute+Graph 实现 **59,648 行 / 252 文件**，
不含测试、文档、样例与第三方源码。
