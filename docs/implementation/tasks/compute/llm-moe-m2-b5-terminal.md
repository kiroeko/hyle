# M2-B5 成熟 Provider 稀疏注意力与压缩状态终局

<!-- MODULE REVIEW: CLOSED -->

施工边界见[B5 closure card](llm-moe-m2-b5-sparse-compression.md)。

2026-09-24，B5 在产品候选
`d9d6a6ef313e60ac2912b35e950a51bf1a4ea138` 上关闭。公共层新增自然布局的
`LightningIndexer` typed descriptor；CPU、CUDA、Apple Metal 和 Vulkan 都 lower 到锁定 ggml 的
`GGML_OP_LIGHTNING_INDEXER` 原版节点。MLA 继续由已有 MatMul/Attention/View 组成，跨层状态复用
Graph 显式端口，pool/tail 使用已有 Pool/TopK/Gather，压缩状态使用已有 MatMul/Conv/Attention 与
状态端口。本批没有新增 HYLE 数学 kernel、CUDA `.cu`、Metal shader、SPIR-V、host sparse loop 或
自研 compressor。

## 实际资格与验证

| 路线 | 已绑定域 | 实际验证 |
|---|---|---|
| CPU | F32 query/key/prescaled weights、F16 broadcast mask、F32 output，`D=128,H=64` | Windows shared：公共 schema、ggml Direct 与 CPU Graph 独立 oracle、两次 replay 全过 |
| CUDA | 同一公共域，native device buffer/stream；原版 backend capability 最终裁决 | Windows RTX 4070：Direct/Graph 全过，无 CPU fallback、无新 `.cu` |
| Apple | 同一 schema 经既有 whole-region seam 到原版 ggml Metal，并补齐 F16 mask carrier | MacBook Air shared 与 Mac mini static 的 provider/Direct/Graph 全过；实机编译并执行 `kernel_lightning_indexer_f32` |
| Vulkan | 同一 schema 经 ggml Vulkan managed region；FP16 storage 与对齐由原版 backend 裁决 | Windows RTX 4070：provider/Direct/Graph 全过，device 下载结果与公共 oracle 一致 |

共同 fixture 覆盖多个 token/key、正负 dot、ReLU、非零 F16 mask 与 stream mask broadcast；公共 schema
还拒绝错误 dtype、rank、shape、D/H 和不可广播 mask。准备结果保留各路线原生图、buffer 与完成对象，
重复提交不重新搜索 Provider、不计时。Apple 首轮验证发现 F16 mask 未进入路线 schema matcher，修复
`InputSpec` 与 context dtype 映射后，shared/static 两种交付均通过。P1 对本批
`NOT_APPLICABLE`：没有冻结完整 sparse/compression 模型片段；P0 已确认无逐 token host dispatch、
无热路径 Provider 裁决或额外同步。

## 有限残差

- 完整 checkpoint 的 absorbed/gated MLA、跨层 sparse selection schedule、ragged pool tail 和
  CSA/HCA 增量 compressor 尚未作为真实模型片段运行，四路线均保持 `UNPROVEN`，归 M3-05。
- 当前公共 Lightning Indexer 域固定 `D=128,H=64`；CUDA 的 `H=32` 扩展、Metal 的 simdgroup 要求
  和 Vulkan 的 FP16-storage 限制继续由各原版 backend 资格决定，不提升成四路线共同合同。
- 模型 loader、服务端稀疏调度、跨设备通信和推理 engine 不属于 Compute/Graph。

这些残差不要求 M2 自研算法；M3 用真实消费图验证组合完整性，只有具名缺口才允许进入 M4。持续
承载 M2 的组件 owner 保持 `AUDITED`；B6 必须另行 SCOPED→AUDITED 后才可修改产品代码。

统一 `loc-report.py --self-check` 在不可变候选上报告 Compute+Graph 实现 **59,749 行 / 252 文件**，
不含测试、文档、样例与第三方源码。
