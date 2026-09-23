# M3 四路线真实模型计算链候选结果

<!-- MODULE REVIEW: CANDIDATE -->

## 裁决

M3 于 2026-09-24 形成不可变候选。产品候选仍为
`68278c999b623edbd0c87f05d3926208ea84f6db`；本轮只增加测试消费者与收据，没有修改公共 API、
Provider 数学、kernel、shader、decoder、quantizer、路由规则或冻结自实现。

第一次审计留下的 134 个 `VALIDATION_NOT_RUN` 已全部用当前源码重新构建并执行，再绑定到具名
Direct/Graph 测试或实际准备期拒绝。最终 64 个需求 × 四路线共 256 格为：

| 结果 | 格数 | 含义 |
|---|---:|---|
| `PASS` | 153 | 具名普通 Direct 与 Graph 收据覆盖该 HYLE 片段、重复执行、原生存储和资源边界 |
| `BACKEND_REJECTED` | 21 | 锁定 Provider 或当前设备明确拒绝具名公共域 |
| `NUMERIC_INCOMPATIBLE` | 78 | dtype、state、codes/scales、activation 或输出合同不相容 |
| `UPSTREAM_MISSING` | 4 | 没有准入 Provider 暴露精确量化 attention/index state region |
| `VALIDATION_NOT_RUN` | 0 | M3 没有留下未运行格 |

[执行映射](../../../../evidence/correctness/compute-llm-moe-m3/execution-map.json)逐格保存测试名和
GoogleTest JSON receipt；[聚合收据](../../../../evidence/correctness/compute-llm-moe-m3/acceptance.json)
核对收据哈希、测试数量、最终状态和本地模型边界。`PASS` 只表示声明的 HYLE Direct/Graph 片段；
模型解析、tokenizer、采样和端到端推理 engine 仍不属于 HYLE SDK，也没有被本卡冒充通过。

## 六组结果

| 批次 | 终局 |
|---|---|
| M3-01 F01--F04/F19 | 69 PASS、15 backend rejected |
| M3-02 F05--F08 | 36 PASS、16 numeric incompatible |
| M3-03 F09--F11 | 24 numeric incompatible |
| M3-04 F12--F14 | 20 PASS、24 numeric incompatible |
| M3-05 F15--F17 | 28 PASS、4 numeric incompatible |
| M3-06 F18 | 6 backend rejected、10 numeric incompatible、4 upstream missing |

M3-08 Provider 完成门仍为 `PASS`：M0 准入的成熟 Provider 在其精确适用域内未接通数量为 0。
这项结论不把精确数值域缺口改写成 Provider 接线完成，也不授权自行实现新算子。

## 本地 Qwen3.5-2B 边界

- 13 个文件、4,571,274,023 bytes 的 size/SHA-256 和 source ID
  `4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155` 保持一致。
- B16 prompt 1/17/128 + 8 步 decode reference 已重新生成并核对；它是外部参考输出，不是 HYLE
  产品通过。
- CPU、CUDA、Vulkan 在真实 `[1,T,16,128]`、`T=1/17/128` 的 GatedDeltaNet 描述上接受 F32；
  Apple 已有 16 heads × depth 128 的 F32 原生执行收据。四路线都对同一真实 shape 的 BF16 首个
  recurrent 边界给出实际拒绝，所以 B16 完整模型旅程仍为 0/4。
- CUDA Marlin 的 Q4-S128、Apple MLX 的 Q4-A64 和 Vulkan IREE 的 dense FP8 是各自精确域，
  不能互相代签四路线整模。Vulkan FP8 Direct/Graph 已执行，但完整 profile 仍在 BF16 recurrent
  边界停止。Q8-B32 没有精确载体。
- grouped expert、mHC 和 Lightning Indexer 的准入 mature fragment 仍是窄 F32 域；公共 schema
  对 BF16 的实际拒绝已进入共同收据，因此需要 BF16/F16 checkpoint storage 的行保留
  `NUMERIC_INCOMPATIBLE`。

## 执行与成本

- 当前源码重建后实际运行：Windows CPU 57、CUDA 64、Vulkan 190 个路线测试；MacBook Air
  Apple 64 个路线测试；共同 operation schema 47 个测试。所有映射用例通过。
- 测试覆盖 Direct/Graph、变化输入、重复调用、旧结果、失败不发布、状态隔离、完成/关闭和原生
  资源；精确测试名由执行映射绑定，未用测试文件数量替代结果。
- 产品候选未变，P0 确认无 API、算法或路由变化；P1 不适用。没有进行冻结自实现竞速。
- Compute+Graph 产品实现 LOC 仍为 **59,749 行 / 252 文件**；测试、文档、vendor、构建产物和
  scratch 不计入。

## 停止

M3 候选已满足执行退出条件，M4 尚未开始。完成终局检查后才可把 registry 转为 `CLOSED`；下一阶段
只能处理本卡留下的具名精确缺口，并继续先查成熟 Provider、已有合法组合或上游版本。
