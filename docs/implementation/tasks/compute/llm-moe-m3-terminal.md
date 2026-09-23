# M3 四路线真实模型计算链验收终局

<!-- MODULE REVIEW: CLOSED -->

## 裁决

2026-09-24，M3 在产品候选
`68278c999b623edbd0c87f05d3926208ea84f6db` 上以
`M3_COMPLETE_WITH_RECORDED_COVERAGE_GAPS` 结束。M3 没有修改产品 API、Provider、kernel、shader、
decoder、quantizer 或冻结自实现数学；唯一新增代码是一份测试侧聚合验收器。

[验收收据](../../../../evidence/correctness/compute-llm-moe-m3/acceptance.json)与
[覆盖矩阵](llm-moe-coverage-matrix.json)已经逐一裁决 64 个需求 × 四路线的 256 格：

| 结果 | 格数 | 含义 |
|---|---:|---|
| `PASS` | 33 | Apple B1 的 21 个完整格，以及四路线 F19 的 12 个运行、资源、失败和交付合同格 |
| `BACKEND_REJECTED` | 21 | 锁定 Provider 或当前设备对具名公共域有明确拒绝 |
| `NUMERIC_INCOMPATIBLE` | 64 | dtype、state、codes/scales、activation 或输出合同不相容 |
| `UPSTREAM_MISSING` | 4 | 四路线均没有准入的精确量化 attention/index state region |
| `VALIDATION_NOT_RUN` | 134 | 有成熟原语或片段，但完整真实模型 Direct/Graph 链未执行，不能写 PASS |

`UNPROVEN` 为 0 表示审计归类完成，不表示 256 格能力已经全部实现。非 PASS 格全部是 M4 的具名输入，
不能由相近位宽、单原语测试、reference 运行或其它路线结果代签。

## 六组模型链结果

| 批次 | 结果 | 最早边界 |
|---|---|---|
| M3-01 F01--F04/F19 | 33 PASS、15 backend rejected、36 validation not run | 已接基础能力继续可用；非 Apple 路线仍有明确 layout/mask/scatter/stable-domain 残差 |
| M3-02 F05--F08 | 16 numeric incompatible、36 validation not run | Q4-S128 只有 CUDA Marlin 精确，Q4-A64 只有 Apple MLX 精确，Q8-B32 无精确路线；Vulkan BF16 attention/RoPE 不相容 |
| M3-03 F09--F11 | 24 numeric incompatible | 本地 Qwen3.5-2B 的 conv/recurrent 值为 BF16、state 为 F32；四条成熟 recurrent 路线只资格化 dense F32 |
| M3-04 F12--F14 | 12 numeric incompatible、32 validation not run | grouped expert/mHC 只签 dense F32；完整 routing→expert→combine、SiTU、AttnRes/MTP 未执行 |
| M3-05 F15--F17 | 4 numeric incompatible、28 validation not run | Lightning Indexer 的窄 F32/F16 域已接；MLA/DSA/CSA/HCA 与跨层状态整链未执行 |
| M3-06 F18 | 6 backend rejected、8 numeric incompatible、4 upstream missing、2 validation not run | 只有 IREE Vulkan 的两种 dense FP8 精确域已接；它们尚未执行完整 2B 链 |

## 本地模型与执行证据

- 用户本地 Qwen3.5-2B 的 13 个文件、4,571,274,023 bytes 已重新逐文件核对 size/SHA-256，source ID
  保持 `4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155`。
- B16 的 prompt 1/17/128 + 8 步 decode reference 已重新生成，三个 safetensors 的 size/SHA-256 再次
  核对通过；receipt 明确标记为外部 reference，不是 HYLE PASS。临时张量在保存收据后清理。
- HYLE 的 16 个“路线 × B16/Q4/Q8/F8”完整模型旅程通过数为 0。B16 在首个 linear-attention
  recurrent 的 BF16 边界停止；Q4/Q8 缺四路线共同精确载体；F8 只有 Vulkan dense primitive，不能
  外推到 recurrent/整模。M3 没有用转换成 F32 常驻或 CPU fallback 绕过这些边界。
- Windows CPU 4/4、CUDA 6/6、Vulkan 4/4 当前 factor 通过；MacBook Air Apple 3/3、Vulkan 4/4
  通过；Mac mini Apple static 2/2 通过。Air 的一次旧 `compute_ops` 程序曾与新 dylib 混用，按登记
  Metal 工具链重新链接后 47/47 通过，确认不是产品 ABI 缺陷。
- 统一 LOC 脚本在不可变产品候选上报告 Compute+Graph 实现 **59,749 行 / 252 文件**，排除测试、
  文档、bench、vendor、构建产物和 scratch，self-check PASS。

## Provider 完成门与下一步

M3-08 `PASS`：M0 准入的 mature Provider 在其精确适用域内未接通数量仍为 0。这个门只证明接入
完成，不证明需求覆盖完成。M4 必须从上述 223 个非 PASS 格中先合并共同根因，继续核对成熟库、合法
组合、版本升级或窄移植；只有仍可复现且成熟方案确实不存在的最小缺口，才能另写 SCOPED→AUDITED
卡。冻结自实现继续不动。

按用户要求，本轮在 M3 终局后停止，不进入 M4。
