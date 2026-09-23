# M2-B6 成熟 Provider 精确低精度格式批次

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- round / owner：M2-B6；`/root`。
- baseline：`b01fadb42f11f83c70cff07a70a0e686b831827c`；[B5 已关闭](llm-moe-m2-b5-terminal.md)，
  Windows canonical 工作树在本卡登记前干净。
- trigger：F18.1--F18.5 的 E4M3/UE8M0/E2M1 精确格式、dense/grouped MatMul 与量化
  attention/index state。M1 已接 Vulkan IREE FP8 与 ggml MXFP4 的有限域，其他路线只登记了
  成熟候选来源，尚未按五行格式合同完成资格裁决。
- stop：逐行核对锁定 oneDNN、cuBLASLt/Marlin/SGLang、MLX、IREE/ggml Vulkan 的 payload、scale、
  activation、rounding 与 output 合同；只把精确相容入口记为 BOUND。相似位宽、整模型浮点常驻、
  自研 decoder/quantizer/kernel/shader 均拒绝。已准入入口完成适用 Direct/Graph、资源与交付门；
  其余留下具名反例和 M3/M4 owner 后关闭 M2。

## SCOPED owner 与有限边界

| owner | 本批允许审计/修改 | 本批禁止 |
|---|---|---|
| `compute_ops` | 复核既有 E4M3 block128/group32 与 MXFP4 格式 identity、size/alignment 和推导；仅在成熟入口需要时修正描述 | 新量化公式、改变已冻结 codes/scales |
| CPU | oneDNN 3.13.2 原版 E4M3/E2M1/E8M0 reorder 与 MatMul/primitive 组合的精确可行域 | HYLE tile decoder、整模型解压常驻、把 reorder 当 MatMul |
| CUDA | cuBLASLt、Marlin `51918e25`、SGLang `06992c92` 的原版 scale/activation/GEMM 合同 | 新 `.cu`、CUTLASS 模板改名自研、SM100 能力代签 SM89 |
| Apple | MLX 0.32.2 `from_fp8/to_fp8/quantized_matmul/qqmm/gather_qqmm` 的精确模式 | 将默认 NVFP4/affine Q4 冒充 MXFP4 W4A8 |
| Vulkan | 已接 IREE block128/group32 FP8 与 ggml MXFP4 的精确子域、原生资源/完成和组合输出 | 手写 SPIR-V、用 F16 activation 的 MXFP4 代签 W4A8 |

共享路由、Mem、构建系统、性能仪器与冻结自研算子不在预期 FIX。模型/config、平台、硬件、
驱动/SDK、路线、量化档位与 Provider 版本继续参与 prepare 冷绑定，submit 不搜索或计时。

## SCOPED 验收

1. F18.1--F18.5 分别保存 payload code、scale dtype/group、activation dtype/group、舍入/饱和与
   输出类型；一个字段不同即为不相容，不能按“FP8/FP4”名称合并。
2. 复用独立 bit decoder 与同 codes/scales oracle；适用入口覆盖 Direct/Graph、两个 invocation、
   压缩输入驻留、旧结果和失败回收。没有精确成熟入口的路线记录源码反例，不制造 mock 测试。
3. P0 记录 packing、decode、workspace、同步和完整转换；未冻结完整 F18 模型片段时 P1 保持
   `NOT_APPLICABLE`。本地 2B 的 24 层/8 步 FP8 旅程仍属 M3-06 汇总，不由单 MatMul 代签。
4. B6 关闭时汇总 M2 source/static/shared 交付证据、九个稳定 profile 的复用/运行边界、全部 residual
   和统一 LOC，然后暂停并向用户报告，不进入 M3。

审计完成后，本卡必须固定有限 `KEEP/BORROW/DEFER/REJECT` 表并转为 `AUDITED`，之后才能修改
产品代码或新增资格测试。
