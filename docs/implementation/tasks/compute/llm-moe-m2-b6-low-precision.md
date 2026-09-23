# M2-B6 成熟 Provider 精确低精度格式批次

<!-- MODULE REVIEW: CLOSED -->

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

## AUDITED：精确成熟入口与有限 FIX

2026-09-24 对锁定 oneDNN 3.13.2、CUDA 13.3 cuBLASLt、Marlin `51918e25`、SGLang
`06992c92`、MLX 0.32.2、IREE Vulkan `6ef606b6` 与 ggml `456172ec` 完成逐字段审计。
[资格收据](../../../../evidence/correctness/compute-llm-moe-m2-b6/qualification.json)保存源码 pin/hash、
本机 CPU/GPU 身份和运行时拒绝码。API 中出现 FP8/FP4 类型不等于当前设备有可执行算法；位宽相同但
activation 位宽、scale dtype/group、packing、舍入或输出不同也不相容。

| ID | 裁决与有限施工 | 验收 |
|---|---|---|
| B6-C0 | `KEEP` 既有五个公共 format identity 与独立 bit decoder；不改变 codes/scales，也不新增量化公式 | `compute_ops` 现有 size/alignment/overflow 测试复用 |
| B6-C1 | `REJECT` 本机 oneDNN FP8 MatMul：i9-13900K 对 group32 UE8M0 与 block128 F32 两种 primitive descriptor 均返回 `unimplemented`；block128×128 weight scale 还不满足 CPU 单 primitive 的分组限制 | 保存 descriptor 探针；不接 reorder+整权重展开回退 |
| B6-C2 | `DEFER` CUDA 精确档位：cuBLASLt 的 `VEC32_UE8M0`、`VEC128_32F`、`BLK128x128_32F` 名义相容，但 RTX 4070/SM89 两种 heuristic 均返回 `NOT_SUPPORTED`；Marlin 是 W4A16，SGLang 原版量化器的匹配 W4A8 GEMM 只在其具名新架构域 | 保存 CUDA 13.3/driver 610.62 探针；未来按 SM/driver/toolkit 冷规则资格，不让 SM89 代签 |
| B6-C3 | `DEFER` MLX `mxfp8` 完整动态链到 M3-06；`mxfp4 quantized_matmul` 是 W4A16，`mxfp4 qqmm` 让 activation 与 weight 同为 FP4，均 `REJECT` 为 F18.3/4 的 W4A8 代用品 | Air 源码核对 mode/group/input contract；本批不增加隐式解码或不同入口语义 |
| B6-C4 | `BORROW/KEEP` 已接 IREE Vulkan 的 E4M3 block128 F32 与 group32 UE8M0 dense MatMul；只补 BF16 输出组合的 Direct/Graph、两 invocation、压缩输入驻留与资源回收验收 | Windows shared 与 Air static 的两个 factor 集中验证；不重写 IREE 程序 |
| B6-C5 | `DEFER` F18.1/2 完整模型/grouped 域及 F18.3--F18.5 精确专家/index state 到 M3-06；若真实链仍缺成熟入口才按 M4 规则处理 | residual 保持 `UNPROVEN`；无 mock、无自研 decoder/kernel/shader |

`KEEP`：公共格式 identity、Vulkan 原版 IREE FP8 与 BF16 conversion、ggml MXFP4 的既有 W4A16
有限域；`BORROW`：只复用精确相容的 IREE dense 程序。`DEFER`：硬件/入口不相容但已有具名成熟候选
的真实模型链资格。`REJECT`：把 oneDNN reorder 当 MatMul、把 MLX/ggml W4A16 或 W4A4 当 W4A8、
把 SM100/107 能力写成 SM89 PASS、整模型浮点常驻和任何新 HYLE 数学实现。

本批产品数学 FIX 为 0；有限改动只有资格测试、覆盖/残差账本和用户文档。P1 在没有冻结完整 F18
模型片段时 `NOT_APPLICABLE`。有限 FIX 已冻结，可以进入 B6 集中验收。

## 终局裁决

2026-09-24，B6 在最终验证候选
`68278c999b623edbd0c87f05d3926208ea84f6db` 上以 `PASS_WITH_RECORDED_RESIDUALS` 关闭。Windows
RTX 4070 shared selector 的 `compute_route_vulkan` 与 `graph_route_vulkan` 两个 factor 均通过；MacBook
Air 的静态链接候选在 MoltenVK 上完成同两组原生测试。Direct 新增验收覆盖 block128 F32 scale 与
group32 UE8M0 两种 E4M3 输入继续组合到 BF16 输出、两次 invocation、压缩输入不变与完成回收；
Graph 在原 IREE FP8 region 后继续执行 BF16 cast 与既有 ggml/native 节点。

[验收收据](../../../../evidence/correctness/compute-llm-moe-m2-b6/acceptance.json)记录候选身份、两机命令和
结果。Air 首次运行只缺 Homebrew Vulkan loader 的动态库搜索路径；加入登记的
`/opt/homebrew/lib` 与 MoltenVK ICD 后，同一静态构建无需修改即通过，因此不构成产品失败。

CPU、CUDA 与 Apple 的精确入口拒绝/延后结论以及 Vulkan 的完整模型、grouped、W4A8 和量化 index
残差继续保持 `UNPROVEN`，owner 为 M3-06。B6 没有新增 HYLE decoder、quantizer、kernel 或 shader，
也没有把相近位宽实现当成精确格式支持。没有冻结完整 F18 模型片段，P1 按合同
`NOT_APPLICABLE`；统一 LOC 为 Compute+Graph **59,749 行 / 252 文件**，不含测试、文档、样例和
第三方源码。
