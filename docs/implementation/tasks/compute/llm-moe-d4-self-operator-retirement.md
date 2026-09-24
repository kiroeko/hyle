# M3-D4：Provider 优先的旧算子批量退场

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- owners：`compute_provider_native`、`compute_route_cpu`、`graph_route_cpu`、
  `compute_provider_marlin`、`compute_provider_cuda`、`compute_route_cuda`、
  `graph_route_cuda`、`compute_route_apple`、`graph_route_apple`、
  `compute_route_vulkan`、`graph_route_vulkan`；需求矩阵仍由 `compute_ops` 的
  [M3 主卡](llm-moe-m3-provider-demand-reconciliation.md)持有。
- baseline：`0128315b3314a637b6370d9488c14bd0b413d213`；Windows canonical 工作树干净且与
  `origin/main` 一致，CPU/CUDA/Apple/Vulkan 的 ggml-native D3 子批均已形成候选。
- reopen trigger：真实模型矩阵与 D3 成熟 Provider 接线已经证明若干旧实现没有目标格式、没有独立
  consumer，或仍在成熟实现之前命中。用户进一步明确要求一次性识别并清退过去在未检查成熟
  Provider 覆盖前写入的 HYLE 算子；只有已有合格同合同领先证据或成熟实现的精确缺口可以保留。
- 正常 consumer：普通 C++ 用户选择 CPU/CUDA/Apple/Vulkan 路线，准备 Direct operation 或 Graph
  region；冷期路由到适用成熟 Provider，重复 submit 不重新搜索、计时或回退到另一设备。
- 主纵切：M3 的 15 个真实 LLM/MoE region，重点是 norm/activation、attention、索引/TopK/scan/
  scatter、dense/quantized/grouped projection 和 recurrent state；反偏置纵切保留 public Compute 的
  checked bounds、溢出、alias、empty tensor 与完成/失败语义，防止删除一个成熟实现未覆盖的语义域。
- scope：一次表驱动审计 CPU native、CUDA HYLE `.cu`、Apple 自有 Metal/旧组合、Vulkan HYLE shader
  和旧 Provider adapter；按实际 route preference、完整语义域、真实 consumer 与既有合格性能证据，
  形成有限 `DELETE_FULL/DELETE_DOMAIN/KEEP_RESIDUAL/KEEP_INFRA/KEEP_WIN` 清单并整批实施；同步删除
  product route、CMake、公开 alias、测试和当前用户文档中的过时接线。
- non-goals：新增或优化 HYLE 数学；为每个算子新建 benchmark；引入新 Provider；删除 SIMD 基础库；
  删除 backend/device/resource/completion、checked boundary adapter 或 Graph planning 设施；把 CPU
  fallback 用于 GPU；改写公共治理或性能仪器。
- 删除规则：
  1. 没有真实模型或独立通用 consumer，仅由 test/benchmark/历史文档消费的实现直接删除；
  2. 成熟 Provider 完整覆盖相同 dtype/layout/shape、checked/alias、状态、完成和资源合同，且没有既有
     合格证据证明 HYLE 在该精确域领先，删除 HYLE 实现或该域的路由；
  3. 成熟替代只覆盖部分合同时，仅保留未覆盖的精确残余，不保留整套低优先级 fallback；
  4. 既有 `WIN` 只保留证据绑定的硬件/软件/shape 域；parity、inconclusive 或无证据都不构成优势；
  5. 原版 SGLang/CCCL/ggml/MLX/oneDNN 程序及其 HYLE 生命周期 adapter 不是自研算子。
- 性能：P0。复用既有合格证据，不启动新 P1。当前账本没有证明 broad HYLE CPU/CUDA/Apple/Vulkan
  算子领先：CPU M32 支持 oneDNN，Apple 是 MLX frontend parity，CUDA 是 SGLang 接入非劣，Vulkan
  是 ggml recipe 接入非劣。删除后 D5 的真实完整片段 P1 仍按主卡执行。
- budget：一次四路线静态审计、一次整批实现、每路线一次受影响 correctness；稳定后统一跑一次
  source/static/shared。算法优化预算和逐算子性能采样预算均为 0；发现无成熟同合同替代即记
  `KEEP_RESIDUAL`，不继续研究。
- stop：有限清单每项都已删除或收窄到具名残余；普通 route 不再优先选择已替代实现；三交付和受影响
  correctness 通过后形成不可变候选。任何需要新数学或新 Provider 的真实缺口退回 M3/D5 具名处理，
  不在 D4 扩张。

## 审计全集

| ID | 路线 | 审计边界 | 冻结输出 |
|---|---|---|---|
| O1 | cross-route | Marlin factor/public alias/default binding、CMake、vendor、tests/docs | 无真实目标或独立 consumer 时 `DELETE_FULL` |
| O2 | CPU | `compute_provider_native` 每个 operation domain 对 ggml CPU/oneDNN/standard-index | 只保留成熟实现未覆盖的精确 public 语义 |
| O3 | CUDA | `cuda_native::provider()` 自有 kernel 对 ggml CUDA、cuBLASLt、SGLang、CCCL、FlashInfer | SGLang/CCCL 原版程序保留；自有 provider 收窄到精确缺口 |
| O4 | Apple | 自有 Metal 字符串 kernel、旧 indexing/reduction 与当前 MLX/ggml Metal lowering | 已不可达或由当前 MLX lowering 完整替代者删除 |
| O5 | Vulkan | HYLE tensor/scan/scatter/SwiGLU/F32 MatMul shader 对 ggml Vulkan/IREE | 原版 ggml region 优先；只保留 Fill/Select/check 等精确缺口与资源 adapter |
| O6 | seam | Direct/Graph roster、component graph、安装导出、validation policy、用户文档、tests | 不留 dead target、public alias 或错误 fallback |

审计只接受当前源码、固定 Provider 上游、真实模型矩阵和已合格证据。旧 CLOSED 记录说明历史事实，
不证明实现仍应保留；旧冻结卡中“M5 前暂停”已由本轮用户明确授权覆盖。本卡仍遵守其中 SIMD 基础库
保留、算法优化预算 0 和精确残余不得误删的约束。

## 预声明验证

- docs/status-only transition：`check-module-closure.py --previous-ref <registry-before>`、
  `check-docs.py`、`git diff --check`；不构建、不跑 P1。
- product candidate：selector 以本卡 baseline 为 base；依选择结果运行 CPU/CUDA/Apple/Vulkan 的受影响
  provider/route/Graph correctness。Windows CUDA/Vulkan 重型目标串行；Apple 使用 Windows 精确 archive
  在两台 Mac 验证，远端副本不反向覆盖产品源码。
- delivery：产品稳定后一次 source/static/shared installed-package consumer；Windows shared DLL 与 EXE
  同目录，避免 UI 弹窗。清理临时 build/archive 后再报告。
- P0：静态确认成熟 Provider 在冷期绑定，submit 无 provider search/timing；保留的残余逐项写明缺少的
 成熟同合同能力。D4 不运行新性能比较。

## SCOPED

本轮 reopen trigger、owner、删除规则、审计全集、预算、验证和停止条件已冻结。下一步仅完成 O1-O6
源码对账并生成机器可读退场 ledger；在 owner 进入 `AUDITED` 且有限删除集冻结前不修改产品代码。

## AUDITED：有限退场集

[机器可读退场 ledger](llm-moe-d4-retirement-ledger.json)已逐项绑定删除对象、成熟替代、保留原因和产品
动作。审计没有找到任何可以跨域保留自实现的合格 `WIN`：CPU M32 证据支持 oneDNN；CUDA 证据只证明
采用 SGLang 的组合路线非劣；Apple 比较的是同一 MLX 程序；Vulkan 比较只证明 ggml recipe 接入非劣且
严格方向不确定。因而不追加测量，也不为旧实现寻找翻盘条件。

有限产品改动固定为：

1. 整包删除 Marlin adapter/carrier/vendor/default route/build/tests/current docs；历史 task/evidence 只保留为
   历史事实，不再参与当前产品构建。
2. 从 CPU native 删除 Scan/TopK 域，普通 CPU route 继续由 ggml standard-index 承载相同 public operation。
3. 从 HYLE CUDA native provider 删除 Scan/TopK/Gather 域，普通 CUDA route 继续由已注册 CCCL provider
   承载；SGLang/CCCL 原版程序和其生命周期 adapter 保留。
4. 删除 Apple 无调用的旧 indexing/reduction/custom ordered-scatter 五个文件；当前 MLX mature lowering
   保持唯一产品路径。
5. 删除 Vulkan 自有 F32 MatMul 优先实现、packing 和专用 SwiGLU shader；plain F32 MatMul 与普通门控
   进入 ggml managed region。通用 tensor shader、checked I32 scan、ordered scatter 和 boundary adapter
   只按 ledger 的精确缺口保留。
6. 同步收口 Direct/Graph roster、组件图、dependency lock、validation policy、安装导出、当前 tests/
   benchmark 和用户文档。历史证据不回写，SIMD 基础库不动。

表外不删除 CPU/CUDA/Vulkan fallback 域，不新增 kernel/provider/性能仪器。实现中若静态或 correctness
证明成熟替代没有 ledger 所写合同，立即恢复该精确残余并记录限制，不扩大成逐算子研究。
