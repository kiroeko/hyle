# CUDA U3 N1：成熟 GPU 权重准备载体

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- owner：`compute_provider_torch_cuda`；普通默认组合接缝：`compute_route_cuda`。
- baseline：`cc7cde66bbe156a6f3601f48062f6eee2f4a1c38`；Windows canonical，保留后续修改。
- 当前需求：[U3 N1](cuda-final-u3-mature-bindings.md) 及[共同目录](../../../architecture/model-compute-catalogue.md)
  的 6 NF4/nested、7 AWQ、5 CT 原 packed 合同；不重新研究模型，不改变 CPU 端口或数学。
- 主纵切：GPU 冷解码原模型权重，保留 dense 常量，原成熟 MatMul／Gather 继续计算。
  正交纵切：变化 payload 的重复 Direct／Graph decode、两独立 invocation、失败和 teardown。
- 公开形状：既有 `ArtifactWeightDecode` 原输入／output 推导；普通用户仅选 CUDA。
  新 `torch_cuda::provider()` 是可组合成熟实现，非 Native 数学。Graph 复用普通 operation adapter，
  不另建执行器或为此改 CLOSED `graph_route_cuda` 产品代码。
- runtime：共享 HYLE 原 CUDA primary context／stream；GPU-only ATen tensor，库私有 allocator／metadata；
  `kManaged`，允许库内分配、阻塞主机，非外部 capture；不承诺 caller quota/no-allocation。
  prepared 保留设备／描述符；每 invocation 私有持有本次结果至 terminal Completion，失败输出不可消费。
- SDK：保留 locked LibTorch CPU common runtime／headers，附加同 2.8.0 原 CUDA leaf。
  CPU-only 构建不发现／下载 GPU SDK、不替换同名 DLL；CUDA-only 不创建 CPU 执行 target。
  Windows 固定官方 cu129 wheel、hash，经原缓存可离线复用；Linux 固定官方原 leaf 与 bundled CUDA
  依赖，资格化前不签通过。外来 runtime/code image 必须覆盖所有 artifact 与 Completion。
- deps：Compute、CUDA backend、compute_ops、Error、Mem；原 recipe 许可证与版本由新 owning seam 锁定。
  静态／共享 SDK dependency shim 与源码使用相同 SDK resolver，runtime staging DLL 邻 exe。
- P0：原 decode 时间／临时空间线性于权重元素；dense output 为显式用户操作，不隐藏整模型缓存；
  原 nibble 顺序、nested block256＋offset、AWQ permutation／zero／scale、CT signed offset／scale 精度不变。
- P1：本 N1 是单一成熟 carrier 的首次正确接线，不制造性能竞争；U4 对完整合格重复域另行裁决。
- 验证：一次批量稳定 candidate，Windows／Linux；18 实际格式合同、F16/BF16/F32、两入口、
  改变输入／重复执行／两个 slot／释放调用方 Context、OOM／资源拒绝／原错误、CPU-only 隔离，
  CUDA-only 与 mixed source/static/shared/offline/搬移／DLL staging、headers 和直接消费者；
  docs／closure／selector／LOC。未运行为 UNVERIFIED，探针不能替代 HYLE qualification。
- 预算：一次源码／SDK 接缝审计，一次批量实现验证，至多一批具名接线修正；源码证据缺失或
  runtime ABI 不成立时具名裁决，不自行写 kernel、不通过修改旧 CPU 合同绕开。
- 停止：上述适用验证完成才可 CANDIDATE／CLOSED；大模型可原内宽分片，不能冒称完整常驻推理。

## 当前审计输入

原 bitsandbytes `833649043474794b8fe7a4136e0c40faf077b2e0`、AutoAWQ
`88e4c76b20755db275574e6a03c83c84ba3bece5`、compressed-tensors
`ac8e2ba82f4e0d5eaa40069f8fc642738f124cd4` 已登记于 CPU 原来源，本轮保持其算术及当前端口。
GPU 库原 native probe 在 RTX4070／torch2.8.0+cu129 完成 NF4 nested 和三 dtype AWQ／CT；
CPU common＋CUDA leaf 的隔离 C++ 进程完成两设备计算，torch_cuda 所需 3202 个 torch_cpu
导出在固定 CPU SDK 均存在。仅证明本机接缝可用，不签 Linux／HYLE／三交付。

Windows 官方 archive SHA256：
`2bc729898e422b9f3da54349eed98f2f0b5dd415434508ee2ab2a13fb021815d`；
现 pip cache 原 archive 与官方身份相同，无需重下 3.57GB。

## AUDITED 有限 FIX

1. GPU leaf SDK/source/runtime 及许可证 owning description；原 CPU resolver/shim 保持。
2. 原 GPU tensor 程序，仅 tensor factory 的 device 与已有外部 stream/context 接缝。
3. Provider prepare/propose/submit、原异常翻译、异步保活、严格资源／capture 明确拒绝。
4. 普通 CUDA 默认组合、Direct／Graph／精度／18合同与三交付批量资格，原 driver/SM guards 不盲删。

SCOPED 提交 `76b9f9db`。上述四项已冻结，允许 owning 产品施工；新 component 从 QUEUED／SCOPED
逐级进入 AUDITED，已有 route 的 AUDITED 范围在本卡有限重定界，旧 U2 架构／事实修复保持。

Windows SDK 只从原 wheel 抽取 CUDA leaf import libraries、GPU headers、原 bundled CUDA DLL 与
原许可证；同名 CPU common 与 torch_python／Python运行环境不引入。通过实际引用原
`at::cuda::getCurrentCUDABlasHandle` 强制加载 torch_cuda 并注册原 dispatch，不依赖 Python import。
该初始化发生在 cold prepare；常量 shift/order 也在 cold prepare 创建、完成后复用，不热上传常量。
原 CPU resolver／config 原文件直接复用，GPU shim 将 CPU common runtime 注册给 GPU owning target，
不改旧 CPU owner 或原 CPU-only 合同。SDK acquisition 是本 factor 的私有机制，不扩公共下载框架。

GPU original math 与 CPU tensor translation 的区别只在原 device propagation：GPU factory 与 inputs
同设备，shift/order 冷驻留；decode 的算术／舍入／原 dtype 完全保持。Prepared 无 payload 缓存；
每次 submit 保留结果在 invocation 至真实 terminal，原 CUDA 外部 stream guard 还原调用方状态。
ATen 原临时释放遵守该 stream 的 allocator 次序；exception 发生后即使有已提交前缀也标 accepted，
让真实 Queue Completion 等待并报告失败，禁止提前释放 borrowed 输入或消费失败输出。
严格 caller workspace／quota 或外部 capture 合同明确拒绝，不伪造资源能力。

Linux 同版本官方 wheel 身份已核对：
`02c7258e917f3043c978b53acf6f02b818db0d0d85db0e58ae578af333b9b4e2`；
其 NVIDIA leaf 依赖仍须原 metadata/pin/hash 固定并实际资格，未运行不签支持或交付通过。

## 七门

第三方路径、需求完整、内部资源复用、P0 成本、C++ 边界、三交付与普通使用均适用；当前全部
UNVERIFIED。不以组件注册或本卡完成冒充产品覆盖。

## Windows 批次进展

`0cd39365339e75a2d0bdce15a8fa065a8f5408ba` 已实现普通默认成熟GPU接线，18原packed合同的
Direct／Graph decode→projection、12精度／资源寿命样例、默认FFN回归及三个公开头检查通过；
见 [原始批次与失败记录](../../../../evidence/correctness/cuda-final-u3-artifact-carrier/README.md)。
原allocator cold初始化／capability传播已修复；无新数学，CPU common／CPU-only resolver未改。
当前仅Windows mixed producer资格，不签Linux、三交付、剩余OOM／严格资源／capture、全模型或性能。

## Linux 批次进展

固定原 Torch cu129 leaf 与14个原 NVIDIA runtime archive 已实际组装；原 CPU common、原 ELF
SONAME／RPATH 与许可证保持。`4d571e1d` mixed producer 的18实际合同两入口＋12精度／生命周期＋
metadata OOM拒绝＋默认FFN共50项、三个公开头及精度预检通过。
`c8db9033` 普通 CUDA-only 的 shared/static 源码和搬移后独立 SDK 各50项通过，离线／loader通过。
严格未知 private-storage预算、require-prepared／require-capture拒绝及普通command重放实际通过；
隔离原库OOM／恢复另2项已通过；Windows三交付及原库OOM尚待，不签本卡整体CLOSED。精确输入与原失败见
[Linux资格](../../../../evidence/correctness/cuda-final-u3-artifact-carrier/README.md#linux-实际资格与三交付)。
