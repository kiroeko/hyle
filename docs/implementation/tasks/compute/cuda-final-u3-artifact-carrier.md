# CUDA U3 N1：成熟 GPU 权重准备载体

<!-- MODULE REVIEW: SCOPED -->

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

## 待冻结有限 FIX

1. GPU leaf SDK/source/runtime 及许可证 owning description；原 CPU resolver/shim 保持。
2. 原 GPU tensor 程序，仅 tensor factory 的 device 与已有外部 stream/context 接缝。
3. Provider prepare/propose/submit、原异常翻译、异步保活、严格资源／capture 明确拒绝。
4. 普通 CUDA 默认组合、Direct／Graph／精度／18合同与三交付批量资格，原 driver/SM guards 不盲删。

## 七门

第三方路径、需求完整、内部资源复用、P0 成本、C++ 边界、三交付与普通使用均适用；当前全部
UNVERIFIED。不以组件注册或本卡完成冒充产品覆盖。
