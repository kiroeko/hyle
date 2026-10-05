# D5 CPU / Apple 官方低精度：新增成熟来源资格

2026-10-05，baseline `5bb3d153bfe137feb6b7e3d22ced40e513b1fd5a`。
这批只签新增成熟程序的来源、原生数值及 CPU C++ 接缝，**不签 HYLE 完整画像、三交付或 P1**。
原 ledger 的 154 RUN_PASS / 20 BACKEND_REJECTED / 2 RUN_FAIL（数值失败）不变。

## 来源及裁决

| 来源 | 固定 revision | 裁决 |
|---|---|---|
| TorchAO | `a701b6a6058720c21f95908b7ae4a24bf0cae1b6`，BSD-3-Clause | 原分组 FP8 量化、rceil 和 FP4 codec 可在 CPU 执行；原 block GEMM 依赖 Triton，所以采用具名 ATen 成熟矩阵组合，不声称有上游完整 CPU FP8 kernel |
| FBGEMM | `b20a2e28de7277dcd3b447e44e3cf9866c908028`，BSD-3-Clause | `quantize_ops_cpu.cpp` 的 FP8 rowwise CPU 入口明确拒绝，不再测试该分支 |
| LibTorch | 官方 `2.8.0+cpu` Windows release | CPU-only DLL 依赖闭包与原生 C++ 调用通过；本机 `2.8.0+cu129` 的 torch_cpu.dll 含 CUDA/CUPTI DLL 依赖，已排除 |
| MLX / ds4 Metal | 沿用现有具名来源 | 尚无区别于旧失败的新完整候选；不重复旧 FP4 探针，不放宽 BF16 中间边界/极小值合同 |

上游原文位于 `upstream-source.tar.gz`，逐文件 SHA256、URL 与许可证在
`source-manifest.json`；对应 [TorchAO 原语](https://github.com/pytorch/ao/blob/a701b6a6058720c21f95908b7ae4a24bf0cae1b6/torchao/quantization/quant_primitives.py)、
[MX 程序](https://github.com/pytorch/ao/blob/a701b6a6058720c21f95908b7ae4a24bf0cae1b6/torchao/prototype/mx_formats/mx_tensor.py)及
[FBGEMM CPU 源码](https://github.com/pytorch/FBGEMM/blob/b20a2e28de7277dcd3b447e44e3cf9866c908028/fbgemm_gpu/src/quantize_ops/quantize_ops_cpu.cpp)。
LibTorch 下载/使用遵循 [官方 C++ 交付说明](https://docs.pytorch.org/cppdocs/installing.html)。

## 已运行资格

- `cpu-probe/receipt.json`：原 TorchAO 函数体不变，仅移除包注册 decorator；
  GLM ordinary / DeepSeek power2 group128 各 107,264 个输入，code 和 scale bits 差异均为零。
  ATen 初步矩阵组合 6,912 个输出一致；不是完整消费签署。
- `cpu-carrier-probe/receipt.json`：原 FP4 decoder 检查全部 256 packed bytes，
  共 512 元素，包含正/负零，逐位一致。32 个 encoded grouped FP8/FP4 投影案例，
  417,792 个 BF16 输出一致；8 个越界 ID 正确拒绝。
  覆盖共享/逐 slot 激活、重复 ID、两个变化输入 replay、模型真实 N/K。
- C++ `build.log/run.log/cxx-receipt.json`：MSVC 14.50、/O2 /fp:strict /MD，
  原生 CPU matmul 与 TorchAO FP4 tensor 序列通过。原 FP8 两模式各 65,280 个输入，
  独立数值校验见 `cxx-numeric-check.json`，codes/scales 逐位一致。
  不链接 Python、torch_cuda 或 CUDA runtime；运行时 DLL 放在 exe 同目录，隐藏子进程及
  Windows ErrorMode 抑制加载错误弹窗。
- 初次编译缺少 grad_mode 头及 const cast 拼写错误，只纠正一次；保留
  `build-initial.log`。CUDA wheel 的 DLL 缺依赖失败独立保留，不能把它签成 CPU 隔离通过。

矩阵候选只在索引之后解码选中专家，保留 A8 转换、原 K128/K32 分块和 scale 乘法/累加顺序。
矩阵内部由 ATen matmul 执行，不用 HYLE 标量点积冒充 Provider。
本批权重是格式合法的非零稀疏诊断 payload；它不能替代完整 MoE、MLA/KDA/状态反馈和资源验收。
参考来自独立有限编码表与原坐标/scale，不从实际输出或目标解码生成。

## 交付与成本限制

CPU 原生库是 shared LibTorch runtime，torch_cpu.dll 约 244 MiB；HYLE static 交付也需要其
runtime，不宣称完全静态进程。线程由成熟 CPU 库的进程 runtime 管理，不能把全局
`set_num_threads` 包装为某个 HYLE Context 的局部设置；探针的 2 threads 仅限验证进程。
分组组合有内部 allocation、选中权重展开和 caller 输出复制，应归 managed replay。
本轮无 P1，无 prepared-only / 内部严格内存上界 / 多种硬件性能资格。

最初 CPU SDK 资格下载仅取得所需 ZIP entries，CRC32 和逐文件 SHA256 已核对；该 subset manifest
不声称全量 SDK archive SHA256。随后取得官方完整 584,320,064 字节 archive，SHA256
`bf0079104e068dcbd60b2f8de589dc725282cbe865a85016cc5b8aa46cab519d`，见
`cpu-sdk-archive-receipt.json`。只展开 header/CMake metadata 和实际 shared runtime/import libraries，
约 323 MiB；未展开不使用的巨型静态库。正式产品仍须完成 CPU SDK 的版本/许可和三交付合同。
产品接线前由当前 owner 固定有限 FIX；SDK DLL 不入仓库。证据文本/源码/输入小于 1 MiB。

## Apple 当前边界

既有 [MLX FP4 反例](../../correctness/llm-moe-d5-real-fragments/apple-fp4-native-boundary/README.md)
显示普通/中点可用、极小 scale 下有非零 BF16 输出被清零；ds4 Metal 还存在 Hadamard→BF16
中间边界差异。只换编译姿势、以完整片段容差覆盖精确 roundtrip 或删除不利输出，均不能解除。
本批不再签重复运行；完整官方 Apple FP4 保持未完成。

`manifest.json` 记录本批原始文件身份；源码/状态维护不触发产品构建或 P1。
