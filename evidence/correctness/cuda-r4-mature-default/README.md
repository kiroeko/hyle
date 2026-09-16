# R4 CUDA 最终成熟默认：正确性与交付

产品候选 `755fb7c76132908e15cc8e2c298c69c227af775a`，tree
`5f91d6d548e6e59406adcdc683c67ee473e59f4b`。SGLang 接入为 `c5848846`；`f7d14870`
冻结最终性能合同。`755fb7c7` 只给 Graph 测试的 gtest 条件宏加大括号，修复 GCC14 WERROR；
产品字节不变。Windows 交付和全 selector 使用 f7；WSL/Mac 使用该 Git archive 加唯一测试补丁，
[身份与差异](test-braces-source-identity.json)可复核，原 GCC 失败日志保留。

## 实际验收

| 范围 | 结果 |
|---|---|
| Windows 定向普通测试 | 三个 CUDA factor 的实际测试与逐头检查，6/6 PASS |
| change-impact selector | f7 全选 106 PASS、5 skipped；755 测试修正增量 4 PASS、106 skipped |
| Windows CUDA-only 三交付 | source、relocated shared、relocated static 各 Direct/Graph FFN 2/2 PASS |
| WSL GCC14 host | off、ASan、UBSan、TSan 各 6/6 PASS；ASan 保留下面的外部驱动例外 |
| CUDA device memcheck | Provider 6、Direct 3、Graph 4 个测试，全部零错误、零设备泄漏 |
| WSL CUDA-only 三交付 | source、relocated shared、relocated static 各 FFN 2/2 PASS |
| 两台 Mac CUDA OFF | 各 none/cpu/core 依赖与安装投影 PASS、两个公共 C++ consumer PASS；不签 CUDA 执行 |

CUDA-only SDK 核对了组件/头/依赖、重定位，以及 Marlin 和 SGLang LICENSE/NOTICE 的原字节。
Windows 实际复制 CUDA/CRT DLL 到 exe 目录。两 Mac 检查 CUDA OFF 不发现 Toolkit、不泄漏 CUDA
导出和新增 SGLang 许可；Mac mini 仅是 correctness/portability 环境，不作为 Formal 性能机。

新增测试覆盖成熟 FP16/FP32 的18组形状、特殊值、双槽变化输入、非法对齐/alias 后复用、
OOM/关闭/完成保活，以及普通 Direct 和真实 capture 的默认选择、separate/非向量尾部能力。
原 C1/C2/C6、共同失败退役的未变化实现继续使用主卡已有证据，没有为本片段重跑 CPU P1。

ASan 使用已经登记的 `protect_shadow_gap=0` 与外部 `libcuda.so.1` leak 限定；三个 GPU 程序
各自核对恰好两笔/280 bytes。没有禁用其余 leak/access 检查；host 插桩不冒称闭源 SDK 或 GPU
device code 已插桩。TSan 无 suppression；device 检查由 Compute Sanitizer 单独执行。

普通构建并行2、重型检查串行。selector 的 Visual Studio SDK 路径发现 oneDNN 自带 `/MP`，
因而将当时仍存活的本次构建进程树限制到两颗 P 核，记录在 Windows 归档中；未改用户进程或
全局电源计划，没有在该阶段采 P1。全局35个 CL 进程快照不当成全部属于该构建的证据。

## 原始材料

- [Windows 日志、增量 selector 与复现脚本](windows-and-recipes.zip)。
- [完整 selector receipts 与日志](selected-validation-raw.zip)、[摘要](selected-validation-summary.json)。
- [WSL 原始日志、编译参数与命令](wsl-raw.tar.gz)。
- [MacBook Air](mac-air-raw.tar.gz)、[Mac mini](mac-mini-raw.tar.gz) 的 CUDA OFF 日志。
- [上游源码/archive/blob 身份](source-identity.json)、[机器汇总](summary.json)、[归档哈希](sha256.json)。
- [统一 LOC](loc.json)：Compute + Graph 全部自有实现 **32,776 行 / 109 文件**，不含测试、第三方，
  self-check PASS；后续只有测试大括号变化，产品计数不变。此规模包含旧实现，不代表完成比例。

本包完成普通正确性、插桩与交付门。最终默认 P1 和原生边界成本在独立证据中裁决，不能由本包
推导性能 PASS 或整个 R4 CLOSED。

后续终局：最终默认 [Direct v3](../../performance/compute-cuda-ffn-p1-v3-755fb7c7-w1-win/README.md)、
[Graph v3](../../performance/graph-cuda-ffn-p1-v3-755fb7c7-w1-win/README.md)与
[原生边界诊断](../../performance/cuda-r4-native-boundary-v1/README.md)均已实际完成；裁决由主卡汇总。
[最终产品候选 LOC 重计](loc-final.json)仍为32,776行/109文件，self-check PASS。原 summary.json 保留
本正确性检查点当时的性能待验收状态，不改写历史快照。
