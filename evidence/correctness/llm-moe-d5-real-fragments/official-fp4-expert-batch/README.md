# D5 官方 FP4 专家批次

候选 `4c86216457e0eb0aa58bc75c3f42086cb1d0fed8`。CPU、CUDA、Apple、Vulkan 的普通
Direct/Graph 共 **24/24 测试组通过，零失败、零跳过**；[机器收据](receipt.json)记录各组和原始
日志哈希。这是共享消费者接线，无产品算子或 Provider 修改，完整画像账本仍为72/176。

## 实际验证边界

| 官方模型 | gate/up 逻辑 `[N,K]` | down `[N,K]` | resident / selected |
|---|---|---|---|
| Kimi K3 | 3072×3584 | 3584×3072 | 17 / 16 |
| DeepSeek V4 Pro | 3072×7168 | 7168×3072 | 7 / 6 |
| DeepSeek V4 Flash | 2048×4096 | 4096×2048 | 7 / 6 |

每入口遍历六个真实形状签名，各两次改变输入、专家成员和顺序。原始 payload 是相邻元素低/高
半字节 E2M1，独立 scale 为每32元素一个 UE8M0 byte；DeepSeek 的 I8/F8_E8M0 和 Kimi 的
U8/U8 只在文件声明上不同，本批均保留原位。量化数值由源 code/scale 独立解码计算，不能通过
目标 ggml decoder 自证。矩阵内容为合成稀疏非零值，完整内宽不缩小，不声称下载了巨型权重。

冷准备只重排位到 ggml MXFP4 block，执行不重复转换，也不常驻展开的 F32 权重。
[golden](converter-golden.json)来自执行固定 llama.cpp
`4c9233c034fc450dcf34c7c0988aebe6da5cdf1a` 的
`conversion/base.py::ModelBase.repack_mxfp4_blocks`；它的 Kimi/DeepSeek converter 均调用该函数。
测试逐字节比较包括特殊 scale 的原位保留；特殊 scale 不据此获得数值资格。
官方 tensor 名、dtype、shape 和源审计 hash 复用
[共享事实](../../../../tests/compute_models/d5_moe_execution_facts.json)。

Kimi 官方完整 MoE 在八入口均通过三次变化输入：全896专家打分、top16、BF16 latent down/up、
原始 FP4 routed experts、SiTU、加权归并、norm 和独立 BF16 shared experts。它验证完整计算
拓扑和上述矩阵格式；norm 使用消费层已准备的 F32 常量（值均可精确表示为 BF16），不代签
官方 norm 文件读取/转换旅程，也不声称原模型的每种部署激活量化策略均已通过。
原有八个 GGUF MoE 同批在全部八入口回归通过，仍保留各角色原格式。

## 运行与成本

Windows 使用原 CPU、CUDA、Vulkan 隔离构建树，依次以 `-j 2` 构建同一路线两入口，再运行
三个批量测试；Apple 从 Windows candidate fast-forward 同步，MacBook Air `Mac16,12` 上串行
构建/运行。驱动和完整 gzip 日志随本收据保存，`VSLANG=1033` 使 Windows Ninja 正确记录头依赖，
DLL 沿用 EXE 同目录部署，无提权。CPU/CUDA 本轮补回上一批修复的缺失 C/C++ 对象；没有重建
新的 NVCC kernel。golden 生成器只执行锁定转换函数，不导入模型 loader 或下载权重。

selector 已生成；本批按 D5 A2/A3 执行上述定向共享消费者和输入工具13项，未把它写成完整
factor/三交付/P1 PASS。文档和 closure checker 通过。统一实现统计
[LOC](loc.json)为 **58,561行 / 232文件**，不含测试和第三方源码。
官方 FP8、attention/MTP 等剩余片段以及完整 D5 交付/P1 仍待完成。
