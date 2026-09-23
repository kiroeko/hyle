# M2-B1 成熟 Provider 基础批次终局

## 裁决

2026-09-23，M2-B1 按[批量执行补充合同](llm-moe-coverage-completion-plan.md#2026-09-23-批量执行补充合同)
结束。产品候选为 `e43b85a5c98425036b3d63ca5104ee1ba3c2306b`；其后的提交只改测试可移植性与文档。

这里的“结束”表示当前已准入 Provider 对 F01--F04/F19 的可用原版能力都已批量接入普通
Direct/Graph，并完成适用验证；上游没有实现或与公共语义不相容的域已形成有限残差。它不把
残差写成 PASS，也不宣称四路线 21 行已经全部覆盖。覆盖矩阵只有 Apple 的 21 个路线格为 PASS，
其余路线格保持 UNPROVEN，跨四路线完整行仍为 0/64。

组件 owner 保持 M2 父任务的 `AUDITED`，供 B2 复用相同 Provider、资源和交付接缝；本页关闭的是
B1 需求批次，不把持续施工的路线组件伪装成 `CLOSED` 后立即重开。B2 启动前仍须把 B2 有限 FIX
绑定到这些当前 owner。

## 已接成熟路径

| 路线 | 已绑定的成熟能力族 | 普通入口与状态 |
|---|---|---|
| CPU | ggml 的 layout/copy/concat、任意轴 Gather、广播、Sum/Mean、L2 region、重复 Scatter-add；标准库 Scan/TopK；oneDNN 字节 Copy/Select 与 dense F32 Softplus | Direct/Graph 共用冷绑定程序；动态索引、两 slot、旧结果、OOM/回收已经验 |
| CUDA | ggml CUDA 的 layout/cast/broadcast/unary/norm/reduce/Softmax region；CCCL Scan/TopK/Gather；成熟 Scatter-add | 原生 stream/buffer/completion；Direct/Graph 普通工厂自动选择，执行期不搜索或计时 |
| Apple | MLX 的 F01--F04/F19 全 21 行 | 已独立 `CLOSED`，21/21 路线格 PASS |
| Vulkan | ggml Vulkan 的 tensor/concat、half 边界、任意轴 Softmax/mask、Scatter-add；IREE BF16 边界 | Windows Vulkan 与 Air/MoltenVK 原生资源、Direct/Graph、重放和失败路径通过 |

## 聚合验证

| 风险 | 结果 |
|---|---|
| Windows 产品真相 | PASS：CPU/oneDNN、CUDA、Vulkan 的既有批次定向门通过；Vulkan 当前候选 Compute 118/118、Graph 34/34，聚焦 9/9 |
| WSL x64 | PASS：oneDNN Release/static+TSan 33/33；CUDA 13.3、RTX 4070 SM89 的 shared 与 static 聚焦门各 10/10 |
| Apple | PASS：Apple B1 的 Air/Mini、三交付与 P1 见独立终局；Vulkan 当前候选 Air static 6/6，使用登记的 Vulkan loader 环境 |
| source/static/shared | PASS：canonical source；CPU 当前 static 与 Windows shared；CUDA 当前 shared/static；Vulkan Windows shared/Air static；未改变的 package/relocation 接缝复用 M1/Apple 已验身份 |
| 资源/失败 | PASS（已接域）：两 invocation、变化输入、旧结果、checked 失败不发布、完成/关闭与配额回收均在路线用例中实跑 |
| 性能 | P0 PASS：packing、layout materialization、workspace、同步和 half 转换均如实保留；B1 非 Apple 不作独立算子速度声明，因此不制造逐算子 P1。Apple 已有完整 region P1 PASS；B2--B6 在完整模型片段上执行适用 P1 |

最后一次验证没有新增产品算法：GCC 14 暴露的 GoogleTest 宏悬空 `else` 警告只通过加花括号修正；
CUDA static 255 个目标编译完成后，五个 factor 的 10/10 CTest 通过。Air Vulkan 首次运行缺少
`DYLD_LIBRARY_PATH`，用项目登记的 `/opt/homebrew/opt/vulkan-loader/lib` 重跑后 6/6 通过；这不是
产品失败。WSL TSan 首次受到 Linux 地址随机化映射冲突，进程局部 `setarch x86_64 -R` 后真实
oneDNN 测试 33/33 通过，没有修改系统设置。

## 有限残差

| 路线 | 精确残差 | 后续 |
|---|---|---|
| CPU | ggml I32 Scan、稳定 TopK 的全部公共域、全 mask/LogSoftmax、Scatter assign、部分特殊值稳定 unary；oneDNN half/BF16稳定函数、任意非连续 reorder/数值 Cast 与 min/max 特殊值域不相容或未准入 | M3 逐格汇总；先查 B2--B6 的完整 ggml/oneDNN region 是否自然覆盖，仍无成熟承载才进入 M4 |
| CUDA | LogSoftmax、Bool mask、Scatter assign及尚未由当前 ggml CUDA/CCCL 域接受的 dtype/layout/index 组合 | 同上；不启用冻结自研兜底来伪造 B1 PASS |
| Apple | 本批无残差；旧不可达自定义 Metal 链不承载这 21 格 | M5 才裁决冻结存量 |
| Vulkan | Scatter assign、Bool mask、稳定 LogSoftmax、非连续/empty 与未准入 storage/index/state 组合 | M3/M4；保留具名上游不支持或语义反例 |

残差不阻止 B2：B1 的退出条件是“已准入成熟能力未接通数为 0”，不是要求 M2 用自研补齐每条
路线的 21 个完整矩阵格。后续完整 region 若覆盖上述残差，矩阵在对应批次凭实际证据更新。

## 规模、清理与下一步

统一脚本在不可变产品候选上统计 Compute+Graph **57,740 实现行 / 249 文件**，排除 tests、
bench、docs、vendor，self-check PASS。B1 不把旧 CLOSED、文档、研究探针或冻结自研算作新覆盖。

B2 现在可以按[M2 B2--B6 批次合同](llm-moe-m2-batch-contracts.md)启动：先一次声明各 Provider 的
M2 primitive/feature 超集，再批量接 F05--F08 的线性、量化线性、Attention/KV 与位置编码 region。
不再为单个算子重配 Provider 或重复冷编译。
