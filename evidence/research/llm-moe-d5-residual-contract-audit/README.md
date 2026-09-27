# D5 残余合同：一次批量源码审计与离线日志重放

baseline `416242f8ef2cd9a83d558b3b7879e4c2c1707be4`；产品未改。
[施工卡](../../../docs/implementation/tasks/compute/llm-moe-d5-residual-contract-audit.md)。
本批没有构建产品、启动GPU或生成新性能样本；下述结论不提高154/176覆盖数字。

## A1：官方低精度要求不是只由权重文件名推测

固定源码、SHA-256及片段见`sources.json`；四份完整config提取见
`official-config-extract.json.gz`，原始库存文件SHA另存。没有重新下载模型或选择新revision。

| 模型 | 实际合同 | 不能替代的路径 |
|---|---|---|
| GLM-5.3/Flash官方 | config为dynamic E4M3、128×128权重块；排除模块保持各自原dtype。SGLang原专家程序含group128激活量化 | FP8权重先解码后直接BF16 GEMM没有动态A8舍入 |
| DeepSeek V4 Pro/Flash官方 | 原`linear`在FP8、FP4权重分支均先做group128 E4M3激活量化；FP4专家权重group32，FP8权重块128 | A16W4/A16W8不能直接签A8W4/A8W8 |
| DeepSeek状态与indexer | 非旋转KV有group64 FP8 roundtrip；indexer/KV有group32 FP4 QAT；BF16存储不代表中间没有量化 | 只支持GEMM或只保留BF16状态不代表整链完成 |

原DeepSeek kernel确实规定FP8 absmax floor `1e-4`、FP4 floor `6*2^-126`、
F32乘倒数后向上取2幂scale、饱和转换以及BF16写回。HYLE公开语义和独立oracle中的这些要求
有原程序依据；本轮没有发现能直接撤掉20格准备拒绝的合同误绑。

llama.cpp固定转换器先将部分官方FP8权重按scale解码，再写GGUF；DeepSeek另有MXFP4专家
重新打包。其GGUF推理支持说明转换后的画像可执行，不能单独证明各backend保留了原官方程序的
动态激活量化。两类画像均留在目标中，不互相替换。

## A2：成熟来源与下一步边界

| 剩余区域 | 已证实边界 | 新发现的具名候选/裁决 |
|---|---|---|
| CPU：GLM与DeepSeek官方8格 | 旧F47的oneDNN指令集条件、reference双重舍入、SGLang AMX限定仍成立 | PyTorch 2.8 C10有直接F32→E4M3 RNE转换，避免F16中转；但尚不是HYLE依赖、没有完整group/scale/GEMM承载，不能把一个标量函数算接入完成 |
| CPU：DeepSeek FP4/FP8状态 | 当前成熟调用尚未通过固定状态合同 | ds4原CPU具有group64 FP8和group32 FP4 roundtrip；函数嵌在整模型文件，FP8符号处理不保留负零，FP4 scale写法也须资格化。可做一批原函数资格，不能直接挪为HYLE数学或宣称补齐group128 GEMM |
| Apple：DeepSeek官方4格 | MLX原FP4候选在低scale域存在已记录FTZ/scale差异，旧预算结束 | ds4原Metal QAT已含正确floor，但它融合Hadamard并非独立通用量化；还需FP4 GEMM及完整状态资格。原Metal是否保留低域BF16数值尚未验证 |
| Vulkan：官方8格 | 当前ggml/IREE原程序没有通过该官方组合的完整资格 | 上述CPU/Metal候选不能当Vulkan provider；本轮没有找到可立即接线、已合格的完整承载，仍是具名缺口，不宣称全生态不存在 |

选择：保留已合格CUDA官方程序及其它154格；不复跑F45/F46/F47原候选，不新增自研算子。
CPU C10/ds4和Apple ds4只登记为待独立有限资格的成熟来源；不为得到覆盖数字而隐藏CPU回退、
host读回、量化合同变化或大依赖成本。它们不能解除Vulkan数值和性能问题。

## A3：Vulkan Q8失败确实触及状态

`vulkan-failure-roles.json`从已有压缩日志定位测试、模型、输出序号和trace；两入口结果相同。
当前消费者与原失败候选相关部分未变。这里统计失败断言，不把断言数量当画像数量。

| 每个入口 | 实际反馈状态失败 | 诊断输出失败 | 最终投影失败 |
|---|---:|---:|---:|
| DeepSeek Flash Q8完整attention | 17 | 7 | 0条已记录 |
| 同画像独立compressor | 456 | 0 | 无独立最终投影输出 |
| Pro/Flash官方后投影compressor语义 | 888 | 0 | 无独立最终投影输出 |

因此不能删掉7条诊断差异就把Q8两格改PASS。官方后投影语义也不能据此签完整FP8/FP4画像。
长位置RoPE问题仍需成熟程序在相同phase、状态反馈合同下修复或资格化；保留原误差门。

## A4：性能测量接缝与明确修正范围

一次批量离线重放24份A/A进程日志，详见`aa-offline.json`：

| 批次 | 原两侧op/s比值 | nanobench配对比值（仅印刷精度） |
|---|---|---|
| Vulkan前次中断批 | PASS | diagnostic PASS |
| Vulkan唯一恢复批 | FAIL | diagnostic FAIL |
| CUDA已完成批 | PASS | diagnostic PASS |

原validator使用两侧各自中位数的比值；nanobench另算逐轮log-ratio中位数。恢复批
`peer-0-1` Qwen的前者为1.22453，后者约1.10742，仍超过原10%单进程门。
配对原始数据没有归档，不能用四舍五入的表格重新资格化或追认PASS。

源码还证实可修正的输入平衡缺陷：两侧`verify`各自归零，然后nanobench分别运行校准，
校准调用次数可能不同。Qwen有2槽，MoE有3槽，`run`每次前进一槽；校准后既没有归零，也不
保证epoch调用数是槽数的倍数。因此同一配对轮可能比较不同输入/专家分布。
例如3槽、两侧校准次数1和2、epoch次数1，会从槽1/槽2开始。这个源码反例成立，
但旧日志不含迭代/槽序列，不能声称它是当时22.45%偏差的唯一原因。

恢复批个别Qwen表格12轮仅约0.03/0.04秒；共享最小校准次数也不能保证每侧每轮达到配置的
8ms。精确每轮次数/时长缺失；不凭这些四舍五入总数反推原始样本。

下一项为独立D5验证入口修正：Vulkan每轮固定6次完整调用（2/3槽的最小公倍数）、
跳过不平衡自适应校准，24配对轮并保留逐轮次数/耗时；仍测完整submit/wait/释放，
保留双执行顺序、双分配顺序、5%因素门和10%gross门。先一次8进程A/A，失败即停，
不直接重跑P1、不修改公共统计核或产品API。CPU/Apple/CUDA旧有效证据不改写。

## 验证与限制

本批脚本`replay.py.gz`只读取固定源码与已有日志，不调用GPU。归档数据有来源SHA。
未通过/未运行的产品要求仍保留；源码候选不等于运行能力，测量缺陷不等于产品缺陷。
