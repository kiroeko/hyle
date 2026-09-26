# D5：十三画像词表与共用语义核心批次

**后续形状资格更正：** 此处3584宽 SiTU 是 Kimi latent 宽，不能代签实际 expert3072、shared6144、
dense33792 的 SiTU；仅保留通用公式证据。更正和实际宽度复核由
[当前 D5 卡](../../../../docs/implementation/tasks/compute/llm-moe-d5-real-fragment-acceptance.md)持有。
原运行日志及其它已签局部资格保留，完整画像 ledger 未因本批增加。

候选：`f0bec3afe856533f9fa366ef556c649b53f75f3a`。Windows canonical；Apple 副本由
Windows Git bundle 前进到同一候选。没有产品代码或 Provider 数学修改。

## 已运行

| 路线 | Direct | Graph | 机器 |
|---|---:|---:|---|
| CPU | 29 PASS | 29 PASS | Windows 产品机 |
| CUDA | 29 PASS | 29 PASS | Windows RTX 4070 |
| Apple | 29 PASS | 29 PASS | MacBook Air Mac16,12 / M4 |
| Vulkan | 29 PASS | 29 PASS | Windows RTX 4070 |

共 **232/232 个 GTest、零失败、零 SKIP**。每入口含26个十三画像 R01 词表测试与三个
参数化核心批量测试；每个核心批量测试内部遍历多个 exact profile。

- R01：十三画像的完整逻辑词表、真实 hidden、逐角色 BF16/Q8_0、两次变化输入、全部 logits、
  跨片 Gather、越界失败及保留输出。先前 CPU 检查点在本候选重新运行，其余三路线首次签本批。
- KDA：Kimi 三画像 H96、GLM Flash 三画像 H64，D128/K4；逐通道有界门控、卷积/SiLU、
  Q/K L2、原生 recurrent state、RMSNorm 与 sigmoid 输出门。八步使用路线的实际上一步状态；
  独立 double oracle 自行推进。卷积窗口随输入滑动。
- SiTU：Kimi 三画像，expert width3584；`4*tanh(g/4)*sigmoid(g)*25*tanh(u/25)`。
- mHC：DeepSeek Pro 一画像、Flash 三画像及 GLM Flash 三画像；H7168/4096、hc4、
  非均匀 Sinkhorn20、pre collapse、独立 branch 与原 residual 的 post combine。

语义配方见 [共用测试](../../../../tests/compute_models/d5_moe_semantic_fragments.h)，
精确模型/权重来源身份见 [画像事实](../../../../tests/compute_models/d5_moe_profile_facts.json)。
公式引用已登记 llama.cpp `4c9233c034fc450dcf34c7c0988aebe6da5cdf1a` 的
`src/models/kimi-k3.cpp` / `deepseek4.cpp` 及 Transformers
`c1b80cf3ee183f94d8b21dfa7e312d1eea4942ad` 的 GLM5 Next 实现。

## 证据边界

核心配方限定 **F32 运行域与投影后输入边界**。未代签 R03 投影、官方 BF16 参数/激活转换、
完整 MoE/MLA/DSA/MTP 或整模型推理；合成非零 payload 不冒充下载的巨型 checkpoint。
完整画像 ledger 仍为 **72/176 RUN_PASS**；本批新增完整格为零。
三交付与适用 P1 **未运行**，保持 D5-A5/A6 待办。

保留 selector plan 和123项 skipped/reused reasons；按批次协议只运行这一组新增消费者，
不声称完成整个 selected-factor suite。`inputs_test.py` 13项、closure、docs、diff 检查通过。
首次 CPU 编译的测试索引窄化警告已在冻结候选前修正，原失败日志保留。

统一 [LOC 报告](loc.json) 为 **58,255 行／228 个 Compute+Graph 实现文件**，不含测试和 vendor；
本批产品实现行数不变。[机器收据](receipt.json) 保留各日志与脚本 SHA-256。
复用现有四条构建树，重型验证串行；没有新建完整 vendor/build 副本。
