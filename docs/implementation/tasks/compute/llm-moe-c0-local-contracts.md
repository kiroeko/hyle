# C0：本地输入、参考计算与有限能力合同

## Closure card

- round / owner：LLM/MoE C0，当前执行者 `/root`；2026-09-17 新 goal 已获用户明确实施授权。
- baseline：`2e85b8942d14a395e148da8b60fd5ea24562f472`，Windows 工作树干净。
- 总合同：[唯一执行计划](llm-moe-coverage-completion-plan.md)。旧 R1–R7 已完成，不继承其进度或队列。
- consumer：独立 C++ 测试消费者将以公开 Direct/Graph 执行本地 Qwen3.5-2B 语言主干及七个大模型的
  具名计算片段。模型解析、固定 token、离线量化与参考运行属于测试工具，不安装到 SDK。
- 主纵切：固定输入 → embedding → 18 个 DeltaNet / 6 个 GQA 层 → tied head → 多步状态更新。
  正交纵切：带 stride/index 的批处理数值变换与独立资源状态，防止把模型层对象引入库。
- C0 范围：校验/同步用户本地输入、固定数值及 case、取得 B16 参考、绑定量化名单/公式、裁决三个
  来源问题、展开 F01–F19 与后续 owner。C1 才开始产品实现。
- 产品 owner：共同 tensor/op 语义 `compute` / `compute_ops`；逻辑 DAG `graph`；原生执行分别归
  CPU/CUDA backend/provider/route 与 Apple/Vulkan route。每个即将修改的 owner 在新卡先依法
  SCOPED → AUDITED；C0 的数据/测试维护不伪造 CLOSED 产品的 reopen 或新产品完成。
- 当前 registry 不变；`compute_provider_ggml` 的 BLOCKED 与 `compute_sched` 的 QUEUED 不自动解除。
- 直接依赖：标准 Python、已有 Torch/Transformers/Safetensors、SSH，以及既有文件清单。
- tier：P0，只审准备阶段资源与测试边界；无 HYLE 产品/API/算法变更，不构建、不运行 P1。
- 预算：三个未决来源合计最多 90 分钟、各最多两个具名实现路径；输入工具与参考至多两轮源码驱动
  修正。超预算记录具体条件，独立本地 2B 能力继续按各自卡推进，不以研究替代验收。
- 预声明验证：本地/WSL/Air/Mini 输入文件大小与 SHA-256；参考严格权重映射、有限输出、固定
  token/状态 receipt；量化编码边界的独立小向量；selector 选择的维护门、closure、docs、diff check。
- stop：本包有限产物都有真实 receipt 且 C1 owner/card 就绪。同步、来源或参考未完成如实列出，
  不签 C0 完成，更不签 HYLE 模型运行 PASS。

## 有限改动集（实施前冻结）

| ID | 问题 / 依据 | 允许改动 | 验收 |
|---|---|---|---|
| C0-01 | 用户指定唯一模型，远端尚无同源输入 | 测试目录的本地校验/复制工具、缓存与接收 receipt；在现有 selector 只登记新增工具测试映射 | 13 个文件逐项大小/hash；临时目录验证后发布，已有正确文件复用；不改 selector/统计算法 |
| C0-02 | 只有模型清单，没有计算 oracle | 独立参考脚本、固定 token / 中间结果 / 状态与工具身份 | 本地 B16 24 层 prefill + 8 步 teacher-forced decode，缺权重/非有限值失败 |
| C0-03 | Q4/Q8/FP8 不能仅按位宽等同 | 固定语言 linear 名单、编码/舍入、精度转换与离线派生合同 | 具体 tensor 名、公式来源、相同 codes/scales，量化损失与实现误差分开 |
| C0-04 | Qwen3.8 输出门、K3 activation quant、MTP 路径尚未资格化 | 每项最多两个具名源码路径，保存 pin/hash/结论 | 条件公式和模型分支回映；无法裁决记录剩余事实，不猜测 PASS |
| C0-05 | F01–F19 仍是粗粒度族 | 有限 requirement/profile/domain/case 明细与 owner 分配 | 无无意义笛卡尔积；每条 MUST 明确四路承载、入口、oracle 和验收包 |
| C0-06 | 新工作尚未绑定产品状态 | 创建首个 C1 卡、合法状态迁移、生成状态表 | 有真实 consumer trigger、源码审计及有限 FIX 后才写产品 |

## 实际输入与工具

- 输入身份：[`qwen35-2b-local-input.json`](qwen35-2b-local-input.json)，完整 source ID
  `4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155`。
- 用户原目录只读；新缓存按 source ID 隔离，验证副本不反写 Windows 产品或模型。
- 当前参考候选工具：Python 3.12、Torch `2.8.0+cu129`、Transformers `5.3.0`、Safetensors `0.8.0`。
  运行 receipt 绑定实际源文件 hash；先验证本地 B16 分支，不用它代签 Qwen3.8 或 MTP。
- Windows 参考进程设置 `SetErrorMode(0x8003)`，软件自带 DLL 闭包；不启动推理服务、改变电源模式、
  请求提权或结束用户 GPU 应用。重型运行和模型传输串行。

## 执行结果

实施中，已完成以下实际准备：

- Windows 用户原始文件与 WSL 本地副本：13 个文件、4,571,274,023 bytes 全部大小/SHA-256 一致。
  [Windows receipt](../../../../evidence/correctness/compute-llm-moe-c0/windows-original.json)、
  [WSL receipt](../../../../evidence/correctness/compute-llm-moe-c0/wsl-original.json)。两台 Mac 尚待接收。
- [B16 参考 receipt](../../../../evidence/correctness/compute-llm-moe-c0/b16-reference.json)：
  本地实际 320 个语言权重、24 层，prompt 1/17/128，各 8 步 teacher-forced decode；保存所有层
  每步末 token、完整 logits、最终 conv/recurrent/KV 状态，均 finite。RTX 4070 Torch 峰值 allocated
  3,811,440,128 bytes、reserved 3,888,119,808 bytes；这是该参考任务的观测，不外推其它 consumer。
  三个原始结果共约 98 MB 留在 receipt 指定外部缓存。实际执行脚本、cases 和模型源码 hash 已保存。
- 测试目录已冻结同源 token 与 132 个待量化 linear 权重（1,296,039,936 元素）；188 个高精度权重
  （585,785,152 元素）按名字保留，包含 embedding/tied head、norm/conv、decay/beta/output-gate
  投影与参数。工具分别校验和执行 reference，不把模型知识放入 HYLE 产品。
- 维护测试发现 Windows 的 `/name` 是 rooted 但 `Path.is_absolute()` 为 false，已按 anchor 拒绝；
  小数据实际覆盖复制、发布、复用、同大小损坏、非法路径和固定合同。未修改 selector 算法，只新增
  测试工具到现有 `tool_suites` 的映射。
- [三个来源裁决](llm-moe-c0-source-qualification.json)已绑定现有 clean/pinned SGLang、llama.cpp
  及官方实现：Qwen 的 swish 是 DeltaNet gated RMS，普通 Attention 另用 sigmoid；K3 的 W4A8
  activation 由 runtime profile 决定，不能用 BF16 路径代签；Qwen、DeepSeek、GLM 的 MTP 分别
  绑定真实数学分支，冻结的 Kimi config 则为 0 个 MTP head。这些结论只关闭来源疑问，未签产品通过。

上述均为输入/参考准备，没有新增 HYLE 产品执行证据，不给能力表或产品验证填 PASS。
有限能力明细与 C1 owner/card 仍在本包内推进。
