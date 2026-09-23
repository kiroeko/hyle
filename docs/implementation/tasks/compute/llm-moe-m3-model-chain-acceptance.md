# M3 四路线真实模型计算链验收

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- round / owner：M3；`/root`。
- baseline：`da62ec9b9f7ee588a3e25918cd7dd6956514711a`；[M2 终局](llm-moe-m2-terminal.md)
  已关闭，Windows canonical 工作树在本卡登记前干净。
- trigger：M2 已把全部准入成熟 Provider 接入普通 Direct/Graph，但 64 个需求域 × 四路线仍只有
  21 个 Apple B1 格有完整 PASS；M3 必须用真实模型消费链区分已覆盖、上游缺失、设备拒绝、
  数值/状态不相容与验证未运行。
- stop：完成 M3-01--M3-06 六组模型链、M3-07 的 256 格审计和 M3-08 Provider 接入完成门；
  未运行项不写 PASS。按用户本轮要求在 M3 终局后汇报并停止，不进入 M4。

## 权威输入与边界

- 唯一模型输入是用户本地
  `C:\Users\kiro\Downloads\Qwen3.5-2B`，source ID
  `4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155`；配置为 24 层语言主干、
  18 个 linear-attention 层和 6 个 full-attention 层，hidden 2048、head dimension 256。
- 旧 C0 B16/量化收据只证明当时参考与派生算法；当前外部制品不存在，必须从用户原始文件按相同
  hash/recipe 重建并出新收据，不能把历史路径写成当前 PASS。制品放在短验证根，跨机只同步同源
  文件，M3 关闭后清理。
- 模型解析、tokenizer、采样、服务调度与推理 engine 只属于测试消费者，不安装进 HYLE SDK。
  产品继续只公开模型无关的 typed Direct/Graph 计算、原生资源、程序和完成对象。
- CPU/CUDA/Apple/Vulkan 自实现数学继续冻结。允许修复的产品缺陷仅限真实消费暴露的成熟 Provider
  schema/lowering、资源/完成、构建/交付和既定冷路由；不得新增 kernel、shader、decoder、量化器
  或用冻结实现兜底。

## SCOPED 批量施工单元

六组共享一份测试侧 safetensors/config 读取、固定 token、独立 oracle、权重/状态 owner、错误注入、
两 invocation 和结果收据。最小施工单元是“模型片段 × 路线成熟 region”，不是单个算子。

| 批次 | 真实消费 | 预期成熟承载 | 关闭条件 |
|---|---|---|---|
| M3-01 | embedding→norm/gate→residual/index/state | M2 B1 四路线 region | F01--F04/F19 的普通 Direct/Graph、alias/失败、两次运行和三交付逐格裁决 |
| M3-02 | dense/量化 FFN、GQA prefill→decode、RoPE/KV | oneDNN/ggml/cuBLASLt/Marlin/FlashInfer/MLX/IREE 已接域 | 同 codes/scales 的 B16/Q4/Q8、真实 head256 与 KV 状态；转换和驻留计费 |
| M3-03 | 18 层 conv/DeltaNet/KDA 与 6 层 GQA 组合 | ggml/MLX 原版 recurrent 与 attention region | 本地 2B 24 层、prompt 1/17/128、8 步 decode、257-token chunk、非零初态/reset |
| M3-04 | 四个目标 MoE 家族的 routing→expert→combine、mHC/AttnRes/MTP | ggml 原版 grouped expert/mHC 与已有组合 | 真实登记宽度的缩小容量片段；空/偏斜专家、ties、共享状态和 MTP carry |
| M3-05 | MLA/DSA/CSA/HCA、shared/pooled index/tail、压缩状态 | ggml/FlashInfer/MLX/Vulkan 已接成熟组合 | Pro/Flash 配置、跨层索引复用、ragged tail、窗口重叠和非零增量状态 |
| M3-06 | FP8/MXFP4/FP8 专家与 index | 已接 IREE/ggml/MLX/CUDA 精确相容域 | 同 codes/scales、独立 bit decoder、量化质量与实现误差分开；不相容设备留具名反例 |

## 审计阶段允许动作

1. 核对已有公开 Direct/Graph 工厂、路线 region、测试支持与 CMake factor，形成一张有限
   `KEEP/BORROW/FIX/DEFER/REJECT` 表；只读模型 config/header 和已登记源码证据。
2. 重新验证本地 13 个输入文件 identity；按需重建 B16 oracle 与 Q4-S128/Q8-B32/F8-B128 制品，
   不下载模型、不启动服务、不把整个模型复制进仓库。
3. 固定测试消费者的 shape、权重集合、误差、资源与三交付批次。开发期按路线一次增量构建；
   Windows 产品真相，WSL/Air/Mini 只作验证副本，重型运行串行、并行度最多 2。
4. 只有审计证明现有成熟接缝无法表达真实链时，才把具名适配缺陷加入有限 FIX；产品改动前本卡
   必须转为 `AUDITED`。关闭组件没有满足 reopen trigger 的事实时不重开。
5. 更新覆盖矩阵只接受不可变候选的实际收据。`PARTIAL` 不冒充 `PASS`；Provider/硬件不相容和
   未运行验证分别记录。P1 只在完整片段/API 固定后运行，不与冻结自实现竞速。

## 预声明验证

- 输入与 oracle：13 文件 size/SHA-256、safetensors header、固定 tokens、所有层末 token/logits/state；
  缺权重、非 finite、hash 变化明确失败。
- 每条适用路线：Direct 与 Graph、两 invocation、变化输入、旧结果、状态隔离、失败不发布、关闭与
  quota 回收；路线原生 buffer/array/stream/queue/completion，不允许 host payload fallback。
- 交付：复用未变 M2 package seam；一个真实 M3 组合消费者分别执行 source、installed static/shared。
- 成本：P0 记录权重 packing/decode、workspace、上传下载、同步、region 边界和压缩驻留；稳定的完整
  模型片段才按卡启动一次适用 P1。
- 维护：change-impact selector、`check-module-closure.py --previous-ref`、`check-docs.py`、统一
  Compute+Graph LOC；阶段结束清理短构建根、传输包和派生测试制品。

审计尚未完成，本卡当前不授权产品实现或覆盖 PASS。
