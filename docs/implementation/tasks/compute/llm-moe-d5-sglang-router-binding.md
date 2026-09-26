# D5-F25：DeepSeek 完整成熟 CUDA 路由程序

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- baseline：`41bfdb2800e17aa852f181d726585726a2087313`，Windows canonical 工作树干净。
- owner：`compute_ops`、`compute_provider_cuda` 当前 AUDITED 范围追加本卡；不重开已关闭
  CPU/CUDA route。共用四路线测试消费者按本卡迁移完整片段观察边界。
- trigger：D5 MoE 批次 CUDA 数值通过，但 DeepSeek 普通 Softplus 命中冻结 HYLE 实现，
  两个聚合用例的成熟 Provider 断言失败；固定 SGLang 已有完整原版融合程序。
- consumer：DeepSeek V4 Pro/Flash，256/384 experts、top6、F32 logits、correction bias、
  sqrt-softplus score、偏置只参与选择、原分数归一化、1.5/2.5缩放。
- scope：接入原版 SGLang 完整 CUDA 程序；CPU/Apple/Vulkan 保留已接入成熟库组合，
  四路线验收同一片段的专家 ID 与权重及其完整 MoE 下游。路线程序不被强制拆成相同 kernels。
- non-goals：HYLE 数学/kernel、原版 Softplus helper 单独提取、改变通用 Softplus/TopK、
  通用 Provider/Queue/路由系统重构、新依赖版本、提前计时或重新比较冻结算子。
- 依赖：已登记 SGLang `06992c92ad4494fe9f9636e75be3e35040512447`、既有 CUDA shared stream
  和 prepared invocation、同一 in-tree 依赖登记及许可证。仅更新该依赖内容 hash，不改构建治理。
- P0：CUDA 一个原版 kernel、冷绑定 launch、无设备 workspace/热分配/路由搜索；输入输出由调用方
  保活至 Completion，可复用同一 prepared，独立并发 invocation；常数偏置转换显式计入 consumer。
- 验证：公共 schema 正/负例；CUDA 普通 Direct/Graph 完整四画像 router 与两个 GGUF MoE；
  tail/large logit、变化输入、近并列/并列、all-zero score、metadata/资源/alias；原四路线片段回归。
  同合同 ID 非并列时精确比较；并列时检查合法成员、无重复、原分数权重，不伪造稳定排序要求。
  稳定后统一 selector/三交付/P1；其它未改变批次证据复用。
- budget：一次有限接线批次、一次有具体失败的修正；不改变已有数值门。
  stop：完整 CUDA 路由成熟路径通过或得到精确拒绝；不把此结果签作全画像/全 D5 完成。

## 源码审计与语义

原文件 `python/sglang/kernels/jit/csrc/moe/moe_fused_gate.cuh` 的小 token /大 token kernels
都保留：score=`sqrt(max(x,0)+log1p(exp(-abs(x))))`，以 score+bias 选专家，以原始 score
求权重；sum>0 时归一化，否则权重零。shared expert 在模型消费者中独立计算，不启用 fused-shared。
准入固定 E256/384、K6；反偏置验证真实大 token 分支，避免仅复制 decode kernel。
原 CUDA 小 token 分支的 ties 不保证最低 ID；模型片段只保证按分数选择合法的 top-k 成员，
不继承通用 TopK 的稳定顺序合同。普通 TopK 完全不变。

公共 `model_ops::SqrtSoftplusRouting{scaling}` 是完整 region 的描述：F32 logits `[T,E]` 和
F32 bias `[E]`，输出 F32 weights `[T,6]`、I32 ids `[T,6]`，plain dense 非空、E256/384、
有限正 scaling。原版 CUDA 程序的融合 carrier 由已有 `sglang_provider()` 承载；其它路线
继续通过已验证成熟原子组合得到相同片段结果，不把这个 CUDA carrier 宣称为四路线通用原子。
官方 BF16 bias 若存在，由消费层准备时显式转成 F32，不按 artifact 标签把 dtype 一概改写。
片段输入的 logits/bias 是有限模型值；非有限数据不承诺有效路由，但不得产生越界访问。

SGLang 的完整数学定义使用正数 sum 归一化。旧共用 ggml recipe 的 F16 最小值 floor 不适用于
资格化极小正 score；DeepSeek 消费者使用 F32 最小 normal floor，仅为零分母提供保护。
sqrt(nonzero F32 softplus) 大于该 floor，因而不会改变任何非零 F32 score 的归一化。
这属于明确模型片段合同，不修改 ggml 或普通 Reduce/Divide。

原小 token kernel 先读取 `shared_original_scores[expert_id]` 再检查 expert_id，NaN 或极端
偏置可使选中 ID 为-1。有限安全修复仅把读取移入既有范围检查；正常输入算术、选择、归约和
launch policy 不变。保留来源与 patch 说明，不把修过的文件称为逐字节原版。

| FIX | owner | 有限决定 |
|---|---|---|
| F25.1 | compute_ops | 完整路由描述、元数据推导、公开边界；不公开中间 Softplus |
| F25.2 | compute_provider_cuda | BORROW 原两个 kernels、原 SUM 归约和 launch policy；仅隔离 namespace/host binding，补已证实读取越界保护 |
| F25.3 | 同一 owner | 现有 SGLang provider 冷准入、资源/alias/完成和 CMake 依赖内容登记 |
| F25.4 | D5 shared consumers | DeepSeek 原分数归一化与四路线完整片段结果；CUDA 用融合 carrier，保留其它成熟组合 |

## 当前 verdict

AUDITED；产品代码尚未修改。本卡不解除旧 CUDA FAIL，也不增加完整画像 ledger。
