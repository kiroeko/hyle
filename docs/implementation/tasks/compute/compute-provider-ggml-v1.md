# Compute ggml provider v1：CPU compiled-region 审计

## Closure card

- component / owner：`compute_provider_ggml` / Compute；round=`compute-provider-ggml-v1`；
- baseline commit：`7d684f033f4ee4cdc53eb40d7f422bd27e1b78c0`；
- trigger 与证据：Compute/Graph CPU 纵切已经关闭，但真实量化推理需要重复执行包含常量权重、block-quantized
  layout 和多个算子的 region。ggml `v0.23.0` / `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`
  公开 graph plan、backend buffer 和 CPU execution 能力；现有 HYLE 单-op provider SPI 与只按 element stride
  描述的 Tensor 不能未经审计就冒充这条 journey；
- 正常 consumer 任务：应用在 CPU 上冷编译一次固定拓扑、固定权重的推理 region，为动态 activation 和每次
  invocation 提供显式 storage/workspace，随后并发重复执行而不在热路径搜索、建图、JIT、分配或隐藏复制；
- 主产品纵切 / 具名来源：Qwen 类 Transformer MLP 的 RMSNorm、quantized gate/up projection、gated
  activation 与 down projection；对照 ggml/llama.cpp、ds4，并检查 vLLM/SGLang 的 weight/layout、动态
  batch/token 和 compiled-region owner；
- 正交反偏置纵切：无模型命名的 caller-owned dense f32 三算子工程 region，验证 API 不是 GGUF、Qwen 或
  某个 benchmark shape 的专用入口；
- 跨模块 public journey 与每段 owner：Mem 拥有原始 storage，Compute 拥有 Tensor/storage bridge、provider
  与 prepared region，Graph 拥有 logical DAG、partition/placement/liveness/alias/workspace 计划，ggml factor
  只拥有被分配给它的 compiled artifact 和 invocation state；模型加载、GGUF、KV cache、batching、sampling
  与 serving 仍由消费者拥有；
- scope：审计 CPU ggml region 的最小有意义合同、量化 layout/constant binding 的 owner、Graph region seam、
  ggml context/plan/buffer 生命周期、显式线程与 workspace、失败/cancel/drain、版本/许可证、依赖和三交付；
  只有审计证明可独立承载后才冻结有限实现集；
- non-goals：GPU backend、把 ggml scheduler/gallocr 变成 HYLE Graph owner、公开 ggml C 类型、完整 ggml API、
  GGUF/tokenizer/model runtime、KV/batching/sampling/serving、Native FFI、`OperationKind::kOpaque` 或字符串属性包；
- 直接依赖 / 直接消费者：依赖在 AUDITED 时冻结；候选方向是公开依赖 `compute`，实现可私有依赖 `graph` 或
  由 Graph 的 provider-neutral region seam 调用，但禁止 Graph/Compute 反向依赖本 factor。直接消费者为
  Compute direct compiled-region journey、Graph partition 和后续各 backend-specific ggml factor；
- 性能 tier 与原因：`P2` 为主，比较完整 repeated region 的 admission-to-completion；`P1` 只诊断 provider
  boundary、copy/workspace 和单 leaf，不用单算子点值决定 region API；
- 适用机器 / 仪器 / material margin：Windows W1-WIN 做 CPU P1/P2 qualification，固定 5%；Windows、WSL
  Linux、两台 Apple 做适用 CPU correctness、headers 和 package。GPU 不由本卡资格化；
- 调查与纠正预算：四个 consumer 各扫描一次；ggml public API、CPU backend、graph-plan、allocator、quantized
  type 与 build/license 各扫描一次；API/owner 最多两轮；实现根因最多两轮；候选后每个正式 workload 一次；
- 预声明验证：block layout/shape/stride/overflow/alignment、immutable weights、caller-owned activation、alias、
  workspace/OOM、unsupported、错误注入、artifact/invocation lifetime、并发、cancel/close/drain、差分、逐头、
  sanitizer、四机 CPU、source/static/shared、按 factor 选择、offline/license、P0 账单及合格 P1/P2；
- stop / blocker 条件：若有意义 region 必须先改变已关闭 Compute Tensor/constant binding 或 Graph partition
  合同，则停止本卡审计并形成精确 reopen trigger；若 ggml 热执行不能隔离隐藏 allocation/copy/global thread
  policy，或三交付无法给出一致 ownership，则记录 blocker，不能用 f32 三算子 wrapper 降低目标。

## SCOPED 待审计项

1. `compute_provider_ggml` 是否只拥有 CPU region；CUDA、Metal、Vulkan、HIP 的 ggml carrier 是否必须分别成为
   backend-specific factor，独立声明设备、buffer、queue 和完成合同；
2. provider-neutral region seam 属于 Compute 还是 Graph；如何让 direct region 与 Graph partition 共用实现，
   同时保持 Compute 不依赖 Graph、Graph 不依赖具体 provider；
3. 当前 `DataType`、`TensorDesc` 的 element-size/stride 模型能否诚实表达 ggml block-quantized type、row size、
   tensor byte span 和 alignment；若不能，最小上游合同是什么；
4. immutable constant weight 如何绑定到 prepared artifact；borrow/copy/prepack 各自的 lifetime、版本和失败语义
   如何显式表达，不能用普通 invocation input 隐藏 persistent packing；
5. ggml context、`ggml_cgraph`、backend graph plan、backend buffer、CPU work buffer 和 HYLE Mem workspace 的
   owner/销毁顺序是什么；能否让热执行保持零 HYLE allocation、零重建、零隐式 copy；
6. ggml CPU threadpool、abort callback、NUMA 与 HYLE scheduler/thread owner 如何组合；cancel、close、drain 和
   concurrent invocation 是什么语义，不能同时启用两套不受控线程策略；
7. ggml 错误/abort/assert/OOM 如何映射到 HYLE Error；哪些失败只能在 prepare 发现，哪些可能在 execute 出现；
8. source/static/shared、按 factor 选择、统一 dependency lock/cache、MIT license、符号隐藏和升级/fallback 如何
   成为一个一致产品，而不是链接 ggml 就静默改变默认 Context；
9. 公平 P2 如何让 HYLE route 与自然 ggml graph plan 在相同 dtype/layout、constant preparation、threading、
   storage、workspace 和 completion 边界比较；P1 只测哪些仍未由源码账单解释的边界成本。

AUDITED 前不得添加 ggml 源码依赖、公开头、产品 façade 投影、feature switch、测试专用操作或 provider 实现。

## 当前状态

本记录只冻结审计问题、预算和停止线。`compute_provider_ggml` 仍是空的内部 factor target；不存在可调用能力，
也不计入 `hyle::compute` / `hyle::graph`。能力表、有限 `FIX` ledger、API freeze、七问、candidate 和验证结果
将在审计得到源码事实后填写，不能由本 SCOPED 卡预判。
