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

## SCOPED 源码事实与有序前置条件（2026-09-12）

- ggml `ggml_tensor` 用 byte stride `nb[]`、`ggml_blck_size(type)` 与 `ggml_type_size(type)` 表达 block
  storage；`ggml_row_size()` 明确要求连续维可被 block size 整除。ds4 也直接按 block-elements、block-bytes
  和 row-bytes 校验 Q4/Q8/K/IQ 权重；这不是测试形状，而是两份真实引擎的持久权重合同；
- 当前 HYLE `TensorDesc` 把 `DataType` 同时当逻辑 scalar 与物理 element size，stride 以 element 计，
  `required_bytes()` 只能计算普通标量 storage。因此它无法无损表示同一个逻辑 f32 MatMul 权重的 Q4_0、
  Q4_K、IQ2 等不同物理编码，也不能靠谎报 `UInt8` 修复；
- ggml CPU 的高层 `ggml_backend_graph_plan_create()` 在冷路径用 `new` 拥有 work buffer，且只浅拷贝
  `ggml_cgraph`；但公开 `ggml_graph_plan()` / `ggml_graph_compute()` 允许 caller 提供 `work_data`。因此
  HYLE 可以保留 ggml metadata/tensor 生命周期并把 per-invocation work buffer 纳入 Mem，而不是接受热路径
  隐藏分配；
- ggml CPU threadpool 在一个 pool 中保存当前 cgraph/cplan，不能未经额外 owner 证明就宣称共享 pool 可并发
  执行多个 Invocation。首个实现必须把 serial 与每-invocation pool 做成显式档位；HYLE `ParallelExecutorRef`
  不能伪装成 ggml threadpool；
- vLLM/SGLang 的 fused gate-up、SwiGLU 与量化 scale tensor，以及 llama.cpp 的 `build_ffn()` 证明 region
  API 应保持 typed op/value 组合；不能引入 `QwenFfn`、GGUF loader 或字符串属性袋；
- CPU 与 CUDA/Metal/Vulkan/HIP 的 ggml buffer、queue、completion 和依赖不同。本 identity 只承载 CPU
  Compute leaf；Graph region adapter 和每个 GPU carrier 在需要时使用独立 factor，不能让 `hyle::compute`
  反向取得 Graph 或所有 GPU 依赖。

由此形成两个顺序明确的真实 reopen trigger：先由 `compute-v3-block-storage-format` 修复 provider-neutral
Tensor storage 描述；再由独立 Graph v3 卡增加 constant intent/binding 与 provider-neutral region partition。
两项关闭前，本卡保持 `SCOPED`，但这不是外部 blocker：可以按依赖顺序继续施工。

## 当前状态

本记录已完成第一遍源码审计并产生有证据的下层 reopen trigger。`compute_provider_ggml` 仍是空的内部
factor target；不存在可调用能力，也不计入 `hyle::compute` / `hyle::graph`。能力表、有限 `FIX` ledger、
API freeze、七问、candidate 和验证结果要在有序前置合同关闭后填写，不能用空 wrapper 预判。

## 前置完成后的阻塞裁决（2026-09-13）

Compute v3 的 block storage 与 Graph v3 的 constant/region seam 已进入当前实现，原先两个 HYLE 前置缺口
已经解除。第二遍审计同时检查 dependency lock 中的 ggml `v0.21.0` / `8599e0ea3756`，以及干净参考仓的
ggml `v0.23.0` / `e91ded11bdcd`。二者都能表达 HYLE 需要的 byte-stride/block weight、CPU
`buffer_from_ptr`、RMSNorm/MatMul/SwiGLU 和 caller-supplied `ggml_cplan::work_data`；Tensor/Graph schema
因此不再是 blocker。

阻塞位于 ggml CPU runtime 的公开 owner/failure 合同：

1. `ggml_init()` 即使收到 caller-owned arena，也始终用内部 `GGML_MALLOC` 创建 `ggml_context`；该 allocator
   在失败时调用 `GGML_ABORT`。全局 abort callback 只先收到文本，随后仍无条件 `abort()`，adapter 无法把
   正常 OOM 转成 HYLE `Status`；
2. `ggml_threadpool_new()` 内部调用 `ggml_aligned_malloc` 创建 pool/workers，当前实现随后直接解引用返回值，
   没有 caller allocator 或 fallible create 结果。多线程创建失败还走 `GGML_ASSERT`；
3. 不传持久 threadpool 时，`ggml_graph_compute()` 在**每次 submit**创建并销毁 disposable threadpool；这会
   隐藏 allocation、thread lifecycle 和潜在 syscall，违反本卡 steady contract；
4. 高层 CPU `graph_plan_create` 自行 `new` plan/work buffer，并且只浅拷贝 `ggml_cgraph`。它既不能使用 Graph
   分配给 Invocation 的 workspace，也不能独立解决 graph/tensor metadata 与并发 Invocation 的生命周期；
5. 包含 `ggml-impl.h`、手工构造 `ggml_cgraph`/threadpool，或复制 CPU kernel 可以绕开部分问题，但会依赖
   私有 layout/静态函数并形成 HYLE 自己维护的 ggml fork。这不是本卡准入的“可升级第三方 region
   provider”，也会重演旧 synthetic metadata bridge，故明确 `REJECT`。

因此当前公开 ggml API 无法同时兑现“可恢复运营失败”“并发 Invocation owner 明确”和“steady submit
不隐藏分配/建线程”三项已接受的 Compute/Graph 合同。将线程数降为 1 也不解决 context/pool allocation 与
disposable-pool 问题；把冷路径 abort 写成限制会破坏 HYLE 工程章程的 OOM/failure 门，而不是诚实的可选
性能取舍。

本组件从 `SCOPED` 裁决为 **`BLOCKED`**，`blocked_from=SCOPED`。它仍保持空 factor：不下载、不链接、不
导出 header、不进入 product façade。恢复条件是一个新的精确 ggml 版本提供可验证的 fallible、non-aborting
caller-owned context/threadpool/plan seam，或者 HYLE 另行登记并审计一个明确维护成本、升级策略和三交付
合同的 source patch factor；恢复后必须重新从 `SCOPED` 冻结有限实现集。Graph 的真实 multi-op region
资格转交 oneDNN Graph 独立 factor，不让本 blocker 污染 Compute/Graph 主闭包。

## 跨 backend 需求校准（2026-09-14）

后续“借 ggml 的算子广度，同时由 HYLE 保留全局计划”的需求不解除上述 **CPU** blocker。锁定版本的
公开 backend graph submit、event 与 provider-owned buffer 足以让 CUDA/Metal/Vulkan delegated region 各自
形成独立审计候选，但这些 backend 的 memory、stream/queue、completion、错误和 build SDK 合同互不等价。
因此不得把本通用名称解释为全部 ggml 能力，也不得以 CPU threadpool 问题否决 GPU 路线。

预发布阶段的正确迁移是：本 record 继续只记录既有 CPU 审计事实；真正开工时新增 backend-qualified
component identity（如 `compute_provider_ggml_cuda`），以公开 API 的单节点或多节点 compiled graph 为
自然复用单位。只有 adapter 实际完成 HYLE Operation/region、storage boundary、失败和 Completion 翻译后，
该 factor 才拥有能力。调用 `ggml_cuda_compute_forward`、`ggml_cuda_mul_mat` 等私有实现符号不属于普通
adapter；若 material 证据要求叶级替换，应另立版本锁定的 source-patch/kernel factor。
