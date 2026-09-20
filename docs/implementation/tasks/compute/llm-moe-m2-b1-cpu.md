# M2-B1 CPU：成熟 Provider 的张量、归一化与状态接入

## Closure card

- component / owner：`compute_provider_ggml`、`compute_route_cpu`、`graph_route_cpu`；`/root`。
- baseline：`b2d2cfa68d1ac65483f8a2f091a9115cd0807baa`；Windows canonical checkout，工作树干净。
- 当前阶段：SCOPED。只恢复本卡三个 owner；有限 FIX 与 API 尚待下面的有界审计冻结。
- trigger：M0 的 B1 真实消费要求普通 CPU Direct/Graph 承载 F01–F04/F19，共21行。
  现有两个普通 CPU 工厂只组装 oneDNN/native，`compute_provider_ggml` 仍是无产品实现的身份。
- 原 BLOCKED 恢复事实：[M1-G](llm-moe-m1-ggml-runtime.md) 已以 `04368a39` 完成独立
  `compute_ggml_runtime` source-patch 的版本/维护/升级、OOM、线程启动回滚、并发与四环境三交付。
  [原卡](compute-provider-ggml-v1.md) 的可恢复 context/pool/plan 前提已满足，依法从 BLOCKED
  回到本卡 SCOPED；不回写旧卡，不直接继承普通 Provider 可用或性能结论。
- 主纵切：目标模型的 embedding→norm/gate→residual/index/state，使用已冻结宽度、头布局和
  动态索引；两次请求重复执行、保留旧结果、共享只读常量，各自拥有可变状态。
- 正交纵切：无模型注解的同类工程张量图，包含广播、非连续布局、两输出、明确错误和关闭；
  普通用户只选 CPU 路线，不能要求排序 Provider、提供 timing profile 或编写 ggml glue。
- scope：21行需求在 CPU 的原版算子/原版组合、统一私有 recipe、Direct/Graph 批量 lowering、
  固定合法默认绑定、原生 CPU 线程/资源、失败/完成、三交付与必要接入成本验证。
- non-goals：新增/研究/优化/比较任何冻结自研数学；其它路线实现；模型 loader/推理服务器；
  公共 selector/性能仪器或 Sched/Mem 重写；重启 oneDNN Q2 旧性能采样。
- 直接依赖：复用已关闭的 Compute/schema、CPU backend、Graph region、Mem、Error 和
  `compute_ggml_runtime`。准确新增边随 AUDITED 冻结；Compute 不依赖 Graph。
- Graph 翻译 owner：普通 CPU route 是本包实际消费者，region 翻译留在 `graph_route_cpu`，
  Compute 私有 recipe 不含 Graph 类型；目前没有必须新增独立 Graph factor 的消费者。
- 性能：P0 先核对准备/执行、常量、workspace、转换、检查与完成账单；不可变候选后进行一组
  必要 P1，比较 HYLE 成熟路径与同一原版成熟完整调用，非劣门5%，不比较冻结 HYLE 算子。
  A/A、双顺序、相同线程与生命周期先资格化；最多15对，只有原仪器合法要求时总上限31对。
- 预算：源码/设计总2小时；前置只读定位保守扣除30分钟，本轮合同与入口核对再预留10分钟，
  新增定向源码/设计审核上限80分钟；不是新的2小时。每个具名产品根因最多两次纠正。
  正常构建和适用测试不冒充调查；到预算固定结论/具名缺口，不无限扩来源或转自研。
- stop：21行 CPU 必需域及全部适用门完成后关闭；上游不接受或数值不相容须先给出原版反例，
  核对已准入成熟组合/后备。未接通仍未完成，不削减需求分母。进入 M5 前暂停等待用户。

## 必须在 AUDITED 固定的决定

1. 单节点 Direct 与多节点 Graph 共用一份私有 operation recipe、类型/布局/参数资格和原版程序；
   不逐算子建立互不相干的 adapter。Graph 只翻译 region 的端口、常量、状态和边界。
2. 默认 caller-thread；明确线程预算、持久 pool 的 owner、用户 `ParallelExecutorRef` 的组合或
   拒绝规则。不能忽略 executor，也不能在热执行临时建线程或把外部 executor 冒充 ggml pool。
3. Q-SHAPE：rank0..6、正 stride/空张量、广播、动态索引、重复 scatter、checked I32 scan；
   ggml rank4 用合法折叠/分段或具名成熟组合证明，不能单凭同名 API 宣称覆盖。
4. Q-NUMERIC：F32/F16/BF16 转换特殊值/舍入、stable unary、L2 epsilon、gated norm 舍入顺序、
   all-masked softmax、TopK ties/NaN/两输出；以 C0 需求和独立 oracle 为准。
5. artifact/slot 的 metadata、plan、pool、workspace、常量和输入/输出借用；两请求独立，
   一个 slot 不并发 run/close，真实终态后才复用/释放；动态失败不发布成功输出或推进后段状态。
6. 准备期应用已固定模型/硬件/软件条件及合法通用基线；执行期不查表、计时、探测或重建图。
   全部持久/每请求资源进入准确成本与配额；未知不是零。
7. 一个 CMake 接入位置复用既有 vendor/runtime owner、lock/cache、许可与 package 投影；
   CPU-OFF 不取得 CPU Provider，私有 ggml C API 不泄漏到用户头和动态导出。

## 需求账本与保留边界

需求真相为 [原矩阵](llm-moe-coverage-matrix.json)；来源与承载候选为
[M0 映射](llm-moe-provider-map.json)和[接入清单](llm-moe-provider-integration-ledger.md)。
本卡覆盖 F01.1–3、F02.1–4、F03.1–4、F04.1–6、F19.1–4，共21行；先审成熟调用的确切域，
再冻结有限 FIX。此处未产生新的 BOUND、correctness 或完整覆盖 verdict。

`compute_provider_onednn` / `graph_provider_onednn` 的旧 Q2 BLOCKED、恢复条件与预算保持；
仅可消费不变的已有能力。`compute_ggml_runtime` 继续 CLOSED，只有具名新资源缺陷才定向重开。
所有 CPU/CUDA/Metal/Vulkan 自研算子保持冻结；SIMD 基础与通用能力保留。

## 预声明验证

- 21行的正常/边界/数值差分，主纵切和无模型注解纵切；Direct/Graph 使用相同输入与 oracle。
- 形状/格式/容量/alias/索引与溢出拒绝；OOM/部分构造、两 slot、旧结果、取消/关闭/失败回收；
  不能因快路径跳过必要动态检查。已验收且身份相同的共同资源故障证据复用。
- Windows、WSL、Air、Mini 的 CPU 正确性；受影响 host 路线 ASan/UBSan/TSan，精确说明覆盖边界。
- 一个真实组合 consumer 批量覆盖 source/static/shared、relocation、隐藏 producer、许可/导出、
  GPU-OFF 与 CPU-OFF；不逐算子重跑同一交付接缝。重型串行、parallel2，DLL 邻置，无 UAC。
- selector 选中门及七门、独立一次候选复核；文档/状态只运行 docs/closure，不启动 native/P1。
- P1 的具体 workload、真实完成边界、静态账单和原仪器身份在 API/正确性固定后登记；未资格化
  不启动完整样本，未运行不得写 PASS，不用本卡解除旧 Q2 失败。

## 当前裁决

SCOPED；无产品实现授权。下一步在剩余80分钟审核预算内固定有限 FIX 和 API/资源合同，
独立提交 AUDITED 检查点后再写产品。M2-B1 CPU 与全部256格均未新增完整验收。
