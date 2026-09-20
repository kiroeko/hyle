# M2-B1 CPU：成熟 Provider 的张量、归一化与状态接入

## Closure card

- component / owner：`compute_provider_ggml`、`compute_route_cpu`、`graph_route_cpu`；`/root`。
- baseline：`b2d2cfa68d1ac65483f8a2f091a9115cd0807baa`；Windows canonical checkout，工作树干净。
- 当前阶段：AUDITED。三个 owner 只实施下列 C1–C6；全部21行的完成门保持不变。
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

## AUDITED：资源与原版计算的批量承载

2026-09-20 UTC，SCOPED检查点 `e63a5d5a` 后核对锁定源
`ggml-456172ec733a-41b7e10fd424`、现有 Provider/region SPI 和 CPU 普通工厂。来源只查本包
接法与上游数学，未研究或比较冻结自研。以下是 backend-domain 审计，不是产品覆盖验收。

| 需求族 | 承载与源码事实 | 当前覆盖/动作 |
|---|---|---|
| F01 布局/复制/转换 | 公共 View 的 rank0..6 语义复用；ggml `ggml_cont/ggml_cast/ggml_concat`；CPU `ops.cpp` 的 DUP 同型 bytes 与 F32/F16/BF16 转换分派已存在 | PARTIAL/BORROW；用描述符资格化可表示 stride/折叠；不把任意 rank6 stride 写成 ggml 已支持 |
| F02 gather/scatter | `ggml_get_rows` 接 I32 索引并支持 F32/F16/BF16/I32；`ggml_set_rows` 接 I32/I64，`ggml_get_rows_back` 原版求重复累加 | PARTIAL/BORROW；动态索引/唯一 assignment、容量/重叠先检查；不让运营输入触发上游 assert |
| F03 浮点/门控 | 原版 `binary-ops.cpp` 的 add/sub/mul/div、`unary-ops.cpp` 与 `ggml_swiglu_split`；标量/行/同形广播及输出舍入可用成熟组合 | PARTIAL/BORROW；转换、repeat/cont 与融合边界计入成本；min/max/Select 不由同名近似拼成已覆盖 |
| F04 norm/归约 | `ggml_rms_norm/ggml_norm/ggml_sum_rows/ggml_sum`；F32 原子组合后显式输出 cast，保留 norm-before-gate 的舍入点；已验收 oneDNN leaf 不变 | PARTIAL/BORROW；L2 用原版平方/归约/加 epsilon/sqrt/div 组合，不能直接使用 epsilon 位置不同的 L2 primitive |
| F19 程序与状态 | `MetadataArena::for_graph` 与 `CpuExecution::prepare/run`；公共 shape、constant、effect、lease 和完成合同复用 | PARTIAL/FIX；不可变 recipe、每请求 graph/pool/plan/workspace，热路径仅绑定地址并执行固定程序 |

### 已确认不能直接代签的精确域

- I32 Scan：`src/ggml-cpu/ops.cpp:1440` 只分派 F32，其余类型 abort；不能以浮点扫描代替
  checked I32，也不能从 `ggml_backend_supports_op` 的 default-true 推导可运行。
- 稳定 TopK：`ops.cpp:8480` 的 argsort comparator 只比较值，未建立 ties/NaN 全序；
  `ops.cpp:8585` 的 TopK 还主动交换前两个结果。不能签本需求的 stable/sorted 两输出。
- all-masked Softmax：`ops.cpp:5674` 在指数和后要求 `sum > 0`，直接全负无穷不满足所需结果；
  不能把无 mask 的普通样例推成全域资格。
- 数组 min/max、Select、任意不可折叠 rank6 stride、受检查的浮点→索引转换，还需要具名成熟
  组合/后备接线。现有 oneDNN adapter 只接 MatMul/norm，不能假称已经接了整个 oneDNN 目录。

这些是21行中的剩余接入责任，**不是缩小 B1 分母、授权自研或关闭本包的理由**。
后备优先核对 M0 已定位的 IREE 标准 Scan/类型/选择程序的 CPU 承载；涉及新 factor/原共享
runtime owner 时先独立 QUEUED/SCOPED/AUDITED。该后备产品不包含在本次三个 owner 的授权中。
本卡仍负责汇总21行；当前先实施原版 ggml 成批接线，完成后再接精确域补位，M3-01 保持未完成。

### API / ownership / 成本冻结

- 新 expert factory 为 `compute::ggml_cpu::provider(Options)`，普通用户继续使用 CPU 工厂。
  `Options` 借用 `ParallelExecutorRef` 并显式设置 `native_threads`（默认1）；普通 CPU options
  同样增加这一线程预算。ggml pool 在 invocation 准备期建立，默认没有额外 OS worker。
- 外部 executor 的 `parallel_for(1, ...)` 执行一个完整 ggml 程序；不会把它解释成 ggml pool，
  也不会按其 concurrency 偷建同数量线程。显式 `native_threads > 1` 才取得额外持久 workers，
  同时设置 runtime 的 `uses_private_threads`；executor backing 活到全部派生对象销毁。
- 同步执行借 CPU Queue 的实际 admission/序列，原版 workers 全部 join 本次工作后才返回终态。
  未开始/已完成 CPU 调用的取消沿既有合同；不宣称可抢占一段正在同步计算的 native kernel。
- 一个 Compute 私有 `Recipe` 描述值/typed operation/端口；Graph 翻译放在 `graph_route_cpu`，
  不暴露 ggml 类型、不新增 Graph→Compute→Graph 环。独立 slot 保有 native graph 和可变指针。
- input/constant payload 借用，artifact 不在 prepare 执行它；每个 slot 预留中间值、workspace 和
  映射记录，绑定时只更新地址及依赖 view。不重建 graph/pool，不重搜 Provider，不进行 timing。
  能绑定原版输出 backing 时直接绑定；必要 cont/cast/repeat 均保留真实成本，不伪造零拷贝。
- 动态索引检查放在运行接纳/使用之前；需要检查内部生成数据的操作保留合法 region 边界，
  不用整个 region 名义跳过检查。状态/effect 不可证明兼容时不合并，Graph 继续管理真实顺序。
- Direct 保留既有合格 oneDNN leaf，其后尝试 ggml，再保留未改旧 fallback；旧 fallback 的执行
  **不计本轮成熟覆盖**。Graph 优先原版合法多节点 region；region-only 配置不抢单节点 leaf，
  独立叶配置按 Direct 顺序绑定。既有 ProviderInstanceId 区分配置，不改公共选择器。
- 这里只签明确支持域的成熟基线，不宣称普遍 SOTA；P1 仍比较相同成熟程序的完整成本。

### 有限 FIX

| ID | 允许的产品改动 | 验收 |
|---|---|---|
| C1 | CPU ggml factory、私有 recipe 与描述符/类型/参数表；复用现有 runtime，接原版 Copy/Cast/Concat、浮点 arithmetic/unary/gate、norm/归约和合格 indexed 域 | 原版调用与独立 oracle 的 Direct 表驱动差分；unsupported 精确拒绝，禁止进入上游 abort 分支 |
| C2 | 每 invocation 的 metadata、持久 pool/plan、原版中间存储及绑定；checked 索引/alias/映射错误/终态和复用 | OOM/部分构造、两个 slot、失败后资源安全、热执行无重建/分配；所有数据检查成本保留 |
| C3 | CPU Graph region 发现/翻译与同一程序生命周期，保留完整原版可融合片段和必要错误/effect 边界 | embedding→norm/gate→residual/state 与无模型图；改变 payload 不重编译，旧结果与两请求独立 |
| C4 | 两个普通 CPU 工厂与 `native_threads`，固定成熟 region/leaf 规则、原版执行器提交 | 普通用户入口、线程预算/调用线程/外部 executor 的真实执行，不要求用户 Provider 排序 |
| C5 | 本 factor CMake、中央实际依赖/公开头 owner、已有 vendor target/link/许可复用；无新依赖下载系统 | CPU-only、CPU-OFF、Graph/Compute 分离、源码/静态/动态与 relocation、无私有 C API 泄漏 |
| C6 | 对应测试、consumer、用户 guide、源码成本账单和独立候选复核；记录全21行逐域差异与下一成熟承载 | selector适用门、四环境CPU、插桩、三交付及必要P1；全部门未完不关闭/不填完整覆盖 |

SCOPED→AUDITED 为独立元数据检查点，随后才改产品。剩余精确域和后备审计继续计入同一总预算，
不因子卡改名续杯；实现细节调查不扩大上述公开/所有权合同。本包与全部256格仍未新增完整验收。
