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
- all-masked Softmax：`ops.cpp:5676` 的标准 `assert(sum > 0)` 在 Debug 是运营输入风险；
  Release 的全负无穷 NaN 与公共结果一致，不能误写成要求全零。完整 mask/类型/LogSoftmax 尚待资格。
- 数组 min/max、Select、任意不可折叠 rank6 stride、受检查的浮点→索引转换，还需要具名成熟
  组合/后备接线。现有 oneDNN adapter 只接 MatMul/norm，不能假称已经接了整个 oneDNN 目录。
- 首轮接线的特殊值核对进一步固定：原版 ReLU 将 NaN 变为0，SiLU/GELU及相应 gate 在
  负无穷上不满足已登记极限语义，Sigmoid/Softplus 的极小尾值也不能直接代签。普通 ggml
  factory 暂不接这些模式，不用有限正常值样例缩小原需求，也不插入自研数学修补。
  plain byte stride 还须满足原版元素对齐；奇数字节 stride 保留为后续成熟承载责任。

这些是21行中的剩余接入责任，**不是缩小 B1 分母、授权自研或关闭本包的理由**。
后备须先核对 M0 已登记的 CPU-I 标准库 scan/sort 与 IREE 标准程序；涉及新 factor/原共享
runtime owner 时先独立 QUEUED/SCOPED/AUDITED。IREE CPU 产品不包含在本次三个 owner 的授权中。
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

## 实现中检查点

CPU 普通工厂、ggml 私有 recipe/Direct、Graph region 翻译和每请求持久 pool/plan 已开始实施。
动态索引的检查表在准备期生成，运行仅检查当前 payload；原版 repeat 直接声明输出尺寸，不为
广播额外分配同大小占位 tensor。Windows MSVC shared、CPU-only、oneDNN-OFF 的7个开发验证
目标已通过：ggml 6项、普通 Direct 5项、普通 Graph 9项用例，另有3个头自足检查及一个实际
executor consumer。覆盖归一化/舍入、rank6广播、strided copy/concat/gather、越界保留输出、
两 slot 重放、OOM回收、多输出原版 region；旧 fallback 通过不计成熟覆盖。
尚未产生完整 selector、跨平台、三交付或 P1 verdict；整个21行包仍为 AUDITED、未关闭。
对应原始日志和产品/测试hash见[开发检查点](../../../../evidence/correctness/compute-llm-moe-m2-b1-cpu/README.md)。
源码/设计预算累计保守记72分钟（前置40、初审17、本次实现疑问定位15），仍余48分钟；
本次构建与用例编写不算新的调查预算，不因后备卡改名重置。

## C1-I 重新定界与有限审计（69351e26 后）

本包21行、三个 owner、公开 API、资源/完成合同不变，owner 保持 AUDITED；先将剩余 indexed
承载固定为下面的有限增量，再开始产品代码。未宣称 whole B1 完成，未修改任何 CLOSED owner。
M0 `llm-moe-provider-map.json` 的 **CPU-I** 已登记 C++ 标准 scan/sort 加 checked host-index
边界，owner 就是 `compute_provider_ggml`。因此 F02.4/F04.6 先接这个既定成熟承载；IREE
local-sync/embedded-ELF 的有限源码核对保留为后备，没有下载/构建新工具或新增 runtime factor。

- 数学算法 BORROW：非并行 `std::inclusive_scan`（I32 先升 I64、再检查所有前缀与最终总和），
  F32 沿原标准 scan；`std::partial_sort` 选择 value/index 全序前缀。排序比较器只绑定公共
  largest/smallest、NaN-last、相等时原下标的语义，不实现排序/scan 内核。
- MSVC 14.50.35717 的 `<numeric>:315–359` 和 `<algorithm>:9573–9593` 已定位原实现；
  后者在给定数组上进行原版 heap/select，无隐藏稳定排序临时分配。其余标准库仍需随平台验证。
- HYLE glue 只负责当前输入的 byte-stride 读取、I32 范围检查、原版算法调用、结果发布和生命周期。
  I64 scan 的长度上限沿现有 INT32_MAX，总和可证明不会溢出 I64。溢出先于输出映射/写入返回。
- 保留独立叶边界：Scan 及两输出 TopK 不塞入 ggml 单输出 recipe。一个不可变参数制品和
  每 invocation 固定 scratch，经现有 CPU Queue 与借用 executor 同步执行；没有私有线程、
  没有热期建图或标准库并行执行策略。输出不 alias，TopK 返回原 storage bits。
- F32/I32 Scan、F32/F16/BF16/I32 TopK；rank1..8、非负 byte stride、任意合法 axis、空非选中
  轴与尾部、ties/NaN/Inf/±0、两个输出、两请求、重复变输入和失败后安全由本卡验收。

| FIX | 有限动作 | 验证 |
|---|---|---|
| C1-I1 | 在现有 CPU Provider 增加 private standard-index program，复用公共 Scan/TopK schema与原标准算法 | 独立宽整数前缀及稳定全排序 oracle；所有适用数据检查保留 |
| C1-I2 | 冷 scratch/stride 参数、内存配额、无私有线程、热复用与明确终态；Graph沿已有leaf SPI | 溢出输出不变、两输出/ties/NaN、OOM、两 slot、无热 HYLE allocation |
| C1-I3 | 普通 CPU Direct/Graph 固定绑定和既有 CMake factor 内部源文件登记 | 普通入口无用户 Provider 排序；原三交付/四环境/P1门仍须完成 |

此增量不包含稳定浮点激活、Select/min/max、L2、Scatter 或新 IREE CPU 接入；它们仍属于21行
剩余责任。本次定界扣6分钟，源码/设计累计78分钟、仍余42分钟；不重置，不启动冻结自研的比较或研究。

### C1-I 实现中检查点

原版标准算法的私有叶程序、普通工厂和 Graph 两输出/Scan→Gather 接线已落地。Windows
CPU-only、oneDNN-OFF、shared 下7个定向目标通过（Provider 11、Direct 6、Graph 11项用例，
另有3个头自足检查及实际 executor consumer）。覆盖非末轴/奇数字节 stride、I32 中途与
最终总和溢出而输出不变、F16/BF16/NaN/±0位保留、稳定 ties、大 I32 比较、两请求重放和配额失败。
该记录只签开发检查点；全21行、所有环境/交付、完整 selector与P1仍未完成，不关闭 owner。

## C1-L 有限接线审计（4776264a 后）

保持原 C1–C6、三个 AUDITED owner 和21行验收范围。此次只补已定位的原版承载，不扩张数学内核：

| FIX | 原版来源与接法 | 必须验证的域 |
|---|---|---|
| C1-L1 | `ggml_get_rows` 接 `[inner, extent, outer]`；`ggml_repeat_4d` 复制 I32 索引到各 outer 行，CPU `ops.cpp:1796–1811` 明确支持 I32 repeat；源和索引的非连续布局先沿原版 cont | Gather 任意合法轴、标量/矩阵索引、重复/首尾/越界、rank6 dense、已准入 aligned stride、空输出；所有当前索引先检查再发布 |
| C1-L2 | `binary-ops.cpp:49–112` 支持逐行标量；将公共列广播正确折为 `[1, rows]`，由原版 binary 或 lhs repeat 执行。L2 直接复用 Cast、乘法、Sum、epsilon、Sqrt、除法组成 Graph region | FP32 累积与末尾显式低精度舍入；零/极小行与 epsilon 在根号内；两 slot、变输入重放和无热 HYLE allocation；单 region 的普通入口 |

CPU `ops.cpp:5105–5216` 的 get_rows 分派支持 F32/F16/BF16/I32；I32 使用原版同宽复制，
不经过浮点数值转换。Gather 仍保留 checked leaf 边界，不能把动态 payload 检查吞进无检查 region。
索引 repeat 的 `outer * index_count * sizeof(I32)` 中间存储、cont 和低精度 cast 均是实际成本；
不能宣称所有索引布局零拷贝。公共 shape/alias/bounds 合同不变。

另确认本卡早先所列 `get_rows_back` **不能直接覆盖 Scatter add**：它先归约更新再合并 base，
而公共合同要求按逻辑顺序更新，F16/BF16 每次相加后立即舍入。即使最终 dtype 相同也不等价。
`ggml_set_rows` 对重复目标的并发写也不能代签 last-update-wins。Scatter 保留为具名成熟承载
缺口，本增量不接、不计覆盖，不以自己实现数学补洞。

本次及前一检查点尚未入账的定向源码核对合计保守扣12分钟；源码/设计累计90分钟，剩余30分钟，
总预算仍为120分钟。实现和预声明验证沿原预算/仪器；不触及冻结自研，也不重开旧 oneDNN Q2。

### C1-L 实现中检查点

原版任意轴 Gather、非连续索引检查、列广播与普通 Graph L2 组合已落地。Windows 同一
CPU-only、oneDNN-OFF、shared 开发配置的7个目标通过（Provider 14、Direct 6、Graph 12用例）。
Gather 验证四种 storage、rank6 dense/非连续源、标量/矩阵/非连续索引、重复/首尾、变 payload
重放、越界输出不变和空索引；大 I32 未经过浮点数值转换。L2 验证 F32/F16/BF16、宽1/7/31、
零/极小行、根号内 epsilon、中间归约结果发布、一个原版 region、两 slot 和无热 HYLE allocation。
原始日志、失败修正和文件身份见 `layout-windows.json` 及其 archive；这里只是开发检查点，
不是完整21行、所有平台/交付/插桩或 P1 的验收。其余精确域保持，不增加256格完成数。

## C1-B 广播与归约的有限审计（b4094ca7 后）

仍在 C1–C6 和三个 AUDITED owner 内，不改变21行分母、公开 shape/数值合同或原 runtime。
`ggml.c:2584–2607` 的 repeat_4d 为每次广播产生真实原版节点；`3854–3873` 的 view_4d
接收 size_t strides，可表达分组转置，再沿原版 cont 物化。没有修改或复制任何数学内核。

| FIX | 冻结接法 | 验证 |
|---|---|---|
| C1-B1 | 对公共 inference 已确认合法的 Add/Sub/Mul/Div 广播，保留同形/标量/行/列原快接法；其余按逻辑轴折为 `[inner, 1, outer]` 并调用原版 repeat，再接同一 binary | F32/F16/BF16、rank0..8已准入布局、交错广播和两操作数顺序；独立坐标 oracle，输入不改写 |
| C1-B2 | Sum/Mean 先将相邻选中轴分组，用原版 view_4d/cont 稳定移至末尾，再调用 sum_rows/mean；half 显式 F32 累积/末尾舍入。空集合用原版 repeat 发布0或NaN | 任意正负轴集合、keep-dim、rank6全部非空轴子集、非连续输入、空集合/空输出、scalar；普通Graph组合及重放 |

只生成冷布局和原版程序，执行不排序 Provider、不建图或计时。每个实际转置/广播的中间 tensor
由同一 invocation allocator 冷分配，生命周期与配额不变；多次物化属于真实成本，不能记作零拷贝。
一般广播每轴至多一侧扩张；归约至多每组一次 cont，原 metadata 容量上界仍覆盖这些节点。
本轮设计及定向源码核对扣12分钟，累计102分钟、剩余18分钟，总预算不重置。其它成熟补位仍待
完成，自研冻结及旧 oneDNN Q2 条件不变。

### C1-B 实现中检查点

任意合法广播和指定轴 Sum/Mean 的原版组合已接入。Windows 同一开发配置的7个目标通过，
包含 Provider 17、Direct 6、Graph目标13项用例执行。归约独立坐标 oracle 覆盖 rank6 的63种
非空轴集、两种 keep-dim、Sum/Mean 和三个浮点类型（756组），另有非连续输入、scalar、
空集合与空输出；交错广播覆盖 rank6/rank8、不同 rank、左右操作数顺序和源数据保留。
普通 Graph 的分组均值→广播相减→方差在一个原版 region 内执行，双 slot 重放及无热 HYLE
allocation 通过。原始记录为 `broadcast-windows.json` 与 archive；不签全21行、跨平台/三交付/
插桩/P1，也不增加完整覆盖格。API/算法尚未全部稳定，未启动性能采样。

## M2-D1：oneDNN 新能力独立接入（已批准）

2026-09-21用户明确批准：“旧性能 FAIL 保留；新能力按独立任务完成接入和验收，后面还是要查清楚
为什么差距这么大”。两个oneDNN owner据此只为[新能力任务](llm-moe-m2-b1-onednn.md)进入SCOPED；
旧Q2仍为性能资格FAIL/产品性能INCONCLUSIVE，不声称其恢复条件已经满足。先AUDITED再改产品。
当前 Q2 的 Direct A/A 两边确实使用同一 `DirectRoute::run()`、同一 oneDNN factory 和相同
submit/wait 边界；本次只读核对未找到新的公平性缺陷，不追加计时。旧失败及恢复条件继续有效。

模型仍需的 Select、比较/掩码与一般布局可以进一步资格化当前锁定 oneDNN 的成熟接口：
`dnnl.hpp` 已有 `binary_select`、比较和 min/max，以及原版 reorder。`math_utils.hpp` 的
soft_relu 使用 log1p/exp；logistic 会截断部分负尾，swish 在负无穷仍有不相容，因此不能因
解除流程阻塞就宣称全部数值域已通过，也不直接用同名 primitive 代签稳定合同。

用户批准的新能力与旧资格分开推进，有限卡固定为：

1. 先对 Select/比较、reorder、softplus 及必要原版组合做类型/布局/特殊值资格；只接实际合格域。
2. 复用既有 oneDNN engine、stream、私有 primitive 与每请求资源路径，接普通 Direct/Graph；
   原 norm/MatMul 数学和旧 Q2 benchmark/门限/FAIL 记录保持，不启动旧自研比较或淘汰。
3. 新映射仍完成正确性、动态错误/资源、四环境、三交付与同一成熟完整调用的成本验证；
   完整语义缺口继续记账，不能以新 scope 名字变成 PASS。
4. 新任务依法SCOPED→AUDITED；调查仍计入本CPU B1总预算（118/120分钟，余2分钟），
   复用已登记源码定位及资格方案，不借换卡续杯。正常原版资格测试、构建/实现单列。

这次状态范围调整仅依据上述用户决定，不改变[旧Q2终局](llm-moe-c1-normalization-aa-qualification.md#q2-终局blocked)
的门限、原始记录或采样预算。新能力仍须实际验收，不能由改状态取得性能PASS。
用户同时要求后续定位同实现A/A差距，已登记[独立原因调查](llm-moe-onednn-aa-cause-followup.md)；
先推进成熟接入，后续有界调查，不提前启动冻结自研比较或重复旧实验。

### 有限原版数值核对

Windows Release 直接链接本轮已构建的原版 ggml 静态库，仅运行无计时的独立语义探针：

- 全负无穷输入的 Softmax 返回三个 NaN；原先“公共要求全零”的解释已由下面合同复核撤回。
- get_rows_back 后加 base 的 Scatter-add 组合，在 F32 base=2^24、F16 base=2048、BF16
  base=256，重复更新 `[1, -base]` 时均返回1；公共逐次更新/舍入合同均要求0。
- 原版 SiLU 在负无穷返回 NaN，ReLU 在 NaN 返回0；Sigmoid(-100/-90) 与 Softplus(-20)
  另观察到小尾值截断。尾值的最终资格还须按适用误差门裁决，不能把“有数值差异”直接等同
  于所有消费者不合格。原版 exp(-100/-90) 能返回非零值，保留为后续成熟组合的源码事实。

该探针不调用冻结 HYLE 算子、不计模型覆盖，也不执行旧 Q2 或任何性能采样。初次独立链接
缺少 Windows Advapi32，补齐系统库后同一探针正常结束；没有产品或 vendor 修改。
源码、命令、库身份与完整输出见 `remaining-numeric.json` 和原始 archive。oneDNN 范围决定
仍待用户回复，当前两 owner 保持原 BLOCKED；此处没有自动恢复或更改统计门。

本轮只读源码/方案核对记14分钟，累计116分钟，仍余4分钟；独立数值验证与构建单列，不进行
无预算的来源扩张。现有21行完成门不变，未运行的全包验收不写 PASS。


### 公共 Softmax 合同复核（2026-09-20）

CUDA 接入核对共同语义时发现：当前[Compute 指南](../../../user/compute.md)明确规定无有效值行
返回 NaN，F04.5/C0 没有改成全零。此前将原版 Release 的三个 NaN 判作数值不兼容是记录错误，
已纠正 `remaining-numeric.json` 并保留原解释、原探针/库身份与原 archive，不重跑取有利结果。
原版 Debug 的标准 assert、尚未接线及其它 dtype/mask/LogSoftmax 域仍未资格化；这不构成完整
Softmax PASS，也不解除 oneDNN Q2。合同核对另计2分钟，CPU本卡累计118分钟、余2分钟。
