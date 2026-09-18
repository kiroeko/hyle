# C1-AA：oneDNN 归一化及原生 post-op 接入

## Closure card

- owner：`/root`；`compute_provider_onednn`、`graph_provider_onednn`。
- baseline：`4a7685523d139a54aab9043bdf3d5b767a2bed20`，工作树干净。
- trigger：用户要求先实际绑定成熟 Provider，再比较、补缺和删除冗余实现。C1 已准入的
  F03/F04 及本地 Qwen3.5-2B 的归一化/缩放/残差片段仍主要走 Native；锁定 oneDNN 已有
  RMSNorm、LayerNorm、mixed parameter dtype 和 fused binary post-ops，现有构建却仅启用
  MATMUL。该新增真实 consumer 域满足前轮新增能力/公开合同 reopen trigger；不是旧卡续跑。
- 主 consumer：普通 CPU Direct/Graph 的 RMSNorm/LayerNorm，及 norm→Mul/Add 片段。
  正交 consumer：多维批量归一化、无参数/带参数、两个 invocation、重绑及原地输出。
- scope：复用 oneDNN 3.13.2 的 normalization primitive 与原生 post-ops；冷绑定、调用方
  workspace、借用 executor、失败回报及普通路线；对应构建/三交付、数值/生命周期与公平性能。
- non-goals：不自写归一化算法，不接全部 oneDNN 算子，不改 SIMD、Sched、公共治理/性能仪器，
  不改既有 BRGEMM/gated-MLP 算法，不自动解除 ggml-CPU BLOCKED，不把局部接入记为 C1 完成。
- 依赖：现有 Compute/Graph/Mem、锁定 oneDNN；真实直接消费者为 CPU/Graph CPU 默认工厂。
  私有 CMake dependency options 属本接入范围；不重写构建系统。
- 性能：P1，完整同合同 Direct/Graph 对比现有 Native；oneDNN 内部分配、转换、同步、Graph
  边界和融合损益全部计费。未完成数值/资源资格前不定赢家；不靠省略检查赢。
- 预算：两条具名机制（原版 Graph partition、primitive/post-ops），源码选择最多60分钟；
  产品按有限 AA01–AA04 实施，最多两轮具名性能纠正；一个稳定不可变候选执行适用 P1。
- 预声明验证：selector 所选 gates；W1 static/shared、WSL GCC14，x64 oneDNN 实际运行；
  独立 FP64 oracle、F32/F16/BF16、2048及不规则宽度、decode/prefill、特殊值、大均值小方差、
  精度/布局拒绝、OOM、两个 invocation/并发/重入/资源失败、Graph 可观测中间输出/分叉边界。
  CMake seam 变化执行适用 source/static/shared installed consumer 与依赖隔离；不代签 ARM。
  DLL 邻置、PATH unset，parallel2、重型验证串行，复用固定构建目录。
- 停止：本轮适用验收齐全才 CLOSED；未运行项如实保留，实质 blocker 具名裁决。C1 其他操作、
  平台/模型验证继续归主卡；不以本轮关闭替代全部范围。

## 有限待审项

| ID | 缺口 | 允许边界 |
|---|---|---|
| AA01 | 构建只启用 MATMUL/SEQ，成熟 norm 未进入产品 | 固定 pin 不变，增加 LAYERNORM、借用 HYLE executor 的 THREADPOOL runtime |
| AA02 | Direct 未绑定上游 norm | 共享私有 immutable primitive plan、每 invocation stream/memory handles、caller scratchpad；F32 accumulation，dense F32/F16/BF16，明确不支持域 |
| AA03 | Graph 未保留原生 norm post-op 融合 | 原有 region 扩展识别 norm→Add/Mul，不吞掉 published/多消费者中间结果，不降低半精度中间舍入要求 |
| AA04 | 没有该实际路线的资格与默认规则 | 正确性/所有权/交付先通过；固定同合同 full-cost 规则及文档，再裁决 Native 重复域是否可移除 |

SCOPED 登记提交为 `98cf0742`。

## AUDITED：机制选择与有限 FIX

选择原版 **normalization primitive + binary post-ops**，不复制其算法。固定源
`71094df4` 的 `dnnl_layer_normalization_forward_primitive_desc_create_v2` 支持 RMS flag、
参数类型和原地输出；`jit_uni_layer_normalization` 与 `ref_layer_normalization` 使用中心化方差，
并直接执行 binary post-ops。原版 Graph 的同类 partition 最终也使用上述 mature kernel。
本批所需链没有 Graph 特有算子；使用 primitive 能明确每 invocation 的 stream/memory handle
所有权，避免引入该 Graph kernel 的线程局部 execution-args cache。并非宣称 primitive 无分配：
`primitive_iface.cpp` 每次转换 execution args 仍有 vendor metadata 分配，全部计费。

AA01 的具体失败证据：该 pin 的 `primitive_iface.cpp::dnnl_primitive_execute` 在普通路径尾部
才调用 after_exec_hook；`cpu_stream.hpp` 直接清空当前 pool，没有恢复嵌套调用前的 pool。
有限修正为构建目录中的单文件 overlay：用栈上 scope guard 保证正常/异常退出都执行 hook 并恢复
旧 pool。固定源 SHA/精确锚点失败即停止，原缓存/checkout不改动；不改算法或另建 patch 框架。
嵌套回归进一步证实 `utils.cpp::activate_threadpool` 在已有 active pool 时不会替换它；scope
进入时必须先保存并清空旧 pool，再执行原 before hook，退出时恢复。否则内层串行 norm 错用
外层 executor，回归得到 `operation invocation already has work in flight`；这是一行受审修正，
没有放宽重入/在途检查。

AA02 的外部分配注入在 MSVC `/EHsc` 下复现进程退出（code3）。该模式假定 extern-C 不抛异常，
不满足 oneDNN C 入口内部 C++ 分配的实际语义，见 [MSVC 异常模式合同](https://learn.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?view=msvc-170)。
仅本 adapter translation unit 使用 `/EHsc-`，保留同步 C++ 异常展开和捕获；不捕获 SEH、不修改
全局编译合同。注入回归逐个失败 vendor 热路径 new，要求 accepted ResourceExhausted、失败槽
退休、其他槽仍可执行；不是从 C API 名称推断不抛。

| FIX | 冻结实现与验收 |
|---|---|
| AA01 | MATMUL 保留，加 LAYERNORM；THREADPOOL 借用既有 ParallelExecutorRef，不创建私有池。冷 primitive 创建保存/恢复 oneDNN 线程局部 concurrency 设置；并发数量可表达性检查，不截断成错误整数 |
| AA02 | 私有共享 plan 保存原版 primitive/descriptor/engine；每 invocation 独立 stream、memory handles 和 caller scratchpad，重绑更新 handles。dense 非空末轴归一化，F32 accumulation，F32/F16/BF16 input/output；scale/bias 可以同输入 dtype 或 F32，但二者类型不同的域明确留给合格 Native。冷提案不创建 vendor artifact；准备再次核对精度/shape。精确 input0 alias，其他重叠仍由共同绑定检查拒绝 |
| AA03 | Graph 识别一个 norm 后最多4个 dense F32 Add/Mul post-ops，保持原版融合；支持 scalar、最后轴向量和全形状 RHS。published/多消费者边界停止扩张，按实际数据流匹配并校验可收缩性。半精度 Direct norm 接入；半精度跨节点链暂不融合，避免少一次中间舍入。既有 BRGEMM gated MLP 不改算法 |
| AA04 | 独立 oracle、类型/形状/错误域、alias、两槽重绑/旧输出、并发/嵌套 executor、allocator/映射故障与普通 Direct/Graph；三交付及 full-cost P1 签完才冻结默认域并清理被完全替代的自有域。SIMD 基础库保留；基于 SIMD 的 Native 算子适配、候选和路由参与同一比较/淘汰，无证据不删除有效支持域 |

公开 API 不新增。普通 `cpu::create_context` 和 Graph CPU 工厂已有相应 Provider 注册；扩展其
冷提案/精确准备即可接入。新增路线在 AUDITED 产品候选中待资格验证，不预称优胜。执行只调用
已绑定 primitive，不搜索/计时。编译/运行第三方异常在 adapter 边界转 Status；HYLE allocator
管理 wrapper，vendor opaque code/metadata 数量未知如实报告 unknown，不宣称受 HYLE quota 全控。
执行同步结束后返回 accepted terminal；失败不重放到另一 Provider。partial creation 用 RAII 回收；
两个 invocation 不共享可变 handles/scratch；映射失效、flush/close 保留首个错误。

七问：① oneDNN 3.13.2/Apache-2.0 pin不变，原生算法 BORROW，升级重新资格化；②模型归一化
和批量数值 consumer 共用普通接口，范围外域明确fallback；③复用 Compute/Graph/Mem和借用executor，
不重造 Sched；④线性归一化、原生融合，记录 vendor 每次metadata成本并做同合同P1；⑤无C/FFI
产品；⑥私有头无vendor类型泄露，source/static/shared与依赖隔离门适用；⑦普通入口不要求选库，
guide记录精度、已验证域和借用executor寿命。

## 首批实现与局部验证

- AA01–AA03 已接入普通 CPU Direct/Graph，原版 norm primitive 与 post-ops 没有改数学算法。
  热路径因 vendor execution-args 分配如实为 **Managed**；prepared-only 不接纳，BRGEMM 仍为 Prepared。
- W1 MSVC 首轮 F32 RMSNorm/LayerNorm、参数/alias、2048及不规则宽度、FP64 oracle、特殊值/
  中心化方差、借用 executor/两个共享 artifact invocation、普通 Graph 融合/可观测边界通过。
  HYLE metadata OOM 逐分配点回收及 vendor warm-new OOM 转终态、坏槽退休/新槽恢复已有回归。
- 本机13900K的 oneDNN F16/BF16 原生 norm 不支持；固定源 `cpu/platform.cpp:112` 和
  `jit_uni_layer_normalization` 要求 AVX512 或 AVX2-VNNI2 等相应资格，不能以 F16C 等同替代。
  最初测试无条件期待 oneDNN identity 被证伪，现记录 ISA_UNSUPPORTED、数值 UNPROVEN；
  默认工厂正确回落 Native，现有半精度 state Graph 独立数值回归通过。没有删检查、添加整张量
  转换或把这部分写成 oneDNN 数值 PASS；对应 ISA 原生资格仍待有设备时验证。
- W1 static/shared 均通过8个 CTest（166个 gtest case，另有独立 executor consumer）；
  static 分配检查通过2例，逐个注入18个 vendor warm-new 失败点，旧 BRGEMM 热路径仍无 new。
  WSL GCC14 static 同样通过8个 CTest/166个 gtest case及独立 executor consumer，分配检查2例通过；
  其 vendor warm-new 为13次，逐点注入均回报失败并退休坏槽。GCC 的断言宏 dangling-else 只通过
  测试外层显式花括号修正，未改产品代码；W1 对受影响测试增量6个 CTest再次通过。
  完整 package/selector 扩展门及公平 P1 尚待完成，owner 保持 AUDITED。
  AA04 的默认优选/Native冗余删除尚未裁决；本节不是 C1 或本轮 CLOSED。

产品仍绑定 `2c9338e1`；[结构化局部证据](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-local.json)、
[原始日志和命令](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-local-raw.zip)、
[统一 LOC](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-loc.json)记录上述边界。
Compute+Graph 全部实现为46,941行/193文件，不含测试和 benchmark；本批验证准备不增加产品 LOC。
已核对 WSL 归档 receipt 后删除79,892,480-byte传输 tar；两个环境继续使用固定源/构建目录。

## AA04 对比前置合同：先比较，再裁决重复实现

本批产品快照 `2c9338e1c3a01d793aeda6aef1c4898709b8fe47`。此节在新增 norm benchmark 前冻结，
不借已有 MatMul/FFN 的 P1 代签。双方分别通过相同 HYLE Direct 或完整 Graph 公共入口，
仅冷准备时显式固定 oneDNN/Native，以测量研发期的候选实现；最终普通用户仍只选择 CPU 路线。

源码预审：`compute_provider_native/native_provider.cc::run_rms_norm/submit_float`、
`compute_provider_native/normalization.cc::run_layer_norm_range/submit_normalization` 与
`compute_provider_onednn/normalization_internal.cc::execute/submit` 均在 CPU 计算、映射关闭、
flush 和 Status 可见后同步返回 accepted terminal；共同 OperationInvocation/Completion 负责相同
绑定/在途检查及失败退休。Graph 两侧均使用 compile→create_invocation→bind→prepare_replay→
submit→wait；一个原生融合 region 对完整 Native 节点链，不能拿 Graph 对裸函数。
没有提交后尚未计费的设备工作、callback 或延迟释放。生命周期/错误/完成合同因此为 SAME_CONTRACT。

| 成本 | oneDNN 候选 | Native 对照 | 计费 |
|---|---|---|---|
| 冷准备/析构 | engine/primitive/JIT、每槽 stream/handles/scratch | 已有 SIMD route 与计划/槽 | 均在稳态外，报告保有资源；不重复 JIT |
| Direct 每次执行 | 公共检查、map/rebind、原版 norm、vendor execution-args new/free（已测18次）、close、completion | 同一公共检查、map、SIMD norm、close、completion，无 vendor metadata 分配 | 全部计入 |
| Graph 每次执行 | 固定 replay、一个 norm+Mul+Add post-op region，vendor 分配同样计费 | 固定 replay、三个完整节点与必要中间存储读写 | 全部计入；两侧都到 wait 终态 |
| 线程与数据 | 默认串行，无私有 pool；同一份只读输入，独立输出/workspace | 相同，Native 不隐式增加 worker | 一颗已登记 P 核；不外推并发性能 |
| 数学与格式 | F32 输入/参数/输出、F32 归约、epsilon=1e-5 | 相同；RMS/LayerNorm 都调用已有 SIMD 能力 | 无 dtype 转换；分别独立 FP64 oracle |

vendor metadata 已归因为成熟 primitive C API 的固定成本，不能用裸 ukernel 或跳过检查掩盖；
融合和向量化收益是否抵消它尚未知，允许有界完整成本比较。Native 当前 F32 norm 本身主要是 SIMD
包装。底层 SIMD 基础库按用户要求保留；该包装及其候选注册、路由没有保留豁免，若同合同
比较落败且适用域被完整替代，就迁移并删除。单一 shape 胜出不代签半精度/无分配等其他必要
档位；这些支持域分别核对覆盖，不能因此连带删除底层 SIMD 能力。

冻结 workload：RMSNorm+scale、LayerNorm+scale+bias 各自的 Direct 和 norm→Mul(vector)→Add(input)
完整 Graph；真实本地 hidden2048，rows1/17/128，对应 decode/短prefill/prefill；一个不规则
批量数值 holdout 为 rows17/width257。BF16/F16 vendor 本机不支持，不纳入这次性能胜负。
双方相同输入、64-byte alignment 和资源，先独立 oracle 后计时；warm submit 内无搜索或计时选库。
先同 workload A/A（两侧都用同一个真实实现），再 A/B 与 B/A，独立交换分配顺序。
使用既有 nanobench compare 和统一仪器；placement 只加少量冷 section 位移，不采用全函数 padding。
预声明 material/NI margin5%，样本15，只有既定统计停止规则要求时升31；两轮具名纠正预算不增加。
未通过 preflight 不启动完整样本；未验证域不写默认赢家，P1 后按完整适用域裁决保留/替换/删除。

### 对比程序准备事实

[固定合同](../../../../bench/contracts/onednn-normalization-p1-v1.json)及
[消费形状来源](../../../../bench/graph_provider_onednn/normalization_shapes_v1.json)已登记。
W1 Release/WERROR 的 primary/placement 两份程序已编译，16个 workload 的双方完整路径都通过
独立 FP64 oracle（`HYLE_BENCH_VERIFY_ONLY=1`，没有计时）。placement 仅增加48-byte `.text`，
link map 证实 Graph submit 等产品函数地址移动32 bytes；没有修改热函数 flags 或全函数 padding。
数值程序使用确定性输入，尚不是本地真实权重 replay；LayerNorm属于正交数值 consumer，
不宣称本地 Qwen 使用 LayerNorm。workload A/A、统一仪器采样和默认规则裁决仍未运行。
按本卡先完成交付前置，再对冻结候选执行 P1；不得把程序已存在计作性能验收或 Native 已淘汰。

### AA04 交付 consumer 迁移

普通 Compute 样例的 RMSNorm 原先直接创建无 workspace 的 invocation；现在按所选 artifact 的
workspace requirement 分配并传入。Graph 样例原先拒绝 mixed providers，这与已授权的同 CPU
Provider 组合冲突，移除该旧实现假设，保留图结构、资源、执行和结果检查。新增普通工厂及显式成熟
region 两次独立消费：RMSNorm→Mul(vector)→Add(input)，两个变化输入的固定 replay、独立 double
oracle；后者保证 installed SDK 确实交付并执行原生融合，即使后续默认优选规则选择别的合法实现。
该改动只在现有 AA04 的样例/验收范围，不改产品 API 或调度检查。

### AA04：Compute 三交付结果与 Graph 样例修正

候选 `58543f8c` 的 Windows CPU Compute 源码嵌入、installed shared/static 独立 C++ consumer
均已实际通过；两种 SDK 都先移走原安装目录，再由新路径构建/运行消费者。原有依赖、feature、
许可证和 shared export 检查全部执行，shared 检查13个 HYLE库/1005个导出。三次 consumer
均使用邻置的 HYLE/CRT DLL、PATH unset 和禁弹窗的进程设置。
[结构化记录](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-compute-delivery.json)和
[原始证据](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-compute-delivery-raw.zip)
保留原始命令、cache、测试日志、安装文件哈希和已知限制。

首个 CTest 启动在 PowerShell 参数转发处失败，测试尚未运行；改为原样传递参数后，从已有构建
恢复，所有检查保持原样。没有因此修改产品或省略验收。证据核对后删除本轮三个重复 build 目录，
共4,534,920,223 logical bytes，保留 relocated SDK、独立消费者和共享源码缓存。
统一脚本对 `58543f8c` 仍统计46,941实现行/193文件，详见原始包中的 `delivery-loc.json`。

Graph 首次 source consumer 编译在新增样例的条件表达式处报 `C2280`：MSVC 尝试复制只可移动
Context 的 Result。样例改为两个显式分支后移动赋值，不改 Result、Context 或产品 API；Graph
三交付仍待实际执行通过，Compute 已完成证据不因样例/说明修改重跑。性能预检另补齐 CRT 邻置
及 DLL 哈希记录，执行前设置禁弹窗；这是 benchmark 启动准备，不是运行过 A/A 或 P1。

完整 selector 扩展门、合格性能、默认优选和 Native 重复域清理仍未裁决。以上结果只完成本批
Windows Compute 交付，不代签 Graph、其他路线、C1 整包或自研算子淘汰。

### AA04：Graph 三交付通过

候选 `0faf698c` 的 Graph 源码嵌入、installed shared/static 独立 C++ consumer 均已实际通过。
源码消费只增量编译修正后的样例，再实际运行 CTest；共享与静态 SDK 完整执行原有配置、构建、
安装、依赖/feature/许可证检查、迁移和消费者测试。shared export 检查16个 HYLE库/1201个导出。
三次消费均运行普通 CPU 工厂和显式 oneDNN 融合 region，使用变化输入及独立 double oracle，
DLL邻置、PATH unset、禁弹窗设置不变；没有重跑已完成的 Compute 交付。

[Graph 结构化记录](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-graph-delivery.json)
及[原始证据](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-graph-delivery-raw.zip)
保留首次样例编译失败、修复后的实际命令/日志、CMake cache、安装产物哈希和统一 LOC。
核对证据后删除 Graph 三个重复 build 目录，共4,555,332,249 logical bytes；连同此前 Compute
共清理9,090,252,472 logical bytes，保留 relocated SDK、消费者和共享源码缓存。

本批 Windows CPU 的 Compute/Graph 三交付至此完成。完整 selector 剩余门、公平 P1、条件优选规则
及 Native 重复域取舍仍待完成；未运行 A/A 或正式采样，也未删除未裁决的自有算子。性能合同静态
校验已通过，Native 对照固定到 `2c9338e1` 的干净验证 checkout，当前 Native 产品子树身份一致。
`0faf698c` 的 Compute+Graph 实现仍为46,941行/193文件，不含测试、benchmark 或第三方源码。
