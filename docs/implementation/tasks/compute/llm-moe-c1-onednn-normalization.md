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

### AA04：扩展验证完成，进入不可变候选比较

本次 selector 从 `9c9ef3be` 选中的112项检查已全部有实际通过记录：`904b1fce` 的96项基础/
逐头检查，`98d3331e` 的组件合同及13个剩余产品的39条源码/shared/static 交付旅程，另复用
前述 Compute/Graph 各三交付。后继仅改文档和 Vulkan OFF 测试注册；该注册修正的4项 selected
检查单独通过，不重复96项。两份 CPU 交付证据逐项核对原 cache 的 Vulkan OFF、未变的 OFF
分支，以及 CPU 产品、全局 CMake、依赖和对应 consumer 字节，未用新提交号无条件继承旧结果。

[选中检查与复用依据](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-selected-validation.json)、
[基础/组件原始记录](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-selected-base-raw.zip)和
[剩余交付记录](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-selected-deliveries.json)
绑定实际 candidate、命令、退出码、日志和安装产物身份。组件合同中的隔离 cache 先因缺少
Highway 归档被离线策略拒绝；复制本机已有锁定归档后原合同通过，未改检查器或下载新版本。
39条交付全部使用邻置 DLL、PATH unset 和禁弹窗设置；header-only 产品仍只验证其真实头与
package consumer，不虚构 producer binary。核对归档后删除39个重复 build 目录，共401,814,419
logical bytes；保留 relocated SDK、独立 consumer 和共享源码缓存。统一 LOC 仍为46,941行/
193文件，见[本候选统计](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-selected-loc.json)。

P1 首次计时前将合同的 W1 generator 固定为已完成正确性构建的 Ninja，编译器仍是同一
MSVC19.50 x64、Release、portable ISA、LTO OFF，CXX flags 与该构建一致。锁定 oneDNN 的
`cmake/platform.cmake` 会加入 `/MP`；Ninja 的每命令单 TU 配合 parallel2，避免项目内多文件
编译绕过低并行约定。该变更发生于任何 A/A/采样之前，不改 workload、数学/完成合同、阈值、
样本、顺序或统计核心。使用短固定 fresh build 目录，并由原统一仪器执行原预检和采样。

比较前另沿 benchmark 的两个 Route 复核：Direct 强校验实际 Provider ID，Graph 强校验实际
Provider ID 和 region 数；两侧均使用自然 submit/wait，完整 norm→Mul→Add 与成熟融合片段
同责比较，冷准备/JIT 均在稳态外，运行分配/映射关闭/完成全部计费。AA01–AA03 与 AA04 的
验证入口已实现，API/算法停止变化；接下来进入 CANDIDATE 性能门。AA04 的条件规则和重复域
取舍尚未签署，任何因此产生的产品修改先依法回到 AUDITED。本节不签 C1 全包、四路线完整
覆盖或算子淘汰；没有运行过的 A/A/P1 仍不写 PASS。

### CANDIDATE 冻结

本轮候选固定为 `4895df0bd70669c138a5ce3cb1a4d0a00b7349cb`，两个 oneDNN owner 从 AUDITED
进入 CANDIDATE。产品 norm/post-op 实现相对 `2c9338e1` 未变；当前候选同时包含已迁移 consumer、
完整验证记录和冻结的 P1 程序/合同。随后 registry/status 说明提交不改变产品候选，统一仪器
仍记录实际干净运行 HEAD。原 C1 的其他 owner 状态不变；默认规则变更或实现删除必须退回
AUDITED 后按具名证据施工，不以候选状态跳过比较。

### AA04：合格 P1 结果与第一轮有限纠正

干净运行 HEAD 为 `e33b6fea4b80d4b82adb6a267dc52ba2ad7a73f9`。
[完整性能记录](../../../../evidence/performance/onednn-normalization-p1-e33b6fea-w1-win/README.md)
保留16项全部分层、原始日志和仪器裁决：资格 PASS，性能 FAIL；4项批量 Direct 胜出，
4项小规模/不规则 Direct 和8项 Graph 落后。一次 n=15 已可裁决，没有升级31或原样追加计时。
双方均走相同公开 Direct 或 Graph submit/wait；Graph 是原版融合片段对相同 Native 三节点链，
不是 Graph 对裸 kernel。初始 CRLF 配置失败、首次 A/A 失败（零产品样本）与已声明的一次相同
重试全部保留；该重试已经结束且额度用尽。原 deep verify 为 QUALIFIED / EXACT / paths_verified。

有限源码调查得到具体产品反例：锁定 oneDNN 的 `jit_uni_layer_normalization.cpp:61` 只接纳
scalar/最后轴广播，批量 full-shape binary post-op 不在优化内核支持集合。
对实际链接库执行64个**只查询冷 descriptor、无执行/计时**的用例，批量 norm→Mul(vector)→
Add(full) 均选择 `simple:any`，norm 或 norm→Mul(vector) 均为 `jit:uni`；改成等价3D描述符
仍相同。`simple_layer_normalization.cpp` 在每个元素调用通用 `ref_post_ops`，与合格样本中
约百倍退化的成本结构一致。最初怀疑的 `ref:any` 由实际 `simple:any` 身份纠正。
这是消费者残差链被过度融合的准入缺陷，不是要求重写成熟归一化算法。

两个 owner 依协议从 CANDIDATE 退回 **AUDITED**；第一轮具名纠正仅为 **AA04-F1：优化融合边界**：

- 冷匹配在导致该退化的批量 full-shape Add/Mul 前停止，保留此前合格的 norm 与 scalar/vector
  post-op 前缀，余下节点沿已有 Graph 组合；单行及可观测/分叉边界保持正确。
- 精确准备查询 vendor 实际实现身份，拒绝已识别的 `simple:any` / `ref:any` 融合制品，返回
  既有 unsupported decline；不吞 OOM/执行错误，不在热路径查询、搜索或改选。
- 迁移普通 Graph 回归和三交付 consumer，证明优化前缀与后续节点的数值、状态、重复执行和资源
  责任；加入单行、批量、scalar/vector/full-shape 与 Layer/RMS 边界的有限回归。显式成熟样例
  仍强校验真实 region，不能用全 Native 假通过。
- API、底层 SIMD、oneDNN 算法、公共构建/性能仪器不改。默认小规模选择和 Native 重复域取舍
  仍属 AA04 待裁决项，不把此修复预称为 Graph 胜出，也不套用 Direct 胜项代签 Graph。

验证仅运行本改动 selector 与已声明的 W1 static/shared、WSL GCC14、相关 Graph 三交付；原有
未受影响的基础/头/交付事实按身份复用。所有形状、阈值和统计门保持，完整 Graph 的 P1 须等
本轮 API/规则稳定、correctness/交付完成后再冻结候选启动；不边改边计时。两轮性能纠正预算中
本项占第一轮，第二轮未使用。达到预算仍无法裁决按原停止规则处理。

统一脚本对 `e33b6fea` 为46,941实现行/193文件，自检 PASS；代码行数不是完成度。

### AA04-F1：实现、交付完成与第二候选冻结

产品修复固定为 `ae270af9ceb948b09601b26f32454eb28d32dff3`：冷匹配保留优化 norm→scalar/vector
前缀，批量 full-shape Add/Mul 沿原 Graph 节点组合；精确准备拒绝已识别的通用融合制品。
运行路径没有新增查询、搜索或字段；公开 API、SIMD、自有 norm 和 vendor 算法均未变化。

[本轮正确性与交付证据](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-f1/README.md)
记录 W1 static/shared、WSL GCC14 static 各7项 CTest：58个 gtest case 加独立 executor consumer，
以及 Graph source、relocated shared/static SDK 的原有完整检查。新增回归覆盖 RMS/Layer、
scalar/vector、单行/批量、full Add/Mul、公开中间值、两个独立运行槽与资源关闭。
静态首次构建被新测试局部变量遮蔽的 WERROR 拒绝，改名后只恢复受影响 Graph 组；原始失败与
未变产品/构建/公开头的复用依据一并保留。当前机器不支持的 vendor F16/BF16 ISA 域仍不冒称数值通过。

普通三交付消费者实际验证 oneDNN 两节点前缀和 Native 残差节点，不能以全 Native 替代。
核验后删除85,063,680字节的 WSL 传输包；保留一个固定的当前交付构建区，用于未完成默认规则的
后续增量验证，不再为同一批次堆积新目录。统一脚本为46,955实现行/193文件，自检 PASS。

两个 owner 进入 CANDIDATE，冻结的是本轮 F1 产品修复；条件默认规则与 Native 域取舍仍未签署。
接下来仅运行已声明第一轮纠正的完整同合同 P1：原16个 workload、8个顺序/分配/placement 分层、
阈值、n=15/31停止策略及统一仪器不变。性能合同只更新6个批量 Graph 项的 owner 描述，使其明确
包含优化前缀、Native 尾节点、每步状态与 Graph backing 的全部成本；未更换输入、语义或对照。
修复版性能在本次冻结时仍未运行，不预称胜出；原始4胜12负与失败记录保持。

旧 P1 build 的实际制品先按原 manifest 哈希归档核验，再清空同一个固定目录供 fresh 候选构建；
原始 raw/manifest/裁决保留，旧绝对构建路径的时点验证不能冒充永久可复验。第一轮只测固定 F1，
不原样追加计时；如结果授权默认规则修改，先依法退回 AUDITED，并按第二轮有限预算实施。

### AA04-F1：性能资格未通过，定向隔离控制问题

干净运行 HEAD 为 `64e1e169`。
[完整记录](../../../../evidence/performance/onednn-normalization-f1-64e1e169-w1-win/README.md)
保留120个产品进程、30个公共控制进程、全部 raw/manifest/contract 及唯一四进程限定诊断。
采样前 A/A 通过；采样后 workload A/A 的256个比值中，一个 oneDNN/oneDNN 的 Direct LayerNorm
1×2048 为1.21435，触发原 gross guard。全部固定因素中位数仍在5%带内，五项公共控制 PASS，
source/artifact/host 身份未变；最终资格 **FAIL**，工程性能 **INCONCLUSIVE**。

限定四进程诊断未复现21%差异，不覆盖失败、不签新赢家，也不据此修改 SIMD/Sched。
deep verify 为 DIAGNOSTIC / EXACT / paths_verified；未新增完整重跑或升级n=31。
F1 正确性及三交付 PASS 保留；本次样本不能签修复后的默认规则或 Native 删除。
统一脚本在 `64e1e169` 为46,955实现行/193文件，自检 PASS。

两个 oneDNN owner 从 CANDIDATE 进入 **BLOCKED**，只隔离该资格问题。
具名 owner、两条调查路径、30分钟预算、FIX 准入及停止/恢复合同见
[AA-Q1 资格恢复卡](llm-moe-c1-normalization-aa-qualification.md)。公共性能核心没有修改；
原定第二轮性能纠正预算尚未使用，不以资格异常增加无界优化或采样额度。
CPU 路线 owner、其它 Provider 的状态不变；本记录不把全局 C1–C7 goal 标成已完成或全局阻塞。

### AA-Q2：具名恢复条件重新定界（SCOPED）

2026-09-19，baseline `8461925b`，owner `/root`。按[Q2有限范围](llm-moe-c1-normalization-aa-qualification.md#q2恢复所需的具名验证记录修复scoped)
处理preflight记录覆盖及实例冷观测缺失：两个oneDNN owner从BLOCKED进入SCOPED，产品仍为
`ae270af9`，原资格FAIL、工程性能INCONCLUSIVE和第二轮产品纠正预算均保留。
只有该受审模块验证改动及其冷验收完成，才固定唯一新实验；没有授权原样重测、改门限、默认
规则或删除Native。其余C1能力继续原卡，Vulkan Y31通过不能解除CPU性能资格。

Q2在`882ecd0e`之后完成[有限AUDITED ledger](llm-moe-c1-normalization-aa-qualification.md#q2-audited固定实现与验收)，
两个owner进入AUDITED，只允许模块验证记录/冷观测与相关验收。F1产品、历史FAIL及原性能门
保持；本轮冷检查通过前不启动计时，记录修复不能冒签性能资格或默认赢家。

### Q2 CANDIDATE：唯一恢复实验

候选固定为`07a69e607d54f9258d655f146e925f1d84ff8b58`；两个oneDNN owner由AUDITED进入
CANDIDATE。11项模块验证测试及两个真实verify-only进程已通过，产品F1、问题数据、oracle、
两个热`run()`、配对调用及原统计门未变；[冷验收](../../../../evidence/performance/onednn-normalization-aa-q2/README.md)
明确不含性能样本。候选后的registry说明不改benchmark制品合同，统一仪器仍记录实际干净HEAD。

现仅允许这一次`onednn-normalization-p1-v1`的W1-WIN原合同完整实验：原16项、8分层、公共控制、
前后A/A、n=15/31停止策略；保留每次preflight的独立原始记录，采样普通路径不启用冷诊断或JIT
dump。沿已有固定`active-cpu/p1`作fresh build，冷EXE已按hash另行备份；新output独立命名q2，
旧失败/证据不覆盖。parallel2、重型验证串行、DLL邻置、禁弹窗，不改电源或请求UAC。

资格再次失败立即按既定条件BLOCKED，不附加原样重测或限定诊断进程；若合格才按其实际结果
裁决原剩余AA04。需要默认规则或删除实现时，先回到AUDITED并受原第二轮产品纠正预算约束。
本节尚无Q2性能PASS，不代签Vulkan/其他C1验收或整包完成。

### Q2 失败与有限接线收口

`0e107a06`的唯一恢复实验已在产品采样前停止：记录目录占位符未展开触发源树身份门，且
16个A/A进程中oneDNN对自身的一项也越过原10% gross门。原始失败与512个实例完整保留于
[Q2证据](../../../../evidence/performance/onednn-normalization-aa-q2/README.md#唯一恢复实验失败)。
两个owner只为[Q2-D目录接线FIX](llm-moe-c1-normalization-aa-qualification.md#q2-唯一实验停止与目录接线纠正audited)
退回AUDITED；修正合同参数与模块落点，加无计时回归后按原停止条件BLOCKED。
产品F1/正确性/三交付不变，未产生产品性能样本；默认规则、淘汰和第二轮产品纠正预算未签署。
不因为这是接线错误而忽略另一项A/A失败或追加同样实验，C1其他独立验收继续。

Q2-D已在`7f625eb4`修复并通过12项Python回归及docs/closure；C++、原门和产品子树不变。
两个owner依法从AUDITED进入BLOCKED，详细恢复/停止条件见[Q2终局](llm-moe-c1-normalization-aa-qualification.md#q2-终局blocked)。
没有重新构建、冷执行或P1；目录修复不签原失败通过，也不决定赢家。统一脚本为48,294实现行/
195文件，自检PASS。CPU局部阻塞不替代C1其他独立验收，整包及64行/256格仍未签收。
