# HYLE Graph：编译并重复执行计算 DAG

Graph 用于把多个 Compute operation 编成可复用执行计划：推导中间值、选择实现、规划 workspace，
然后用不同输入重复执行。只需要一个固定 operation 时先用 [Compute](compute.md)。

普通 CPU/CUDA 路径使用共享 Queue 片段；[Vulkan](vulkan-compute.md) 与
[Apple](apple-compute.md) 同一 Runtime 的固定纯图也可通过实际 ExecutionSource 进入同一
Executable／Invocation，内部继续使用原生 Program、结果与完成机制。其余混合运行时仍在迁移。
项目内的引擎层拥有模型加载、请求循环和 offload 策略；Graph 保持模型无关，亦可独立使用。

先看下面的两节点例子和[生命周期](#四个生命周期对象)，再按需要选择：

- 单 CPU 或单 CUDA：对应普通入口；最简单的固定设备图无需手写 target/storage/lane 表。
- CPU 与一个 GPU：[显式片段组合](#cpu-与-gpu-组合)。
- 可变 batch/token：[独立与派生维度](#独立与派生维度)与[精确变体](#有限精确变体)。
- 内存上限、低热开销、capture：按后文编译与 replay 选项检查实际计划，不从逻辑图推断零分配。
- KV/conv/recurrent 快照：[状态版本](state-versions.md)；Graph 不隐式决定何时发布模型状态。

## 第一次图执行

以下片段接在 Compute 指南的 host buffer/tensor view 创建之后，放在返回 `hyle::Status` 的函数中。
`input_view`、`output_view` 描述相同的 F32 张量，使用 `mem::system_resource()`；
它们的 backing 由调用者保活。图执行 RMSNorm → SiLU，只将最后结果发布到外部输出：

```cpp
#include <array>
#include <utility>
#include <hyle/graph/cpu.h>
#include <hyle/graph/executable.h>
#include <hyle/error/result.h>

namespace c = hyle::compute;
namespace g = hyle::graph;
HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
HYLE_ASSIGN_OR_RETURN(auto x, builder.input(input_view.description()));
HYLE_ASSIGN_OR_RETURN(auto normalized, builder.add(c::RmsNorm{}, x));
HYLE_ASSIGN_OR_RETURN(auto y, builder.add(c::Unary{.kind = c::UnaryKind::kSilu}, normalized));
HYLE_RETURN_IF_ERROR(builder.publish(y));
HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());

HYLE_ASSIGN_OR_RETURN(auto context, g::cpu::create_context());
HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
HYLE_ASSIGN_OR_RETURN(auto compiled, g::compile(logical, context));
HYLE_ASSIGN_OR_RETURN(auto slot, compiled.executable.create_invocation());
HYLE_RETURN_IF_ERROR(slot.bind(x, input_view));
HYLE_RETURN_IF_ERROR(slot.bind(y, output_view));
const std::array queues{g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));
HYLE_ASSIGN_OR_RETURN(auto done, slot.submit(queue));
HYLE_RETURN_IF_ERROR(done.wait());
// 成功后读取 output_view；中间 normalized 的存储由计划管理。
```

保留 `compiled` 与 `slot` 就可更新原 input buffer 的内容并重复 submit，无需重新建图。
换绑定或 reset 后重新准备 replay；并发请求使用独立 slot 与可变 backing。
完整多节点、变化输入与 CPU/GPU 工程见[Graph sample](../../samples/package/graph/cpp/)。

## 四个生命周期对象

```text
GraphBuilder -> immutable Graph
Graph + CompileEnvironment/CompileOptions -> Executable + CompileReport
Executable -> one Invocation per concurrent lane
Invocation.bind(...) -> prepare_replay(...) -> submit(...) -> Completion
```

- `GraphBuilder` 是单 owner、可失败的逻辑构造器。`ValueId`、`NodeId`、`DimensionId` 和 `EffectResourceId` 带 graph identity，
  异图 ID 会被拒绝；node 只能读取已经产生的 value，因此数据环不能被构造出来。
- `Graph` 是不可变、可共享且与具体设备无关的逻辑 DAG；它不持有 tensor payload、Context、Queue 或
  provider artifact。
- `Executable` 是不可变、可共享的 exact specialization 或显式声明域的 general 路线。它固定 provider
  artifact、placement、physical value、segment dependency 和 workspace layout；域模式使用安全上界。
- `Invocation` 是 move-only 的单次并发 lane，独占 bindings、workspace、Completion/capture slots 和 replay
  route。同一个 Executable 的并发请求必须使用不同 Invocation。

## 普通 CPU 入口

```cpp
#include <hyle/graph/cpu.h>

auto context = hyle::graph::cpu::create_context();
if (!context) return context.error();
auto compiled = hyle::graph::compile(logical_graph, *context, options);
if (!compiled) return compiled.error();
```

默认入口自动组合 CPU 的 region 和 leaf 实现，不要求用户排序 Provider。
完整合法片段可以交给原库融合，不必为了使用 Graph 把每个节点拆成独立调用。
算子、布局、中间精度与 managed 程序限制见 [Compute](compute.md#cpu-与模型格式接入)。

可通过 `compute::cpu::DefaultContextOptions` 注入借用的 allocator、同步 executor 和
`native_threads`。该线程预算不统一控制 ATen 等原库的私有线程池。
需要接入已有 Scheduler 时，看[持久 executor 示例](../../samples/package/graph/cpu_executor/)；
Context、executor、Scheduler 和外部 backing 都需遵守各自的借用及关闭顺序。

## 普通 CUDA 入口

CUDA Graph 与 Direct 使用相同的默认计算规则，链接 `hyle::graph`。普通单 GPU 程序显式选择
CUDA device resource，无需手写 storage/target/lane ID 表：

```cpp
#include <hyle/graph/cuda.h>
#include <hyle/compute/backends/cuda.h>

auto context = hyle::graph::cuda::create_context();
if (!context) return context.error();
auto memory = hyle::compute::cuda::MemoryResource::create(
    *context, hyle::compute::cuda::MemoryKind::kDevice);
if (!memory) return memory.error();
auto queue = context->create_queue();
if (!queue) return queue.error();
auto compiled = hyle::graph::cuda::compile(logical_graph, *context, memory->ref(), options);
if (!compiled) return compiled.error();
auto run = compiled->executable.create_invocation();
if (!run) return run.error();
// bind 全部外部 input/output；options.constants 借用持久权重 owner。
auto ready = hyle::graph::cuda::prepare_replay(*run, *queue);
if (!ready.ok()) return ready;
auto done = run->submit(*queue);
if (!done) return done.error();
return done->wait();
```

默认 PreferCapture + RejectHostWait；通过 `replay_route()/replay_report()` 查看实际 capture 或
command 路线，也可显式 RequireCapture。Graph 规划中间值与 workspace，Provider 保持自己的
cuBLASLt/ggml CUDA/kernel artifact；captured 热执行直接重放原生程序。改变同地址 buffer 的内容可
重复执行；bind/reset 改变绑定会使 preparation 失效，需要重新冷准备。两个并发 run 各有独立
scratch/captured pointers，外部 input/output 不应产生未排序的冲突访问。

CUDA的Scan／TopK以CCCL primitive为基础，Graph为每个invocation规划独立workspace。
宽多行TopK1使用冷准备的分块选择与每行合并，保NaN、稳定选序和原值位模式，可用于
`kRequirePrepared`及capture；每个invocation持有独立workspace。单行仍用设备归约。
其它非空Scan／TopK路径仍声明managed commands，`kRequirePrepared`会跳过这些候选，prefer-capture如实报告
节点的commands路线。Gather也使用已准备的CUDA kernel，可参与capture；checked模式先在设备验证全部索引，
错误时保留该 Gather 的输出，在Completion边界报错并退役该invocation。I32 Scan的溢出摘要仍需要host等待，
以`blocks_host=true`声明；发生上述错误后的新请求使用新的invocation。
`HostWaitPolicy`控制跨队列依赖的host等待，不取消Provider内部的必要检查；普通单队列入口仍可
使用默认选项。执行期不临时改选Provider。

完整 dense FFN 与变化输入示例见
[CUDA Graph FFN](../../samples/package/graph/cpp/cuda_ffn.h)；支持域见
[CUDA 默认规则](compute.md#普通-cuda-入口)。资源和常量 owner 借用至最后在途 Completion 的
真实终态；失败槽需要重建。高级同 GPU 多 lane 使用既有 `CompileEnvironment`，普通 helper
不推断跨 stream 依赖。`graph::compile(graph, context)` 的通用简写使用 host storage，CUDA
普通代码使用上面的 `graph::cuda::compile`，显式提供设备资源。

## CPU 与 GPU 组合

HYLE 提供CPU与一个GPU的显式粗片段组合；引擎负责模型加载、offload比例、KV和请求循环。
构建选择对应CPU/GPU路线，冷编译显式给出执行端点、存储与placement。
CPU+CUDA可使用普通`graph::compile`装配环境，也可用`hyle/graph/cpu_cuda.h`中的
`cpu_cuda::compile`冷环境helper。两者返回相同`CompiledGraph`，执行均使用普通`graph::Invocation`。
`hybrid replay`表示capture与普通命令组合，与CPU/GPU放置是不同的轴。

下面接在编译、创建CPU/CUDA Queue和上传`WeightBatch`之后。`cpu_cuda::compile`默认CUDA lane为0、CPU lane为1；
自行装配环境时按实际lane表绑定：

```cpp
#include <hyle/graph/cuda.h>
#include <hyle/graph/result.h>

HYLE_ASSIGN_OR_RETURN(auto run, compiled.executable.create_invocation());
const std::array queues{
    graph::QueueBinding{graph::ExecutionLaneId{0}, &cuda_queue},
    graph::QueueBinding{graph::ExecutionLaneId{1}, &cpu_queue}};
HYLE_RETURN_IF_ERROR(run.bind(input_id, input_view));
HYLE_RETURN_IF_ERROR(graph::cuda::bind(run, weight_id, weight_batch, 7));
HYLE_RETURN_IF_ERROR(run.bind(output_id, output_view));
HYLE_RETURN_IF_ERROR(run.prepare_replay(queues));
HYLE_ASSIGN_OR_RETURN(auto sources, run.input_source_report());
HYLE_ASSIGN_OR_RETURN(auto done, run.submit(queues));
HYLE_RETURN_IF_ERROR(done.wait());
HYLE_ASSIGN_OR_RETURN(auto result, run.result(output_id)); // GraphOutput
HYLE_ASSIGN_OR_RETURN(auto readonly, result.tensor());
```

`graph::cuda::bind`保留实际权重selection及其上传依赖，允许上传仍pending；只读owner与分页资源附件独立，
不需要另一种Invocation或Results。实际CUDA Queue在首次读取时导入依赖，同batch多port和同Queue多lane去重；
不能原生导入的读者使用显式HostWait，`HostWaitPolicy::kReject`在冷准备拒绝该路线。
直接发布输入或其metadata View也会等待真实source就绪。成功设备后缀实际退休后保留原上传的晚错诊断；
失败或取消的图不会额外等待未消费的外部上传。reset或重绑释放相应selection，旧GraphOutput仍阻止槽被覆盖。

`input_source_report()`报告保留source与设备／主机等待路线；`replay_report()`报告共同片段和搬运事实。
配置共享`ResourceBudget`后，`resource_report()`按实际StorageDomain报告usage与boundary范围；
边界可能已被计费，不能直接加到live字节。完整工程见[CPU+CUDA sample](../../samples/package/graph/cpu_cuda/README.md)。

需要提交线程异步返回并推进独立CPU/GPU支路时，冷调用[prepare_progress](#可选主机异步推进)，
继续使用同一Invocation和显式QueueBinding；纯GPU或允许同步提交时无需Scheduler。
并发请求各建Invocation和可变workspace／输出，实际copy/compute重叠取决于原生路线与硬件。
取消或首错停止后继，已接纳工作仍须实际退休。释放旧结果与Completion后才能重放或重绑。

CPU+MLX/Metal与CPU+Vulkan也以共同环境、Invocation和GraphOutput组合，分别见
[Apple sample](../../samples/package/graph/cpu_apple/README.md)与
[Vulkan sample](../../samples/package/graph/cpu_vulkan/README.md)。原专用合同、流水线和分页风险消费者也已迁入共同入口；
这些定向检查不代表全部模型和数学域已覆盖。
原生私有字节、UMA和任务／线程成本按实际合同报告，不能从有限依赖表推断整条调用零分配。

## 原生精确形状与副作用

共同编译入口的 `kFiniteExact` 只覆盖明确列出的精确点；例如 bounds 为 1..4、表只有 1 和 4，
选择 2 会返回 Unimplemented。在 `create_invocation` 传入形状，得到已编译入口的原执行槽；
reset/submit 不改变形状，不运行 JIT。各槽继续拥有自己的原生资源和旧结果权限。

```cpp
const std::array small{graph::ShapeBinding{batch, 1}};
const std::array large{graph::ShapeBinding{batch, 4}};
const std::array points{graph::ExactShapeVariant{small}, graph::ExactShapeVariant{large}};
auto family = graph::compile(logical_graph, environment,
    {.shape_policy = graph::CompileShapePolicy::kFiniteExact, .exact_variants = points});
auto invocation = family->executable.create_invocation({.shape = large});
```

各原生常量必须对所列每个形状都合法，任一点校验/准备失败则整表失败，原 caller tensor 不被消费。
共同编译先验证完整形状表，并累计编译及槽配额；共享 ResourceBudget 仍按实际资源收费，
MLX/driver 私有字节仍 UNKNOWN。精确表不证明 general/capture/零复制。
CPU＋Vulkan 的固定／有限形状数据链已接入共同入口，配置实际 host、device 与 staging 域即可选择双向
搬运；原生 Copy 在冷绑定录制，同址改内容后可直接重复 submit。当前原生复制要求跨度和偏移满足4字节对齐，
原生数学域仍由实际 lowerer 决定。Apple也通过实际Source准备共同固定／有限形状入口，
不能拿局部数据链推导全部混合／分页／custom 路线已经覆盖。
pure control 前驱通过冷可达性保留，并在原生完成/barrier 边界排序；只有控制用途的输出可能需要
物化，Apple managed region 与 hybrid 会有显式等待。hybrid `prepare_replay` 的
`HostWaitPolicy::kReject` 仍可拒绝这些拓扑。外部 effects/resource ports 仍需真实 qualified carrier，
GPU native admission 明确拒绝未资格 carrier。CPU+Apple/Vulkan 可将 effect 显式放在
`kCpuTarget`，用 `CompileOptions::effect_bindings` 声明实际固定 footprint；原 CPU Provider
与 Graph 校验负责端口、alias 和执行。零输出 effect 不会被剪掉。不同资源 ID 的重叠 footprint、
跨片段 tensor/constant alias、缺失/重复 binding 都拒绝；未知 native input/weight 的 alias
无法核对时拒绝与 named effects 组合，caller 可选原 checked host binding。
接受后失败保留实际写入的前缀，停止后继，不能假报 rollback。固定 footprint/backing 由 caller
保活并跨 invocation 同步。普通公开 C++ 消费示例见
[public_effect_consumer.cc](../../tests/graph/public_effect_consumer.cc)。
CPU+CUDA 的既有 effect 能力保持原规则；有限形状表使用同一份累计编译配额，不能按点重置。
该形状表不增加未声明的动态域、任意 GPU mutation 或 capture 能力。

## 当前运行时能力

| 能力 | 当前合同 |
|---|---|
| DAG | typed、多输入/输出 node；共享 value、分支、多个 published output 与显式 control dependency |
| 外部资源 | graph-local `EffectResourceId`、按 schema 端口绑定、read/write 冲突验证与固定 backing footprint；独立域不加边，未知副作用保守排序 |
| shape | concrete `TensorDesc`，或独立 min/max/multiple 维度、有限 sum/product/exact-quotient/broadcast 派生维度及 equal/less-equal guards；同一 `DimensionId` 共享值 |
| specialization | 默认 exact：compile 绑定全部独立维度；declared-domain：create_invocation 绑定。派生值由程序计算，Invocation 总有具体 `SpecializationSignature`；域根没有虚构的 exact signature |
| execution target | `Context` 是执行者；node placement 只选择 target，不把 allocator 或 Queue 混入 identity |
| storage domain | exact `MemoryResourceRef` 加 workspace hard budget；value placement 与真实 copy 只由 storage 决定 |
| execution lane | shared lane 绑定 target、role 与 caller Queue；delegated provider 为每个 island 产生无需 QueueBinding 的 synthetic lane |
| placement | node/value 可 required 或 preferred；其余按输入 locality、边界数和稳定 target/storage 顺序确定性选择；prepare 尝试有硬预算 |
| region partition | provider 在冷路径提出非空 node-ID 集合，支持单节点、连续和非连续 region；与 leaf 共用有界冷搜索，完整覆盖且联合收缩无环后才准备 artifact；绑定首个完整可行计划，计数不证明性能最优 |
| boundary | canonical tensor 只有在真实 storage residence 改变时才产生 `Copy`；换 target 或 lane 本身不复制 |
| memory plan | 每个 `(logical value, storage)` 是一个 physical instance；默认只复用 happens-before 已证明有序的 lifetime，显式允许时才为硬预算增加可报告的顺序边 |
| scheduling | 冷路径先做确定性 lane-aware Kahn 调度，再建立 happens-before 和 workspace；同 shared lane 尽量连续录入以减少 Queue segment |
| replay | `prepare_replay()` 冷验证 bindings、transfer、Queue 与 delegated endpoint identity、dependency route 和 Completion 容量 |
| Provider replay 成本 | `CompileOptions::provider_replay` 在冷编译筛选 leaf/region；默认允许所有合法类别，严格模式只接受 prepared |
| Provider 准入 | 全局 `allowed_providers/denied_providers` 和逐节点 `node_providers` 硬约束同时筛选 leaf/region；可指定家族或精确 Context 配置槽位，不因融合或回退而放宽 |
| capture | `kCommands`、`kPreferCapture`、`kRequireCapture`；capture 只适用于 shared Queue segment，delegated island 不能伪装成 Queue capture |
| report | `CompileReport` 分列 target/storage/lane/segment/runtime owner；`ReplayReport` 给出 exact endpoint 后的 ordered/imported/host-wait 事实 |

Provider-private layout 只能留在一个 region 内，公开 region boundary 当前必须回到 canonical `TensorDesc`。
因此 Graph 不伪造一个没有 layout producer/consumer 合同的通用 `Repack` step；将来某个真实 provider 需要
跨 region opaque layout handoff 时，必须先增加有版本的 boundary-layout 合同和可执行 repack owner。

对于 ggml、MLX 或 TensorRT 这类自己 host 图/engine 的 runtime，adapter 在 compile 冷路径把一个
HYLE region 翻译成 provider-native artifact；单节点 region 可以像函数一样重复提交，多节点 region 可以
保留融合。Graph 不会因为库已链接就自动发现这些能力，也不会进入第三方黑盒逐 kernel 替换。
实际 Source／Provider 提案自身能力，冷编译核对所选存储、端口和完成依赖；普通入口负责装配已支持的实现。
当前普通CUDA Graph入口已组装`graph::ggml_cuda::provider()`：在Compute指南限定的浮点张量域，
把可合法收缩的Copy/Cast/Concat、广播/Unary、norm/归约及无mask/浮点mask Softmax节点集合交给同一原版ggml程序；无关节点的插入位置不拆散片段，
离开再进入集合的路径不强行融合。它可与同设备cuBLAS片段串接，共用真实Completion。
每个invocation持有独立原生资源，旧结果不被其他槽覆盖；prefer-capture对该managed程序使用
commands，ggml保留自己的内部capture。它不是全算子覆盖或其他平台接入完成的声明。
普通SDPA节点亦复用Compute的完整原版attention程序；Linux优先适用的FlashInfer，
适用的BF16域随后选择原生融合程序，再由ggml承载其它域，具体类型/head/数值边界见
[Compute指南](compute.md#普通-cuda-入口)。SDPA作为独立operation执行，不被ggml张量region接管。
多节点张量region保留逐节点的半精度舍入和发布中间值。
需要外部 CUDA capture 的消费者，应在编译时用
`CompileOptions{.provider_replay = ProviderReplayPolicy::kRequirePrepared}` 排除 managed 程序，
再由 `prepare_replay(..., {.mode = ReplayMode::kRequireCapture})` 验证实际 capture 支持。
prepared 本身不是 capture 保证；编译完的 managed artifact 不会在 replay 时临时改选 Provider。
普通允许 managed 的入口继续保留原版 ggml 内部融合/capture 和真实 commands 完成边界。
`kCommands`只规定Graph外层的重放方式，不保证Provider内部已经预热。当前ggml CUDA程序会在本槽图属性稳定后延迟建立自己的capture；
性能测量应分别保留每槽首次执行、内部准备和稳态调用，不能把槽0已预热视为槽1也已预热。

第三方的 `compile` 不必然意味着每次调用零分配：Provider 分别声明 prepared/managed/unknown 成本，
严格使用者可在 compile 要求 prepared-only；它与 `ReplayMode` 的 command/capture 选择互相独立。

Region 由数据流而非插入顺序定义。例如 gate/up MatMul 与 gated activation/down MatMul 之间插入无关
node，或者先创建 up 再创建 gate，不应阻止 oneDNN 识别同一 MLP。反之，如果 data/control 路径离开该集合
后又回到集合，就不能把它收成一个 execution island；多个各自合法的集合也须通过联合无环检查。
普通用户不需要手工分区。实现 Provider 时可以用 `is_region_contractible()` 检查候选：合法但不能收缩返回
`false`，空集、重复或异图 ID 返回错误；Graph 在接收 proposal 后仍独立验证。连续集合保留快速路径。

leaf/region 的结构提案共用 `compute::ProviderRouteCostFacts`，workspace、persistent、copy/repack、sync 和
dispatch 都允许明确 unknown，不能写零补齐。默认 `kAllowProviderRegions` 让 region 与 leaf 同池选择；
从最小未覆盖 node 开始，先看 preferred target，再按配置 target/Provider 顺序，
最后才以覆盖数与 dispatch 打破等价次序。只有本次候选都声明 dispatch 时才比较该维度；否则按稳定
候选顺序。它不会因为另一 Provider 的 region 更大或 dispatch 更少就绕过前排可用 leaf，也不是推理性能排名。
旧的“相对本 Provider fallback 节省若干 ns”入口和 `kRequireEstimatedBenefit` 已移除，不保留别名。
这套低层可行性规则保留真实资源约束；默认 Provider 和量化实现由已构建路线的规则匹配，
prepare/compile 按实际 dtype、布局、设备及资源条件决定支持或拒绝。
专家可显式筛选 Provider、约束具名节点或关闭 region，这些约束本身不证明性能最优。

### 指定某个节点的实现，其它节点仍可外包

例如用户已经配置了一个自定义实现和成熟 Provider，希望某个 MatMul 节点只能由自定义实现准备，
其它节点仍参与正常选择。约束与 target/storage placement 独立，不用拆出第二张图：

```cpp
// context、graph、matmul_node 已创建；slot 是调用方配置的 Provider 槽位。
auto family = context.provider_identity(slot).value();
auto instance = context.provider_instance_id(slot).value();
const std::array required{
    NodeProviderConstraint{matmul_node, family.id, instance}};
auto compiled = compile(graph, context,
                        CompileOptions{.node_providers = required});
```

省略 `instance` 时，同一 `ProviderId` 家族的任何已配置实例均可；提供它则只允许该 Context 的那一个
槽位。重新创建 Context 会得到不同实例 ID，不要持久化 ID 或把它当作可跨进程的版本/性能身份。
`NodeProviderConstraint` 是必需条件，不是软偏好；不相容的 region 不会参与 placement 或创建 artifact。
若没有完整可执行路线，compile 返回错误，不能为凑齐计划换用其它实现。安全资源拒绝后的重选同样守约。
它约束的是准备逻辑 node/region 的 Provider。未融合 View 由 Graph 自己处理，没有 Provider：若对 View
指定必需 Provider，只有完整覆盖它的合格 region 能满足，否则 compile 失败。全局 allow/deny 不关闭
metadata View；Graph 插入的跨存储 copy/sync 仍由 backend 完成，不能把 Provider filter 当“设备禁用”。

全局筛选用 `CompileOptions::allowed_providers/denied_providers`，元素都是家族 `ProviderId`：allow 空表示
不限制，deny 优先，重复 ID 无额外语义。全局被排除的 Provider 不会被 Graph 查询 extension、发现候选或
准备算子，但 Context 的配置/构造已经发生，筛选不是 library 卸载或安全隔离。未知 allow-only 家族、
不匹配的必需实例或与 placement 冲突的约束不会被默默忽略。没有候选满足约束时返回 `Unimplemented`；
已匹配路线的资源拒绝可以保留 `ResourceExhausted`，真实 Provider 错误继续按原错误返回。无效家族、
异图节点和重复节点声明返回 `InvalidArgument`。

这些 span 只借用至 `compile()` 返回，之后可以释放或修改。筛选/索引只存在于冷编译，Invocation 不保留
它们，也不在 submit/replay 再查策略。普通用户全部留空即可。硬约束和 Context 安装顺序
本身不证明性能排名；具体默认组合以路线验收为准。

### 何时要求 prepared-only

CPU自研Native MatMul已退役。普通矩阵仍可绑定成熟库；高rank只有managed候选或严格内存
政策无法接受成熟库未知上界时，compile会明确拒绝该请求，不能要求Native后备或自动放宽政策。

若热路径不能接受 Provider 实例化执行 metadata/array 或内部 shape/cache 工作，在冷编译时明确要求：

```cpp
auto compiled = compile(
    graph, context,
    CompileOptions{.provider_replay = ProviderReplayPolicy::kRequirePrepared});
```

这个约束同时作用于 leaf 和 region：managed/unknown 候选不能影响 target/region 选择，也不会创建
artifact；仍可选择其他合格 Provider，没有合格路线则 compile 失败，不在 submit 时放宽。默认
`kAllowAny` 保留已安装 Provider 的合法路线，但不把 unknown 解释成零成本。prepared 也不等于 shared
Queue 或可 capture：已预备的 delegated artifact 同样可以合格。

实现 region Provider 时，在 `RegionProposal::replay_class` 提前声明类别。known 值必须与资源提案和
artifact 一致；unknown 只能在 allow-any 下细化，不能先以 unknown 进入严格计划再补声明。此策略只约束
Provider 成功 replay；逻辑 Copy 以及含 View 的 Provider region 同样受筛选。它不替调用方 executor/callback、
编译器插入的 backend transfer、失败诊断或 Completion 的独立合同证明零开销，也不把声明本身当成
性能资格化证据。

## 单 target/storage/lane：自然短路径

固定 shape、单 Context 的调用不需要显式写 environment 或 replay options：

```cpp
#include <array>
#include <hyle/graph/executable.h>

using namespace hyle::compute;
using namespace hyle::graph;

auto created = GraphBuilder::create();
if (!created) return created.error();
GraphBuilder builder = std::move(*created);

auto x = builder.input(x_desc);
auto weight = builder.constant(weight_desc);
if (!x || !weight) return hyle::Status::InvalidArgument("graph inputs");

auto normalized = builder.add(RmsNorm{}, *x);
if (!normalized) return normalized.error();
auto output = builder.add(MatMul{}, *normalized, *weight);
if (!output) return output.error();
if (auto status = builder.publish(*output); !status.ok()) return status;

auto graph = std::move(builder).finish();
if (!graph) return graph.error();
const std::array constants{ConstantBinding{*weight, weight_view}};
auto compiled = compile(*graph, context,
                        CompileOptions{.constants = constants});
if (!compiled) return compiled.error();

auto invocation = compiled->executable.create_invocation();
if (!invocation) return invocation.error();
if (auto status = invocation->bind(*x, input_view); !status.ok()) return status;
if (auto status = invocation->bind(*output, output_view); !status.ok())
  return status;

// 单 active shared lane 会自动走同一套 cold prepare；后续相同 Queue 可直接复用。
auto done = invocation->submit(queue);
if (!done) return done.error();
return done->wait();
```

`constant()` 只声明逻辑 value；`CompileOptions::constants` 提供 payload。默认 `ConstantPolicy::kBorrow`
不复制字节，源 storage 必须不可变，并活过 Executable、全部 Invocation 和 in-flight Completion。即使
Provider 另有 packed representation，也不因此缩短借用期。

模型加载后要释放 staging buffer 或 mapping 时，显式选择物化：
下面的借用 view 写法适用于同步即时搬运；CUDA 等 queued-direct 路径须提供下文的 owner 与 ready lease。

```cpp
const std::array constants{ConstantBinding{
    *weight, weight_view, ConstantPolicy::kMaterialize}};
auto compiled = compile(*graph, context, {.constants = constants});
if (!compiled) return compiled.error();
// 此后可释放 weight_view 的源 allocation/mapping；不可再通过旧 view 访问它。
// Executable 及其 Invocation/Completion 共享编译时创建的不可变副本。
```

物化把完整 TensorDesc 所需的字节复制一次，保留 dtype、encoding、shape、strides 和 padding；它不是
重新量化或 Provider packing。目标为 constant 的 external storage 域（默认 0，也可由
`ExternalResourceContract`/value placement 指定），无需与源相同。配置的某个 target 必须支持这条
实际 Queue 复制或原生冷搬运；缺少搬运能力即失败。原生冷搬运所需 staging 计入编译峰值，零字节常量不提交复制。

源必须在调用前 ready，并在整个 `compile()` 期间保持不变；冷编译等待自有复制到终态，不执行用户
Graph，也不替用户等待此前独立 Queue 的工作。最终 Provider prepare 只看到有效副本；候选重试复用
资源、描述和对齐要求都兼容的 ready 副本，换资源或提高对齐要求则准备相应副本。编译失败不修改源，
也不发布未完成的副本。queued-direct 冷复制在首次提交前要求源 ready lease、源／目的 StorageOwner
和实际 metadata allocator owner；缺少时返回带原因的 FailedPrecondition，不静默降为 Borrow。
同一 target 的复制合成一条原 QueueSequence。等待错误若未证明实际退役，已接纳批次保留所用源、
canonical 副本、资源及预算到原完成机制确认退役；调用者可释放自己的 owner/lease 句柄。
同步借用路径的副本 `MemoryResource` 对象仍须由调用者保活；提供 StorageDomain::owner 时则由副本共享该所有者。

同一次 compile 的 general/exact 或 finite-exact 路线，按同一 Graph slot、精确描述、实际目标资源和足够的
分配对齐共享 canonical 副本；不同目标资源分别持有，不按设备名字或 coherence domain 混用。
取得的 exact 子路线可以比父表活得更久。不同 Executable 编译之间不隐式共享。
Region Provider 可通过下述入口共享兼容 packing，未接入的 Provider 仍独立持有并计费。每条路线的
`materialized_constant_bytes()` 都描述它借用的 canonical 副本，不能把各路线的数值相加当成物理总量。

同一计划的不同存储域消费同一常量时，各域需要的不可变副本也在编译期间准备，重放不重复复制。
metadata View 共享目的域中的完整常量根，再按具体形状和偏移取子范围。region 与 packing factory 取得的是
本片段实际使用的副本。`kBorrow` 的主绑定仍借用原源，其寿命要求不因已有其他副本而缩短。
Apple 原生源可同步复制 ready 主机／共享 Metal 来源至真正可写的共享 Metal 分配，无需 CPU Context；
readonly 原生导入只能作源。Vulkan 冷能力支持主机上传、下载和四字节对齐的同 Runtime D2D，
原生尾部 padding 仅供完整逻辑根使用，裁短的范围不能借此扩大权限。实际源读取、设备完成或目标写入失败即返回原错。

### 常量所有权与不可变 packing

需要让冷准备的失败前缀独立保活时，可将真实资源移入
`compute::StorageOwner::adopt(context, std::move(resource))`；资源的 `.ref()` 保持同一身份。
把该 owner 放进 `StorageDomain::owner`，将上传完成的 Buffer 交给 `owner.adopt_buffer()`，再把
返回的 ready lease 与 owner 放进 `ConstantBinding`。这不复制权重，lease 只读且必须覆盖绑定的
真实 resource/handle/子范围。CUDA 简单入口也接受 `compile(graph, context, owner, options)`。
默认采用系统 metadata allocator；自定义 allocator 传其真实 `std::shared_ptr<Allocator>` 给
`StorageOwner::adopt` 的第三参数，`.ref()` 必须与原 Context allocator 完整匹配。
已有 `LeasePool` 的完整独占 writer 可交给 `owner.freeze(pool, std::move(writer))`，
零拷贝取得保活真实资源及 allocator 的只读 lease；成功后 pool／writer 均为空，失败均保持。
该入口拒绝不同资源、无法保活的 allocator 和已经附有 lifetime owner 的 pool。
手工构造 ready lease 时，其 metadata 或 ResourceBudget 的实际 allocator 必须由该 owner 持有，
或为进程系统 allocator；仅保留不拥有 allocator 对象的 AllocatorRef 不足以满足异步寿命。

`region_provider.h` 的 `RegionCompileRequest::constant_artifacts` 只在 `compile` 回调有效，resource query
不提供它。`get_or_create(constant, declaration, factory, context, &view)` 返回 Compute 的显式 prepare
outcome；非 prepared 应按原 kind 传播，不能把 factory 的真实 OOM 当成可以换路线的预算拒绝。

`ConstantArtifactDeclaration` 声明非零 `ConstantPackingKey::format`、全部配置 discriminator、已登记的
storage、packed bytes、独立的 optional persistent metadata bytes，以及所供 allocator 之外的 private
resident/peak 上界。persistent 未知不等于零；它是制品分类计数，不能拿 private peak 代替或与完整编译
配额再相加。Graph 自动绑定本次 family、
具体 Provider instance、logical constant/描述和 storage；同 key 不得改大小或上界。两个 token 数量不同
的 exact regions 可以复用同一权重格式，但不同格式、设备实例或 storage 不误共享。普通 Graph 用户
无需设置缓存或添加开关；采用该入口的是 Provider 作者。

factory 仅在 miss 执行，接收 `ConstantArtifactPrepareRequest` 并返回拥有 replay 全部内容的
`ConstantArtifactPayload{state, destroy, retire_preparation}`，不能借用 caller scratch 或递归使用 access。
同步准备直接结束；`queue_preparation=true` 时，将 accepted prefix 记录到 request.sequence，
由 Graph 完成原序列并在首次等待前保留真实所有者。先用 `can_prepare_queued` 查询资格，缺少 owner
时保留原 runtime 运算；提交后错误按原错误传播。失败 partial payload 同样保活到实际退役，
可选 retire_preparation 此时释放 native 临时程序/workspace，destroy 随 immutable payload 最后释放。
等待报错不等于退役。成功的 `ConstantArtifactView` 是只读借用，直到使用它的 compiled region
销毁都有效；可变 Invocation/workspace/capture 不在此共享。不同 compile 不以源地址做全局缓存，borrow
常量原有生命周期也不缩短。

受管 packing 不再重复填写 `RegionArtifactProperties` 的 persistent/packed 字段或 region 自身 private
compilation bounds；allocator 请求已自动计费，额外私有上界由 declaration 单独准入。单路线报告计它
实际使用的 unique packed/persistent bytes（含 unknown），family 总报告只计实际共享 owner 一次；step 的 packed 字段仍只表示
该 step 独占的 Provider packing。编译候选放弃时，先销毁 region 再释放共享引用；发布后没有热 cache。

### 限制编译制品的常驻存储

一张变体表会同时保留多个制品。使用 `compiled_storage` 限制它们累计的 canonical 常量、Provider
声明的 persistent bytes 和 region 额外 packing，而不是只限制每条路线：

```cpp
CompileOptions options;
options.compiled_storage.max_bytes = 256 * 1024 * 1024;
auto result = compile(graph, context, options);
if (!result) return result.error();
const auto storage = result->report.total_compiled_storage();
// 有限预算默认拒绝未知量，成功时 total_bytes() 有值。
const auto compiled_bytes = storage.total_bytes().value();
```

`compiled_storage()` 是单路线；`total_compiled_storage()` 是 general 与全部 exact 的集合。同一张表中
确实共享的 canonical 副本与受管 packing 只计一次；独立 Provider 制品/packing 每份都计。子报告包含该子路线
独立保活所需的常量，即使父 Executable 已释放；报告本身不保活 payload。普通算子的 persistent facts
也通过 `step().persistent_bytes` 如实传播。求和溢出是错误，不会饱和成一个看似合格的数值。

`max_bytes = SIZE_MAX`（默认）不限制此项；有限上限可以触发冷路线重选。所有路线都不能满足则失败。
后继变体只取得余量，不因 variant 另开预算；不保证对所有变体组合做全局最优搜索，失败也不会偷偷丢掉
请求的某个形状。这个检查发生在制品准备后，**不是限制准备过程分配峰值的 allocator quota**。

某些 Provider（例如当前 oneDNN adapter）无法报告完整 persistent 量，有限预算默认拒绝。确实愿意接受
部分约束时显式设置 `options.compiled_storage.unknown = UnknownStoragePolicy::kAllow`；仍检查
`known_bytes()`，但 `unknown_provider_count` 非零时 `total_bytes()` 返回空，不能宣称满足完整总额。

本项不包含 Graph/Compute 控制 metadata、allocator rounding、借用的外部 backing、Invocation 的
workspace/capture、编译临时量或进程其它分配，也不是 OS RSS/物理显存预算。borrowed 常量没有新增
编译存储，并不代表它不占内存。编译冷峰值用下面的独立配额；capture 私有完整上界仍未交付。单个 storage domain 的
workspace 仍可独立用 `StorageDomain::workspace_budget_bytes` 限制。

### 限制整个编译过程的新增存储

需要同时控制编译临时峰值与最终保留量时，用 `compilation_storage`，不是上面的 artifact 分类限额：

```cpp
CompileOptions options;
options.compilation_storage = CompilationStorageLimits{
    .max_resident_bytes = 256 * 1024 * 1024,
    .max_peak_bytes = 384 * 1024 * 1024};
auto compiled = compile(graph, context, options);
if (!compiled) return compiled.error();
auto storage = compiled->report.compilation_storage().value();
```

集合包括 Graph/Compute 经所供 allocator 取得的 metadata、canonical 常量请求字节，以及 Provider
在 prepare 前声明的额外私有 resident/peak 上界。它不会再加一次可能已经经过 allocator 的
`persistent_bytes`。general、全部 exact 子路线、候选搜索临时量和报告共享一个配额；canonical 副本只
计一次。Provider 临时上界在准备期间预留，成功后只留下 resident；拒绝/失败先清理实际对象再归还。
Region 的候选发现与资源查询也显式取得本次编译 allocator；其 scratch 必须在回调返回前释放，不能回用
Context allocator 绕过额度，或在这些尚未准入私有上界的阶段创建私有堆状态/制品。

`max_peak_bytes` 在受管请求前准入；`max_resident_bytes` 在编译 scratch 已释放、整张变体表发布前检查。
后者失败时整个 family 回滚，不会静默少返回几个指定变体。报告是发布时的快照，不是之后释放子对象时
不断变化的计数；父/子报告都给出相同的 family 快照，单独保留报告不会保活 tensor payload。
Invocation 使用独立 runtime allocator/配额，不继续消耗 compiler quota。

任一上限有限时，未知 private 上界默认在 Provider prepare 前拒绝，可触发安全的冷路线重选；默认都为
`SIZE_MAX` 时只观察。显式 `unknown = UnknownStoragePolicy::kAllow` 只限制已知部分，
`has_unknown_storage` 会保守记录已尝试路线的未知量，失败后不能把历史峰值改写成完整总量。Native
编译分配全部走所供 allocator，所以额外 private 为 0；oneDNN 等未给出上界的 Provider 仍是 unknown。

这仍不是 OS RSS 或物理显存配额：输入 Graph、借入的外部 backing、Context/Queue/backend/driver 控制
开销、allocator rounding、错误诊断以及 Invocation/capture 不在编译集合内。私有声明可能是保守上界。
未配置时不创建 compiler tracker；配置后的成本在冷编译、分配和释放，不给普通 submit 增加扫描或锁。

### 限制所有并发 Invocation 的运行存储

需要控制同一张编译图同时创建多少运行存储时，启用独立的 family-wide quota：

```cpp
CompileOptions options;
options.invocation_storage = InvocationStorageLimits{.max_bytes = 256 * 1024 * 1024};
auto compiled = compile(graph, context, options);
if (!compiled) return compiled.error();
auto run = compiled->executable.create_invocation();
if (!run) return run.error(); // 超额或 backing OOM：没有接纳工作，已回滚。
auto usage = compiled->executable.invocation_storage_usage().value();
```

general 和全部 exact 子 Executable 共用额度，单独保留子 handle 不会得到另一份预算。计费包括
Invocation 的 allocator metadata、LifetimePlan 的实际 backing 请求字节和 Provider 额外私有存储上界；
共享编译常量不在这里再扣一次。Native 的额外私有量为 0，因为其实例分配全部经过所供 allocator。
域路线的 workspace 按该路线已规划的上界预留，不按当前 token 点假装减小 backing。

配额不包含外部输入/输出、Context/Queue、编译常驻、错误诊断、栈或 allocator/driver 内部开销，不是
OS RSS/实际物理显存上限。metadata 按 requested bytes，额外 private 按声明上界保守预留；禁止重复计算。
有限上限默认拒绝未声明的 private 上界，可在冷编译改选其它合格路线；愿意只约束已知部分时显式选择
`UnknownStoragePolicy::kAllow`。`has_unknown_storage` 是保守的粘性标志，任何一次未知实例/capture 尝试
都会留下它，不能因对象后来释放就把历史 peak 当作完整量。

调用完成不等于归还存储：保留 Invocation 供下一次 replay 时仍占额度；丢弃正在运行的 Invocation 或
Completion 也不会提前归还，必须等真实终态和最后 owner 释放。单独保留的 specialization metadata
会继续占用自己的字节。创建失败可重试，多个线程同时创建使用同一严格额度；usage 各字段可以来自
并发变化的不同瞬间，硬 admission 不会因此超限。

目前 backend capture 的私有占用没有完整上界。有限且拒绝 unknown 时，`kRequireCapture` 在接纳前
返回 ResourceExhausted，`kPreferCapture` 改用合法 commands 路线；显式允许 unknown 才可带着不完整标记
捕获。未配置 `invocation_storage` 不创建 tracker，正常 submit 不新增逐项计费/分配/搜索。

## 独立与派生维度

例如消费者按 batch × tokens 提供一块连续的 activation，用派生行数描述它，不要再创建第三个独立
symbol 然后要求调用者重复维护乘积：

```cpp
auto batch = builder.dimension({.minimum = 1, .maximum = 8});
if (!batch) return batch.error();
auto tokens = builder.dimension({.minimum = 1, .maximum = 4096});
if (!tokens) return tokens.error();
auto rows = builder.dimension_product(Dimension(*batch), Dimension(*tokens));
if (!rows) return rows.error();
if (auto s = builder.require_dimension_less_equal(
        Dimension(*rows), Dimension::fixed(4096)); !s.ok()) return s;
auto spec = TensorSpec::dense(DataType::kFloat32,
    std::array{Dimension(*rows), Dimension::fixed(128)});
if (!spec) return spec.error();
auto input = builder.input(*spec);
if (!input) return input.error();
auto output = builder.add(RmsNorm{}, *spec, *input);
if (!output) return output.error();
if (auto s = builder.publish(*output); !s.ok()) return s;
auto graph = std::move(builder).finish();
if (!graph) return graph.error();

const std::array shape{ShapeBinding{*batch, 2}, ShapeBinding{*tokens, 128}};
auto compiled = compile(*graph, context, CompileOptions{.shape = shape});
// 只绑定 batch/tokens；实际输入、输出 TensorView 是 [256, 128]。
```

- `dimension_sum` 适合 concat 输出轴；`dimension_product` 适合展平后的 extent；
  `dimension_quotient` 要求除数非零且整除，不向下取整；`dimension_broadcast` 要求两值相等或其中一个为 1。
  零尺寸合法，广播 `(0, 1)` 得到 0，但具体算子仍可能不支持空输入。
- 操作数只允许非负常量或同 Builder 已存在的 ID；表达式天然无环，不读取 payload。无递归表达式解释器、
  用户 callback 或通用符号求解器。固定矛盾在构造/finish 时拒绝，其余约束在 exact compile 或 domain
  Invocation 创建的具体求值阶段检查。
- `dimension_bounds()` 给的是保守范围；guards 可以排除范围中的组合，各轴上界未必同时合法。过宽范围
  导致整数溢出时必须缩窄声明，不能依赖 guard 自动收紧。派生 bounds 由系统计算，不接受独立覆盖。
- `dimension_count()` 枚举全部 ID；`independent_dimension_count()` 和 `dimension_is_independent(id)` 用于
  识别调用者需绑定的部分。重复、缺失、外来或派生 ID 的 `ShapeBinding` 都会失败。signature 按 ID 顺序保存
  **全部**已解析值，不能把其 `binding_count()` 当成调用者的输入维度数。

上述默认写法仍生成 exact Executable：更换具体 shape 需要重新 compile，或由消费方缓存多个 Executable。
若 Provider 对整个范围给出覆盖证明，可显式编译 general 路线，随后仅在冷创建时绑定实际尺寸：

```cpp
auto compiled = compile(*graph, context,
    {.shape_policy = CompileShapePolicy::kDeclaredDomain});
if (!compiled) return compiled.error();
auto invocation = compiled->executable.create_invocation(shape);
if (!invocation) return invocation.error();
// bind 的 TensorView 仍是实际 [256, 128]，不是范围上界。
// Invocation 的 shape 固定；用 submit(queue) 重复执行，reset 不改变 shape。
```

这里的 `shape` 仍只绑定独立维度；domain compile 的 `CompileOptions::shape` 必须为空。
`is_shape_domain()` 区分域与 exact；域根及其报告的 `specialization()` 无效，实际 signature 从 Invocation
取得。报告中的 descriptor、workspace 和 copy 字节是保守上界，各轴最大值可能被 guard 排除，不能把它
当成合法输入。每个 Invocation 都有独占 workspace，尺寸变小不会重新规划或缩小预留；需要紧凑内存时
选择 exact 编译、下面的预编译 exact variant，或缩窄声明域。新尺寸会有冷 metadata/执行状态分配，
不应在每次 token 的 submit 中创建。

当前 Native 仅对 dense float32、固定 hidden width 的 RMSNorm 承诺完整域，可带固定 scale；动态 prefix
可引用派生维度。静态节点可组合原 exact Provider/region。动态节点没有完整域实现时 compile 明确失败，
不拿上界单点冒充通用路线；当前 exact-only region 不融合任何动态节点。常量描述在一个 Executable 内
固定，动态尺寸的外部值应建成 input。完整域制品与上界 exact point 制品仍分别准备。

### 动态 View 与生产者推导布局

直接 Compute 的 `View` 仍使用具体参数。Graph 中需要随 batch/token 数变化的 reshape、slice 或 broadcast
时，使用 `ViewSpec`：它保存同一 Builder 的 `Dimension`，编译时绑定为真正的 Compute View。比如把
`[batch, tokens, 128]` 展平、归一化后恢复：

```cpp
const std::array original{Dimension(*batch), Dimension(*tokens),
                          Dimension::fixed(128)};
const std::array flat_axes{Dimension(*rows), Dimension::fixed(128)};
// rows 由上面的 dimension_product(batch, tokens) 产生；input3d 是 original 形状的输入。
auto flatten = ViewSpec::reshape(flat_axes);
if (!flatten) return flatten.error();
auto flat = builder.add(*flatten, input3d);
if (!flat) return flat.error();
auto normalized = builder.add(RmsNorm{}, *spec, *flat);
if (!normalized) return normalized.error();
auto restore = ViewSpec::reshape(original);
if (!restore) return restore.error();
auto output3d = builder.add(*restore, *normalized);
if (!output3d) return output3d.error();
```

`ViewSpec::slice(axis, start, count, step)` 的 start/count 也可以是独立或派生维度；axis/step 固定，step 必须
为正。`permute(axes)` 固定轴顺序，`broadcast_to(dimensions)` 推导零步幅；负 axis 按输入 rank 归一化。
所有 ID 都必须来自同一 Builder。`add(ViewSpec, input)` 自动推导输出，不要求用户手工重建 strides。

- View 不复制 payload。reshape 仍要求既有 Compute 支持的连续 plain 布局和相同元素数；非连续
  slice/permute/broadcast 后需要实化时显式插入 `Copy{}`，不暗中分配或搬运。空切片保留源步幅，不能
  因元素数为零就把它与规范化 empty-dense 描述视为相同。
- 显式指定操作输出时，`TensorSpec::inferred(type, dimensions, format)` 只约束形状、类型与 storage
  format，真实 strides/block axis 由生产者推导。它只能作为输出，不能给 input/constant 使用；即使
  dimensions 全固定，`is_symbolic()` 仍为 true，表示布局待推导。原有 dense/concrete 输出合同不放宽。
- View 链支持 exact 与 declared-domain。完整域编译会证明 reshape 元素数关系、slice 边界和 broadcast
  兼容性；flatten/head split 可用既有乘积/整除表达式。动态 slice 常用
  `end = dimension_sum(start, count)` 配合 `require_dimension_less_equal(end, extent)`（step=1）。
  仅有相同上下界的独立符号不表示相等；无法用有限表达式/guard/区间证明时返回 Unimplemented，可改为
  exact 或补充真实关系，不能通过添加不成立的 guard 隐藏输入错误。非连续输入的 reshape 仍须先 Copy。
  Native Copy 支持 plain bounded-strided 输入并输出 canonical dense；不暗中加入 Copy。
  任意非 View producer 的 inferred layout 尚无通用完整域证明，需显式输出布局或使用 exact。
- 域内的具体 strides/offset 在冷创建、绑定时求出，Invocation 重放复用它们。共享 workspace 按根 backing
  的寿命和上界规划，不将不可能同时出现的最大 offset/span 相加；普通 exact 对象不增加 offset sidecar。
- `Graph::operation_kind(node)` 对所有节点有效。尚未绑定的 View 用 `view_spec(node)` 查看表达式；
  `operation(node)` 返回 FailedPrecondition，不返回假固定参数。输入和参数都已具体化的 ViewSpec 会在
  构图时折叠为普通 Operation，此时反过来用 `operation()` 查询。查询结果借用至 Graph owner 释放。

公开可运行的 slice→permute→broadcast→Copy→reshape 例子见
[`custom_kernel` sample](../../samples/package/graph/custom_kernel/README.md)，source/static/shared 使用相同 API。

### 有限精确变体

常见尺寸需要 exact Provider、独立 workspace 布局或精确 capture，而其它合法尺寸仍要能运行时，在
同一次冷编译中列出有限表；它不是自动增长的缓存：

```cpp
const std::array common_shape{ShapeBinding{*batch, 2}, ShapeBinding{*tokens, 128}};
const std::array variants{ExactShapeVariant{common_shape}};
auto compiled = compile(*graph, context,
    {.shape_policy = CompileShapePolicy::kDeclaredDomain,
     .exact_variants = variants});
if (!compiled) return compiled.error();
auto run = compiled->executable.create_invocation(shape);
if (!run) return run.error();
// exact_variant_index(): 命中时为声明序号，否则 nullopt，使用 general。
auto selected = run->executable(); // 独立保活实际路线，用来查询 Queue/workspace 合同。
```

- 表及绑定只借用至 compile 返回；每项必须完整绑定独立维度、满足原域和 guards。绑定顺序不限，重复
  concrete point 拒绝；序号按声明顺序稳定。普通 exact 编译不接受该表。
- general 必须可编译，所有指定 exact 路线也必须准备成功；任一失败则整次失败并清理，不返回缺项表。
  有限点不能替代完整域证明。exact 路线可采用只支持单点的 Provider/region，general 的限制仍独立成立。
- 创建 Invocation 先验证 shape，再按完整维度有序查表，最坏 `O(D log K)`；命中只实例化已有 exact
  计划，未命中实例化已有 general。两者都不重查 Provider、不 JIT、不扩表；warm submit 不查表。
- 顶层 `Executable/CompileReport` 的描述、workspace、队列及并发查询仍描述 general；
  `exact_variant_count()` / `exact_variant(i)` 枚举独立保活的子路线/报告。命中后应查实际路线，不把
  general 的队列/并发约束套到 exact 路线上。子 Executable 用无参数 `create_invocation()`，已固定尺寸。
- 所有路线共用累计 `max_*` 编译预算（per-provider cap 仍是每次发现的局部上限）。`search()` 描述当前
  路线，`total_search()` 是 general 与全部 exact 的计数之和及限制标志并集，没有虚构的总执行时间。
  前面路线花完预算可能让后续路线失败；`kAllowLimited` 不允许缺失指定变体。
- 常量副本共享不等于全部内存共享。workspace 仍由各 Invocation 独占，现有 storage workspace 上限是
  逐路线、逐 Invocation 的，不是所有变体和并发的总上限；不能据此宣称完整 aggregate budget 已交付。

只支持有限尺寸、没有完整域路线时，同一个入口改用 `CompileShapePolicy::kFiniteExact`，仍传
`exact_variants`，不另传单个 `shape`。整表在准备任何子路线前校验，全部成功后才发布 family；未命中
直接返回 Unimplemented，不临时编译或要求先有 general。这适用于只支持固定形状的原生实现。
父 `Executable/CompileReport.has_execution_route()` 为 false，只保存形状索引和已编好的子路线，
其 step／workspace／queue 查询不代表可执行路线；通过 `create_invocation(shape)` 或 `exact_variant(i)`
取得实际路线后查询。累计编译预算、共享 Invocation 配额和常量的原所有权合同仍有效。

## Symbol 与 target/storage/lane

共同编译按每个实际 source 的计算、资源和搬运能力选择路线，不要求所有 target 属于同一 `Device`，
也不需要用后端 ID 为 CPU＋GPU 开特例。默认 storage、原 owner、跨域 copy 和完成依赖仍逐项检查；
某个组合缺少实际桥接时明确拒绝该路线。普通单 Context 调用 `compile(graph, context, options)`；
下例展示需要显式资源约束时的同 CUDA 设备双 Context。

symbolic graph 在 compile 时产生 concrete specialization；Graph 本身不维护无界 cache。推理引擎可用
`SpecializationSignature` 管理 decode/prefill 等有限 Executable 集合：

```cpp
auto batch = builder.dimension({.minimum = 1, .maximum = 64,
                                .multiple_of = 1});
const std::array logical_shape{Dimension(*batch), Dimension::fixed(hidden)};
auto spec = TensorSpec::dense(DataType::kFloat16, logical_shape);
auto input = builder.input(*spec);
// ... add operations and finish the Graph ...

const std::array storage{
    StorageDomain{.resource = host_resource},
    StorageDomain{.resource = device_resource,
                  .workspace_budget_bytes = device_budget},
};
HYLE_ASSIGN_OR_RETURN(auto source_a, ExecutionSource::acquire(cuda_context_a));
HYLE_ASSIGN_OR_RETURN(auto source_b, ExecutionSource::acquire(cuda_context_b));
const std::array targets{
    ExecutionTarget{.source = source_a,
                    .default_storage = StorageDomainId{1}},
    ExecutionTarget{.source = source_b,
                    .default_storage = StorageDomainId{1}},
};
const std::array lanes{
    ExecutionLane{.target = ExecutionTargetId{0},
                  .role = ExecutionLaneRole::kCompute},
    ExecutionLane{.target = ExecutionTargetId{1},
                  .role = ExecutionLaneRole::kCompute},
    ExecutionLane{.target = ExecutionTargetId{1},
                  .role = ExecutionLaneRole::kTransfer},
};
const std::array shape{ShapeBinding{*batch, 1}};
const std::array resources{
    ExternalResourceContract{.value = *input,
                             .storage = StorageDomainId{1}},
};
auto compiled = compile(graph,
                        CompileEnvironment{.storage_domains = storage,
                                           .targets = targets,
                                           .lanes = lanes},
                        CompileOptions{.shape = shape,
                                       .external_resources = resources});
```

`NodePlacement` 选择 target，`ValuePlacement` 与 `ExternalResourceContract` 选择 storage；前两者可以表达
required 或 preferred。required route 不可执行时 compile
失败；preferred route 不可执行时按稳定候选顺序继续。冷 proposal 的 Unimplemented 表示没有对应能力；
真正 prepare/region compile 使用共同 `compute::ProviderPrepareOutcome`，只有明确的 unsupported 或
candidate-local resource-budget decline 可在回收部分 artifact 后回退。kind=Error 即使 code 是
Unimplemented/ResourceExhausted 也立即失败，不能从错误码推断 rollback。非法 kind/status 同样失败。
当前编译分别发现一次 region 与普通算子候选；普通算子按允许的 node/target/Provider 配置查询，即使前排
Provider 或 region 最后胜出，后排候选也可以被发现。`propose` 只返回冷能力与资源要求，不准备重型制品；
明确 unsupported 被跳过，其它查询错误立即传播，不能因该 Provider 最后未选中而吞错。
region 的 boundary、资源要求、runtime 与效果声明也在发现阶段查询并缓存；required target 不允许的
候选不做资源查询。查询不代表后面一定会 compile 该候选，Provider 不应在这里创建重型制品。
资源要求非法或后排查询出现真实错误，同样在准备任何制品前失败；明确 unsupported 只排除该候选。
某个 region 的已选资源路线，或某个 leaf 的 prepare 被明确安全拒绝后，释放本次构造的制品并排除该
资源组合，再从原候选池重新放置/分区；同一 Provider 的其它兼容资源仍可使用，也可改选另一 Provider、
另一 node set 或另一个允许的 target。禁止
leaf 回退时仍可改选完整 region 覆盖；允许 leaf 时也可以使用已安装的 operation 路线。重选不重复发现或资源查询，
不重置任何预算。region 联合成环、某个完整组合无 storage/copy 路线时，非递归搜索回溯到下一组合，
不会仅因该组合失败就淘汰其中所有候选。leaf/region/target 与 input/output/workspace storage、copy
承载 target、shared lane 使用同一有界搜索。默认先保持 locality/preference/default 次序，找到完整
可行计划即返回。它不保证穷尽所有组合或全局最优。
先构造完整 leaf/region/copy 的物理步骤和 storage 路线（包括最终输出 copy），之后才准备选中的制品；
因此后续已知无能力/无合法 storage/copy 承载 target 时，不会先给前面的步骤做 JIT/packing。最终 prepare
只使用已选路线，仍经 Compute 校验；不能原地换 Provider/target/storage，明确 decline 必须整图重规划。
这可能重新准备已成功的前缀，全部计入累计预算。不承诺提前证明任意 Provider 的真实 prepare 必定成功：
精确 workspace/persistent、lane/内存和实际 Queue transfer 仍需后续校验。只有完整可行计划才能返回。
最终 lane/schedule/workspace/compiled-storage 校验也在同一冷重选事务中：已准备路线缺可用 lane，完整
workspace 无法满足显式 storage 预算/允许的有界序列化，或编译存储超限/包含被禁止的未知量时，释放
本次制品及其调度/别名/内存决定，再尝试下一组合。
这不永久排除组合中的每个 Provider；所有路线都无解仍失败。真实 OOM、Provider/alias 合同错误不会被
按 Status 名称猜成“可以换路”。胜出报告不保留被拒绝路线的 delegated lane、序列化边或实现键。
`max_placement_attempts` 限制跨所有构造尝试的实际冷准备次数；若
在形成完整可行计划之前耗尽这个上限，仍然失败，不能返回部分图。
`max_route_search_states`（默认 65,536）限制累计 node-cover 状态及有多个合法值的 storage/copy-target/
lane 分支；重放同一选择前缀和只有一个合法值的决定不重复扣除。不在重选时归零，也不提前分配笛卡尔积。
尚未找到可行计划就用尽，返回 ResourceExhausted。恰好用满且没有省略必需搜索不算受限。
state 上限必须非零；prepare 上限为零只可能接受不需 Provider 制品的纯 metadata 路线。两者跨变体累计
扣除。候选表容量、搜索工作量和真正准备次数不是同一个预算。
找到完整可行方案后即结束准备，不为检查未使用替代方案继续消耗 state/prepare 预算。
发现阶段省略候选仍按 `search_policy` 裁决；真实 OOM/Provider 错误立即返回，未完成计划不会成为 Executable。

`View` 的 required 输出域会反向约束其 input 0 的物理副本：必要时先 copy 输入，再在目标域上建立零拷贝
View。报告明确显示 copy；若 alias 根在内部副本，输出由 Invocation 持有，用 `output(value)` 取得，
不能再绑定一个 caller 输出。只想把最终切片复制到 caller 存储时，用 `ExternalResourceContract`，
不必要求 View 本体换域。

`max_operation_proposals`（默认 16,384）限制本次编译接收的普通算子候选总数，独立于实际 prepare 次数；
禁止 operation fallback 时不发现 leaf，该上限也不使用。`max_region_proposals`、
`max_region_proposals_per_provider` 和 `max_region_proposal_nodes` 分别限制 region 总数、每配置实例数量与
复制保存的 node-ID 总数。默认 `search_policy = CompileSearchPolicy::kAllowLimited`：达到
容量时停止扩大相应发现，只用已接收的候选生成并验证完整计划；per-provider 上限不阻止还有总额时
探索下一家。成功报告的 `search().optimization_limited()` 明确说明发现是否被截断，可再查看
`*_limit_reached` 字段、实际接收数和 prepare 次数。这不是全局最优或性能资格声明。

需要拒绝受限发现时使用 `CompileSearchPolicy::kRequireUnexhausted`；它不是要求无限搜索。恰好用满
容量但没有任何候选/provider 被省略，不标记受限。两种政策都不放宽 coverage、required placement、
workspace 等硬约束，也不吞真实 Provider/OOM 错误。
候选上限不是“保证找到任何存在的可行解”：当前普通候选按 node/target/Provider 的稳定顺序接收，过小
的上限可能让后面的节点无可用候选，此时 compile 失败而非返回不完整 Executable。

Provider 的 `RegionProposalWriter::try_add()` 返回 `Result<bool>`：true 表示已复制并排序，false 表示
容量不够，应结束当前发现并返回 `Status::Ok()`；unexpected 必须原样传播。没有部分接收一个 node set，
Provider 可以使用临时栈数组，不必把它们保活到 compile 结束；不能把自身 OOM 转换成 false。

非默认 storage 的 external value 应提供 `ExternalResourceContract`，让 compiler 能选择真实 transfer route；不能让
Graph 从偶然地址猜 host/device locality。普通 canonical boundary 使用 Compute byte transfer，不隐藏
pageable staging、repack 或同步 copy。

每个 `StorageDomain::workspace_budget_bytes` 都是硬上限。默认
`WorkspaceReusePolicy::kExistingOrderOnly` 保留原有并发；若现有 partial order 下的峰值超过预算，compile
返回 `ResourceExhausted`。只有调用者显式选择 `kAllowSerialization` 时，compiler 才会在有界、精确的
happens-before 分析内给独立 lane 加顺序边并重新规划 workspace：

```cpp
auto compiled = compile(
    graph, environment,
    CompileOptions{.workspace_reuse =
                       WorkspaceReusePolicy::kAllowSerialization});
if (!compiled) return compiled.error();

// 非零表示为了预算牺牲了多少条原有并发关系；实际 peak 仍看
// workspace_bytes() 和各 storage report。
const auto added_order = compiled->report.serialization_edge_count();
```

该选项不会保证任何预算都可满足：预算小于单条不可拆 lifetime、分析规模超过内部硬界或不存在无环排序时仍
失败。预算本来足够时不会加边，`serialization_edge_count()` 为零。

## 编译约束与固定绑定

`compile` 按当前路线的既定候选规则，绑定第一个满足全部硬约束的完整计划。它不接收用户
timing profile，不执行用户输入或计时，也不保留多个已准备制品来比较性能。研发资格与具体
默认规则属于各路线实现；`CompileReport::search()` 的计数描述本次可行性规划，不证明最优性。

资源、region 覆盖、shape、alias、workspace 及累计编译预算仍是硬约束。只有明确的事务化拒绝
才允许回收本次制品并尝试下一种可行组合；真实错误立即返回。可行性规划只在冷编译发生，
Invocation 重放不搜索 Provider。

需要同一个 Executable 的多个 Invocation 并发运行时，可以在编译时要求这一能力：

```cpp
auto compiled = compile(
    graph, context, {.require_concurrent_invocations = true});
if (!compiled) return compiled.error();
```

该选项排除不支持并发 Invocation 的制品，不预留并发槽位，也不承诺吞吐量。每次并发执行仍需
独立 Invocation、输出和 workspace；外部可变资源的冲突由调用者同步。与必选 Provider 等约束
冲突且没有完整可行计划时，编译返回失败。

## 冷态 replay 路由

多 active shared lane 必须显式准备 Queue 集合；synthetic delegated lane 不接受 QueueBinding：

```cpp
const std::array queues{
    QueueBinding{.lane = ExecutionLaneId{0}, .queue = &cuda_queue_a},
    QueueBinding{.lane = ExecutionLaneId{1}, .queue = &cuda_queue_b},
    QueueBinding{.lane = ExecutionLaneId{2}, .queue = &cuda_copy_queue},
};

auto status = invocation.prepare_replay(
    queues, ReplayOptions{.mode = ReplayMode::kPreferCapture,
                          .host_wait = HostWaitPolicy::kReject});
if (!status.ok()) return status;

auto replay = invocation.replay_report();
if (!replay) return replay.error();
auto done = invocation.submit(queues);
if (!done) return done.error();
return done->wait();
```

`prepare_replay()` 是冷态 commit point。在任何 segment 提交前，它会：

1. 验证全部 external bindings、exact resource 与 Queue/target identity；
2. 对每个真实 copy 查询 immediate/queued-direct/peer capability；
3. 取得 shared Queue 和 delegated provider 的 exact completion endpoint，把每条 dependency 固定成
   ordered endpoint、同步 source、非阻塞 imported wait 或 host wait；
4. 按物理 Queue 汇总并预留 Completion slots；
5. 按 replay mode 建立 command、capture 或 hybrid route。

默认 `HostWaitPolicy::kAllow` 允许不兼容 backend 的显式 host fallback；延迟敏感部署应选择 `kReject`，在
进入 warm path 前拒绝这种 topology。host wait 不会藏在 report 里，也不会在持有 destination Queue/provider
submission admission 时等待上游。`ReplayReport` 提供四类 dependency 数和三类 transfer 数；
`dependency_route(segment, index)` 可查询精确边。

`kPreferCapture` 在不支持、不安全或 delegated 的 segment 上保留普通 route，最终可报告 `kCommands`、
`kCaptured` 或 `kHybrid`；`kRequireCapture` 遇到任一 delegated 或不能 capture 的 command segment 就失败。
当前 RTX 4070 纵切已验证：同一 CUDA target 的 compute/transfer lane 共用一个 device storage identity，只有
实际产生 segment 的 lane 要求 Queue；兼容 CUDA stream 可用 event 非阻塞导入前缀，单 Queue 的兼容 copy
segment 可 cold capture 并 warm replay。普通 CUDA FFN 还验证了 cuBLASLt/HYLE kernel 的
同 Queue 计算 capture；这些单 Queue 证据不承诺一般 multi-stream capture。

## 冷路径、热路径与调度取舍

- compile 做 exact shape 解析或完整域上界/覆盖验证、provider proposal/prepare、region 选择、target/storage placement、
  boundary materialization、lane scheduling、happens-before、liveness/alias/workspace；复杂度有显式预算，不读取历史 benchmark 常数作为 placement
  oracle。`kAllowSerialization` 的额外搜索只在冷编译且实际超过 storage hard budget 时发生；warm path
  只消费已经固化的 segment dependency。
- `create_invocation()` 实例化 workspace、provider invocation、预定长 metadata，并固定 delegated endpoint；
  域模式还求值具体 shape，选择已编译 exact 变体或冷实例化 general 制品；不复制整份计划或重选 Provider。
  应按服务并发槽创建并复用。
- `prepare_replay()` 做 Queue/endpoint/transfer/dependency/capture 资格化；bind、reset 或稳定地址变化会使其失效。
- warm `submit()` 只遍历 emitted steps、segment dependencies 与 bindings：不遍历原始 DAG、不推 shape、
  不搜索 provider、不 JIT、不增长 Graph heap、不重新查询 transfer/dependency capability。backend
  finalization/event 数按 segment 增长，不按 node 增长。
- 调度器优先减少 host 录入与 event 边界，不声称求解一般 DAG 的全局最优设备并行调度。需要跨请求公平、
  CPU worker 或 cancellation policy 时，在消费层组合 Sched；Graph 自身不启动线程池。

## Report 怎么读

`CompileReport` 是可复制的冷态事实副本，不保活 heavyweight artifact。它可枚举：

- `search()` 的 `operation_proposals/region_proposals`、已接收 region node 数、累计 prepare 次数、
  `route_search_states` 尝试状态数、`route_plan_rejections` 不可行组合数、
  `region_route_rejections/operation_route_rejections` 候选拒绝数和是否因
  容量省略了候选/provider；正常拒绝不等于搜索容量耗尽；
- `node(index)` 的原节点 ordinal、OperationKind 与 OperationTraits，包含精度及累加 dtype，
  symbolic View 也有独立条目；不保留 custom Operation payload；
- specialization、target/device/backend、storage workspace/budget，以及每 lane runtime mode/role/Queue/capture 需求；
- physical value、external/constant/workspace/alias ownership；
- `constant(index)` 按 Graph 常量顺序给出主绑定的 value、borrow/materialize policy、storage、TensorDesc，
  owned bytes 包含本路线为该常量持有的全部 canonical 副本。两个 count 统计主绑定政策，
  `materialized_constant_bytes()` 汇总实际副本字节；借用主绑定也可能有其他域的副本。报告不保活 allocation；
- operation fallback、provider region 和真实 copy step；
- leaf 的 `kRegionDeclined` 表示本轮曾安全拒绝包含该 node 的 region，不是 Provider 执行出错后重跑；
- shared/delegated segment、dependency、为 workspace budget 新增的 serialization edge、potential
  synchronization 和 concurrent-invocation 能力；
- provider/version、实际配置实例 `provider_instance`、dispatch、workspace、persistent/packed/borrowed
  bytes、blocking/threading 与 replay 类别。

`step(...)->implementation` 保留 Provider 自行声明的稳定实现键，与临时 prepare token、配置实例编号都
不同；`policy` 和 `preference` 记录这次冷绑定采用的版本化 HYLE 规则。unknown 不会被自动补成上述编号，
Graph 插入的 copy 和未融合 View 也没有 Provider implementation/policy。它们用于识别实际绑定的实现和
决策，不代表单独完成了合格计时或性能排名。

Provider 名称/版本文本由报告复制，释放 Context、factory 或 Executable 不会令报告悬空。
`provider()` / `step()` 返回值中的文本 view 借用报告共享状态，不能活过最后一个报告 handle；复制/移动
报告不复制执行资源，也不缩短这份共享文本的生命周期。

`persistent_bytes == nullopt` 表示 provider 无法报告，不是零。`delegated_segment_count()` 是 provider-owned
island 数，不是 hidden Queue 数。CompileReport 中的 `kQueueBindingDependent` 也不是已经发生 host wait；
具体 Queue/provider endpoint 固定后的真实路线只看 `ReplayReport`。

Provider operation/region 的 `step(...)->runtime.replay_class` 保留 Compute 声明的 prepared/managed/unknown
事实（选型见 [Compute](compute.md)）。它不从 region 大小、shared/delegated 或 capture 反推；unknown
不能视为 prepared。compile 验证 proposal/artifact 一致性，并按显式 prepared-only 政策筛选候选。
未融合 View 报告 `kView`：没有 Provider 制品、workspace 或 dispatch，`persistent_bytes` 为零，仍保留
DAG/control 依赖和 alias lifetime。它不要求 host 可访问存储；纯 View 图无需安装 kernel Provider、
无需 Queue，绑定后可直接 `invocation.submit()`。operation fallback 的关闭与 Provider 准备配额不关闭
这些元数据步骤。Graph copy 由 backend 负责，两者都不能用 Provider replay 字段替代自身成本合同。

`step(...)->provider_instance` 区分同一 Provider 家族的多个配置槽位：operation 来自 PreparedOperation，
region 来自实际绑定的 Provider extension，Graph copy 与未融合 View 没有 Provider 实例、implementation
或 policy。
ID 本身不保活 Context/artifact；报告可留作诊断，不能用
旧 ID 授权提交或当成跨进程、跨构建的完整性能 key。详见 [Compute 的身份与成本说明](compute.md)。

`CompileStepReport::node_count` 是该 step 覆盖的逻辑节点数量；`node_begin` 是最小 ordinal，不总是一个
连续 range 的起点。只有 `nodes_contiguous` 为真时才能按 begin/count 展开；通用代码使用
`CompileReport::step_node(step_index, node_index)` 查询精确节点集合。真实 Copy step 的 node count 为零。

### 原生路线报告

Apple 与 Vulkan 的原生 `CompileReport` 是可选冷态快照：复制环境、版本、输入输出、原节点序号、
数值 traits、固定规则和边界，不保活 Graph、Program、常量、custom payload 或设备 owner。
每次 compile 先 reset；报告分配失败也会拒绝编译。同一 report 不能被并发写入。
不传报告时不构造这些诊断数据。原生 dispatch、managed region 与内存字段各按
[Apple](apple-compute.md)和[Vulkan](vulkan-compute.md)的具体含义解释，不从逻辑大小猜实际驻留。

## 外部可变资源：声明访问与顺序

普通 tensor 的输入/输出依赖不需要资源 ID。custom operation 还访问外部状态时，先按
[Compute 资源 schema](compute.md#custom-operation-的外部资源访问)声明每个端口的 read/write，再把端口
绑定到 graph-local `EffectResourceId`。例如更新一个外部累积缓冲区后读取它：

```cpp
// update_op / observe_op 是已由 context 中的 Provider 支持的 custom Operation；
// 均有一个资源端口，分别声明 Write 和 Read，此例没有 tensor 参数。
using namespace hyle::graph;
auto state = builder.effect_resource().value();
auto update = builder.add_node(update_op, {}).value();
auto observe = builder.add_node(observe_op, {}).value();
const std::array ports{state};
HYLE_RETURN_IF_ERROR(builder.bind_resources(update, ports));
HYLE_RETURN_IF_ERROR(builder.bind_resources(observe, ports));
HYLE_RETURN_IF_ERROR(builder.depends_on(observe, update));
auto graph = std::move(builder).finish().value();
const std::array backing{EffectResourceBinding{
    state, EffectResourceFootprint::memory(state_buffer.view())}};
auto compiled = compile(graph, context, CompileOptions{.effect_bindings = backing});
```

`finish()` 不会按添加顺序暗中插入依赖。同域写→读、读→写、写→写必须已有 data/control path；多个
并行 readers 后的 writer 必须依赖全部 readers。read/read 和独立资源无需串行化。缺端口绑定、异图 ID
和冲突未排序都会失败；失败后可以在 Builder 更正绑定或补 `depends_on()` 再 finish。重复端口可指向同一
ID，`Graph::node_resource_count/node_resource/effect_resource_id` 可查询逻辑声明。未知外部副作用使用
`kExternalSideEffect`，与全部具名访问及其它未知副作用保持保守顺序。

`effect_bindings` 是**访问范围证明，不是执行参数传递或资源所有权**。custom 实现仍须自行取得对应资源，
且声明必须覆盖它实际访问的范围；此入口不自动安装 custom execution。地址化内存使用 Mem 的
`ConstBufferView`，按 resource/allocation/offset/size 检查；不同 ID 的范围不得重叠，即使两个声明都只读。
需要 alias 时复用同一 ID；同一 backing 的不重叠 subview 可以使用不同 ID。非地址化外部对象使用
`EffectResourceFootprint::object(canonical_owner)`；不能用两个包装对象的地址掩盖同一 owner，也不能用
object 掩盖本应与 tensor 检查重叠的内存。

这个检查依赖 Mem 的 canonical allocation identity，不探测 OS 物理页。若同一内存经多个 handle/resource
重复导入或映射，调用方必须统一声明它的 backing 身份，不能把不同包装当作独立性证明。

已使用的每个资源需恰好一个有效 backing；未使用的 ID 可不绑定。Graph tensor input/constant/output
不得与 effect memory 隐式重叠，compile 检查常量，`Invocation::bind()` 检查输入和输出，不因绑定顺序
不同而漏检。需要从可变资源产生数据时，用 custom operation 的显式输出建立 SSA value，不把相同内存
重新伪装成普通输入。

编译会复制声明而不保留传入的数组/span；footprint 本身仍借用，实际 backing、resource provider 或外部
owner 必须活过该 Executable、全部 Invocation 和 in-flight Completion。footprint 固定，不随 tensor
rebind/reset 改变。不同 Invocation 使用同一外部可变资源时，仍由调用方 Queue/Completion 排序；
`supports_concurrent_invocations()` 不代表用户资源天然无数据竞争。

Region 编译收到按域合并的访问集与 backing 声明。Provider 必须在资源提案与最终制品中一致承诺
`preserves_effects`；缺承诺的路线在创建制品前排除，前后矛盾是合同错误而非可忽略 fallback。资源顺序
沿已有 data/control 边进入 region 收缩和 lane/Completion，不增加全局 registry、锁或 submit 扫描。

## 所有权与常见陷阱

- `ExecutionTarget.source` 是可复制的 `ExecutionSource`，保留实际 Compute Context 或 native Runtime，
  plan 和内部 artifact 继续保留同一 owner。销毁或 `close()` 原端点后，已有 Invocation 的新 submit
  同样失败；重复执行需要保持原端点开放。内部保活保证制品与已接纳工作可以安全完成和释放。
  `StorageDomain` 中 `MemoryResourceRef` 的 provider
  仍是 caller borrow，必须活过 Executable、Invocation 和全部 in-flight work。
- Tensor view 不拥有 payload。输入、输出、shared Queue、external resource 与 native kernel 参数都必须活到
  Completion 真终态；delegated provider invocation 由 Graph Completion 保活，丢弃 pending Completion 不会隐式等待。
- `output_ownership()` 为 `kCaller` 的计算结果绑定 mutable `TensorView`；`kSourceAlias` 借用原输入/
  常量，`kInvocation` 使用本次运行的私有 storage，`kExecutable` 则是已物化常量或它的零拷贝 View。
  后三种不接受重新绑定；成功完成后用 `Invocation::output()` 取得借用 view，不得活过对应 owner。
  原生生成结果也属于 `kInvocation`，直接使用原生 backing，不重复申请 Graph workspace。
  `GraphOutput::description()` 不要求 Mem 表示；`tensor()` 只返回真实存在的内存投影，独立只读 owner 用
  `storage_lease()` 导出。旧 GraphOutput 或导出的原生 lease 仍会阻止覆盖。
- `reset()` 保留已规划 workspace，但清 external bindings；in-flight 时失败。失败终态后的 Invocation
  已退役，submit/bind/reset/prepare_replay 均拒绝；需从 Executable 创建新槽。未接纳工作的拒绝仍可
  修正参数后重试。设备/Provider 的恢复限制另按其 owner 合同处理。正常 bind/reset 后 replay route
  失效，多 shared lane 必须重新 `prepare_replay()`。
- 不同 published outputs，以及 output 与仍存活 input，不得绕过 plan 形成危险 overlap。只在相同
  storage/resource、最后一次读取且 provider 明确允许时选择 in-place；其它 partial overlap 被拒绝。
- 外部资源按上述域和 read/write 规则排序；纯 node、独立域与同域 read/read 不要为了“看起来顺序一致”
  手工串行化。固定 effect backing 与每次运行可重新绑定的 tensor 是不同生命周期。
- Graph 可以保存合法 blocked/opaque TensorDesc，但没有理解该 encoding 的 provider 时 compile 会返回
  `Unimplemented`，不会把压缩 bytes 当普通 scalar storage。

source、headers + static、headers + shared 都选择 `hyle::graph`。三种交付是同一 C++23 surface，不是
Native FFI，也不承诺跨编译器/STL 的 C++ ABI。

## 外部租约绑定与只读结果

同一个 `Invocation` 可在创建时通过 `InvocationOptions::resources`，或在空闲时调用
`prepare_resources()` 冷准备租约管理；接口随 `hyle::graph` 交付，不需要额外构建开关。
独立消费者见 [`resource_leases`](../../samples/package/graph/resource_leases/)；同一 CMake 工程只链接
`hyle::graph`。输入可以绑定 `ReadLease`，独占输入/输出可移动一个 `WriteLease`；描述、资源与 alias
仍由原计划检查。

```cpp
auto run = executable.create_invocation({
    .resources = hyle::graph::ResourceOptions{}}).value();
HYLE_RETURN_IF_ERROR(run.bind(input_id, input_description, input_read_lease));
HYLE_RETURN_IF_ERROR(run.bind(output_id, output_description, std::move(output_write_lease)));
auto done = run.submit(queue).value();
if (auto status = done.wait(); !status.ok()) return status;
auto result = run.result(output_id).value();
auto readonly_tensor = result.tensor().value();
```

示例的 `executable` 已按原 Graph 合同编译，pool、Context、资源、Queue 与 allocator 保持有效。
成功写绑定才接管原 WriteLease。每段已接受的 native use 自动保活；观察失败或丢弃 Completion
不能提前复用 backing。混合使用普通 TensorView 绑定时，该绑定明确借用存储，调用方负责保持其有效。

`result()` 返回保留原 Invocation 和绑定数据的 `GraphOutput`；`output()` 返回借用的 ConstTensorView，
两者所有权不同。旧 GraphOutput 在时，replay/reset/rebind 会拒绝；普通 caller/source-alias 输出
可用 `result.storage_lease()` 导出 ReadLease 再交 Direct，独立导出的读者同样阻止对应块覆盖。
`output()` 的借用视图本身不保活 backing 或阻止 reset/rebind；其可用期和覆盖规则仍取决于原存储 owner，不能把它当成结果租约。
内部 workspace 与物化常量保留原 Graph owner，`tensor()` 可读，`storage_lease()` 不伪造独立 Buffer 所有权。

需要把完成的状态留在缓存时，可显式调用`result.storage_lease(cache_budget)`，将原完整backing纳入保留预算。
这不复制数据、不改变读权限；同预算／同存储域内，能证明同一backing的别名共用一份费用，
原池已有费用直接保留，无法证明跨owner身份时保守分别收费。只想保留状态时，对状态／carry调用，
不必把分数、诊断输出和全部中间值都纳入缓存预算。超额返回失败，原结果及其权限仍可使用。
费用跟随原租约域及其在途使用，可能晚于移除会话或释放最后一个业务句柄才退还。
这是完成结果的保留准入，不预留Provider尚未执行的分配，也不约束驱动、原生缓存或未知私有峰值。
需要独立存储租约时，可在保持计算顺序和舍入边界的前提下，把纯元数据 reshape 放在最终 Cast 等物化操作之前，发布并绑定物化结果；最终 View 若根在私有 workspace，仍不能独立导出。
`close()` 停止新接纳且不隐藏全设备 wait；`drain()` 关闭接纳后等待本实例实际退役并保留首错，
旧结果仍可读取。需要继续重放时等待本次 Completion，不调用 close/drain。

Invocation的资源准备和`CompileOptions`中的原程序／workspace预算各按其实际范围计费，
保留结果预算由上述导出显式选择，不把它们当成整设备内存上限。
控制与 public Completion 观察须串行化。每实例一个冷完成观察槽；旧 Completion 未观察/放弃时拒绝新提交，
drain 能主动观察原 Completion，并等待真正的 accepted prefix 退休。
编译时借用常量、固定 effect footprint 及 Context/resource/allocator 保持各自原寿命合同。

CPU+CUDA同样使用本节的GraphOutput与资源合同，pending权重的typed绑定见
[CPU与GPU组合](#cpu-与-gpu-组合)。
Vulkan 的固定纯图可先把实际端点装入普通编译环境，后续仍使用上述绑定、准备、提交和 GraphOutput：

```cpp
auto source = graph::vulkan::execution_source(native_context).value();
const std::array domains{graph::StorageDomain{device_memory.ref()}};
const std::array targets{graph::ExecutionTarget{source, graph::StorageDomainId{0}}};
auto compiled = graph::compile(logical_graph,
    graph::CompileEnvironment{.storage_domains = domains, .targets = targets});
```

这里的 Context 与 MemoryResource 按 [Vulkan Compute](vulkan-compute.md) 初始化，资源须活过所有槽与导出 lease。
当前已接通创建时／冷绑定后可用的真实 backing、native device 上的借用常量，以及 `kMaterialize` 的 CPU→Vulkan
常量物化。物化在 compile 内按源批量完成，复用 staging／命令资源；返回成功后可释放源 buffer，submit 不再上传。
staging 按批中最大所需容量计入编译峰值，实际读取／原生完成失败直接返回。
Apple同样用`graph::apple::execution_source(native_context)`装配实际端点，保留原FIFO晚生成结果；
可传`{.custom = lowering}`接纯custom算子的冷lowering。捕获资源须活过冷编译，重放不调用lowering。

两端的`graph::apple::bind`／`graph::vulkan::bind(run, input, batch, id)`保留真实WeightBatch和选中范围，
绑定不等待上传；原提交在实际reader处按HostWaitPolicy处理依赖。共享Metal存储可直接供CPU读取，
不要求CPU reader一定落在普通主机allocation；copy报告应按实际physical reader和每条搬运边解释。
需要主机异步推进时继续使用下面的可选服务。原生私有分配与managed执行成本仍须按实际报告判断。

CUDA、Apple、Vulkan还提供`bind(run, span<const graph::WeightBinding>, batch)`，一次绑定同一上传代的
权重、scale与显式ID映射。全部输入的tensor、owner与最终alias范围检查通过才提交绑定；失败保留原绑定与准备状态。
用`WeightSlots::upload_slices()`可从分立tensor装配有界bank；选择哪些专家、何时换入与global/local映射由调用者决定，
Graph只执行普通checked Gather和grouped算子。装配、字节报告与打包要求见[有界专家权重工作集](compute.md#有界专家权重工作集)。

普通owned-layout分页通过`batch::prepare`／`batch::bind`绑定实际页权限；Apple／Vulkan的typed
`submit(run, queues, std::move(update))`把原CopyBatchUpdate控制交给同一次Graph提交。
实际退休并释放结果与Completion观察后，用相应`take_update(run)`回收；接纳前拒绝保留调用方原update，
producer失败不发布可写页或成功结果。其余历史专用消费者仍在迁移，不代表所有原生数学域均已补齐。

## 可选主机异步推进

需要CPU计算不占用提交线程，或让独立CPU／GPU片段同时推进时，构建选择`graph;graph_progress`，
应用链接`hyle::graph`与`hyle::graph_progress`，包含`hyle/graph/progress.h`。
进度服务公开提供Scheduler与BlockingPool的依赖。仅选择`graph;sched`不会启用该服务；
SDK即使同时安装了它，仅`find_package(Hyle COMPONENTS graph)`也保持普通Graph的依赖闭包。
仍使用同一个Invocation、QueueBinding、绑定和结果接口：

```cpp
// scheduler 和 blocking_pool 由应用持有；先按原合同绑定、prepare_replay。
HYLE_RETURN_IF_ERROR(hyle::graph::prepare_progress(run, scheduler, blocking_pool));
auto done = run.submit(queues).value();
HYLE_RETURN_IF_ERROR(done.wait());
auto result = run.result(output_id).value();
```

两个运行时及原借用资源须活过所有已接纳工作和`drain()`。准备只增加固定推进元数据，
每次提交仍使用Scheduler原有任务／stack分配。`Invocation::drain()`支持Scheduler fiber等待；
交接后的`Completion::wait()`继续遵守原Compute完成对象的等待合同。
取消不会抢占已经进入的CPU算子或GPU工作。不需要主机异步推进时沿用普通提交即可。

## 可选主机跟踪

使用 `init.py --trace` 开启已有 Tracy client，观察真实 compile/submit/wait 的主机范围，
安装与运行设置见 [Obs](obs.md)。
默认 OFF 不求值跟踪参数或创建 profiler 状态；没有逐 kernel GPU timestamp。Apple/Vulkan 的原生
报告与 owner 执行范围见对应路线指南。跟踪开启时的运行成本不使用关闭状态的性能结论。
