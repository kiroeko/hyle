# Paged / ragged attention 与显式批次准备

本页描述当前F5 C++ surface。当前资格与限制以[模块状态](../implementation/module-status.json)
及[本轮验证计划](../implementation/tasks/compute/foundation-f5-final-validation-plan.md)为准；
这页不承诺各平台相同吞吐、capture或全部native paged。普通接入只选Compute/Graph产品，
不依赖内部factor target。

## 选择执行路线

| 路线 | 数学与公开入口 | 准备与成本 |
|---|---|---|
| Linux CUDA | 原FlashInfer C++ native paged/ragged；BatchProgram、BatchInvocation、PreparedBatch；graph::batch::provider/Invocation | 每layout显式H2D及原plan；不同topology不能靠shape/address猜复用；native private bytes未知 |
| CPU | MaterializedBatchProgram；MaterializedBatchInvocation / graph::batch::MaterializedInvocation | 固定原CPU数学参考，显式gather、GQA展开、F32矩阵、mask与Copy；Direct逐stage同步wait |
| Apple | apple::MaterializedBatchInvocation；graph::apple::prepare_materialized_batch | 原MLX Program/worker，固定两device metadata banks、原F2上传；managed replay/私有storage未知 |
| Vulkan | vulkan::MaterializedBatchInvocation；graph::vulkan::prepare_materialized_batch | 原native Plan/command/fence与ggml承载；固定两metadata banks，opaque scratch/driver仍未知 |
| Apple page16，当前 G01 草稿 | NativePagedBatchProgram；apple::PagedBatchInvocation | 原MLX vector/2pass/NAX直接页读取；单I32表、两个mirror banks；实际Air F16/BF16数学通过，风险/交付/性能未收口 |
| Vulkan page16，当前 G01 草稿 | NativePagedBatchProgram；vulkan::PagedBatchInvocation | 原FullF32 scalar flash直接页读取；实际Air F16数学通过，风险/交付/性能未收口；BF16未签资格 |
| Windows CUDA native batch | 当前未准入 | 必须给具名unsupported，不能把WSL结果或CPU参考说成Windows native paged |

Apple/Vulkan参考档真的在原GPU上执行，但它们把页/packed KV显式gather/materialize后做
数学；索引/掩码、GQA复制、payload/wait的费用必须计入。需要原native paged吞吐时，
不能只看接口形状相同而假定性能已经对称。没有submit隐式plan、JIT或自动CPU fallback。

Linux CUDA的公开factory随普通产品传递，消费者只链接hyle::compute/hyle::graph；
cuda.h在实际HYLE_COMPUTE_FLASHINFER资格下供应原factory头，未准入平台不供应该API。
完整公开D256 Direct/Graph用法见[可编译消费者](../../tests/compute_batch/public_batch_cuda_consumer.cc)，
其输入与结果通过原lease接口读写，数学仍由锁定FlashInfer执行。

## Apple / Vulkan 原生 page16 接入

当前实现与未决资格按[G01 活动卡](../implementation/tasks/compute/inference-g01-native-paged-read.md)
和模块状态核验。两路线使用相同 owned schema/layout/table 与确切 frame 合同，保留各自
原生 Context/MemoryResource/Results。完整可编译入口见
[Apple 消费者](../../tests/compute_batch/native_paged_apple_consumer.cc)与
[Vulkan 消费者](../../tests/compute_batch/native_paged_vulkan_consumer.cc)。普通 consumer
仅链接 `hyle::compute`；不传第四个 raw tensor，也不把整 arena 变成 readonly lease。

```cpp
namespace native = hyle::compute::apple; // Vulkan 改为 hyle::compute::vulkan
auto recipe = hyle::compute::NativePagedBatchProgram::prepare(schema, {
    .max_host_metadata_bytes = 1u << 20,
    .scale = scale,
    .causal = causal,
}).value();
auto slot = native::PagedBatchInvocation::create(
    context, recipe, device, staging, {
        .max_metadata_backing_bytes = 4u << 20,
    }).value();
HYLE_RETURN_IF_ERROR(slot.prepare_batch(layout));
HYLE_RETURN_IF_ERROR(slot.bind(query_read, key_frame, value_frame));
auto submitted = slot.submit().value();
auto result = submitted.wait().value();
```

`schema` 固定 paged/page16、packed Q 容量与真实 physical page 容量；D64/128/256，
Apple F16/BF16，Vulkan F16。非空 Q 需有 KV，causal 需 logical KV≥Q。空 request 跳过。
逻辑重复页可以使 KV 长度超过 physical 容量；root shape 不随逻辑长度扩大。K/V frames
须由对应 PagedArena.read 与本次同一 owned layout/generation 创建；bind 拒绝混代或
foreign schema，不扩大未引用页权限。已 ready 页可共享，未引用页仍可独立更新。

当整个布局的 Q rows 与 page refs 都为零时，调用 `slot.bind(query_read, {}, {})`。
私有原生程序只绑定真实 Q 容量并返回空 view；不读取或伪造 K/V storage。这个零域不接受
其它代际的 live frames，旧空结果同样阻止 slot 更新。PagedArena.read 的原非空页权限
合同保持不变；reference 或 Graph 的空域须按各自资格判断。

每次 prepare 显式冷建原生 topology，并通过原 F2 完成单 I32 表的上传。当前上传与
staging 容量均为 `4 * max_page_refs`，device mirrors 为其两倍，固定 backing cap 还需
三组64byte对齐的最坏 padding；report 分别提供 uploaded/staging/device/backing bytes。
无需全历史 KV gather、按 Q/head 展开或上传 additive mask。原 query/output/MLX compiled
metadata、shader scratch 与 driver 存储仍是额外费用，private bytes 可以 UNKNOWN；
不由这个入口签 delta planning、capture 或吞吐完全对称。

Context/device/staging/injected allocator 保持地址稳定，活过所有 native work 和结果；
控制调用外部串行，多个 slot 独立。旧 result 阻止同 slot prepare；close 停止新工作，
drain 沿原实际退休观察，已接纳错误或未知完成不许可提前释放页 guard。结果按本 route
原 Results/readonly import 使用。新的 accepted-prefix/未知退休、三交付及性能验收仍
待活动卡完成，source 数值通过不能代签这些门，也不能代签新的 CPU+GPU 联合分页。

## Graph 的执行 lane 绑定

原生 batch coordinator 的 `Invocation::create` 接完整活跃 lane 集合；每项是
`graph::batch::LaneBinding{lane, &context, &queue, host_workspace}`。单 lane 也传一个元素的 span；不再接
旧单 Context/Queue 签名。只列 `executable.requires_queue(lane)` 为真的 lane，每个一次。
Context 必须拥有该 Queue，且 Queue 对应 Graph 编译时同一 target；同 device 不是同 Context。

```cpp
const std::array lanes{
    hyle::graph::batch::LaneBinding{gpu_lane, &gpu_context, &gpu_queue, gpu_staging},
    hyle::graph::batch::LaneBinding{cpu_lane, &cpu_context, &cpu_queue}};
auto invocation = hyle::graph::batch::Invocation::create(
    executable, lanes).value();
```

数组与 Context 指针只借用冷创建调用，原 Queue bindings 被复制；各 batch region 按原
Graph lane 选择自己的原生计划/metadata/执行 Queue及合法host staging；另一个Context的
CUDA pinned/device carrier不能假装属于这个lane。非batch辅助lane无需host workspace。
资源与 Context/Queue 沿原 lifetime
合同存活，不能在 native work 或旧结果仍被保留时销毁。prepare_batch 后 bind 输入/输出，
首次 submit 才沿原 Graph prepare_replay 处理完整已绑定 topology；仍无 submit 隐式 batch plan。
此接口正在 G02 验证，不代签 Apple/Vulkan hybrid native 分页或新吞吐/capture资格。

## 容量、有效区与版本

先创建独立的BatchSchema。BatchCapacity固定storage、dtype、request/Q/KV/page/ref容量、
query/kv heads、head_dim、page_size。Paged与ragged是不同schema，F16/BF16采用F32
accumulation；GQA必须整除且在实际route资格域内。

BatchLayoutVersion由schema.version创建：query_indptr描述packed active Q；
ragged kv_indptr描述packed KV行；paged kv_indptr描述每request页引用区间，
page_indices为physical页ID，last_page_lengths为最后一页有效长度。快照拥有host数据，
调用返回后caller可释放原数组。相同shape但新值仍是新generation，不能用地址当版本。
错indptr/page/tail、overflow、foreign schema在接纳前拒绝。

物理Q容量与active Q不同；只消费有效行，inactive output没有数学意义。尾padding不是KV。
空布局的host元数据可以有明确定义，但native数学会按该route域明确接受或拒绝；不能从
schema接受空元数据推断每个kernel也接受空执行。

## CPU公开旅程

可编译的完整例子见[公开CPU消费者](../../tests/compute_batch/public_batch_cpu_consumer.cc)：
paged/ragged×F16/BF16、GQA、乱页/tail、有效区和old output。核心顺序是：

```cpp
auto context = hyle::graph::cpu::create_context().value();
auto queue = context.create_queue().value();
auto recipe = hyle::compute::MaterializedBatchProgram::prepare(schema, {
    .max_kv_per_request = 4,
    .max_host_metadata_bytes = 1u << 20,
    .max_logical_intermediate_bytes = 1u << 20,
    .scale = 0.125f,
}).value();
auto invocation = hyle::compute::MaterializedBatchInvocation::create(
    context, recipe, queue, {
        .max_payload_bytes = 1u << 20,
        .max_workspace_bytes = 1u << 20,
    }).value();
HYLE_RETURN_IF_ERROR(invocation.prepare_batch(layout));
HYLE_RETURN_IF_ERROR(invocation.bind(query, ragged_keys, ragged_values));
auto done = invocation.submit().value();
HYLE_RETURN_IF_ERROR(done.wait());
auto output = invocation.output().value();
```

实际产品代码检查每个Status/Result；例子用value压缩展示顺序。paged用
bind(query, PagedArenaRead keys, PagedArenaRead values)而非整个arena的普通readonly lease。
CPU Graph用同recipe创建graph::batch::MaterializedInvocation，prepare/bind/submit，
从output().storage_lease()取得readonly输出；普通Graph对象没有新增warm字段/规划。

## 原生Apple / Vulkan公开旅程

[公开native消费者](../../tests/compute_batch/public_batch_native_consumer.cc)只用公开API：
创建本route Context、device/staging MemoryResource；原F2上传Q/K/V并取得成功ready lease；
创建typed Direct controller或同route的Graph factory；prepare_batch(layout)上传同一
owned generation，再bind和submit。OwnedResult.wait成功后只借ResultLease访问原native
数组/tensor。旧ResultLease/Completion仍存活时prepare必须拒绝，不能覆盖旧结果。

Context、resource owner、Queue和借用allocator必须晚于所有相关native work、结果、
exported storage和F1 uses销毁。控制对象thread-compatible，caller串行控制；独立slots
可以独立使用。不要在另一个线程还prepare/bind时同时控制同一个对象。

## 页写入、append与F4发布

PagedArena冷分配连续backing和独立page权限。acquire取得具体physical页writer；经原
F2 clone/append成功后complete/publish，read(layout, exact ready pages)产生精确页frame。
未引用的页可以写，仍有旧reader/native use的页不能覆写；整个arena的subview不是独立writer。

原F2已消费page writer时，保留PagedPageWriter，等成功ready输出后调用complete；若放弃
这个已消费的控制对象，arena不会猜测该页已退休或重新ready。失败/无法完成观察的页按
已声明合同保持不可再acquire，显式close/recreate arena；不能把它当普通未提交writer归还。

F4 StateStore/StateDraft列出完整K/V/layout及其他需要保留的成员。共享干净页，dirty页显式
clone到新writer；只把所有成功ready成员finalize为candidate，再由StateHead.publish用
expected-base切换。失败copy不产生ready writer/lease，不能送进F5，也不能发布F4 head。
取消不能回滚已经实际写过的底层字节，不能在失败后伪造整页CoW成功。

Ragged多request append可能需要重排后续request的offset。当前真实两request例子显式
clone有效prefix、append新行并初始化unused capacity；这不是独立物理页CoW，成本账必须
把tail初始化和重排算进去。F5不解释token/page eviction/admission或模型scheduler策略。

## 错误、关闭与费用

prepare/update可能copy、host wait或调用原native planner；它们属于显式准备，不是免费
warm submit。BatchPlanReport记录generation、metadata上传、workspaces、replan/reuse、
split与host waits；MaterializedBatchCosts是保守logical上限，包含view/alias等，不能当RSS。
native controller报告staging、两份device mirror、固定backing和上传字节；未知private
metadata/scratch/driver费仍是UNKNOWN，硬拒未知预算按原资格明确拒绝。

失败和unknown观察都不产生成功输出。public terminal或释放public result/token不代表
accepted native prefix已退休；drain按实际guard/fence/worker等待并保留原错误。正常放弃
可在真实退休后成功排空，取消停止未开始工作。close/drain后的对象不重用；错误slot的
复用必须遵守原route失败合同，不靠清除一个cached status假装恢复。

当前F5 batch没有已资格的capture；稳定地址、固定batch size或MLX shapeless本身不能
证明capture topology。需要不同容量/算法域时冷创建新的受资格program/slot，明确计预算。
