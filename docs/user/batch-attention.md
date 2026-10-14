# Paged / ragged attention 与显式批次准备

本页描述当前 C++ surface。当前资格与限制以[模块状态](../implementation/module-status.json)
及[本轮验证计划](../implementation/tasks/compute/foundation-f5-final-validation-plan.md)为准；
这页不承诺各平台相同吞吐、capture或全部native paged。普通接入只选Compute/Graph产品，
不依赖内部factor target。

## 选择执行路线

G04/G05 共同形状正在实施，尚未签发新的编码/窗口/不同 V 维原生资格；进度见
[活动卡](../implementation/tasks/compute/inference-g04-g05-cache-attention-domain.md)。
普通旧 plain 旅程使用显式的新字段，库不保旧签名转发。schema/descriptor接受一种形状，
不代表所选native核已支持；当前未接通的组合在准备时返回Unimplemented。

```cpp
using namespace hyle::compute;
const auto plain = CacheStorage::plain(DataType::kBFloat16).value();
BatchCapacity capacity{
    .query_type = DataType::kBFloat16,
    .output_type = DataType::kBFloat16,
    .key_storage = plain,
    .value_storage = plain,
    .max_query_rows = 17,
    .max_kv_rows = 64,
    .max_pages = 4,
    .max_page_refs = 4,
    .query_heads = 8,
    .kv_heads = 2,
    .qk_head_dim = 256,
    .value_head_dim = 256,
};
auto schema = BatchSchema::create(capacity).value();
auto keys = PagedArena::create(schema, BatchCacheRole::kKey, resource, options).value();
auto values = PagedArena::create(schema, BatchCacheRole::kValue, resource, options).value();
auto query_desc = batch_query_description(capacity).value();
auto output_desc = batch_output_description(capacity, capacity.max_query_rows).value();
```

Q/output类型、Dqk/Dv和两份cache表示分别固定在schema中。页面字节与有效前缀按对应
key/value role计算；tensor描述不携带外部FP8 scale，frame绑定还校验其原cache解释。
CacheStorage的plain/q8_0/float8 factory生成不可变值，format/data_type/scale用访问器读取。
Q8_0是末depth轴block32，34 bytes/block；copy producer使用真实U8 carrier，不能把
blocked逻辑描述塞进原plain native result-source端口。FP8 E4M3FN使用固定正有限scale；
改变scale需要新schema/cache身份，encoder与attention应使用同一spec。

BatchSchema.version只检查结构和有效行：非空Q需要非空KV；full noncausal允许Q>KV。
causal Q<=KV在recipe/prepare的数学资格检查。window_left=-1为无界，0为self-only，
正值为bottom-right位置之前含左边界的past count；本批仅causal window或full noncausal。
Materialized参考保显式gather/mask费用，支持plain几何与窗口，明确拒绝编码cache；
它不替代直接压缩驻留读取。具体native组合以实际资格为准。

| 路线 | 数学与公开入口 | 准备与成本 |
|---|---|---|
| Linux CUDA | 原FlashInfer C++ native paged/ragged；BatchProgram、BatchInvocation、PreparedBatch；graph::batch::provider/Invocation | 每layout显式H2D及原plan；不同topology不能靠shape/address猜复用；native private bytes未知 |
| CPU 参考 | MaterializedBatchProgram；MaterializedBatchInvocation / graph::batch::MaterializedInvocation | 显式gather、GQA展开、F32矩阵、mask与Copy；Direct逐stage同步wait |
| CPU page16 | 原CPU BatchProgram / BatchInvocation；详见[CPU批次入口](compute.md) | 原ggml flash直接读取确切物理页，有限F16/BF16资格已收口；Materialized路径另为显式参考算法 |
| Apple | apple::MaterializedBatchInvocation；graph::apple::prepare_materialized_batch | 原MLX Program/worker，固定两device metadata banks、原F2上传；managed replay/私有storage未知 |
| Vulkan | vulkan::MaterializedBatchInvocation；graph::vulkan::prepare_materialized_batch | 原native Plan/command/fence与ggml承载；固定两metadata banks，opaque scratch/driver仍未知 |
| Apple page16，有限域已收口 | NativePagedBatchProgram；apple::PagedBatchInvocation | 原MLX vector/2pass/NAX直接页读取；单I32表、两个mirror banks；actual Air F16/BF16数学、风险、三交付与同合同P1与终审通过 |
| Vulkan page16，有限域已收口 | NativePagedBatchProgram；vulkan::PagedBatchInvocation | 原FullF32 scalar flash直接页读取；actual Air F16数学、风险与三交付与同合同P1/终审通过；BF16未签资格 |
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
原 Results/readonly import 使用。新native accepted-prefix/未知退休与三交付已验；宿主san有活动卡记录的插桩边界，
不签shader/driver。两route同合同P1与有限终审通过；Windows仅定向
compile/link通过。这些native入口证据不能代签新的CPU+GPU联合分页。

## CPU 与 Apple/Vulkan 联合分页

当前接缝的[原生有限域终审](../implementation/tasks/graph/inference-g02-native-paged-terminal.md)
已完成实际Air数学/风险、四份relocated SDK、宿主san和Windows定向公开链接；原G02
共同轮的Windows CUDA/Vulkan未决门仍保留，不能由本native域签整个G02完成。
完整普通调用见[公开联合消费者](../../tests/graph_batch/native_paged_joint_consumer.cc)，只链
`hyle::graph`，需要配置 `cpu;apple` 或 `cpu;vulkan`。沿原GraphBuilder描述CPU前后段：

```cpp
namespace gb = hyle::graph::batch;
auto descriptor = gb::native_paged_attention(recipe, initial_layout).value();
const std::array ports{ready_query, physical_key_input, physical_value_input};
auto page_node = builder.add_node(
    hyle::compute::Operation::custom(descriptor), ports).value();
// finish graph；原 node_placements 显式要求 CPU 前后段。
const std::array bindings{gb::NativePagedNodeBinding{
    page_node, recipe, initial_layout, {.max_metadata_backing_bytes = 4096}}};
auto program = joint::compile(graph, environment, {
    .node_placements = cpu_nodes, .paged_nodes = bindings}).value();
auto slot = program.create_invocation(scheduler, cpu_queue).value();
const std::array layouts{gb::NodeLayout{page_node, layout}};
HYLE_RETURN_IF_ERROR(slot.prepare_batch(layouts));
HYLE_RETURN_IF_ERROR(slot.bind(source_query, query_description, ready_host_query));
HYLE_RETURN_IF_ERROR(slot.bind(physical_key_input, key_frame));
HYLE_RETURN_IF_ERROR(slot.bind(physical_value_input, value_frame));
HYLE_RETURN_IF_ERROR(slot.prepare_replay());
auto result = slot.run().value();
```

`joint`选cpu_apple或cpu_vulkan；Environment、Graph ID、Context/Queue、资源和scheduler的
自然借用规则沿原联合入口。冷binding与layout由plan拥有，compile的spans只借本调用。
prepare_batch需要全部live页节点各一次，同一K/V输入的所有节点须使用相同owned版本。
固定active packed输出shape；请求分配、页顺序/共享/tail变化显式cold prepare，active Q
总量改变需编译另一固定shape。程序当前沿原managed native replay/真实边界等待，不承诺
capture、低成本delta planning或吞吐完全对称。

K/V必须是Graph外部inputs，且所有live读者只能是qualified页节点的K/V ports；不能发布
整capacity或交给CPU/dense/query节点。普通tensor/ReadLease/weight绑定不能绕过exact
frame。错误node/schema/resource/generation、重复/漏layout、变化输出shape在接纳前拒绝。
拒绝frame替换保旧正确绑定；拒绝prepare后必须重新完整prepare，不混用新旧tokens。
页输入与本轮尚未资格的CPU effects组合返回unsupported；普通nonpaged effects仍沿原入口。

CPU query完成及原upload之后才接入原page数学；每episode消耗一次页准备。全Q0/ref0
的固定图只需要真实Q，KV不用frame或假root，native空结果可继续到CPU空后段。旧结果阻止
prepare/reset；cancel/close/放弃只有已接纳prefix实际退休后才释放所有引用页权限。
未引用页可独立更新。`replay_report`给页fragment数、table uploaded/backing bytes和
metadata host waits；Q/结果copy、冷native/transfer更新、wrapper metadata及native private
UNKNOWN分别按原报告/活动卡计，不声明零warm allocation或总GPU内存硬上界。

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
F2 clone/append成功后调用complete(ready, initialized_rows)或publish(initialized_rows)，
显式提交连续有效行前缀；值须在[0,page_size]内，0表示尚无有效行。read(layout, exact ready pages)
检查每个引用所需的有效行数，非尾页需要整页，重复/共享页必须满足所有request后才产生精确页frame。
首次取得页或旧页回收为新generation时前缀为0；acquire(ready)与未提交writer归还保持原前缀。
initialized_rows()只返回已提交前缀，写入与copy退休不会自动扩大它。由调用方在真实写成功后声明。
未引用的页可以写，仍有旧reader/native use的页不能覆写；整个arena的subview不是独立writer。

`arena.pending_read(layout, ready_pages, page_updates)`可冷构造独立的
`PagedArenaPendingRead`：ready页保原权限，pending页记录已消费writer的原页身份、代际
和已提交初始化前缀。`PagedPageUpdate{&page, destination_id}`只在构造调用中借用page；
原copy控制必须持有该目的页。frame保精确layout与arena，但不产生ready lease或public字节
读取权限；原ready-only bind/run不接受此类型。receiver须验证本次accepted copy覆盖新增
要读的bytes、实际native排序与全引用页guard，并合并来源结果。

Apple/Vulkan 的 `PagedBatchReceiver` 是显式冷扩展，移交一原 `PagedBatchInvocation`，
保同一 native invocation、queue 和结果槽。`max_source_completions` 按原更新中已接纳
phase 与实际使用的 producer completion 数量设定；不足时拒绝，不消费 Update。
先创建 receiver、准备本次 layout，再提交原 F2，最后将 Update 与精确 pending K/V
一起移交；两个 frame 可同时包含继承的 ready 页和本次 pending 目的页：

```cpp
auto receiver = native::PagedBatchReceiver::create(std::move(slot), source_capacity).value();
HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout)); // explicit cold metadata upload
auto copied = copy.submit().value();
auto update = native::CopyBatchUpdate::take(std::move(copy), std::move(copied)).value();
auto keys = key_arena.pending_read(layout, ready_keys, key_updates).value();
auto values = value_arena.pending_read(layout, ready_values, value_updates).value();
{
  auto submitted = receiver.submit(query, keys, values, std::move(update)).value();
  auto result = submitted.wait().value(); // source and actual suffix retire here
  // result and submitted.completion() use the same original result/aggregate.
}
update = receiver.take_update().value();
page.lease() = update.take_mutable(destination).value();
page_ready = page.publish(initialized_rows).value();
copy = update.release_invocation().value();
```

调用失败且 suffix 未接纳时，传入的 Update 保留或原样归还；接纳后失败/取消/unknown
不发布成功结果，并保全部 source 与页 guards 至实际退休。控制访问外部串行；旧
OwnedResult/ResultLease、pending 工作或尚未回接的 source 阻止 receiver 复用。
frame 与 native binding 有显式冷 metadata 分配，原 metadata prepare 的等待和 native
private unknown 保留。这个 Direct 入口覆盖两 native 路线；Graph与late native producer组合的已准入G03
范围现见[最终裁决](../implementation/tasks/compute/inference-g03-final-closure.md)。

共同 `hyle::compute::PagedBatchReceiver` 移交原 `BatchInvocation`，沿其原队列、
PreparedBatch 和输出 buffer 消费同样的 pending frame。CPU 与已资格的 Linux CUDA
原生路径已有 source/static/shared 双请求、双槽、64步公共旅程；CPU 宿主风险与 CUDA
实际来源/退休边界也已定向验证。Graph与late native producer组合见G03最终裁决。
CPU 队列只接受已实际完成的 inline 来源，异步来源由原 backend 拒绝，不隐式 host wait。

```cpp
auto receiver = compute::PagedBatchReceiver::create(std::move(slot), source_capacity).value();
HYLE_RETURN_IF_ERROR(receiver.prepare_batch(layout));
auto completion = receiver.submit(query, keys, values, std::move(update)).value();
HYLE_RETURN_IF_ERROR(completion.wait()); // merges original source and native suffix
auto output = receiver.output().value();
// Output retains the original capacity; layout.facts().query_rows gives active rows.
auto active = output.view()->subview(0, active_output_bytes).value();
update = receiver.take_update().value();
```

`output()` 不等待：只有聚合成功并实际退休后才能发布。没有接纳 prefix 的拒绝保留
或归还原 Update；已接纳 wait/math prefix 的错误通过真实 Completion 返回并保 guards
至实际退休。旧 output reader 阻止原 buffer 覆盖，尚未回接的 Update 阻止 receiver
复用；外部 PreparedBatch ticket 与 Graph 外部输出控制器不能移交给这个 Direct 扩展。

### Graph pending 与 native producer 输出

`graph::batch::PendingInvocation` 移交原 Graph batch coordinator，在其原队列和输出槽内
组合 Update；pending frame 只绑定已准备的 paged 节点 K/V 输入。Q 用 ready lease，外部
输出用 writer，metadata 由原 prepare_batch 绑定。CPU/CUDA source 的双请求/双槽/64步 F16/BF16
与required CPU+CUDA的同形状源码/静态/动态交付、生命周期已验证；全部已准入路线及
限制见[G03最终裁决](../implementation/tasks/compute/inference-g03-final-closure.md)。

```cpp
auto pending = graph::batch::PendingInvocation::create(std::move(original), source_capacity).value();
HYLE_RETURN_IF_ERROR(pending.prepare_batch(node_layouts));
HYLE_RETURN_IF_ERROR(pending.bind(query_id, query_desc, query));
HYLE_RETURN_IF_ERROR(pending.bind(key_id, keys)); // PagedArenaPendingRead
HYLE_RETURN_IF_ERROR(pending.bind(value_id, values));
HYLE_RETURN_IF_ERROR(pending.bind(output_id, output_desc, std::move(output_writer)));
{
  auto completion = pending.submit(std::move(update)).value();
  HYLE_RETURN_IF_ERROR(completion.wait());
  auto output = pending.output(output_id).value();
  // output.storage_lease() retains the original output storage.
}
update = pending.take_update().value();
```

来源排序在原 shared sequence 中先接纳，随后取得精确 pending 页 guards；原公共
completion 合并来源与实际消费结果。来源失败不发布成功输出。旧 Graph output/completion、
实际 pending uses 或未回接的 Update 阻止复用，不能只释放一个 public handle就覆盖原槽。
两 native 联合 Graph 在原 Invocation 上冷调用 `reserve_pending_sources(capacity)`，再
prepare_batch、bind(PagedArenaPendingRead)、submit(Update&&)、take_update。实际两路线的
双请求/64步、两个页节点共享同一 Update与 CPU 前后段已通过数值消费、生命周期和
源码/静态/动态交付。一个固定图的 active Q 输出行数保持固定；2行 decode与18行
混合 prefill可冷准备两个图，每个图各两槽，合计四个冷槽。这些槽/metadata准备/跨CPU边界
的 checked等待均计成本。它不创建第二个 Graph 或 native 执行器。

CUDA wrapper 可先用 `BatchProgramOptions.resource={}` 声明 schema，在其 Context 上创建
实际 device/pinned资源，再通过显式 CompileEnvironment固定这些 storage domain。F2、lane
workspace和页 arena使用同一个 Context的资源；其它 Context的同设备资源不能冒充这个身份。

Apple/Vulkan 原 F2 可冷绑定同一个已接纳 native producer 的多个输出；以下 `native` 指
`compute::apple` 或 `compute::vulkan`。每条描述必须与实际输出的 dtype、dense shape及字节
尺寸一致，output_index 指原结果中的输出序号：

```cpp
auto produced = producer.submit().value();
const std::array sources{
    native::ResultCopyBinding{key_source_id, 0, key_desc},
    native::ResultCopyBinding{value_source_id, 1, value_desc}};
HYLE_RETURN_IF_ERROR(copy.bind_source(sources, std::move(produced)));
HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases, ranges));
auto copied = copy.submit().value();
update = native::CopyBatchUpdate::take(std::move(copy), std::move(copied)).value();
// Transfer update to the pending consumer; no producer.wait() is required.
```

同一次 copy还可分别绑定两个独立 producer结果：各调用一次 `bind_source`，使用互不重复
的 resource ID；随后一次 prepare/submit，移交一个 Update给消费者。每个 accepted结果
只保自己的原 completion，不为 K/V或两个 page节点复制 producer执行或来源观察者。

成功绑定才移交整个 OwnedResult与它的唯一 completion owner，拒绝保留入参。输出逻辑尺寸
不构成 ready/F1 权限：Apple 在原 worker 的 preflight 取得实际 backing 并完整重验范围/别名
后才 blit；Vulkan 可冷取得实际地址，但来源排序与成功仍由原 producer合同裁决。foreign
Runtime、未资格的动态 queue prefix及不匹配输出拒绝。producer 槽须等原结果/storage/source
全部引用与实际工作退休后回收。两路线的实际多输出 Abs→F2→Direct pending attention source
旅程已验证，独立 Abs多输出→两阶段范围重排→CPU reduction的64步source也已验证；这个
正交消费者在最终 F2成功后显式搬运256字节至host，不声明 GPU reduction流水或零copy。
Apple/Vulkan 新接口的上述 Direct/joint/正交消费者已通过 source 与迁址 static/shared；
已准入G03路线资格与不支持域以最终裁决为准。

原F2已消费page writer时，保留PagedPageWriter，等成功ready输出后调用complete；若放弃
这个已消费的控制对象，arena不会猜测该页已退休或重新ready。失败/无法完成观察的页按
已声明合同保持不可再acquire，显式close/recreate arena；不能把它当普通未提交writer归还。

F4 StateStore/StateDraft列出完整K/V/layout及其他需要保留的成员。共享干净页，dirty页显式
clone到新writer；只把所有成功ready成员finalize为candidate，再由StateHead.publish用
expected-base切换。失败copy不产生ready writer/lease，不能送进F5，也不能发布F4 head。
取消不能回滚已经实际写过的底层字节，不能在失败后伪造整页CoW成功。

G03 的[精确增量有限路径](../implementation/tasks/compute/inference-g03-dirty-page-terminal.md)已验收。`arena.acquire(page_ready)` 只在这个原整页
没有其它 reader/native use 时消费其 ready lease，返回同 generation 的 writer；拒绝
保持入参不变。共享页先沿上面的显式 clone 路线处理，不能原地覆盖旧版本。

F2 只写 dirty range 时，`output` 仍不能把未写 bytes 算成成功 copy。保留 page writer，
在实际成功退休且 submission 已释放后，用 `copy.unbind_mutable(id)` 取回原 WriteLease，
放回 `page.lease()`，再 `page.publish(initialized_rows)`。未写部分是 caller 原来已初始化的内容，成本
只计本次 dirty bytes；新页仍须初始化所有将被读取的 bytes。示例顺序如下：

```cpp
auto page = arena.acquire(page_ready).value();
HYLE_RETURN_IF_ERROR(copy.bind_mutable(destination, std::move(page.lease())));
// bind source/producer and prepare exact dirty ranges with the original F2.
{
  auto submitted = copy.submit().value();
  HYLE_RETURN_IF_ERROR(submitted.wait());
}
page.lease() = copy.unbind_mutable(destination).value();
// This example overwrites existing rows; an append must declare its new prefix.
page_ready = page.publish(page.initialized_rows()).value();
```

`unbind_mutable` 不等待、不产生 readonly output；retained submission/后继ordered read use、pending/unknown/
failure、closed 或已 publish 的 binding 都拒绝。成功使旧 replay preparation 失效。
它会沿原 F2 作非阻塞退休观察；仅放弃 submission 时仍保留 PagedPageWriter，后续
调用可在成功实际退休后取回权限。不能销毁 page writer 后猜测 arena slot 已恢复。
这里仍有显式 host observation；G03 的异步 producer→attention、G06 decode planning 与
G09 流水须分别看各自验收，不能由这段精确更新调用推断已完成。

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
共同 pending Graph每次 `prepare_batch` 清除上次 pending输入，后续 bind会为 reader和lane
表请求 bounded metadata；外层 reserve不等于每步零分配。warm submit不新增来源协议节点、
不复制KV或创建队列；完整 prepare/bind/submit费用应分别计入。CPU+CUDA图的 GPU→CPU
边界可沿原 replay合同执行 checked host wait，因此 submit不保证非阻塞。G06规划与G09
流水分别处理这些成本，不从 pending页接纳推导完全异步或 capture资格。

CUDA 已接纳 native work 的 request_cancel 可返回 Unimplemented；必须检查返回值，
不能把拒绝取消当成已经取消或退休。停止提交后继，等原实际工作退休后再释放相关权限。

当前F5 batch没有已资格的capture；稳定地址、固定batch size或MLX shapeless本身不能
证明capture topology。需要不同容量/算法域时冷创建新的受资格program/slot，明确计预算。
