# Paged / ragged attention 与显式批次准备

## 选择执行路线

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
plain F32 与 Q8_0 是不同表示，即使逻辑 `data_type()` 都是 F32；不能仅按 dtype 推断缓存编码。
Q8_0是末depth轴block32，34 bytes/block；copy producer使用真实U8 carrier，不能把
blocked逻辑描述塞进原plain native result-source端口。FP8 E4M3FN使用固定正有限scale；
改变scale需要新schema/cache身份，encoder与attention应使用同一spec。

BatchSchema.version只检查结构和有效行：非空Q需要非空KV；full noncausal允许Q>KV。
causal Q<=KV在recipe/prepare的数学资格检查。window_left=-1为无界，0为self-only，
正值为bottom-right位置之前含左边界的past count；本批仅causal window或full noncausal。
Materialized参考保显式gather/mask费用，支持plain几何与窗口，明确拒绝编码cache；
它不替代直接压缩驻留读取。具体native组合以实际资格为准。

当前CPU实现可以用 `model_ops::CacheEncode{storage}` 编码实际新行：输入是dense
F16/BF16/F32，输出描述通过 `infer_single_output` 得到。Q8输出是U8真实carrier，
FP8输出是E4M3FN carrier；encoder只写所给的新行，页面的准确range和发布仍由批量搬运管理。
CPU attention可直接读Q8_0或普通分页E4 K/V，保留原页权限、布局更新和队列。
CPU 也支持同为 F32 的 Q／plain K／plain V／output，直接读取被引用的原页，不按 query 展开整套 K/V。
Q27 的 24／4 heads、D256、跨页、partial tail、causal／非 causal 与窗口已有[独立数值检查](../evidence/correctness/mtp-f32-paged-cpu-2026-10-24.json)，不是完整 Q27 模型通过证明。
Q8 K/V同格式、Q/output同F16或BF16；当前有限几何是(64,64)/(128,128)/(256,256)/(64,128)。
编码输入与Q8 attention查询必须有限，Q8的absmax/127不得超过65504。
Q8编码先把block scale舍入成F16；若其bits为零，整个34-byte block规范为全零。
这也覆盖极小的非零输入，避免倒数溢出、FTZ差异和非有限数值转整数。非零 F16 scale 使用 F32 scale/inverse 和 ties-away 规则；零 scale 块的 payload 可能与原 ggml 参考不同，但解码值同为零。
实际payload违反范围
由原Completion返回InvalidArgument，失败invocation/slot按原合同重建。原CPU核还会把Q
转成Q8 dot表示，精度应按这次额外转换评估。

普通分页E4现已接CPU、CUDA、Apple和Vulkan：使用原`CacheStorage::float8(scale)`，
K/V可有各自固定scale；无需新格式或先展开整套KV。CPU boxed、Vulkan FullF32与Apple Steel
直接对已引用页的E4值解码、乘各自F32 scale，在原attention中累加并最终舍入输出。
Apple仅将活跃Q／partial转成F32。共享回归覆盖BF16 Q/output、page16、D64→128、GQA、
多请求、窗口、双槽旧结果及固定Q1历史增长；原Q8/plain邻居继续通过。
`CacheEncode`的F16/BF16/F32输入先以F32除scale，再SATFINITE最近偶数编码；NaN/Inf失败，
失败输出不发布。标量允许正有限次正规数。任意scale的跨后端归约不保证逐bit相同，
CUDA的scale位置仍如下所述。compact MLA的混合CKV/KPE布局和具体制品的scale导入另行核定，
不能由普通分页支持推导其已接通。见[定向数值与寿命回归](../evidence/correctness/paged-e4-kv-reference-2026-10-21.json)。

同一E4 spec也可经成对或独立`CacheEncode`生产者、原`CopyBatchUpdate`和pending页frame进入后继attention；
成功路径只等最终消费，不要求先host等待encoder或copy。绑定失败保留未被消费的原Completion；
撤回已绑定writer前须观察其生产者的真实退休，不能用“copy尚未in_flight”推断生产者没有在途工作。
CPU／CUDA／Apple／Vulkan的[有限旅程](../evidence/correctness/e4-paired-pending-reference-2026-10-21.json)覆盖BF16 D64→128、双请求、六轮交替生产、旧结果与NaN失败；失败旅程使用独立context，不代表同context重试。

Windows/Linux CUDA 的 FlashInfer 入口可用同一 CacheEncode 操作编码 E4M3FN 新行。准备时显式传入
CUDA device resource，BatchProgram的host_workspace使用同设备原pinned-host resource。
K/V均为FP8，分别固定scale，Q/output同F16或BF16，几何与窗口合同如上；有限输入的编码
采用nearest-even及finite saturation，NaN/Inf在接受提交后的原Completion返回InvalidArgument。
原attention直接读FP8字节；Kscale折入softmax scale，Vscale在F32 partial/merge后乘入并
转换最终输出。当前split-KV使用原planner计费的F32 partial workspace，最终scale/cast
已合入merge；非split在原输出变换中完成，没有额外最终scale/cast dispatch，也不展开
整套历史KV。
其中修复了原native FP8 Dqk64/Dv128漏载K的问题：仅tile内K行补至128 bytes并填零，
global cache与QK数学保持64维；额外shared为CTA_TILE_KV*64 bytes。原始依赖cache不变。
原生 CUDA batch 当前限 SM89，Q/output 为 F16/BF16；Dqk 为64/128/256，Dv 相同或64→128，
plain 同类型 K/V 另支持512→512，scale须为正有限值。GQA要求两种head数为正且query heads
可被KV heads整除，没有额外比例名单；实际承载仍受上述数学域、完整Q/output与ragged KV容量的
原生整数索引范围、设备限制及planner/workspace容量约束。576→512 的普通内核超过当前SM89共享内存上限，仍明确拒绝。
其它域返回具名 unsupported。FP8 的 KV payload 为 BF16 的一半，但读取转换、
中间结果和 metadata 仍占资源；比较完整调用时要包含双方实际的展开、规划与等待成本。

FlashInfer 的 F32 accumulation 不表示所有中间值都是 F32：QK 点积、在线统计和 PV 累加
使用 F32，未归一化的指数权重 tile 则先舍入为 Q 的 F16/BF16 类型，再用于权重行求和与 PV。
因此，同一 BF16 Q/K/V 的结果可能不同于保留 F32 概率的 CPU 路径；这与
`allow_bf16_partials` 无关，后者只允许额外缩窄 split-KV partial 的存储。
冷报告的 provider、query dtype 与 `BatchPlanReport::bf16_partials` 应分别解读，
不能把该字段为 `false` 当作概率 tile 采用 F32 的证明。
[Q03 的有界定位](../evidence/correctness/qwen35-nvfp4-attention-boundary-2026-10-24.json)
使用完全相同的实际 Q/K/V，独立重现了这一边界；它不代替完整生成任务的质量检查。

Apple原MLX入口可编码Q8_0／E4新行；Q8在paged/ragged、E4在普通paged attention中直接读取压缩carrier。
Q/output为F16或BF16，支持上述有限几何、窗口与full noncausal；空输出仍按Dv给出形状。
为避免合法Q8值先转half溢出，编码缓存路径只将活跃Q转换为F32，在原vector/二pass/steel
tile中解码和累加，最后转输出类型。活跃Q/output转换费用为O(Q*Hq*(Dqk+Dv))，二pass
还有原F32 partial；没有整历史KV解码。窗口核限制实际扫描范围，plain同维无界路径仍复用
原核。输入NaN/Inf或Q8 scale溢出在接受提交后的原Results返回InvalidArgument。
plain F16/BF16 另支持512→512、576→512。高维统一选已有 compact full tile，包含Q1，
避免默认大 tile 在32 KiB线程组内存设备上先创建失败；保留实际管线资源检查，不展开历史KV。
这里的高维扩展不包含Q8_0。

Vulkan原Plan可以编码Q8_0／E4 carrier；普通paged可直接读取E4，Q8的F16/BF16 paged/ragged路径在原FullF32 scalar核中
直接按页解码，独立Dv与窗口无需整历史KV展开。NaN/Inf和编码scale溢出由原Results报错；
普通 Copy/SetRows 保持原规则。Q、output与plain KV可分别选择F16/BF16，K/V类型相同。
plain F16/BF16另支持512→512、576→512，沿同一 FullF32 shader 直接读原生页／ragged KV；
高维 Q8_0 尚未纳入该承载。

## 逐行压缩历史的页存储

584B主KV行和68B索引行使用 `RowCacheBatchCapacity` / `RowCacheBatchSchema`，
不必填写attention heads或伪造Q/K/V dtype。容量固定encoding、请求/选择行/KV行上限、
物理页数、页引用数和page_size；`row_cache_paged_description(capacity)` 返回真实U8
`[max_pages,page_size,row_bytes]`。对应codec见[Compute指南](compute.md#逐行量化的历史存储)。

`schema.version(HostBatchLayout)` 创建 `RowCacheLayoutVersion`：query_indptr把选择行分到请求，
kv_indptr索引各请求的页引用，page_indices与last_page_lengths描述逻辑历史。空历史允许有选择行分组，
但不因此授权任何非空ID；逐query因果可见性仍由消费者表达。

`PagedArena::create(schema,resource,options)` 创建同一页owner体系的行存储；原acquire、
F2更新、publish/complete和COW规则照常使用。`arena.read(layout,ready_pages)` 返回
`PagedRowCacheRead`，严格持有本布局需要的原整页、代际和初始化前缀，拒绝缺页、多页、外来页或子范围。
它不导出whole-arena ReadLease，也不能传给普通attention入口。

CPU/CUDA可用 `hyle/compute/row_cache_paged.h` 的 `PagedRowCacheGatherInvocation`，
直接从该ready frame按dense I32 `[T,K]` 逻辑ID读取，输出BF16 `[T,K,D]`。
每个T行由query_indptr归属请求，ID从该请求历史的0开始；重复选择合法。

```cpp
auto reader = PagedRowCacheGatherInvocation::create(
    context, queue, schema, ids_desc, {.resource = resource}).value();
HYLE_RETURN_IF_ERROR(reader.prepare_layout(layout)); // Explicit cold metadata copy/wait.
auto frame = arena.read(layout, ready_pages).value();
// output_writer is caller-owned and sized from reader.output_description().
{
  auto done = reader.submit(frame, ids_ready, output_writer).value();
  HYLE_RETURN_IF_ERROR(done.wait());
}
auto selected = output_writer.publish().value();
```

私有映射与原checked decoder同Queue执行，先检查全部逻辑ID才写选中结果；越过请求尾部的ID
即使落在物理合法页也会失败。实际Completion保留准确页/IDs/输出/临时量；保留旧Completion会阻止slot复用。
同版本prepare复用metadata，换版本显式copy并等待；`max_backing_bytes`是可选上限，默认自动计算私有容量。
当前物理行容量须不超过INT32_MAX+1，失败slot需重建。该入口只接ready页和ready IDs。
Graph中的动态IDs可用 `graph::batch::add_row_cache_gather(builder, schema, ids)`：返回的宏提供
`bank_input()` 与选中BF16的 `output()`，继续接原Cast、View或attention。用同一source Graph、
通过原route编译后，用 `executable.create_invocation({.resources = graph::ResourceOptions{}})`
创建唯一Invocation。冷调用 `graph::batch::prepare(invocation, graph, queues, node_layouts,
row_bindings, options)`，其中 `RowCacheBinding` 为 `{macro, layout}`；然后用
`graph::batch::bind(invocation, macro.bank_input(), frame)` 绑定准确页。宏内部的容量与metadata
输入由库管理，禁止发布、另加reader或alias；资格来自实际选中的Map和checked codec制品。
宏创建失败可能已经追加输入或节点，应重建builder。准备阶段按版本同步更新metadata，沿原Graph
调度与同一个Invocation保活；source Graph仅借给冷校验，selected输出只实体化选中行。CPU/CUDA已有TopK→跨页解码→Sink
的定向消费，仍须遵守原输出保留、prepare及失败slot规则。

首次prefill可能没有压缩历史，或可见行少于TopK宽度。CPU/CUDA及Apple/Vulkan原生Graph可用
`add_row_cache_gather_padded(builder,schema,ids,ranges,C)`，将它的 `reader()` 放进同一Invocation的
`RowCacheBinding` 集合。`ranges` 是I32 `[T,2]` 的逐query、请求内半开区间；`C` 是逻辑score列数。
ID `-1` 或合法列中落在区间外的项得到BF16正零及F32 `mask()` 的负无穷，真实可见项的mask为0。
将mask接入attention，不能只靠零向量排除padding。其他非法列、range或metadata仍使checked链失败。
非空query的宏要求选择宽度K>0；完全空分支也可直接省略。空历史frame不取得页面读取权限，
但arena容量仍存在。失败Graph的mask可能已由前序Map写入，不能消费失败输出或假定整图回滚。
Apple/Vulkan ready路径保留真实device bank，支持Map/Decode同region或分别执行；layout元数据由库冷准备，
换版本须显式重新准备，同一已准备版本复用镜像。两端已有FP8 padding→Sink及FP4行/mask/转换定向消费，
Apple也支持上述普通非padding gather。旧结果持有时不能复用原槽，空历史不授予任意页权限。
Apple/Vulkan原生pending行页的普通Graph入口见下文。

CPU/CUDA及Apple/Vulkan普通Graph支持 `add_row_cache_indexer(builder,FP4_schema,query,weights,ranges,C)`：
Q和权重的形状/数值合同同[四输入评分](compute.md#逐行量化的历史存储)，输出F32 `[T,C]`。
列号与ranges都从各自请求历史的0开始，`C` 覆盖每个请求的完整历史，并非物理页容量或tile选项。
把该宏的 `reader()` 放进同一 `RowCacheBinding` 集合，准备layout、`bank_input()` 绑定原准确frame；
内部metadata仍由库管理。它可直接接TopK和上述padding宏，不必先解码完整历史。
准备和frame须复用同一 `RowCacheLayoutVersion`；用相同数组再次调用 `schema.version()` 会创建不同版本。
Apple/Vulkan也已实现该五输入分页评分，由实际所选制品声明bank1的受限读取事实。
原生pending继续使用下文同一typed submit/take_update入口，不增加另一套评分执行槽。

CPU/CUDA Graph也可消费正在追加的行页：`arena.pending_read(row_layout, ready_pages, updates)`
返回 `PagedRowCachePendingRead`，它只记录原页身份与所需初始化前缀，并不授予ready读取权限。
在同一 `graph::batch::prepare` 中设置 `PrepareOptions::max_source_completions`，
再用free `graph::batch::bind` 将pending frame绑定到宏的 `bank_input()`，显式准备原Invocation replay，
用 `graph::batch::submit(invocation, queues, std::move(update))` 提交原 `CopyBatchUpdate`。提交会核对原writer、
generation、准确页集合和新增行的完整逻辑字节覆盖，在decoder所在lane等待原producer/copy顺序后
才准入页读取。失败不发布输出；接受的prefix与原控制器一直保留到实际退休。沿原
`graph::batch::take_update(invocation)` / `take_mutable()` 取回writer后，只能publish实际已初始化的前缀。
空footprint仍用ready frame；Direct ready reader不接收pending frame。

Apple/Vulkan普通Graph使用同一pending frame、`max_source_completions`、`bind`和显式replay准备。
接入原生复制来源时，Apple使用 `compute::apple::CopyBatchUpdate` 与 `graph::apple::submit` /
`graph::apple::take_update`，Vulkan使用对应的 `compute::vulkan::CopyBatchUpdate` 与
`graph::vulkan::submit` / `graph::vulkan::take_update`。来源属于原Context/Runtime，不能换成shared Queue提交入口。
FP8 D512已有padding→Sink的融合/分拆消费，FP4 D128已有解码行、mask及后续转换消费；
FP4的维度不满足模型Sink的D512合同。页guard随原native工作实际退休，旧GraphOutput仍阻止来源回收；
回收Update后须reset/重新准备才能开始下一次pending episode，不能将旧pending frame直接重放为ready。
Apple原FIFO Job在读页前检查真实Update的全部phase；接纳后来源失败使该Job跳过数学计算。
失败不发布页内容，已失败的原Invocation仍遵守重建规则，回收Update不代表数据已初始化。

Apple/Vulkan使用各自原生Context时，分别包含 `hyle/compute/apple_row_cache.h` 或
`hyle/compute/vulkan_row_cache.h`，调用对应namespace的 `PagedRowCacheGatherInvocation`。
`create(context, device, staging, schema, ids_desc)` 接真实device/staging资源，随后仍是
`prepare_layout(layout)` 与 `submit(frame, ids_ready)`；结果为原native `OwnedResult<Results>`，
`wait()` 返回原结果lease，输出由runtime持有。Apple的两个资源可复用同一shared资源。
这些入口不要求host读取IDs，保准确页权限；每新版本冷准备会整份F2上传metadata并等待，
每次native绑定和结果仍有metadata费用。资源/allocator沿原借用合同保活到实际退休。
两端的专属reader目前只接ready Direct；普通Graph另有上述ready与pending入口。

## 紧凑 MLA：只存一份 cache

如果模型的 V 就是 CKV 的前512项，使用独立的 MLA schema，避免把它复制成另一份 V。
`kCkv512Rope64` 的每行是 `[CKV512, KPE64]`，Q同样是576项，输出512项；
`kCkv512` 则没有后64项。两者的 KV heads 都是1，query heads由容量指定。
`query_type`、`output_type` 与 `cache_storage` 分开声明；缓存的CKV、KPE和V前缀共享同一解释与固定scale。
当前紧凑缓存接scalar F16／BF16／E4，blocked cache不属于这个物理布局。模型投影与吸收矩阵仍由调用图表达。

```cpp
MlaBatchCapacity capacity{.query_type = DataType::kBFloat16,
    .output_type = DataType::kBFloat16,
    .cache_storage = CacheStorage::float8(0.5F).value(),
    .layout = MlaCacheLayout::kCkv512Rope64,
    .max_query_rows = 4, .max_kv_rows = 32,
    .max_pages = 2, .max_page_refs = 2, .query_heads = 64};
auto schema = BatchSchema::create_mla(capacity).value();
auto cache = PagedArena::create(schema, BatchCacheRole::kMla, resource, options).value();
// Create an owning layout with this schema; initialize/publish its actual pages.
auto program = BatchProgram::prepare(context, schema, program_options).value();
auto invocation = BatchInvocation::create(program, queue).value();
auto frame = cache.read(layout, ready_pages).value();
auto prepared = invocation.prepare_batch(layout).value();
auto completion = prepared.run(query, frame).value();
HYLE_RETURN_IF_ERROR(completion.wait());
```

同一个 schema 不接纳独立的 `kKey` / `kValue` arena或旧三payload调用。完整 owning
cache也可通过 `MlaBatchInputs{query, cache_lease}` 提交；ragged容量以同一结构声明。
普通分页 Graph 使用 `graph::batch::paged_mla_attention(layout, options)`，只有 Q 与紧凑 cache
两个载荷端口，metadata 由选中的实现管理。低层 `compute::PagedMlaAttention`／`RaggedMlaAttention`
仍显式接 Q/cache/metadata，遵守原容量合同。窗口与 `BatchProgramOptions` 一致：-1无界、0仅self、
正值为包含的past count，仅causal允许窗口。CPU paged/ragged支持该窗口；其他route仍按实际原生资格检查。
Provider资格与schema结构分别检查，
不能由某一种路线的成功推导其他路线、编码cache、显式mask或learned sink已接通。

CUDA的单cache追加可沿原 `PagedBatchReceiver` 使用
`submit(query, pending_cache, std::move(update))`，它接纳原复制来源并在同一Queue排序消费；
独立K/V的旧overload不接纳MLA。source容量包含本次复制的phase与实际producer completion，
例如两phase＋一个producer需要3个source slots。精确页权限、输出保留和失败退休规则与普通分页一致。

原plain路线保留：CPU／CUDA paged/ragged及Apple／Vulkan paged的F16、BF16。
BF16 Q／输出加uniform E4缓存的512／576布局已接四后端的paged/ragged原路径，
缓存每行分别为512／576字节，V为同一行的前512项；不建立另一套V bank或等价页表。
CUDA先将当前E4 tile乘原F32 scale后舍入为BF16，保F32累加／split／merge；SM89使用原Q64调度及KV16，
576维共享内存为99KiB。Apple保KV与计算F32，仅用原Q类型存共享Q tile，576维共享内存为27KiB。
具体设备与几何仍在准备阶段检查；[18项新增与10项相邻回归](../evidence/correctness/compact-mla-e4-reference-2026-10-21.json)
只证明有界数学、资源与寿命，不代表整制品、Linux、质量、吞吐或新E4 capture已验证。

共同Graph的MLA descriptor可表达full causal/noncausal及上述causal window，CUDA当前仅承载无窗口路径，
E4还要求BF16 Q／输出；Q／输出类型独立可描述不表示所有组合均可执行。
CUDA原MLA scheduler将KV长度／split固化在计划中，长度增长需要显式冷重规划；只有物理页ID变化可直接复用。
CPU及两原生GPU的paged Q1增长已核原reader复用；Apple／Vulkan delegated ragged无公开page-plan报告，
其warm/fresh数值与寿命已核，但不据此宣称没有冷重规划。
动态per128 scale＋BF16 KPE的656／528字节DSA格式、query-specific selection、显式mask和learned sink另行承载。

CUDA MLA 的 workspace 按实际调度工作项和 split partial 数量计费，默认 partial 为 F32。
无 split 的计划可以不需要 float workspace；slot 容量仍须为正，并按配置预留，
默认容量不会随需求自动缩小。可根据 `BatchPlanReport` 调低
`max_float_workspace_bytes` / `max_int_workspace_bytes` 后重新创建 program/slot。
例如已测的小计划可用16字节 float、4 KiB integer容量；BF16/H64、Q1/KV8192 的
一个真实 split 计划则需要4,202,496字节 float、2,172字节 integer。这些数值只适用于
对应计划，不是其他 batch 的上限。少于实际需求时 prepare 返回 `ResourceExhausted`。

## 允许 BF16 split-KV 中间结果

`BatchProgramOptions::allow_bf16_partials` 默认为 `false`。设为 `true` 允许 CUDA
FlashInfer 在 BF16 Q/output、FP8 K/V、paged page16、Dqk/Dv=256、causal 无界且实际
CTA64/split-KV 的路径存储 BF16 中间结果。FP8 KV 不展开，局部累加、softmax 和 merge
累加仍为 F32；中间舍入可能丢失默认路径保留的抵消残差。其他路径保持原实现。
`BatchPlanReport::bf16_partials` 表示本次计划实际是否采用它。

该选项固定在 program 中；改变选项需要重新创建 program、计划和 replay。Graph 的
`PagedAttention` / `RaggedAttention` descriptor 与 batch provider options 必须设置相同值。
两份 descriptor 的 schema ID 已升为 v2；更新时一起重建相关 Compute、Graph、provider 库及消费者程序。

工作区需求也按实际中间表示计算。两请求 Q=64/128、KV=1024/768、GQA=8/2 的 D256
例子，需求由 21,053,440 降为 10,567,680 bytes（score 保持 F32）。slot 仍按
`max_float_workspace_bytes` 预留 backing；需要同时调低该容量，例如本例设为 `11u << 20`，
才能减少实际显存预留。该容量不是其他形状的通用上限，不能只打开选项就宣称预留减半。

## 在普通Graph内追加K/V

用`graph::batch::PagedAppend`表达无tensor输出的写页节点，输入是本轮新行。
`PagedAppendPages::create`接原arena、目标layout、旧ready帧及新页writer／独占尾权；
`batch::prepare`之后，用带bank input的`bind_append`连接后继reader，并显式调用`depends_on(reader, writer)`。
只写缓存时使用不带bank input的重载，整图可以没有公开tensor输出。原生执行源不需要调用方提供Compute Queue。

真实工作退休后先释放公开结果，再用`take_append`取回原事务；只有整图成功才能`ready()`发布新页帧。
失败不发布，脏页按原writer的退休／`discard()`合同恢复。CPU／CUDA已有普通F16/BF16路径；
Apple／Vulkan已接原生追加，F16/page16/D64及四输入容量reader的BF16/page16/D256均有本批实际检查。具体reader的dtype／形状仍由所选实现检查。
完整调用和独立数值预期见[原生追加消费者](../../tests/graph_batch/native_paged_append_consumer.cc)及
[CPU追加测试](../../tests/graph_batch/paged_append_graph_test.cc)。

Apple／Vulkan普通Graph也可使用`compute::PagedAttention{capacity}`的Q/K/V/metadata四输入形式。
metadata由`batch::prepare(invocation, queues, NodeLayout...)`绑定，消费者不另建或上传页ID表；
reader直接保留原metadata租约的页ID子范围。冷编译固定物理Q/output行数R，NodeLayout可有A≤R个活跃行，
只计算活跃行并将输出后缀逐位置零；无需伪造查询或按每个T另建Graph。
用法见[同一有限图的容量消费者](../../tests/graph_batch/native_paged_capacity_consumer.cc)。
Direct `NativePagedBatchProgram`的owned layout／私有metadata banks仍按下文合同使用。

## 编码新行并接入原分页更新

完整可编译旅程见
[CPU/CUDA共同消费者](../../tests/compute_batch/encoded_page_common_consumer.cc)与
[Apple/Vulkan消费者](../../tests/compute_batch/encoded_page_native_consumer.cc)。先用与schema
相同的CacheStorage构造K/V编码节点，输入只包含本轮dirty rows；通过
`infer_single_output(Operation::custom(model_ops::CacheEncode{storage}), inputs)`取得实际
carrier描述，不能把逻辑blocked cache描述当作字节输出。

公共Graph invocation由调用方绑定输出storage时，同Graph的K/V输出可绑定同一个真实
WriteLease的两个不重叠子视图。原LeaseUsePool写票附着该Graph的原Completion，再把
该writer和Completion交给批量搬运的`bind_mutable`/`bind_producer`。两份独立Graph则保留
两套真实writer与Completion。CopyRange使用各自K/V row bytes与实际子视图offset，
随后将同一CopyBatchUpdate和pending frames交给Direct或Graph receiver，观察最终结果。
编码 producer 的原生 Program 可将 `OwnedResult<Results>` 交给原 result-source 入口；
后续共同 Graph 用原 `Completion`／`GraphOutput` 观察，完整调用见夹具。

旧reader仍存活时，首次更新须显式克隆对应尾页；其后只有新行需要编码和复制。
成功完成后才能从原Update取回目的writer并发布有效前缀；结果仍被持有时遵守原slot
覆写限制。错误可能在source准入时同步拒绝，也可能由接受后的最终Completion返回；
失败路径均不得发布新页面，原reader保持有效。CPU的同步拒绝不等于GPU异步失败资格。

| 路线 | 数学与公开入口 | 准备与成本 |
|---|---|---|
| Windows/Linux CUDA | 原FlashInfer C++ native paged/ragged；BatchProgram、BatchInvocation、PreparedBatch；原Graph Invocation + batch::prepare/bind | 上述 SM89/格式/形状域；显式active/delta上传；原schedule等价时复用plan，跨界重规划；native private bytes未知 |
| CPU 参考 | MaterializedBatchProgram；MaterializedBatchInvocation / graph::batch::MaterializedInvocation | 显式gather、GQA展开、F32矩阵、mask与Copy；Direct逐stage同步wait |
| CPU page16 | 原CPU BatchProgram / BatchInvocation；详见[CPU批次入口](compute.md) | F16/BF16，含plain512/576→512；原ggml flash直接读取确切物理页。Q1可更新运行时页表/长度，保原graph/pool/workspace；Materialized路径另为显式参考算法 |
| Apple | apple::MaterializedBatchInvocation；graph::apple::prepare_materialized_batch | 原MLX Program/worker，固定两device metadata banks、批量搬运上传；managed replay/私有storage未知 |
| Vulkan | vulkan::MaterializedBatchInvocation；graph::vulkan::prepare_materialized_batch | 原native Plan/command/fence与ggml承载；固定两metadata banks，opaque scratch/driver仍未知 |
| Apple page16 | NativePagedBatchProgram；apple::PagedBatchInvocation | 原 MLX 直接页读取；F16/BF16，含plain512/576→512，单 I32 表与两个 mirror banks |
| Vulkan page16 | NativePagedBatchProgram；vulkan::PagedBatchInvocation | F16/BF16，含plain512/576→512；原 FullF32 scalar flash直接页读取，Q转F32，KV保原存储 |

Apple/Vulkan参考档真的在原GPU上执行，但它们把页/packed KV显式gather/materialize后做
数学；索引/掩码、GQA复制、payload/wait的费用必须计入。需要原native paged吞吐时，
不能只看接口形状相同而假定性能已经对称。没有submit隐式plan、JIT或自动CPU fallback。

CPU `BatchInvocation::update_batch` 在 Q-indptr 相同、每 request 为0或1 query时，允许
tail、跨页和页 ID 重排更新原 native leaf；ragged 则更新每 request 的实际 KV extent 和
packed offset，仍用三个 packed-capacity Q/K/V read ports。保持原 graph、线程池及 workspace，单查询的
全零 causal mask 不分配。普通调用不需要另写 decode helper。请求集合或 Q 拓扑变化
仍真正重规划，mixed/prefill 沿原冷规划合同；旧 output/ticket 仍阻止更新。
caller 创建的新版本仍拥有 active-sized snapshot，metadata delta 上传也可有批量搬运 等待，
不能把计划零分配解读为整步零分配或零等待。
Apple/Vulkan原生ragged沿各自`prepare_ragged_batch`返回的Program/Invocation，使用公开
`update_ragged_batch(slot, layout)`，固定schema/完整Q拓扑下更新KV长度与packed起点，
成功后重新bind原三个容量输入。调用外部串行，旧结果/在途/失败/关闭受原门保护；非法布局
保旧绑定。变化长度时，Apple 仍可能 managed 建图，Vulkan 仍重新编码；不因此取得 capture 能力。

Windows/Linux CUDA 的公开 factory 随普通产品传递，消费者只链接 hyle::compute/hyle::graph；
cuda.h在实际HYLE_COMPUTE_FLASHINFER资格下供应原factory头，未准入平台不供应该API。
完整公开D256 Direct/Graph用法见[可编译消费者](../../tests/compute_batch/public_batch_cuda_consumer.cc)，
其输入与结果通过原lease接口读写，数学仍由锁定FlashInfer执行。

## Apple / Vulkan 原生 page16 接入

两路线使用相同 owned schema/layout/table 与确切 frame 合同，保留各自
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

`schema` 固定 paged/page16、Q 行数上限与真实 physical page 容量；D64/128/256，
Apple／Vulkan均有F16/BF16承载。非空 Q 需有 KV，causal 需 logical KV≥Q。空 request 跳过。
用 `recipe.input_description(layout, port)` 取得本次端口描述：Q 为活动 packed 行数，K/V 与页表
保留固定容量；`input_count()` 返回 recipe 端口数。查询不创建 version、owner 或 metadata。
上游投影、归一化与跨域复制使用实际 Q 描述，无需补齐到最大行数；原有足够大的存储可继续复用。
逻辑重复页可以使 KV 长度超过 physical 容量；root shape 不随逻辑长度扩大。K/V frames
须由对应 PagedArena.read 与本次同一 owned layout/generation 创建；bind 拒绝混代或
foreign schema，不扩大未引用页权限。已 ready 页可共享，未引用页仍可独立更新。

当整个布局的 Q rows 与 page refs 都为零时，调用 `slot.bind(query_read, {}, {})`。
私有原生程序只绑定零行 Q 并返回空 view；不读取或伪造 K/V storage。这个零域不接受
其它代际的 live frames，旧空结果同样阻止 slot 更新。非空布局仍要求精确页权限。
Vulkan 的零字节描述符需最小物理载体：能共享已有合法 backing 时复用，否则冷分配四字节并计入原预算，
逻辑形状仍为零，不增加 Q 计算。

原生分页 controller 在 schema、完整 Q indptr 与空输入端口域相同时复用 Program/Invocation；
KV indptr、tail 和 page IDs 可在显式 prepare 中更新。Q 拓扑变化，或从“Q/KV全空的一端口”
切换到“Q空但保留KV的四端口”时仍冷建。page IDs 由 device 表消费。每个 mirror
记录自己的有效页表，只通过批量搬运 上传实际变化区间并清除缩小留下的旧值；相同表可零上传、
零上传等待。staging 容量仍为 `4 * max_page_refs`，device mirrors 为其两倍，固定 backing
cap 还需三组64byte对齐的最坏 padding。report 的 uploaded bytes/waits 是当次实际费用，
staging/device/backing bytes 是容量；`reused_native_plan` 表示通过原结果/退休门后的计划复用。
controller 创建时经批量搬运 分别清零两 bank，显式冷费用为两次全容量 copy 与完成等待；
它们不计入以后每次 prepare 的 delta 报告。device 不要求 host-mappable。
`NativePagedBatchVersion::indices()` 是显式、可并发的固定容量导出，首次分配填零并缓存，
失败可重试；普通 version 与 controller prepare 不必生成这个全容量 blob。
Apple 每个 Invocation 独立保留当前 layout，初始长度使用原 compiled 路径，其它长度
在原 worker 上组合原 MLX fast primitive；不逐 tail 重新 Context.compile，但仍有 managed
图构造费用。Vulkan 更新当前 Invocation 的逻辑长度/表偏移，并让原录制失效后重新编码；
原 graph 与 storage 保留。`reused_native_plan` 不表示 capture 或 native 零分配。
无需全历史 KV gather、按 Q/head 展开或上传 additive mask。原 query/output/MLX compiled
metadata、shader scratch 与 driver 存储仍是额外费用，private bytes 可以 UNKNOWN；
delta 上传、计划复用与 capture 分别按原生接口检查和报告。

Context/device/staging/injected allocator 保持地址稳定，活过所有 native work 和结果；
控制调用外部串行，多个 slot 独立。旧 result 阻止同 slot prepare；原结果与实际退休门
通过后的 prepare 释放旧 input import，同 topology 也可重新取得准确页 writer；caller
仍须释放自己保留的旧 frame/reader。计划与原 Vulkan output bridge 可保留。close 停止新工作，
drain 沿原实际退休观察，已接纳错误或未知完成不许可提前释放页 guard。结果按本 route 的 Results/readonly import 使用；未知退休不能当作已完成。

## 普通变长图与冷布局准备

`compute::RaggedAttention` 使用容量描述和四个普通Graph输入：Q、K、V、metadata。
通过 `graph::batch::compile` 编译，用每个节点的 `NodeLayout{node, owned_layout}` 冷调用
`graph::batch::prepare`，再绑定Q/K/V、输出并调用原 `prepare_replay` / `submit`。
metadata由库按实际计划的输入资源绑定只读租约；普通bind不能覆盖，变更布局应重新完整prepare。
多个节点共用metadata输入时须复用同一个owned版本，而非重新调用version生成相同数组。

CPU/CUDA共享批次实现与Apple/Vulkan原生GQA实现均沿此流程；Q/K/V可由前序Graph节点生成。
普通Paged/Ragged图的Q和输出按具体行数R分配，schema的`max_query_rows`只是上限C；
冷准备要求布局的活跃行数A满足`A <= R <= C`。`batch_query_description(capacity, R)`与
`batch_output_description(capacity, R)`构造匹配端口；原单参数query helper仍构造C行。
同一符号图可用`kFiniteExact`预编译R=128和R=1，在创建Invocation时选点，各槽保留独立资源与旧结果权限。
原PreparedBatch Direct输入仍按其容量合同；owned分页recipe的Q/output精确等于活跃行，显式MLA端口仍有独立约束。
未活跃输出后缀[A,R)为零，A=R时不增加清零命令。CPU沿原同步执行序列写入，CUDA在原stream中追加可捕获的清零，
Apple/Vulkan按R-A构造；不得写入实际TensorView以外的schema容量或backing余量。CPU/CUDA的F16 ragged、BF16 paged
已覆盖A1/R4、A1/R1/C128、A0/R4及重复执行的输出护栏，CUDA要求capture；见[修复实测](../evidence/correctness/batch-output-tail-2026-10-20.json)。
原生直接ragged接口仍返回活跃行。
Apple/Vulkan同query分组的KV更新复用原程序，query分组变化在prepare中冷重建，共同submit不构图。
Apple原managed runtime对非初始KV长度仍有原生array构造和求值费用，复用Program不表示零分配。
物理输出的补零与拼接有实际计算和存储成本；R可小于C，无需把schema上限作为每个入口的真实分配。
这一接线不扩张各原生实现的dtype/几何能力，也不表示Apple/Vulkan已有compact RaggedMLA。

## 同一分页图用于 CPU 与 GPU

[Apple/Vulkan 分页消费者](../../tests/graph_batch/native_paged_joint_consumer.cc)使用普通
Graph 的 CPU→原生分页→CPU 流程。选择 `graph` 产品，配置 `cpu;apple` 或 `cpu;vulkan`，
链接 `hyle::graph`；需要异步推进时再显式选择 `graph_progress` 产品、链接 `hyle::graph_progress`，
冷调用 `prepare_progress`。按[普通 Graph 接线](graph.md)提供实际 CPU Source 与
`graph::apple::execution_source`／`graph::vulkan::execution_source`，以及真实存储域：
Apple 使用主存与原 shared Metal 资源，Vulkan 另提供实际 staging 资源。两个 K/V
外部输入的资源合同均指定为实际 arena 资源；只给报告要求的 shared lane 绑定 Compute Queue。

```cpp
namespace g = hyle::graph;
namespace gb = g::batch;
auto descriptor = gb::paged_attention(initial_layout, {.scale = scale}).value();
const std::array ports{ready_query, physical_key_input, physical_value_input};
auto page_node = builder.add_node(
    hyle::compute::Operation::custom(descriptor), ports).value();
// Finish graph; placements describe the CPU pre/post nodes and native page node.
auto program = gb::compile(graph, environment, {.node_placements = placements}).value();
auto slot = program.executable.create_invocation({
    .resources = g::ResourceOptions{}}).value();
const std::array layouts{gb::NodeLayout{page_node, layout}};
HYLE_RETURN_IF_ERROR(gb::prepare(slot, queues, layouts));
HYLE_RETURN_IF_ERROR(slot.bind(source_query, query_description, ready_host_query));
HYLE_RETURN_IF_ERROR(gb::bind(slot, physical_key_input, key_frame));
HYLE_RETURN_IF_ERROR(gb::bind(slot, physical_value_input, value_frame));
HYLE_RETURN_IF_ERROR(slot.bind(output, caller_host_output));
HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));
auto completion = slot.submit(queues).value();
HYLE_RETURN_IF_ERROR(completion.wait());
auto result = slot.result(output).value();
```

同一节点与构图用于 CPU/CUDA 批次能力及原生 Apple/Vulkan 分页实现；改变实际 Source、资源域与放置，
不改变载荷端口。节点只拥有初始 layout 和 scale/causal/window/精度许可，layout 已持有 schema。
不需要为普通 Graph 创建 `NativePagedBatchProgram` 或另登记原生 recipe。`gb::compile` 将批次能力
装配进原 Graph 编译器，结果仍是同一 `Executable`／`Invocation`；metadata、工作区与原生计划
由选中的实现冷准备，使用原共享预算和槽 allocator。shared Queue 的诊断用 `gb::report`，
原生实现用 `gb::native_report`；都在没有在途工作和旧结果的冷边界查询。
最终 CPU 输出使用普通 Graph 的调用者存储合同；每槽可复用独立的 host buffer，须保活至工作和结果退休。

改变页顺序、共享、tail 或请求分配时，先释放旧结果，再调用 `gb::prepare`、绑定新页帧、
`slot.prepare_replay`。`gb::prepare`保留普通输入／输出绑定、lease和不可变上传来源，撤销旧布局的metadata、
准确页权限及来源准备信息；稳定端口只需首次绑定，实际更换载荷时再bind。显式`slot.reset()`仍清空全部绑定。
全部 live 页节点各提供一次 layout，共用 K/V 输入须使用同一 owned
版本；固定 active packed 输出 shape，总 Q 数改变须编译另一 shape。原生计划可复用时
保留实际 backing，Q 拓扑改变可冷替换 backing，并重新绑定下游。固定 layout/frames/普通
bindings 不变时可直接重放原 Invocation，仍检查实际退休与旧输出 owner。

布局不变、没有在途工作或旧 Graph 结果时，可用 `gb::bind` 替换 ready 页帧，再显式
`slot.prepare_replay`。绑定拒绝保留旧帧与旧 replay；绑定成功才使旧 replay 失效。
这不包含 pending 来源替换，也不保证下一次 replay 准备失败后回滚整个原生程序。
旧 native import 可能保有旧页，直到实际重绑或释放；不能据此假定旧页已可写。

K/V 必须是外部 inputs，所有实际读者须为已选分页制品的相应 ports，不能发布整 capacity、
添加 dense/copy/query 读者或用普通 lease 代替精确 frame。错误 schema/resource/generation、
重复或漏 layout、变化输出 shape 会拒绝。拒绝 prepare 后须重新完整准备布局和页帧，稳定绑定仍保留；不能沿用旧页权限。
未引用页可独立更新。metadata 初始化和变化表上传仍有真实冷 F2 等待，native/private 成本
保持实际报告，不承诺 capture、零 warm allocation 或整机显存硬上界。
Apple plain F16/BF16 的统一 F32 partial 内核仍是未完成项；输入／输出 dtype 资格不等于中间精度保证。

共享 Queue 路线可复用同布局、资源和实际队列的已上传 metadata；各节点仍预留自己的
mirror/staging 容量。冷准备借用的原 Queue 外壳须保持地址稳定，不能将热提交持有 QueueRef
理解为所有嵌套复制／规划对象都允许迁移外壳。

全 Q0/ref0 时，`arena.read(empty_layout, {})` 保留真实 capacity 身份，不取得任何页 ReadLease。
只有所选实现确实没有 K/V 读取时才准入这种 frame；普通非空算子不能借此
绕过页权限。未引用页仍可持有 writer，零 footprint 不代表零 arena 分配。

Apple/Vulkan pending frame 同样通过 `gb::bind` 绑定，先在 `gb::prepare` 的 `PrepareOptions` 中设置
`max_source_completions`，冷预留真实来源控制容量。完成普通绑定与 `slot.prepare_replay(queues)` 后：

```cpp
{
  auto completion = g::apple::submit(slot, queues, std::move(update)).value();
  HYLE_RETURN_IF_ERROR(completion.wait());
  auto result = slot.result(output).value();
}
update = g::apple::take_update(slot).value();
```

上例使用同 Runtime 的 `compute::apple::CopyBatchUpdate`；Vulkan 对应调用
`g::vulkan::submit`／`take_update`，接 `compute::vulkan::CopyBatchUpdate`。
准确 pending 页覆盖在移交前核对，
原 native submit 排序来源并准入真实页读取；Graph 沿原 Completion 合并来源和消费后缀。同步拒绝保留
调用者 Update；接纳后的来源或消费失败须等真实退休、释放结果与 Completion 后才能取回原控制。
Apple 沿原 Runtime FIFO／Job 排序；Vulkan 当前只接原 fixed native queue prefix，
async/IREE 来源仍按其原资格拒绝。

实际消费者覆盖 ready 与全 pending 双节点链；逐端口 ready/pending 混合及错误注入未据此完成数值验证。
Q 输入按当前 owned layout 的活动行数描述，decode 不需要为 prefill 容量补齐上游计算或传输。
两端均沿本节同一原 Invocation 执行；格式、数值与失败资格仍分别按实际实现检查。
`request_cancel()` 不提前释放已接纳的页 guard；`close()` 关闭新接纳，`drain()` 等待该槽
实际退休并保留首错。原生 generated 输出由实际 owner 保活，成功后通过 `GraphOutput`
观察；CPU caller 输出仍使用调用者绑定的存储。两者均遵守旧结果阻止同槽覆写的门控。

## Graph 的执行 lane 绑定

普通Graph与typed batch使用同一 `graph::Invocation`、执行lane和结果槽。Queue绑定为
`graph::QueueBinding{lane, &queue}`。只列 `executable.requires_queue(lane)` 为真的lane，每个一次。
Context 必须拥有该 Queue，且 Queue 对应 Graph 编译时同一 target；同 device 不是同 Context。

```cpp
const std::array queues{
    hyle::graph::QueueBinding{gpu_lane, &gpu_queue},
    hyle::graph::QueueBinding{cpu_lane, &cpu_queue}};
auto invocation = executable.create_invocation({
    .resources = hyle::graph::ResourceOptions{}}).value();
HYLE_RETURN_IF_ERROR(hyle::graph::batch::prepare(
    invocation, queues, node_layouts, preparation));
// Bind complete ordinary inputs and caller outputs, including exact page frames.
HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues, replay_options));
auto completion = invocation.submit(queues).value();
```

`batch::prepare`只冷配置原selected region的native plan/token和metadata，不创建另一个Context、
Invocation、执行cursor或Queue副本。各batch region按原Graph lane使用真实Queue；
`PrepareOptions::host_workspace`必须满足实际provider和Context的资源合同。
资源与 Context/Queue 沿原 lifetime
合同存活，不能在 native work 或旧结果仍被保留时销毁。首次prepare后bind输入/输出，后续布局更新保留这些绑定；
显式调用原 `prepare_replay` 处理完整已绑定topology；submit不隐式进行batch规划。
可选的 `graph::batch::prepare_replay(invocation, queues)` 只为selected native slot准备capture；
它不替代原 `Invocation::prepare_replay(queues, replay_options)` 的整图提交路由准备。
Apple/Vulkan 分页均沿上述同一 lane 接口；这不增加各实现的 capture 能力。

## 容量、有效区与版本

先创建独立的BatchSchema。BatchCapacity固定storage、dtype、request/Q/KV/page/ref容量、
query/kv heads、head_dim、page_size。Paged与ragged是不同schema，F16/BF16采用F32
accumulation；GQA必须整除且在实际route资格域内。

BatchLayoutVersion由schema.version创建：query_indptr描述packed active Q；
ragged kv_indptr描述packed KV行；paged kv_indptr描述每request页引用区间，
page_indices为physical页ID，last_page_lengths为最后一页有效长度。快照拥有host数据，
调用返回后caller可释放原数组。相同shape但新值仍是新generation，不能用地址当版本。
错indptr/page/tail、overflow、foreign schema在接纳前拒绝。

版本快照只拥有实际有效数组。需要将布局放入固定 State/Graph schema 时，显式调用
`layout.metadata()` 导出固定容量 readonly blob：首次调用分配、填零 inactive 区并复制
有效值，后续调用共享同一 backing。导出可并发，失败保留原版本并可重试；普通
`BatchInvocation::prepare_batch/update_batch` 不需要消费方先做这个全容量导出。

物理Q容量与active Q不同；只对有效行计算attention，未活跃输出后缀按合同清零。尾padding不是KV。
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
创建本route Context、device/staging MemoryResource；批量搬运上传Q/K/V并取得成功ready lease；
创建typed Direct controller或同route的Graph factory；prepare_batch(layout)上传同一
owned generation，再bind和submit。OwnedResult.wait成功后只借ResultLease访问原native
数组/tensor。旧ResultLease/Completion仍存活时prepare必须拒绝，不能覆盖旧结果。

Context、resource owner、Queue和借用allocator必须晚于所有相关native work、结果、
exported storage和资源租约 uses销毁。控制对象thread-compatible，caller串行控制；独立slots
可以独立使用。不要在另一个线程还prepare/bind时同时控制同一个对象。

## 页写入、append 与状态发布

PagedArena冷分配连续backing和独立page权限。acquire取得具体physical页writer；经原
批量搬运 clone/append成功后调用complete(ready, initialized_rows)或publish(initialized_rows)，
显式提交连续有效行前缀；值须在[0,page_size]内，0表示尚无有效行。read(layout, exact ready pages)
检查每个引用所需的有效行数，非尾页需要整页，重复/共享页必须满足所有request后才产生精确页frame。
首次取得页或旧页回收为新generation时前缀为0；acquire(ready)与未提交writer归还保持原前缀。
initialized_rows()只返回已提交前缀，写入与copy退休不会自动扩大它。由调用方在真实写成功后声明。
未引用的页可以写，仍有旧reader/native use的页不能覆写；整个arena的subview不是独立writer。

不同会话已保留ready帧时，用`arena.reframe(target_layout, source_frames)`组合批次或投影回单个请求：

```cpp
const std::array sources{alice_keys, bob_keys};
auto batch_keys = key_arena.reframe(batch_layout, sources).value();
auto alice_only = key_arena.reframe(alice_layout, std::span(&batch_keys, 1)).value();
```

来源必须是同一arena的真实ready帧；目标只能选择来源布局与原页guard共同覆盖的物理行前缀，
不能借whole-page guard或已增长的frontier扩大旧帧的逻辑范围。结果只持有目标实际需要的页，
不复制缓存字节，也不保留来源的无关页；此调用有显式metadata分配，来源在成功或失败后均不变。
同名强类型重载支持`RowCacheLayoutVersion`／`PagedRowCacheRead`；pending／writer仍须沿原退休和成功发布链，不能通过重组成为ready。
帧投影只处理页面读取权限；混合recurrent模型仍须同时保存对应位置的conv／GDN状态。

批次容量中的`max_query_rows`与`max_kv_rows`限制所有请求的行数之和；`max_page_refs`限制页表项总数，
共享物理页在各请求的页表中仍各占一项。`max_pages`单独限制实际物理页池，不能把每请求上下文上限当成批次总KV容量。

`arena.pending_read(layout, ready_pages, page_updates)`可冷构造独立的
`PagedArenaPendingRead`：ready页保原权限，pending页记录已消费writer的原页身份、代际
和已提交初始化前缀。`PagedPageUpdate{&page, destination_id}`只在构造调用中借用page；
原copy控制必须持有该目的页。frame保精确layout与arena，但不产生ready lease或public字节
读取权限；原ready-only bind/run不接受此类型。receiver须验证本次accepted copy覆盖新增
要读的bytes、实际native排序与全引用页guard，并合并来源结果。

Apple/Vulkan 的 `PagedBatchReceiver` 是显式冷扩展，移交一原 `PagedBatchInvocation`，
保同一 native invocation、queue 和结果槽。`max_source_completions` 按原更新中已接纳
phase 与实际使用的 producer completion 数量设定；不足时拒绝，不消费 Update。
先创建 receiver、准备本次 layout，再提交批量搬运，最后将 Update 与精确 pending K/V
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
private unknown 保留。这个 Direct 入口覆盖两条 native 路线；Graph 的 pending 输入组合见后文。

共同 `hyle::compute::PagedBatchReceiver` 移交原 `BatchInvocation`，沿其原队列、
PreparedBatch 和输出 buffer 消费同样的 pending frame。CPU 与受支持的 Windows/Linux CUDA
原生路径已有 source/static/shared 双请求、双槽、64步公共旅程；CPU 宿主风险与 CUDA
实际来源/退休边界也已定向验证。Graph与late native producer组合见最终裁决。
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

同一原 `graph::Invocation` 的资源附件保存来源controls和准确页frame，执行仍用原cursor和
输出槽。pending frame只绑定已准备paged节点的K/V或单MLA cache端口。Q用ready lease，外部
输出用writer，metadata由 `batch::prepare` 绑定。source容量包含producer和全部copy phases。

```cpp
auto invocation = executable.create_invocation({
    .resources = graph::ResourceOptions{}}).value();
preparation.max_source_completions = source_capacity;
HYLE_RETURN_IF_ERROR(graph::batch::prepare(invocation, queues, node_layouts, preparation));
HYLE_RETURN_IF_ERROR(invocation.bind(query_id, query_desc, query));
HYLE_RETURN_IF_ERROR(graph::batch::bind(invocation, key_id, keys)); // PagedArenaPendingRead
HYLE_RETURN_IF_ERROR(graph::batch::bind(invocation, value_id, values));
HYLE_RETURN_IF_ERROR(invocation.bind(output_id, output_desc, std::move(output_writer)));
HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues, replay_options));
{
  auto completion = graph::batch::submit(invocation, queues, std::move(update)).value();
  HYLE_RETURN_IF_ERROR(completion.wait());
  auto output = invocation.result(output_id).value();
  // output.storage_lease() retains the original output storage.
}
update = graph::batch::take_update(invocation).value();
```

来源排序在原 shared sequence 中先接纳，随后取得精确 pending 页 guards；原公共
completion 合并来源与实际消费结果。来源失败不发布成功输出。旧 Graph output/completion、
实际 pending uses 或未回接的 Update 阻止复用，不能只释放一个 public handle就覆盖原槽。
Apple/Vulkan 普通 Graph 的 native pending 接线见[同一分页图用于 CPU 与 GPU](#同一分页图用于-cpu-与-gpu)，
使用 `batch::prepare`、`batch::bind` 和对应的 typed `graph::apple::submit/take_update` 或
`graph::vulkan::submit/take_update`。两端均保留真实原生 Update、来源排序和控制归还。
一个固定图的 active Q 输出行数保持固定；2行 decode与18行
混合 prefill可冷准备两个图，每个图各两槽，合计四个冷槽。这些槽/metadata准备/跨CPU边界
的 checked等待均计成本。它不创建第二个 Graph 或 native 执行器。

CUDA wrapper 可先用 `BatchProgramOptions.resource={}` 声明 schema，在其 Context 上创建
实际 device/pinned资源，再通过显式 CompileEnvironment固定这些 storage domain。批量搬运、lane
workspace和页 arena使用同一个 Context的资源；其它 Context的同设备资源不能冒充这个身份。

Apple/Vulkan 批量搬运 可冷绑定同一个已接纳 native producer 的多个输出；以下 `native` 指
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
不构成 ready/资源租约 权限：Apple 在原 worker 的 preflight 取得实际 backing 并完整重验范围/别名
后才 blit；Vulkan 可冷取得实际地址，但来源排序与成功仍由原 producer合同裁决。foreign
Runtime、未资格的动态 queue prefix及不匹配输出拒绝。producer 槽须等原结果/storage/source
全部引用与实际工作退休后回收。两路线的实际多输出 Abs→批量搬运→Direct pending attention source
旅程已验证，独立 Abs多输出→两阶段范围重排→CPU reduction的64步source也已验证；这个
正交消费者在最终 批量搬运成功后显式搬运256字节至host，不声明 GPU reduction流水或零copy。

批量搬运已消费page writer时，保留PagedPageWriter，等成功ready输出后调用complete；若放弃
这个已消费的reservation，arena不会猜测该页已退休或重新ready。
失败或取消后，保留原PagedPageWriter，释放原copy/producer控制及所有页面权限，再调用
`writer.discard()`。若lease()仍由自己持有，先显式reset它。discard不等待：原物理范围仍被
writer、reader或实际native use保留时返回Unavailable，可稍后重试；公开错误或取消返回
不构成退休凭证。成功后原writer重新持有同物理页的新generation，initialized_rows为0，
必须重写并publish实际初始化的前缀。旧失败内容不会发布，也不保留此前的有效前缀。
未知退休继续阻止恢复；若已丢失原reservation，仍须显式close/recreate arena，不能凭page ID夺回权限。

状态版本 StateStore/StateDraft列出完整K/V/layout及其他需要保留的成员。共享干净页，dirty页显式
clone到新writer；只把所有成功ready成员finalize为candidate，再由StateHead.publish用
expected-base切换。失败copy不产生ready writer/lease，不能送进 attention，也不能发布状态版本 head。
取消不能回滚已经实际写过的底层字节，不能在失败后伪造整页CoW成功。

`arena.acquire(page_ready)` 只在这个原整页
没有其它 reader/native use 时消费其 ready lease，返回同 generation 的 writer；拒绝
保持入参不变。共享页先沿上面的显式 clone 路线处理，不能原地覆盖旧版本。

批量搬运 只写 dirty range 时，`output` 仍不能把未写 bytes 算成成功 copy。保留 page writer，
在实际成功退休且 submission 已释放后，用 `copy.unbind_mutable(id)` 取回原 WriteLease，
放回 `page.lease()`，再 `page.publish(initialized_rows)`。未写部分是 caller 原来已初始化的内容，成本
只计本次 dirty bytes；新页仍须初始化所有将被读取的 bytes。示例顺序如下：

```cpp
auto page = arena.acquire(page_ready).value();
HYLE_RETURN_IF_ERROR(copy.bind_mutable(destination, std::move(page.lease())));
// bind source/producer and prepare exact dirty ranges with the original 批量搬运.
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
它会沿批量搬运 作非阻塞退休观察；仅放弃 submission 时仍保留 PagedPageWriter，后续
调用可在成功实际退休后取回权限。不能销毁 page writer 后猜测 arena slot 已恢复。
这里仍有显式 host observation；不能从精确更新本身推断整条 producer→attention 流水完全异步。

Ragged多request append可能需要重排后续request的offset。当前真实两request例子显式
clone有效prefix、append新行并初始化unused capacity；这不是独立物理页CoW，成本账必须
把tail初始化和重排算进去。批次接口不解释token/page eviction/admission或模型scheduler策略。

## 错误、关闭与费用

prepare/update可能copy、host wait或调用原native planner；它们属于显式准备，不是免费
warm submit。BatchPlanReport记录generation、metadata上传、workspaces、replan/reuse、
split与host waits；MaterializedBatchCosts是保守logical上限，包含view/alias等，不能当RSS。
共同 BatchInvocation 只上传有效字段中改变的区间，并清零缩小有效区留下的旧值；同一
版本且原 plan 可复用时，metadata 上传与准备 host waits 都可以为零。不同 generation
即使数组值相同，仍须更新版本字段。报告的 staging/upload bytes 是实际搬运量，不代表
比较有效数组所需的 CPU 扫描量。共同 Graph 中多个 node 读取同一个 metadata 输入时，在首次上传前共享镜像；每个 node 保留自己的 plan/workspace。Apple/Vulkan原生controller各自持有index banks，
每个bank按自身旧值做delta，原生规划和费用沿各自报告。
FlashInfer 的显式 update 在同 schema、完整 Q indptr 下，检查原计划实际 chunk/CTA/window
及每个有效 request 的 chunk 数；调度表等价时复用原 plan，边界变化时真正 cold replan。
KV offset、page IDs、tail 仍由原 runtime metadata 消费；该判断扫描 requests，批量搬运
另计 active metadata delta。零Q可经完整owning Q/K/V入口
执行checked no-op，显式准备仍可
计一次立即completion的host wait；attention的PagedArena.read继续要求非空准确页集合。原生private/
driver费用仍为UNKNOWN。

native controller报告staging、两份device mirror、固定backing、实际上传字节和原生计划复用；未知private
metadata/scratch/driver费仍是UNKNOWN，硬拒未知预算按原资格明确拒绝。每次prepare创建
的小NativePagedBatchVersion owner可经recipe allocator分配；caller创建layout的active
backing/control、generation CAS与拷贝也计费。另一个invocation allocator无新增attempt
只证明该分配域的观察。显式metadata()/indices()首次生成全容量blob，重复缓存访问仍锁。

失败和unknown观察都不产生成功输出。public terminal或释放public result/token不代表
accepted native prefix已退休；drain按实际guard/fence/worker等待并保留原错误。正常放弃
可在真实退休后成功排空，取消停止未开始工作。close/drain后的对象不重用；错误slot的
复用必须遵守原route失败合同，不靠清除一个cached status假装恢复。
共同pending Graph每次 `batch::prepare` 清除上次pending输入，并冷预留来源、reader、lane及页guard容量；
bind仍可能创建准确reader proof元数据，外层reserve不等于每步零分配。warm submit不新增来源协议节点、
不复制KV或创建队列；完整 prepare/bind/submit费用应分别计入。CPU+CUDA图的 GPU→CPU
边界可沿原 replay合同执行 checked host wait，因此 submit不保证非阻塞。规划与
流水分别处理这些成本，不从 pending页接纳推导完全异步或 capture资格。

CUDA 已接纳 native work 的 request_cancel 可返回 Unimplemented；必须检查返回值，
不能把拒绝取消当成已经取消或退休。停止提交后继，等原实际工作退休后再释放相关权限。

`PreparedBatch::prepare_replay(inputs)`或query＋paged ready
frames显式录制，之后仍调用原`run`；Graph batch在prepare/bind后显式`prepare_replay()`，
再原submit。录制checked-retire但不执行用户tensor或消费run token；每次真正执行重新准入
当前权限，idle artifact不持page read权限。原native plan复用且Q/K/V、metadata、output
storage完全相同时可保留捕获；真replan使其失效，地址变化须显式重新准备，run不暗中录制。
零Q及未资格provider拒capture，原普通run/no-op保持；driver/private费用仍UNKNOWN。
稳定地址、固定batch size或MLX shapeless本身不能证明capture topology。不同容量/算法域
仍须冷创建新的受资格program/slot并计预算。

支持共享录制的原生实现现可在普通Graph的`Invocation::prepare_replay()`中与前后算子合并录制；
仍先调用`batch::prepare`并绑定当前输入。实际计划不变的metadata内容更新可复用录制，但每轮都重新取得当前权限。
计划、载体、工作区或绑定变化会在资源替换前使旧录制失效；准备失败不能继续提交旧请求。
`ReplayReport`区分录制与复用次数；只有一段录制也可能仍是hybrid，例如末尾还有未录制的采样操作。
