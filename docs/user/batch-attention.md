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
| Windows CUDA native batch | 当前未准入 | 必须给具名unsupported，不能把WSL结果或CPU参考说成Windows native paged |

Apple/Vulkan参考档真的在原GPU上执行，但它们把页/packed KV显式gather/materialize后做
数学；索引/掩码、GQA复制、payload/wait的费用必须计入。需要原native paged吞吐时，
不能只看接口形状相同而假定性能已经对称。没有submit隐式plan、JIT或自动CPU fallback。

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
