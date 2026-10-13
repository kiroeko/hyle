# G01 Vulkan 驻留端口的真实零拷贝消费

状态：SCOPED，2026-10-13；owner `compute_route_vulkan`。
baseline `702b6fed97ba1fe7dd2c4d80e992aeb7423f92d3`；原CLOSED F5 candidate与证据保留。
完整授权来自[十项goal](../project/inference-gaps-goal-execution-2026-10.md)。

真实触发：`vulkan_region.cc` 的 `lower_managed_region` 对每个plain输入无条件比较
canonical dense descriptor，差异即加入warm Copy并重写recipe输入。G01 Provider已经
设置native nb，公开数学/旧输出测试却仍可经这个全KV copy通过；“支持view”的数值PASS
不能签直接驻留消费。RR合格P1仍FAIL，decode比0.719–0.727，原FAIL不抹除。

主消费者是原BF16/F16/F32中的Vulkan已准入F32/F16 BSHD GQA8/2 D64/128/256
容量cache、有效长度、batch2、seq非零offset及重复decode；正交为padding与异步资源
owner重新绑定。库不拥有cache策略。原region、copy、queue和退休设施继续复用，不写
新shader/math/allocator/runtime。只修这个端口边界，不重审其他CLOSED算子。

审计区分原flash的vec4行/byte-origin资格与一般TensorDesc合法性：同一dtype且depth
连续、正row-major padding；三个已准入D；原F32/F16 vec4读取需要可表示的pitch/offset。
原shader的F32 vec4/F16 vec4读取及head/batch字节pitch按16-byte保守资格。公开原测试
的D64/128/256自然容量布局均满足；非vec4 padding保原显式Copy路线和费用，不能把
全KV Copy隐藏为零拷贝。原 `ggml_vk_tensor_subbuffer(...allow_misalign=false)` 要求
descriptor origin满足实际minStorageBufferOffsetAlignment；不允许隐藏的round-down。
动态来源的旧adapter正因此无条件先Copy，连dense Q也每步复制。现必须显式在提交
接受前验证该byte origin，不能仅删Copy并等原assert崩溃。

有限拟FIX：VP-01仅当端口所有native消费者都为已准入SDPA且pitch满足原vec4资格时，
保原strided descriptor/资源，不canonicalize；VP-02复用原descriptor alignment准入，
view origin/owner变化仍由原Region invalidation和真实退休处理；VP-03公开CompileReport
反证原native Copy数以及batch/readonly/slot/old-output/rebind数学。其他operation的
canonical adapter及未资格padding路线不扩张。实现前须AUDITED固定以上域与成本。

验证：先实跑原有容量view的非零native_dispatch_count；修后原72个数学episode与64步
rebind通过且合格域native_dispatch_count=0；非vec4 padding保数学及显式Copy费用。
适用source/static/shared搬迁、宿主ASan/UBSan/TSan和一次相同原P1合同，复用现有树，
Windows≤4 jobs且不跑本机重型GPU、Air实际native验证；各heavy步骤串行。公共工具不改。
调查/实现/验收各2小时检查点，用户常设自动延长，所有原FAIL/TIMEOUT/elapsed保留。
仅这一端口范围裁决且适用门全过才CLOSED，不代签全G01、多请求或整个goal。

SCOPED实跑反证：原公开CompileReport在batch1和batch2均报告3个native dispatch，即
两个cache Copy和一个query origin adapter；新增零Copy断言实际FAIL，原日志保留于
`.scratch/inference-gaps/g01/resident-port-probe-run-1.log`，不是源码推导性能PASS。
精确origin准入需要原Provider的私有Region绑定检查配合；该owner仍为CANDIDATE，
在实际修改前按同一有限合同退回AUDITED，runtime的原native shader/Work candidate不改。
