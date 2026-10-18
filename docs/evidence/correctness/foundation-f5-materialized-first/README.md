# F5 显式materialize共同层与CPU首批

AUDITED有限实现checkpoint，不是candidate/全F5。common固定typed recipe与owned layout版本
生成I32 indices/F32 additive mask；实际Win MSVC19.50/WERROR共同23项与六头独立TU通过。
容量/整数flatten overflow、wrong schema/per-request上限/empty拒绝、causal/noncausal、乱页/
非整尾/GQA、owned旧版本/只读F1导出、OOM/native-use晚退及metadata allocator归零均覆盖。

CPU实际Windows和WSL GCC14.2自然Compute/Graph source消费者各8域、16次独立double oracle：
paged/ragged ×F16/BF16 ×causal/noncausal，Direct原operation冷prepare全部steps后才run，
Graph从同recipe用原编译器/native region消费，没有复制数学循环。未使用物理页和非整尾
padding填NaN，active输出保持正确；masked容量索引重复真正有效KV行，不读取arena padding。
inactive Q容量的输出不定义，不纳入数学验收。F32 score/softmax/value累积后转回原dtype。

每query/GQA head显式gather，固定KVbound24。样例host metadata1920bytes、K/V gather98304，
所有逻辑intermediate保守上界paged553984/ragged529408；含alias view重复计量、F32转换、
permute Copy及独立输出Copy。不是native resident/scratch精确值，Provider/driver仍UNKNOWN。
需要caller显式选择和固定byte caps，没有自动fallback或warm JIT/prepare。实际每步host wait
由参考消费方显式执行，不将本Direct组合原型称高性能/capture或完整批次控制器。

WSL同identity旧source-build缓存、Windows旧自然source-parent/build缓存复用。Windows原
graph-windows gate扩新CPU消费者目标，继承该根原失败/累计时间；不是另开root重置预算。
common原失败及所有resume、Windows首Graph编译FAIL亦保留；原first证据未改写。所有精确
命令、输入hash、原日志hash、累计budget在receipt/三个raw archive。库按内存j4/j2/j1，
机器重型串行，无旧算子或全F4测试回放。

共同recipe/metadata/Graph构造和CPU实际承载已证明。路线Direct组合控制器、Apple/Vulkan
实际native Direct/Graph、同版本F2上传、append/read/F4状态分支/旧版本、其他故障/SDK/san/
P0/guide/终审未完成；不能以CPU通过代签其他route或整F5。
