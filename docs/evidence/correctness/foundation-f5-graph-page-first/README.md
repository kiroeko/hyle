# F5 原Graph逐页消费与append/F4首批

AUDITED有限checkpoint，未签candidate/全F5。实际WSL RTX4070/GCC14.2/NVCC13.3，
原FlashInfer native、Graph v19 shared region与原Completion/LeaseUseBatch执行。F16/BF16
各旧→新→旧三episode共六个独立uniform SDPA oracle（0.25/0.375/0.25）通过；原paged短
与ragged长域双节点/共享metadata/双slot的12oracle保持通过。普通CPU Copy/物理View alias/
直接publish whole capacity三种真实Graph分别被private projected binding拒绝。

Graph batch只冷预留页guard额外容量；原LeasedInvocation bindings尾部guard-only项不含
公开ValueId，原submit同一F1 use batch读取全部actual pages，原accepted prefix退休统一
保活。没有普通Invocation/v19常态字段、额外warm分支或第二executor。冷reset移除guard
项；普通调用者没有额外容量与guard项。private冷资格扫描所有实际physical readers，
只允许allow-list中同exact ProviderId的selected batch regions；copy/其他region/alias/
published容量拒绝。cold reserve overflow也实际拒绝。

高层batch协调器只允许same-schema、完全相同layout版本的K/V frame；query/metadata/
混合generation/同prepare内覆绑拒绝。原region slot保两immutable frames；shared_submit
检查原数学输入storage精确身份。prepare只在原idle/result/completion门通过后清除帧与
permissions，部分失败不能提交混合token。actual old result阻止prepare，close后释放
coordinator的escaped result仍保原slot/结果，最后metadata quota为0。

同一Graph原Queue的F2两phase显式clone→append依赖，只clone脏K/V页1024bytes并append
两行512bytes；whole K/V arena4096bytes未复制。保旧frame/F4版本时free物理page2可写，
used page1不可写。成功实际ready后完整F4目录dirty替换与expected-base发布，clean页
原block共享，新layout IDs0/2；原native update同indptr reuse scheduler，无warm规划。

Windows原source/static树实际产品链接与运行通过；modified Graph owner TU沿实际宏/
编译选项加/WX通过，新graph_batch原/WX通过。batch.h与detail/region_access.h各独立TU
在实际CPU+CUDA presence profile、/W4/WX及原wd4324合同通过。不是installed SDK/
迁址/loader或Windows native paged数学资格。共同32项/八compute头证据仍复用
[上轮逐页收据](../foundation-f5-paged-arena-first/README.md)：本轮未修改其共同实现。

新negative fixture错误使用View构造器的GCC编译FAIL保留；改用原View::slice冷工厂后
通过，无产品API放宽。原所有失败/input history/准确命令/hash/累计预算见raw archives。
同root/tree/cache复用，重型跨机器串行，j4/j2/j1按available memory、test1；没有全库
或旧算子回放、公共工具改写或预算重置。

CPU/Apple/Vulkan reference逐页桥、完整组合控制器/ragged append、其他producer/error/
unknown/alias/OOM/cancel风险、三交付/san/P0/guide/终审仍待完成，不由此签整个F5。

冷alias拒绝的O(V)修正及其相同预算复核见[追加收据](../foundation-f5-graph-page-recheck/README.md)；
本首批receipt与raw archive保持原样。
