# Foundation A 候选 P0 结构与 LOC

product candidate546b0d2468e5de29031c34668f4afa4bc31b9a17；候选登记9d0aa797a为产品等价
不可变快照，中央所有权选七CANDIDATE owner。统一LOC2.08/self-check PASS：全库代码
2,471,964行/6,021files，自实现不含tests170,581行/695files，七owner去重27,286行/83files。
原始JSON给出选择、排除、身份与自检，不累加历史各module统计。

冷 exact snapshot 借原shape program/guards/operation inference/generation；V点准备
O(V(N+E+D)+V²D)，其中查重显式pairwise。选择只在create_invocation之前，O(D+VD)，
不在submit查表/JIT/扩cache或猜padding；slot仍为原固定artifact与native owners。
native分析O(N+E+C)，control/effect端点只冷分片；原commands/MLX/Queue/Sched负责执行/退休。
普通CPU+CUDA/general compiler不改执行算法或原全域证明。

全图footprint冷校验复用原used mask/pairwise alias和SmallVector0；bind对named effects做
O(R)范围检查，纯图submit无新增扫描。没有新的global lock/worker/reaper，也没给每Tensor
添版本字段。固定footprint由caller保活/跨invocation同步，accepted prefixerror没有rollback。

control前驱额外物化、managed completionwait和边界copy是实际成本，沿现fragment/replay/
storage facts呈现；MLX/driver私有bytes保留UNKNOWN，严格预算拒绝未知。不能把不同native
吞吐/capture/Queue当作对称承诺；opaque native input未能证明named effect alias时明确拒绝。
family-wide compiler配额在新exact-only表中不能逐点重置，使用共享ResourceBudget或明确拒绝。

生成代码/调用边：改动都在冷shape/control/compile与bind规则，未修改SIMD或数学hotkernel、
native dispatch/完成核心；warm执行仍沿原artifact回调。普通SDK header TU和dumpbin/otool/
dyld/ldd核对对应C++符号/真实runtime，适用证据见各交付record。没有要求另一轮纳秒codegen
点值或P1来决定公共布局；P0结构成本通过不宣称parity/SOTA，P1/P2本卡未准入。
本目录只签P0/LOC，不代签尚缺的平台/SDK/correctness格或最终CLOSED。
