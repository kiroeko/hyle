# G01 CPU 多请求 ragged 原生读取

状态：AUDITED，2026-10-13；owner `compute_provider_ggml`。
baseline `af9dc884448249ef0ad87d28525e5303ec087401`，单请求产品candidate d91a与全部
三交付/san/独立性能证据保留，CPU P1正从独立clean副本取样，不受本轮产品施工影响。
授权来自[十项goal](../project/inference-gaps-goal-execution-2026-10.md)。

真实缺口：普通CPU provider没有kBatchPlanningExtensionId；现有BatchProgram只能选FI，
CPU多请求只能走按Q/H重复物化的reference。BatchProviderVTable已经有qualify/create/
update/plan/run及完整lease/retirement组合，原CPU internal::compile/PreparedVTable已经
有成熟SDPA/独立slot，不能另造queue、数学或controller。

主consumer：两request Q1/17、KV31/47、GQA8/2 D256，F16/BF16 packed NHD ragged，
预填充/decode、重排与独立slots；留出D64 noncausal与request加入/结束、empty request。
CPU数学→原batch attention→CPU数学沿原Graph batch消费者检验。Paged/native GPU与G02
精确页桥仍待后续，不能以本批替代整个G01。

有限拟FIX：CR-01 Provider内置batch extension与原Graph extension并列查询，依赖现有
compute_batch合同；CR-02 cold plan按owned layout构建每request BSHD strided Q/K/V与
原SDPA prepared/slot，输出直接写原packed caller slice，warm仅原mapping/enqueue，无
KV Copy、Q/H expansion、metadata扫描或plan；CR-03 exact layout/generation、完整
topology更新与原allocator/失败回滚，复用BatchInvocation旧结果和F1/F2实际退休。
每request一个原数学计划，复杂度O(requests) metadata、O(QHD)必要query/output、原flash
读取历史；native private pool UNKNOWN不写0。API保持原BatchProgram/PreparedBatch。

预声明：实际CPU public BatchProgram三形状/两请求oracle、readonly/tail、两slots/旧输出、
foreign schema/资源、empty/reorder、prepare OOM与失败后close；native/raw图源码确认不
含K/V cont/gather/重复materialization。复用CPU树、source/relocated static/shared、适用
ASan/UBSan/TSan与已冻结真实同合同性能，only selected factors/direct consumers；不全库。
CPU单请求路径相同输入投影保原证据，new batch入口必须独立实跑。调查/实现/验收各2h
检查点，按用户常设授权延长保日志，Windows≤4jobs、heavy串行，公共工具不改。
实现前AUDITED固定域/ABI/有限FIX；适用门全过才关闭本批，多请求其它路线仍保总卡。

## 审计冻结

CR-01内置kBatchPlanningExtensionId，原Graph扩展照常转发；中央图增加既有
compute_batch私有依赖（其仅依赖compute/ops/mem，不回指CPUprovider，没有环）。不改
公共BatchProgram签名或BatchProviderVTable，公共build核心不改，只登记真实组件边。

CR-02只准ragged、F16/BF16及原已资格F32、D64/128/256，Hq/Hkv整数GQA和非空KV。
原packed NHD为每request创建BSHD rank4 descriptor，subview只移byte origin，原CPU
compile/PreparedVTable保depth连续/正strides并直接flash读KV，输出写packed Q对应slice。
同一原Queue admitted callback顺序调用已构造slots，原同步runtime完成后才归还；没有
另建Completion、queue、数学、KV gather/cont/copy或warm metadata遍历。

CR-03cold plan从owned host layout固定query/KV ranges、数学图与私有slot，cold O(requests)
metadata、原causal O(Q×KV) mask及O(QHD) query/output允许；KV staging=0。拓扑key
覆盖全部query_indptr/kv_indptr和原immutable容量/scale/causal，相同key只换generation，
改变则cold重建，旧output/票据/真实uses沿原BatchInvocation拒覆盖。empty Q request跳过；
Q>0且KV=0冷拒绝（原flash无该数学档）。allocator失败清理partial prepared/slot，按options
float workspace上限检查已知冷payload/plan bytes，未知private线程池仍明示，不签capture。
