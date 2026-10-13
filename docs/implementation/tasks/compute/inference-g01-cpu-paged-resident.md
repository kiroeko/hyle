# G01 CPU准确页直接驻留读取

状态：CANDIDATE，2026-10-14；owners `compute_provider_ggml`、`compute_ggml_runtime`，baseline
`118a8947518f02e0c02aa51918f9d7d15c421ed6`。原CPU resident/ragged973与Vulkanbc0有限
CLOSED均保留；本轮不是重新研究全库，也不把CPU页资格代签Apple/Vulkan/联合G02。

真实trigger：原BatchProgram/PreparedBatch已经能接准确PagedArenaRead，但ggml CPU
native batch只接受ragged；F5参考档按Q/head gather历史，不满足原G01的结构目标。
主consumer是两请求GQA8/2 D256、F16/BF16、page_size16、跨页/尾长/任意页重排；
holdout D64/128、noncausal、共享物理页、重复页参考/旧reader、empty与两个slot。
目标沿原BatchProgram→原Queue→原ggml FLASH_ATTN_EXT直接读取已就绪物理页；
CPU真实数学，无每步整历史copy/gather，无Q/head复制KV，不新增Mem页管理器。

固定来源：vLLM98f72111 CPUAttentionInput/block_table在原kernel内选physical block并
保ISA各自cache layout；llama.cpp cbe28bb3 get_k/get_v只给有效n_kv/strides，不伪造任意
physical pages可用一个stride；SGLang7224fce1原FlashInfer消费owned页表和tail。当前HYLE
CPU-only锁ffa4e8b8，mixed core锁456172+CPU IQ1组合，不能把参考快照当成所链源码。
本机vLLM原CPU kernel依赖Torch/自己的调度、ISA特定packed layout，不能直接把HYLE
NHD页当其承载，另建一套调度或完整repack不作为本有限FIX。配置身份再核：CPU-only
实际用ffa4e8b8，混合GPU闭包用456172+CPU IQ1；两原锁都须认证同一地址适配，不能
只改456便签CPU-only能力。

拟有限FIX CP-01：原CPU batch qualify/plan/run扩同F16/BF16 D64/128/256的paged域；
冷Owned layout固定每请求页表/tail/validKV，三输入仍原Q与完整K/V地址容量，输出仍
实际packed Q；所有读取权由原PreparedBatch的exact page guards承担，不公开raw arena。
CP-02：借原ggml CPU flash数学，私有地址读取适配只把逻辑K/V行映射至physical page。
不复制vec-dot/softmax/online积累/reduction公式；原single/ragged无页表的路径与SIMD
档保留，不把通用Provider所有normal对象加分页metadata。页索引/长度/overflow/owner
在冷阶段验证，热路不扫描host metadata，不重建图或另开Queue/线程池。
CP-03：P0直接原K/V roots及读范围/成本、原预算/partial OOM/accepted-prefix/旧输出，
真实source/relocated static/shared、宿主风险与有限同合同成熟peer性能。G03 producer、
G04编码、G05附加语义、G06增长规划和G10总pool预算不在本卡顺手加入。

API继续原BatchCapacity/BatchProgram/BatchLayoutVersion/PreparedBatch/PagedArenaRead，
没有新公开policy对象。确需source patch只放已登记runtime patch资产并认证原pin/hash/
license/三交付；公共checker/构建系统/统计核心不在本轮改动。若原manifest组合工具
无法表达本改动，先独立记录工具条件，不能借本模块无限扩建工具。

预声明：两request独立double oracle、三D/两dtype、至少64步tail/跨页/加入/重排/共享页，
未引用页可写、foreign resource/schema/generation拒绝、旧result/两slots/关闭、受控quota/
逐分配点失败与真实已接纳错误；确认无KV copy/扩大读取域。Windows/WSL定向CPU，
MacBook Air只按实际CPU/工具链身份签CPU portable域，不能借其GPU签CPU。三san适用
宿主风险与原公开consumer source/static/shared/逐头（仅API变更才重复对应头），普通
selector只真实owner/直接consumer。P1在不可变candidate后用原公开消费边界与成熟
原数学、固定正常/留出形状和阈值；未跑不得写PASS或把原ragged P1继承为paged。
预算：调查/实现/验证各2h检查点，按常设授权延长保旧FAIL/累计，不因短超时block。
Windows≤4jobs、重型串行，原CPU/Vulkan/Apple closed域不重复性能；本卡只先SCOPED，
AUDITED还需冻结私有地址参数、完整原核读点与metadata/生命周期后才可改产品。

## 私有承载冻结

CP-02只扩原ggml CPU flash的地址读取：私有src[5]为冷owned I32页表，原op_params
有效长度/新增private page_size参数给逻辑行数；四个K/V地址点（one_chunk与原tiled）
映射physical page，原dot/online softmax/SIMD/reduction公式不改。循环与split/reduction
统一使用有效逻辑长度，原CPU workspace预估始终取prefill/decode最大，不按physical
rows缩水，重复页造成logical>physical也须安全。原physical K/V tensor与其真实span/
容量保留，不伪造超出ggml view的地址描述；mask只按logical长度冷构造，数学沿原mask。
普通无页表原路径保持原语义与算法；vendor patch只进原已登记runtime资产并更新锁
身份，原组合工具不改。原单请求/ragged回归与原片段原生ISA仍需要影响验证。

CPU原private Recipe增加具名内联PagedRead描述，限Batch cold owner生成，三输入为
BSHD Q及flatten physical NHD容量K/V，输出仍Q shape。描述只保存页表borrowed span/
logical_length/page_size/scale/causal，既有Slot.layout在cold plan前保owned version，
clear先销毁Requests/Program再drop layout。paged完整topology复用必须比较Q/KV indptr、
page IDs、tail；复用时保旧owned topology供原native graph借用，report显式写当前新
generation，outer PreparedBatch仍核当前layout frame/资源/schema身份。不会把span借
到caller栈，也不让普通ragged Request/Program增加页字段；页table通过既有owned
layout保活，额外ggml tensor只有显式paged节点，private费用如实记录。

本卡API继续原公共Batch入口；CP-01/02/03固定。supports/infer拒未资格dtype/D、空Q
以外无KV、非finite scale、overflow与无效schema/capacity。原Ragged math/Factory不换
Provider，不修改compute_batch/Graph公共形状。G02准确joint页权限仍独立，不代签。

实现草稿已提交：原private PagedRead与原Batch扩域，两个原CPU core的11处小上下文
页地址/有效长度适配和NOTICE/锁身份均认证；原CPU-only ffa与mixed456组合工具运行
成功，19项IQ1组合预算不改。普通Program/Request未加页字段，冷layout owned借用与
完整topology比较；新normal页条件的实际性能不能由源码冒称零成本，需候选门裁决。
Windows复用CPU树四jobs定向compile/source数学，非零Q/K的独立double oracle三D/
两dtype/causal及noncausal、任意顺序/共享页/tail实际PASS；原ragged三数学/风险与
预算回归PASS。旧测试仍要求paged不支持的FAIL保留并迁移；new hook缺GTest配置
FAIL与首次新64步fixture未释放PreparedBatch ticket的FAIL也保留，不改原生命周期。
64步/后续交付/风险/性能尚未裁决，当前仍AUDITED，不签CANDIDATE或整个G01。

不可变candidate `3cf78c88b`（完整身份见registry）。Windows最终五tests PASS：原四
ragged plus新paged consumer，三D/两dtype/causal/noncausal非零Q/K independent double，
64步同topology新generation复用且位相payload与输出一致；另64步真实增长/跨页/请求
结束重入/重排/尾长独立oracle全部PASS。计划私有资格实际核原KV roots、src[5]页表
origin/count与有效长度/page_size，不由API名或正确结果猜没有KV物化。未引用页可写、
共享页仍不可写、旧output/ticket/native work阻止复用；原正交budget/OOM/accepted
error退休回归保。最终源run40.843s，编译增量4.359s，<=4jobs，无GPU运行。
所有此前配置/旧unsupported断言/ticket生命周期/fixture shadow警告FAIL保留，修测试
不改原容差/数值/退休日志。CP-01/02实现和scope内普通consumer固定，进入CANDIDATE；
CP-03的三交付/适用风险/定向性能继续，尚未CLOSED，原全G01/G02其它路线不代签。

WSL GCC14 ASan+LSan实际14tests PASS（编译64.638s、运行24.347s），包括原
resident/ragged负空间；实际provider/native_batch与原vendor ops均有address flags。
该次编译出现冷计划keys uint64与queries int32比较警告，CP-01仅补已验证非负query
长度的显式uint64转换；provider退回AUDITED，runtime candidate不变。原3cf源码/
ASan记录保留，后续增量与最终candidate身份单独记录，不把警告或未跑门省略。

最终provider candidate `1e9cdfd45307ea4a15b92374377ad2b268203193`，runtime3cf产品未变。显式冷
转换后Windows同五tests再次PASS（增量2.750s、run40.860s），不改变语义/数学/
费用；后续风险与交付以该最终产品身份记录。原3cf门与警告原日志保留。

2026-10-14扩已有风险fixture到真实paged域：两request反向页表逐分配点失败
均无live allocation，极小cold预算拒绝；已接纳数学错误不发布output，释放实际
页reader后两物理页均可重新acquire，保原Queue错误/drain语义。Windows定向四风险
tests PASS（编译4.157s、run0.328s）；产品candidate1e9未变。初fixture把8192 payload
误作含sentinel总额度导致Bad expected access，Windows与ASan原FAIL保留，修为
8193并显式检查Result，不改数学/错误退休期待；最终风险/交付继续。

ASan增量第4次run仍FAIL的原因已定位为验证副本mtime：canonical修fixture发生
在前一个远程编译期间，copy2保旧时间导致Ninja未重编该test（build4明确no work）。
第4次记录保留为STALE_BUILD，不能代表已修输入；私有同步改为只有changed bytes
用验证机当前mtime，不碰未变文件/公共工具。补Result检查触发实际重编后，ASan
定向accepted分页错误退休PASS（20.261s编译、0.418s运行）；全14tests复验进行。

两槽有限补验：主slot保重排/短tail结果与其native页表，另slot同时用原topology
运行并逐bit核两份各自结果，之后独立close；Windows最终同五tests PASS（5.875s
增量、40.421s run）。两原core数学已实际运行：CPU-only ffa在WSL三san各14PASS，
MacBook Air M4/arm64/AppleClang21/macOS27.0.1在CPU+Apple shared源树实际mixed456
原CPU数学14PASS（5.299s配置、13.088s构建、25.505s run），不跑GPU/不代签G02。
新两槽test的三san/SDK/Air最终串行补验继续，四jobs；产品candidate1e9不变。

最终两槽扩展的Linux三san各14tests PASS；Windows relocated static SDK的同一
非零double分页/64复用/64演化/双topology槽与原Direct/Graph ragged/resident三consumer
PASS。首shared SDK linker FAIL为私有runner只build底层hyle_compute/hyle_graph因子，
没有build公开hyle::compute/graph facade全部CPU roots，install沿旧tree残留static库；
SDK1保失败prefix/导出文件/链接raw，不归为新产品缺陷。私有producer改为只build链接
公开facade的普通consumer，补真实CPU闭包；复用同tree，SDK2新prefix防覆盖FAIL。
没有public构建系统/registry修改，没有CUDA/Apple全局构建。原52.407s与后续累计保留。

最终CP-03 correctness/risk/delivery：最终两槽test三san均各14PASS，Windows
relocated static/shared均三公开consumer PASS。shared2真实facade闭包构建134.344s，
pagedrun40.593s；Mac最终mixed456 CPU14tests25.157s总（1.145s增量/23.320s run），
实际ops SHA d7a2b6d1f4dfe9f57cce2f7e41ed2e46247168039715f2081a67254fb4a7d7c4，
含页地址helper及marker，源/编译/两原锁身份相互核验。见
[风险](../../../../evidence/correctness/inference-g01-cpu-paged-risk/README.md)与
[三交付及mixed core](../../../../evidence/correctness/inference-g01-cpu-paged-delivery/README.md)。
所有普通门终态、无活动重型job。CPU准确页仍CANDIDATE：定向P1未运行，不能
继承原ragged P1或签整个G01；原G01 GPU页/G02联合/G03等继续原十项goal。

P1消费冻结于2026-10-14：只改benchmark adapter/合同，产品1e9/runtime3cf不变。
原独立CPU数学用干净ffa4原AVX2 DLL，实际compute函数所属image核pristine ggml-cpu，
tensor336/cplan56/src offset/BF16 enum同锁ABI核验。metadata沿独立原ggml graph recipe，
不调用candidate compile/lower；pool创建/plan/compute/free全从原DLL，workspace原caller。
对照domain为同ready历史不可变快照的warm attention：保原确切页owners全程不变，
冷按valid logical rows建立单独contiguous cache，额外copy/存储D256 3670016bytes、
D64 39936bytes显式记账；其它backing/topology拒绝。两arm都原Batch/metadata update/
F1 exact pages/CPU Queue/Completion/输出lease返回、两slots/单native线程。此P1不能
代签producer/growth/reorder生产或G03/G06；结构目标无warm KV历史copy沿实际产品证据。
三point nonzero double及readonly预检PASS，max_abs <=6.11e-5，原8e-3不变；原DLL加载
缺dependency/非native Windows路径FAIL均保为private adapter预检，修loader/依赖而非数学。
原stats/controls/16AA/order/allocation/placement/n15/31/5%全部保，合同及实际DLL/source/
SDK/header identities在正式前固定。[消费合同](../../../../bench/inference_resident_sdpa/cpu_paged_p1/consumer-provenance.md)。
selector共享bench unknown_paths条件仍保原BLOCKED计划，不改工具、不跑全库。

2026-10-14原P1已terminal1695.110s，AA16/产品120/controls30、control与原verify0。
消费provenance JSON哈希仍旧，正式资格DIAGNOSTIC，不能签合格性能；原summary显示
D256两点非劣PASS但相对1 LOSS约1.3-2.1%，D64约0.8094-0.8170非劣FAIL，只作为
诊断如实保留。[原证据](../../../../evidence/performance/inference-g01-cpu-paged-w1-p1/README.md)
封存未改原manifest。合同所有三workload与实际JSON/markdown byte hashes、真实immutable
alternate cache ownership描述已校正，直接调用原product-evidence evaluator三点PASS；
所有统计/样本/controls/16rounds/5%不变，未借此重标旧数据QUALIFIED。

runtime CP-A01退AUDITED，provider1e9保持CANDIDATE：P0源码证明当前四K/V地址点在
显式page_size16每次做动态64-bit除/余；真实CPU主域page16可用完全等价shift4/mask15。
有限FIX只原私有地址helper，normal非paged早return与generic正页原映射保；不新增
字段/metadata或改dot/softmax/SIMD/reduction，不改公共API/layout/数学或由诊断点选路。
这是原CP-02地址承载成本审计修正，旧诊断不作为公共合同变动授权。两原CPU锁同改
并认证原manifest/许可，复用source/两个core数学、三san/SDK直接消费者；不可变后
用已纠正完整合同重新P1，保1695s与旧FAIL，不以file/log目录重置预算。无需全库验证。
