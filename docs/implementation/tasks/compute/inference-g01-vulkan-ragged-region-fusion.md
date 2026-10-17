# G01 Vulkan ragged 原region分段纠正

状态：route/Provider/runtime有限轮均CLOSED，2026-10-13；owner
`compute_route_vulkan`、`compute_provider_ggml_vulkan`、`compute_ggml_runtime`。已验不可变candidate
`0e9e011b8c8ff4a37e2bebc5ca8e4408be5e9368`的FAIL保留；最终不可变candidate
`bc0ba41f714f6529cb1c583bee59d482946e7589`含cold admission修正与RF-04记录资格。
当前权威record为[有限terminal](inference-g01-vulkan-ragged-terminal.md)，本卡保施工/失败
历史，不授权重新施工；整个G01未完成。
baseline `dbcba4b15`，原产品280，原Provider单请求3d92有限CLOSED保留。真实trigger是
已准入两request F16短D64 noncausal [合格P1](../../../../evidence/performance/inference-g01-native-ragged-vulkan-m1-p1/README.md)
0.8656–0.8885、非劣FAIL；[原NR轮](inference-g01-native-ragged-layout.md)不签完成。

短只读实际结构诊断发现candidate两原flash region＋另一个managed Concat，3个原生
region/提交段；独立原shader peer七dispatch合在一个原command提交。每request原输入
View和output Reshape打断冷managed region融合；它们本身无数学或设备同步需求。
不是增加样本或换shape/容差可以解决的资格问题，不改原锁/flash math/精度/shader。

主旅程为GQA8/2 Q1/17 KV31/47 D64，holdouts保长KV D256及长prefill、8requests、
empty/reorder/旧result与源readonly/tail。拟有限FIX RF-01：ragged冷recipe先生成全部
外部capacity views，再生成各request原SDPA与dense output reshape、最后一次Concat。
RF-02：Provider只接原dense plain同dtype/rank<=4的reshape View到原ggml reshape；
route只在已开始且明确原ggml managed region内跨这类零计算reshape，不把输入views
当新managed数学/读者，不改变公共默认Provider政策或其它View路线。
RF-03：确认两request只有一个region、KV roots/offset不变、无wholeKV copy，数学/寿命/
失败及三交付定向验收，原同合同性能验证修正后的不可变candidate。

公开API/layout/ownership/Queue/错误/actual retirement不变。原region只因冷recipe更
完整而合并；单request高效域/其它算法不重开。八request若超过原managed Concat arity
仍用原native suffix并保真实accepted-prefix错误反证，不为了拼接顺手改shader。
不新增运行时决策/指标/构建系统/公共工具；所有未签页/G03/G06/G10保原总goal。

预算为一次有限机制诊断和2h检查点，正常编译/验证按常设授权追加并保原FAIL/elapsed。
只用实际Air，Windows定向<=4jobs编译；重型串行。source原四tests＋结构单region反证、
原source/relocated static/shared、适用三hostsan真实TU、直接hybrid原消费者；API不改
不重复头集合，复用已有同产品public header证据；原同合同d9a54a85、5%/A/A/controls/
n15/31不变。原产品280失败永久保留，不复跑CPU/Apple P1或全库算子。

审计决定：RF-02仅原region内部dense F16/F32 reshape，输入已由该region生产；外部
capacity Slice/Reshape仍走原Plan alias，不把独立普通View或任意slice/stride提升为
Provider数学。Provider所有正常dtype/rank/extent与实际device checks保留；reshape借
ggml_reshape_4d和原restore_shape，无新copy/数学。原route已开始managed段后才跨此
reshape，控制边界仍阻断。recipe先外部views后math保证原SDPA直接外部KV分类不被
中间View假作other reader，避免破坏原3d92的zeroKV-copy合同。一次Concat仍在原arity
内融合，八request仍原native suffix；旧failed-prefix test继续适用。

七问：依赖仍原ggml/MLX/loader锁与许可；有限消费者与原长/短holdouts保；既有数学/
Queue/资源/退休复用；冷线性重排、warm减少提交段；无FFI/ABI新面；源码/static/shared
实际交付及宿主风险定向；公开guide不需要新API，补实际cost事实。真正零region-copy与
数学/生命周期、非劣必须实跑，审计不签PASS。

实现已提交0e9e011b8：全部外部Q/K/V slice/reshape先生成，SDPA之后的相邻dense
reshape仅留在已开始的原managed段。Provider同dtype/rank<=4/extent域借原shape
restoration，不添加math或shader；普通独立View起点仍原alias路径，不修改成熟默认
政策。实际Air四source tests两次PASS，新结构反证两request仅一个region且KV根/
byte offset仍指capacity inputs；原八requestnative suffix真实失败退休仍PASS。
candidate前固定此实现；交付/风险与原合同新P1待验，不由source PASS代签性能。
实际短结构诊断也确认candidate managed dispatch3→1、外部temporary blocks3→1，
原SDPA＋原reshape＋一次Concat，peer原七dispatch仍一个native command。这里是
结构事实，不是已资格的吞吐PASS；不修改原schema/math/精度或原短D64 workload。

候选实际门已通过：Air source四tests、实际relocated static/shared各两public tests，
原ASan/UBSan/TSan树各四tests且region.cc/vulkan_program.cc/vulkan_batch.cc及tests
compile flags核验；原CPU+Vulkan exact/control/state consumer增量build6.558s、
run3.188s PASS。Windows同parent/build只重编三TU/链接，四jobs5.125s，不跑本机GPU。
原同三shape/F16独立double/readonly/tail comparator oracle2.819s PASS。全部已验门
见[RF证据](../../../../evidence/correctness/inference-g01-vulkan-ragged-region-fusion/README.md)；
旧private编译getter猜错失败保留，只修诊断消费既有ProgramState，不改公共API。

同合同d9a54a85 formal fresh3/source0006944b7已启动，product0e9e011b8，原peer/shape/
oracle/5%/16paired rounds/min1/4ms/n15/31/A/A/controls全部不变。只因为产品结构已改
进行新候选验证；原fresh2 FAIL928.101s、全部之前成本保留，不因新tree重置预算。
新performance gate尚未终态，不提前签PASS；Apple/CPU原P1不重跑。

候选复检定位段尾规则漏了外部View排除：主分支虽拒独立View作为managed起点，段尾
普通matches_mature_default也可能允许它并入，与本卡只跨相邻内部reshape的负空间
不一致。route单独退AUDITED补同样的!View条件；Provider数学与原NR实际输入不改。
正式source0006944b7/产品0e9e011b8独立不被修改，正在运行的任务保原预算并取得实际
终态；不能把此观察当成停止/重新采样许可。最终cold admission范围由新candidate
明确绑定，并反证没有引入input KV Copy；性能复用需给实际相同NR lowering身份。
新增source-only冷负空间反证：独立第二input的dense reshape位于两Copy region之间，
必须仍是第二input根的原alias与两独立region，不被前一managed段吞成新增producer。
它只验证本RF-02边界，待同formal任务終态后串行执行，不扩全局算子/性能形状。
普通selector已选route及原Graph/native hybrid直接消费者，但净diff引入commit归因
失败，虽当前owner为AUDITED且父提交15dff已先登记；按独立工具记录（已删除的历史文件：`docs/implementation/tasks/build/validation-selector-shared-bench-paths.md`）
保存原BLOCKED plan，不修改公共checker。实际门按本卡精确执行；不会扩大成全库重开。

RF0e9 formal实际终态：929.176s、A/A16/n15产品120/controls30，QUALIFIED、verify0，
性能仍FAIL。D64比0.8018–0.8260稳定LOSS约17–20%，比原280更差；原[第二次合格失败](../../../../evidence/performance/inference-g01-native-ragged-vulkan-rf-m1-p1/README.md)
全部保存，3段→1不等于费用目标完成。cold边界最终source五tests已PASS，后续不借
这两次门重复采样。根因源码已定位：原native_work_impl::fixed_attention_graph拒第二个
tagged flash和GGML_OP_CONCAT，因此合并图反而每次managed编码，不能原固定记录重用。
这形成原runtime单请求a400有限CLOSED之外的真实multi consumer trigger；需runtime
依法SCOPED/AUDITED后修有限whitelist，原hyle_record_safe/实际GC无event/退休与bind
origin变更失效仍是权威，不推广一般capture或去掉实际安全资格。

RF-04 scope baseline6a6ae3cdc：runtime a400原单请求CLOSED保留，真实trigger为两flash/
Concat已准入消费者的固定记录缺口。唯一允许产品改动是private fixed_attention_graph
接受多个原FullF32 tagged flash和原GGML_OP_CONCAT；原node白名单其它项、最后批次
fence、record-safe/无event/semaphore和实际pending/retire/失效流程均不改。Provider
依文件归属退AUDITED，runtime先SCOPED再AUDITED；测试不授权直接改CLOSED产品。
新增原native 64步两flash＋reshape＋Concat、当前payload变更与step32明确invalidate
反证，先复现recording_ready false，再验原四native tests与新五route tests。复用原
source/static/shared与三hostsan，只补真实插桩TU/必要私有test；不改变数学/shader/
强peer/合同/阈值。public头未改，逐头证据按输入相同复用；Windows四jobs只定向编译。
正常验证按原2h检查点/常设追加，保两次原P1 FAIL928.101+929.176s，不重置成本。

RF-04审计与实际反例：原native test编译PASS，实际81ms即在首次wait之后确认
recording_ready=false；只使用现有NativeDevice/MetadataArena/NativeGraph/NativeWork，
没有新增诊断面。fixed graph完全owned、host输入固定resident范围，Concat仅原GPU
storage math；所有flash必须同FullF32 tag。接受多个tagged flash仍不能绕过实际无
memcpy/memset、无wait/signal semaphore、无GC event/timeline的record_safe门；bindings
changed仍invalidate，failed/native lost仍只在退休后cleanup。运行时不新增字段/资源/
线程/分派；只是冷assessment一次从单flash计数限制改成至少一flash资格，并允许
原Concat opcode。源only红反例保rr-red日志；runtime审计提交后才改产品。

RF-04实现只有fixed_attention_graph两行变更；Actual Air新增64步两flash/Concat原生
记录test、原四native风险tests与五route tests共十项PASS，冷外部View alias/单region/
KV roots与实际八requestsuffix错误退休均保。phase/source输入SHA核验，产品数学/原
shader/公共API不改；API/算法稳定后冻结bc0，三交付/三san/原消费者及同合同性能待验。

bc0候选实际source、搬迁static/shared两public tests各PASS；三san各十tests PASS，
包括实际ggml-vulkan.cpp编译插桩（其中包含变更的native_work_impl）、provider/route
TU和两组tests，不借旧san代签新Work资格。原CPU+Vulkan consumer增量build10.093s、
run3.805s PASS。短三shape oracle/readonly/tail+16轮诊断3.212s PASS；D64参考比
1.00x、CI[0.99,1.02]仅diagnostic，不写正式performance PASS。实际证据封存后，同
原contractd9a54a85/peer/5%/n15/31/controls再次验证这一已修normal-cost候选；不是
对相同candidate追加姿势。两次原合格FAIL永久保留，原CPU/Apple性能不重复。

最终[RR门证据](../../../../evidence/correctness/inference-g01-vulkan-ragged-recording/README.md)
已封存11组source/红绿/交付/风险/原消费者记录与三实际插桩清单，Windows定向编译
9.734s PASS。正式fresh4/clean source4f224878a、productbc0、原contractd9a54a85已
串行启动，结果必须按实际终态裁决。此前冷规则source8ec已验证，但0e9性能FAIL，
不需要再做其“PASS复用”论证；旧源码identity检查只保诊断，最终性能绑定bc0。

统一LOC绑定最终bc0、explicit三owner实现目录联合边界，self-check PASS：全库code
2,883,171，自实现不含测试175,077，route/Provider/runtime联合实现11,857（不把这个
联合数冒充某一个factor独立实现LOC）。只统计不可变candidate，不为状态/证据提交
重复全库统计；剩余适用性能门仍等待实际终态。

最终正式P1 actualTERMINAL：931.800s、n15产品120/controls30、A/A16，QUALIFIED、
5%非劣PASS、原verify0。[最终证据](../../../../evidence/performance/inference-g01-native-ragged-vulkan-rr-m1-p1/README.md)
保方向mixedD256 WIN、长prefill/D64 INCONCLUSIVE、原schema/margin和所有旧FAIL；三个
owner按上述terminal有限CLOSED，停止本域重复采样。总goal仍继续原分页与其余G项。
