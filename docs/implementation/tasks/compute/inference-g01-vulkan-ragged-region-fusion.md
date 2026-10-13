# G01 Vulkan ragged 原region分段纠正

状态：两owner CANDIDATE，2026-10-13；owner `compute_route_vulkan`、
`compute_provider_ggml_vulkan`。不可变candidate `0e9e011b8c8ff4a37e2bebc5ca8e4408be5e9368`。
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
