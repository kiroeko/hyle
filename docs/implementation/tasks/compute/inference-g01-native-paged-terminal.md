# G01 Apple/Vulkan 原生 page16 有限轮终审

状态：CLOSED，2026-10-14；五 owner 为 compute_batch、compute_route_apple、
compute_route_vulkan、compute_provider_ggml_vulkan、compute_ggml_runtime。不可变产品
candidate `520027a57ff41da96ac908ad0d35896e2dda9734`；baseline、NP01–05、授权与原失败
见[施工卡](inference-g01-native-paged-read.md)。只签本轮原生 page16 有限域，整个 G01、
联合 G02 与 G03–G10 未完成。

## 七问与七门

1. 原锁 MLX1f8e74 与 ggml456172、许可及 patch 身份不变；SELF 只有 owned topology/table、
   薄 typed 接入与原数学读点地址映射。MLX vector/2pass/GQA/steel/NAX/dsplit 及 Vulkan
   原 FullF32 scalar 读点审计见[原读点证据](../../../../evidence/architecture/inference-native-paged-read-audit/README.md)。
   不新增 attention 数学、executor、queue 或 CPU fallback。
2. Apple F16/BF16、Vulkan F16、GQA8/2、D64/128/256、page16、causal/noncausal；实际 Air
   [Apple 短/长域](../../../../evidence/correctness/inference-g01-native-page-apple-stage/README.md)
   与[Vulkan](../../../../evidence/correctness/inference-g01-native-page-vulkan-stage/README.md)
   independent double、乱序/共享/尾页/空 request、logical KV 大于真实 physical capacity、
   双 slots/旧输出、64 次演化通过。Apple 长域另覆盖 logical4111/8207 与 H16/2。
   M4 只签实际选中的 kernel 路径，不外推其它硬件；Vulkan BF16 全链仍未签。
3. 借原 MaterializedNativeState/PagedMetadataPolicy、CopyBatch、两 mirror banks、
   exact frames、LeaseUsePool 与原 native retirement；只一张 I32 页表，没有按 Q/head
   gather 全 KV。普通 nonpaged 数学和对象 layout 不增加页字段。owned version/read lease
   不自成保活环；Context/device/staging/injected allocator 仍须稳定且活过 work/results。
4. 冷 table 创建/上传、原 native replacement、metadata wrapper、query/output 和真实 wait
   都计费。layout 变化目前 cold prepare，private native 总上界 UNKNOWN；不签 G06 delta
   planning、capture、producer/growth 性能。两个同生命周期 P1 结果见下文。
5. 公共 NativePagedBatchProgram/Version 与两 route PagedBatchInvocation；私有原生地址
   参数和 projection 不成为普通 consumer 的 raw metadata API。无兼容转发、另一控制器
   或 FFI。普通 reference 入口与原 Graph/资源消费者负空间通过。
6. [四份 relocated static/shared SDK](../../../../evidence/correctness/inference-g01-native-page-sdk-stage/README.md)
   只链接 hyle::compute，实际数学/零域/冷 OOM 和十二 standalone headers 通过；source
   与[Windows MSVC 定向链接/三头](../../../../evidence/correctness/inference-g01-native-page-windows-compile/README.md)
   通过。Windows 没有 executable/GPU run，不代签实际 Windows 数学。
7. [使用页](../../../user/batch-attention.md)与两个公共消费者说明 prepare/layout、确切
   frames、submit/wait/owning output/close/drain、资源借用、费用及限制；用户不手写页表、
   扩大整 arena 权限或重复提交循环。

[实际风险](../../../../evidence/correctness/inference-g01-native-page-risk-stage/README.md)
覆盖 metadata/native 拒绝及 accepted prefix 错误、unknown poll/实际 wait、cancel/abandon、
Apple52/Vulkan55 冷逐分配 OOM、foreign resource/generation、精确 backing budget、全空域。
失败无输出、槽不复用，所有引用页只在 actual retirement 后释放，未引用页独立可写。
[六组宿主 sanitizer](../../../../evidence/correctness/inference-g01-native-page-host-sanitizers/README.md)
最终通过、累计480.558s。Apple 原两次 TSan FAIL 保留；独立无 HYLE/MLX Metal probe 复现
上游 completed-handler 不透明交接，最终只在验证头副本标注同一实际 command buffer 的
commit/completion release/acquire，无 suppression、产品/cache 改写。仅签有此显式模型的
宿主结果；无 LSan、shader/driver/private-pool 插桩资格。

终审按 frozen product SHA、真实 root/逻辑有效域/table origin、所有 kernel 读点、旧结果/
失败退休、预算/UNKNOWN、SDK 公开 target 与 cold comparator 费用逐项核查，未发现本已验
域的未决问题。scope、architecture、correctness、delivery、usability_docs、red_team、
engineering_performance 七门 PASS。selector 中央文件保守全选与共享 bench 归属问题仍保
原计划及独立工具卡；不借本轮重写工具或重开全库。

## 同合同性能与停止

clean source `66af56852d0602ce14cdfde9aba3de373a7472c6`、product520；实际 Mac16,12/M4/
macOS27.0.1/AppleClang21，M1-MAC engineering、interactive QoS，独立 Formal 硬件未签。
[Apple P1](../../../../evidence/performance/inference-g01-native-page-apple-m1-p1/README.md)
938.810s，contract938628de；[Vulkan P1](../../../../evidence/performance/inference-g01-native-page-vulkan-m1-p1/README.md)
1493.406s，contract5bd7e044。各16 A/A、120产品/30 controls、n15；QUALIFIED、三个5%非劣
PASS、verify0。两活动共2432.216s，不重置预校验配置 FAIL 或以前的 source/risk/SDK FAIL。

| 同合同场景 | Apple candidate/peer | Vulkan candidate/peer |
|---|---|---|
| Q1/17 KV1024/768 D256 causal | 1.4978–1.5052，WIN | 6.4314–6.6996，WIN |
| Q64/128 同 KV/D 长 prefill | 1.1053–1.1097，WIN | 5.8188–5.9560，WIN |
| Q1/17 KV31/47 D64 noncausal | 1.0481–1.0555，实用 band INCONCLUSIVE、方向 WIN | 2.0305–2.0691，WIN |

peer 每 request gather 一次有效 K/V，再用原相同 fused MLX SDPA 或独立 pristine pin 的
原 FullF32 scalar shader；candidate 直接页读取。两边都计显式 prepare/native replacement、
table F2 上传、exact guards、两 contexts/slots、submit/wait/packed owning result release；
不将这个有限合同的比率解释为全模型 token/s 或不同完整引擎的优势。

停止重复本 native 有限域的性能/数学验证。reopen 只允许已验数学/地址/table provenance、
owned version、资源费用或失败退休的可复现缺陷，以及新真实消费机制。CPU+GPU 联合
只读 Q/frame 接入、G03 producer、G04/G05/G07 dtype/数学、G06规划、G09流水/G10预算等
仍须由各自活动卡实施验收，不由本 terminal 代签整个 goal。

LOC 采用原 loc-report/cloc2.08、candidate520 self-check：全库2898994、自实现不含测试
175806；compute_batch1011、Apple4355、Vulkan8944、ggml Vulkan provider2680。runtime402，同候选五份测量单独保留；vendor patches、测试和 benchmark 不算模块自实现 LOC。
