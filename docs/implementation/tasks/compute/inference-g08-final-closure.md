# G08 输出数值选择与可定位随机数有限终审

状态：**G08 ACCEPTED，2026-10-17**。G07–G10 goal完成**2/4**，原十项累计**8/10**。
本轮关闭mem、compute_ops、compute_provider_native、compute_provider_ggml、compute_provider_cuda、
compute_route_apple、compute_route_vulkan、graph_route_vulkan八owner的有限范围。
G09联合流水、G10 CPU预算尚未开始；完整Qwen文本CLI及与llama.cpp的token/s比较在本goal之后。

baseline为`85bcfa4ebc29cc4d2ed5fb122074204aca675d58`，产品candidate为
`5103c13d8ed119b06aa64e799f9768aea45ab913`。
[施工卡](inference-g08-output-selection.md)保留审计、固定S1–S4/M1与原失败费用；
最终43项产品/依赖输入hash仍匹配候选。候选后只有benchmark、消费验证、证据、文档与状态变化。

## 有限结果与消费合同

| 能力族 | 最终结果与边界 |
|---|---|
| S1原TopK(count=1) | CPU去掉V宽Entry scratch；CUDA/Apple/Vulkan以原生归约替代完整排序。保值的存储bits、首index tie、NaN最后与原合法axis/stride/dtype；其它count保原路线，不增加greedy facade |
| S2原子随机字 | `sampling_ops::random_bits(W)`，dense U32[R,4]显式key/draw→U32[R,W]。固定Philox4x32-10官方known-answer、进位、高位与前缀；四路线逐bit一致，无隐式counter推进或全局generator |
| S3复合数值选择 | `categorical()`/`categorical(true)`消费F16/BF16/F32 dense log-weights与显式counter，返回I32 ID和可选F32所选log probability。NaN/+Inf/全-Inf行checked失败、不发布结果；零R合法。CPU CDF与GPU居中Gumbel各自可重放，ID不承诺跨CPU/GPU逐bit相同 |
| S4组合与持续执行 | 原Direct/Graph/native完成与拥有结果；Qwen实际V248320、65步、两slots、请求增减/重排/旧输出、失败恢复、独立端口和三个CPU+GPU joint消费者。只回读选中小结果，没有全词表D2H或采样政策下沉 |
| M1下层缺陷 | 修正Mem超自然对齐块的inline/out-of-line释放，沿原mimalloc aligned owner；自然对齐路径不变。实际原244项与三公开回归、SDK消费和风险验证通过；原坏baseline会崩溃，只签P0，不造可比速度 |

[原固定审计](../../../../evidence/architecture/inference-g08-output-selection-audit/audit.json)绑定
SGLang7224fce1、vLLM98f72111、llama.cpp cbe28bb3与Random123 v1.14.0/726a093。
BSD-3-Clause LICENSE/NOTICE随公开头依赖交付。原Context/Queue/Invocation/Completion/workspace/
native stream/worker/fence继续持有唯一生命周期；不另建线程池、结果owner或地址cache。
temperature、top-p、penalty、grammar、EOS和请求seed政策由应用组合。

## 七门与七问

| Gate | Verdict | 实际依据 |
|---|---|---|
| scope | PASS | 原S1–S4/M1有限集合、主/正交消费者与固定FIX，无新模型域或全库扫描 |
| architecture | PASS | 原typed custom schema和原资源/完成承载；[源码复核](../../../../evidence/architecture/inference-g08-output-selection-audit/red-team-source-checkpoint.json)，未扩公共统计/构建工具 |
| correctness | PASS | [普通实现检查点](../../../../evidence/correctness/inference-g08-implementation-checkpoint/receipt.json)、[Air数值](../../../../evidence/correctness/inference-g08-air-implementation-checkpoint/receipt.json)、[设备/host](../../../../evidence/correctness/inference-g08-device-host-checkpoint/receipt.json)、[Air host最终](../../../../evidence/correctness/inference-g08-air-host-facets/receipt.json)的适用事实合并，历史IN_PROGRESS不冒充单份总PASS |
| delivery | PASS | [最终18配置](../../../../evidence/delivery/inference-g08-sdk-final/receipt.json)：Windows CPU/CUDA/Vulkan、WSL CUDA、Air Apple/Vulkan各source/static/shared，独立头及Direct/Graph/joint/Mem真实消费者、迁址与license |
| usability_docs | PASS | [Compute](../../../user/compute.md)、[Apple](../../../user/apple-compute.md)、[Vulkan](../../../user/vulkan-compute.md)公开入口/错误/费用/寿命与选型限制；本轮docs/closure检查 |
| red_team | PASS | 分离源码终审、[独立消费复核](../../../../evidence/architecture/inference-g08-output-selection-audit/independent-consumer-review.json)无MUST、[实际artifact补证](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-artifact-postreview.json)与最终四路线原始结果复核；晚补证流程偏差如下保留 |
| engineering_performance | PASS | 原时序/P0账与四路线九项合格P1均过原5%门；S2/S3不造无同合同counter peer的速度结论 |

1. **第三方**：固定上述三引擎原归档与native依赖，借Random123计数方程/known-answer并保许可；无依赖升级或新运行时。
2. **完整性**：原TopK两输出、随机字、可选selected scalar和全部持续/错误旅程分别验收；主Qwen形状与正交分布消费者均有证据，未代签整模型或所有native数学域。
3. **复用**：复用Mem/原executor/Queue/native状态检查与真实退休；没有全局RNG、第二套完成协议或引擎sampler facade。
4. **性能**：固定原HYLE85bc同存储/完成/两输出/步内释放，完整warm调用比较；四路线严格方向与工程门分开保留，未知private上界不由时间推断。
5. **FFI**：原C++23 typed/native桥承载两个数值factory；不增加C ABI、跨语言runtime或总type-erasure入口。
6. **交付**：18真实配置及公开头/运行时依赖身份通过；外置LibTorch缓存依赖与Windows tracing限制明确，不宣称hermetic全SDK。
7. **易用**：原prepare/Graph接普通Operation，caller明确持有key/draw和结果；文档说明零shape、错误、可重放及CPU/GPU ID差异，采样政策保持在应用。

## 性能、风险与终审限制

[完整性能分账及全部CI](../../evidence/performance/inference-output-selection-g08-2026-10.md)绑定
CPU benchmark5fbc、其余47b与原85bc完整provider/lowerer。每路线八strata各n15、120产品进程及30控制，
实际workload A/A、五controls、qualification/stock verify/原5%非劣门全部PASS；九形状严格方向均WIN。
Apple K3约1.043倍、原exact verdict为PARITY，未把不同口径互换或将小差异归因为K1归约。
相对旧HYLE的R1/R16吞吐比：CPU约1.22–1.27，CUDA2.36–3.40，Apple K1为2.51–9.58，
Vulkan4.40–41.92；只在固定两个R、V248320和对应profile成立，没有整模型/llama.cpp速度结论。

原源码完成/资源账在benchmark构造前固定，真实workload A/A在正式采样前通过；独立review、
实际text大小及反汇编记录在正式采样后补齐。相同已执行binary的只读补证确认调用未消除、双方
优化/ISA设置一致、纯布局控制没有material代码膨胀：CPU execute移动16B，CUDA计时callback
移动16B但GenericArm地址未动，Apple/Vulkan execute与callback移动20B；text只增加32/0/20/20B。
**保留未在preflight附齐记录的流程偏差**，作为工程终审证据修正接受，不回填prospective独立预签，
也不声称符合完整Formal release流程。未收集hardware counters或证明GPU代码地址敏感性。

四路线正式阶段合计2411.786秒、连已记录诊断合计2870.065秒；原controller墙时2455.485秒单列。
原编译、预检guard和n1不足样本的FAIL保留，未因换目录重置。CUDA172MB原含binary ZIP在receipt
指定本地内容寻址缓存，live qualified binaries仍保留，提交raw含全部样本/身份；其它三路线归档含
binary。证据为engineering，CUDA/WSL与Air未绑定Formal稳定machine身份，不冒充native Linux。

适用CUDA memcheck/racecheck、Apple Metal、Windows Vulkan真实validation及四host scoped
ASan/UBSan/TSan均有实际PASS；Air Vulkan没有Khronos layer，不把未运行的该层写PASS。
宿主san不证明device/driver/private pools，LSan/TSan模型和private/native cache上界仍按原凭据限制。
Windows只有实际imports/staged DLL身份核对，没有把它写成运行时module tracing。
新数值操作O(RV)扫描、可选normalizer和必要native临时/allocator费用明确，未承诺零分配、零等待、
硬内存上界或新增capture资格。末次retained native结果/runtime/cache析构双方DEFERRED，非完整teardown计时。

[统一LOC](../../../../evidence/architecture/inference-g08-output-selection-audit/loc-terminal.json)固定
不可变`fb725567da912d82575a9787de3b2ed4c2e0719f`，self-check PASS：全库3825680行，自实现不含测试182302行，
八owner唯一实现联合42679行/208文件；产品字节与5103相同。
这是该snapshot的实现总量，不是新增LOC或引擎剩余代码估计；原8349757快照仍保留作历史。

到此停止G08日常施工及重复P1/SDK/风险。仅可复现的S1–S4/M1正确性/寿命/交付缺陷、真实新消费需求、
公开合同/算法/依赖变化或合格同合同material回归，才以新卡合法reopen。G09/G10继续本goal，
完整Qwen3.5-2B full CUDA文本CLI、MTP/多模态关闭与prefill/decode token/s对比另行交接。
