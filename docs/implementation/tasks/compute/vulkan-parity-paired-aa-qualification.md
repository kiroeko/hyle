# Vulkan parity：同轮配对A/A的具名恢复资格

<!-- MODULE REVIEW: AUDITED -->

父轮[Vulkan新goal](vulkan-cpu-cuda-apple-parity-closure.md)，baseline
`cf240534f0365764e890c713f32174599e61f7d2`；owner为本轮性能验证维护者。
本卡只修模块局部测量/资格口径，产品FIX零、公共统计核心FIX零；route/Graph旧BLOCKED
在本卡实际PASS前保持。不是恢复已耗尽的旧采样，也不追认历史P1。

## 原因、时序与固定判断

已有[balanced-AA](llm-moe-d5-vulkan-measurement-balance.md)明确记录：同一IQ/MoE自比较
两侧各自中位数之比1.11169，但相同round的比值中位数1.01392。固定六次调用覆盖原2/3
槽周期，CompareResult24rounds按同轮交错；输入相同、native完成及输出释放在计时内。
公共证据标准本就规定主统计为paired ratio/delta中位数。旧reader仅解析两侧各自op/s，
丢失配对关系，不能将该估计量的FAIL直接视作产品性能退化。

本次事前固定新primary：每round从两arm相同iterations=6的seconds_per_call相除，
按标题/arm/round精确一一对应；取24个ratio的中位数作为一个进程点。保留全部原时间、
原表中两侧op/s和旧marginal ratio作diagnostic，不伪造旧结果或静默替换op/s字段。
原5%因素中位/10%单process粗不稳定门保持；signed ratio无abs择优，反向顺序仍按
同一candidate/peer语义规范。原CompareResult.relative只作独立交叉检查，不挑有利值。

## 一次有限任务

复用当前`C:/hv/vp1`已编译产品库及原source，原Vulkan D5 consumer
两具名完整fragment、原输入/权重/精度/资源verify、两角色Graph/Graph和Direct/Direct，
每角色双执行/双分配顺序共8process；24round/每arm每round6调用，原affinity/QoS策略。
初次，仅局部validator读取已有真实paired-round日志；不写新benchmark数学、不改产品
或全局仪器，不在资格卡里运行n15/n31。实际DLL由CMake邻置，进程隐藏、ErrorMode禁弹窗。

适用静态preflight：两arm同route/same task/相同slots/输入集合/ready权重/完整native终态，
warmup及verify结束后位置归零，pair每round均覆盖完整cycle，无热搜索。GPU上下文、workspace
和输出resource各自独立；本A/A不比较Graph owning lease与Direct caller buffers的不同服务。
后者若正式A/B，必须重新核对合同，不让本A/A代签等价。

先一次verify-only/接口预检，build2；编译/日志接线具名错误仅一次纠正。随后唯一8process
资格批次，重型串行；正确性/资源/round完整性任一失败停止，AA不通过也停止，不加轮数/
重测，不改电源或要求更严格桌面静默。所有实际native环境/二进制/SDK/candidate与raw SHA
记录；后验结果不重置预算。

若PASS：该具名配对测量修正和原同拓扑双顺序AA满足旧resume_condition，route/Graph
可依法重新SCOPED本轮真实覆盖与有限FIX；旧D5聚合结果仍未闭合，不直接CLOSED。若FAIL：
明确未决条件/原数据，继续可独立的成熟Provider审计，不借此把整goal假完成。
后续正式P1应使用同一预声明paired口径和统一统计/controls/raw verifier；必要adapter
必须独立有界tool task，不在本卡扩写统计系统。此卡通过只签A/A资格，不是performance WIN。

初次误向普通root producer `C:/hv/vp1`请求仅存在于external工程的D5 target，配置/产物
未失败，启动target不匹配；改用现有external consumer只增量编译该源，不重建产品SDK。
局部reader严格拒绝缺round/重复/错arm/不完整输入cycle，不改变公共hyle-perf统计核。

external consumer随后的真实编译发现旧SDK公开enum只有十格式，而当前共同fixture引用
新增CPU/GPU枚举；旧SDK/现header不可混用。唯一有界接线改为root的maintainer-only D5
benchmark target，链接当前已编译产品/原GTest/nanobench/ggml；只在对应GTest target存在
时提供，没有新产品/公开依赖或空target。既有producer增量只编译一个consumer，旧失败
原始日志保留；不重建旧SDK、无新benchmark数学/额外采样预算。
