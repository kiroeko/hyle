# F5 原故障门批准追加与实际退役修复

2026-10-12 用户回复“追加预算”；原CUDA Direct root/config/tree的consumer只追加一次
180秒，总360秒，保留原consumer-resume10 TIMEOUT和全部历史elapsed。合法恢复提交
f73ae8761 SCOPED→9a7359a2d AUDITED；本证据只签OBS-01/DRAIN-01，不签完整F5。

fixture先warm相同ragged native domain，再在provider已接受kernel之后、原finish_sequence
记录CUDA completion event之前插host Gate。stdout证明进入drain阶段。原unknown观察仍
转发到真实CUDA event/reaper与原F1 guards，不提供虚构退休证明或产品测试开关。

consumer-resume11/12实际输出`drain waited=0 guards held=1`，然后fixture析构触发
CUDA resource-owner `live()==0` fatal；这是原drain提前返回的运行反证及失败清理问题，
不是一个干净的最终断言FAIL，原日志/exit -6均保留。产品局部修复为显式drain等待原
LeaseUsePool及native pending actual退役，release通知原atomic waiter；保留首个copy/native
错误，没有新增字段、线程、轮询器、Queue/runtime或公共头/API。

最终consumer-resume15两episode通过：unknown poll与unknown wait各在真实CUDA prefix
被挡住时保留三input guards；cancel/close不误称退休；公共Completion和token释放后drain
仍等待。Gate释放后guards归零、in_flight为false、原错误保留，不发布输出/复用closed slot，
metadata quota0。async开始锚排除尚未启动线程导致的假等待结论；80ms只判断Gate阻塞，
不作性能结果。首次prelaunch Gate hang根因仍未证明，不能把静态peer线索改写成运行结论。

[receipt.json](receipt.json)记录原root、完整输入hash/机器/toolchain/stage和累计预算：
configure16.403/build476.227/consumer190.072秒。新增consumer约10.043秒，原TIMEOUT仍在
[旧收据](../foundation-f5-native-fault-timeout/README.md)，本次raw.zip也完整保留历史。
consumer-resume13/14单episode修复PASS保留，不替代最终双episode结果。

源码影响计划`.scratch/foundation-f5/retirement-resume-plan.json`为READY；维护工具未改。
按实际owner/直接消费者串行复核：Win common33项、CPU Direct/Graph矩阵与页/ragged、实际
RTX Vulkan Direct/Graph矩阵与页/ragged全通过；原WSL CUDA Graph同slot/两region/页/ragged
通过，收据见[cuda-graph-recheck.json](cuda-graph-recheck.json)。独立公头未变化，复用匹配
原九头/Graph/native头证据；未跑原全算子/全库、P1或A/F4重审。Apple无新增源码/API seam，
Windows selector按实际平台skipped，并未将未跑Apple写成新的PASS。

本门PASS后compute_batch保持AUDITED。accepted H2D/plan/run其他失败、suffix/abandon完整
组合、actual HYLE D256、18 installed SDK格、san/P0/guide/终审等原未完成项全部保留，
不能以这两episode代签；goal尚未完成。
