# 统一入口/冷路由/CPU+GPU基础机制终审

本轮七包原scope保持：C1/C2/R1/C3已在用户接受边界关闭；H1/H2已关闭；H3也已补齐原生组合并关闭。
这份表只对照当前合同，不重开旧模块，不代签推理引擎、整模型容量或增量依赖图。

| 要求 | 当前来源/证据 | 裁决 |
|---|---|---|
| 唯一路线集合，自动依赖、无普通Provider菜单/旧开关双轨 | build-system/中央组件图，C1/C2/R1施工记录、C3终局与真实SDK requirements | 已实现/已验 |
| build identity与native硬件/driver版本、模型/制品、全部tensor格式/shape/resource条件冷匹配；hot不计时重选 | compute-routing-system/route_policy及各原生prepare；C3终局、H1/H2/H3的immutable plan/固定input目录 | 已实现/已验，未知硬件采用合格默认，不泛称WIN |
| 自然Direct与Graph、成熟完整regions和路线原资源/程序/完成 | CPU/CUDA/Apple/Vulkan原CLOSED；H1 cpu_cuda、H2 cpu_apple、H3 cpu_vulkan及可运行普通样例 | 已实现/已验 |
| 普通CPU权重大于staging仍直接参与混合数学 | 每包主纵切CPU16KiB权重与512-byte native边界、CPU norm/projection→GPU完整FFN→CPU，正交数据管线 | 三组合真数学通过 |
| 有界两槽/IDs/格式/capacity、准备与原生传输、旧owner保护 | cuda/apple/vulkan weights，source/contracts及Win/Air新expert ID7→8 same plan | 已实现/已验 |
| 实际native依赖、可重叠基础机制 | H1 CUPTI、H2 Metal单dispatch/blit、H3原VkQueue copy/dispatch query；真实payload与oracle | 已资格化本机机制，不承诺独立DMA/任意任务收益 |
| 共享host/device/staging预算、packing/metadata/所有slots/旧leases、unknown/partial与实际backing去重 | 原ResourceBudget/Mem与每组合resource_report；各contracts strict、OOM、rollback、旧引用保活 | 已实现/已验；driver/引擎外部账单不伪造总预留 |
| prepare后copy/repack/host等待/原生队列路径诊断，改绑失效与host-wait reject | 各ReplayReport/CompileReport、prepare_replay和reset/bind负例 | 已实现/已验 |
| 失败/取消/close/abandonment、接纳prefix排空、成功lease/晚到、same-device与坏shape/ID/overflow | 每包source/contracts；H3还真实copy接纳后拒绝handoff并据copy fence退休 | 已实现/已验，取消不抢占已接纳GPU |
| source/static/shared、公开头/资产/真搬迁/DLL同exe、genuine缺路线 | H1 Win/WSL、H2 Air及平台OFF、H3 Win/Air五消费者每交付及WSL适用build/install/link/headers，三端缺路线实际拒绝 | 已验；WSL Vulkan GPU运行保持UNQUALIFIED |
| Windows唯一产品真相、限内存低并行、串行、[skip ci]推送与有界清理 | Git candidate/各archive+sync/guard与terminal process receipts，实际三端记录 | 遵守；验证副本未反向改产品 |
| 不扩数学/万能executor/第二后台runtime、不实现loader/KV/offload策略/多GPU/增量图 | 仅新增各路线粗边界/迁移/预算/报告；源/头与任务负空间；DEFER-GRAPH-INCREMENTAL保留 | 保持scope |

具体终局以H1/H2/H3 task及module-status为权威。本机WSL物理Vulkan GPU的十二旧项和H3新增GPU
数学/寿命/overlap没有运行，用户接受UNQUALIFIED，不将硬件缺件变成全库阻塞或假PASS。
保留provider-private未知成本、固定pure DAG/CPU+一个GPU、明确host completion与借用寿命。
统一脚本对最终产品候选97f9332ac执行self-check重新核实：Compute+Graph68,204行/301文件，
全库非测试实现163,431行/652文件；文档、旧CLOSED和探针本身不计新产品能力。
最终registry60组件均CLOSED，四H3 owner从CANDIDATE合法关闭，未扩大其它CLOSED owner。
所有重型run已终态，原sdk-staging在三端真实不存在；保留搬迁SDK/源码缓存/原始日志与收据。
本轮最后定向清理Windows2,402,309,781字节、Air641,336,448字节的临时static构建与运输archive，
使用固定边界脚本并保留删除收据；未删除产品源码、用户改动、旧验证副本或已验SDK。
七包基础库goal完成后停止；DEFER增量依赖图、模型loader/引擎/多GPU和进一步算子扩张等待下一阶段。
