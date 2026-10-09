# Vulkan V3：成熟默认、条件绑定与有限成本裁决

<!-- MODULE REVIEW: AUDITED -->

baseline `755dec4c5`；owners `compute_route_vulkan`、`graph_route_vulkan`，必要的
原region接线归 `compute_provider_ggml_vulkan`。V2 Windows 104/559/80/34/45/八族及
原IQ1 bank资格继承；CPU/CUDA/Apple CLOSED、共享Compute/Graph与公共仪器不改。
主纵切仍为当前固定模型片段，正交为公开typed语义/失败/未知条件默认。

## 审计输入与有限范围

- D1：`vulkan_internal.h::route_facts` 未被任何prepare/compile使用。复用共同
  `RouteSelector/RouteMatchInput` 接入现有冷承载，不让用户排序或提供timing；
  平台、实际device/driver/API、编译器/源码版本、provider版本、模型/制品/phase/region
  和每个端口保留条件表达。未知条件匹配合法成熟默认，真实capability仍由原承载检查。
- D2：清点native tensor中的数学与存储/检查机制，按原ggml/IREE source和完整合同
  判定重叠/唯一/冗余；成熟完整替代无明确优势的self退出，不为保self追加优化。
  有效存储、NaN/RNE、checked索引、资源、未知device独有能力不误删，SIMD不动。
- D3：只对真实同合同重叠或Graph消费者成本运行有限P1。旧Recorder直接路径由caller
  供输出、Graph返回owned lease，不能直接比较。先冻结完成/输出拥有/失败/临时成本
  等价的原生peer旅程；没有同合同就如实NATURAL_ROUTE，不能为通过临时删掉服务。
- D4：API/算法/default稳定的不可变candidate才一次P1，先相同workload A/A与双顺序；
  事前paired口径、n15、5%门、placement，源/成本先行，失败不追加样本或改系统模式。

source/接口审计预算30分钟；有限批接30分钟，纠正20分钟一次。P1最多一次正式批次，
预检不合格即停止该比较，单carrier不造竞赛。适用未验不能PASS，AA失败不认产品FAIL。
Windows同producer build2/test1、重型串行、DLL邻置/隐藏child；已过V2大矩阵不重测。
验证只新默认/条件/真实重叠、必要资源与直接Graph消费者delta；三平台/三交付仍V4。
必须selector、closure明确previous-ref、docs/diff与统一LOC。本文SCOPED前段只读，
具体原算子漏绑定及删除表固定后再AUDITED，不扩模型、算子研究或公共治理。

## D1 有限FIX已审计

现有RouteFacts保留vendor/device/name/driver/API/subgroup，但未接线、缺编译SDK/原
provider源码版本，空文本还会成为InvalidFacts。仅修改route owner：Environment冷事实
补SDK/ggml/IREE源码身份，route_facts改为稀疏有效事实；既有所有原typed/capacity
admission前接入固定ggml/IREE默认selector，PlanStep输入/输出、workload全字段传入共同
matcher。默认selector不约束特定机器/模型；窄条件可用共同matcher，未知facts不得命中
窄规则、仍可匹配合法成熟默认。GPU代数未可靠提供则未知，不猜SM或借CUDA探测。
matcher只prepare执行，native执行状态/资源/融合形状不变，没有新用户偏好或hot metadata。
Graph实现说明的旧十格式名单同步当前原承载范围，不改接口责任。

source/contract审计还证明Relu/Gelu原shader虽存在，但完整公开NaN/Inf/负尾语义不同；
不能凭正常有限输入计时认完整替代。native低feature/严格存储等独有域暂不删除，D2表逐
域继续源裁决，只有真实完整重复才比较或清退。D3的旧不等责配对仍不运行。

首批验证只新条件适配的组合字段/unknown facts/default、两入口ggml/IREE实际链与
workload保留、Graph报告和必要重放；V2相同typed/math大矩阵不重跑。公共matcher与
共享模块不改，CMake只配置现有生成头里的锁定事实，不新增依赖/用户开关。

## D1当前实际结果

`7cda5d729` 新冷匹配与稀疏事实已接线，锁定SDK/ggml/IREE身份由唯一dependency lock
配置进既有生成头。Windows Direct10/10、Graph10/10 PASS：组合条件与未知事实默认，
真实ggml规范与本地权重、IREE IQ1/FP4/FP8、跨Provider链及重放/lease/cancel/close/OOM。
原cold admission/融合/资源/数值规则保持；没有新数学，匹配不进入submit/run。
编译首失败仅为新测试Operation的显式构造器，已一次修正；产品编译和定向测试通过。
原V2大矩阵未重测。[raw收据](../../../../evidence/correctness/vulkan-parity-v2-catalogue/README.md)
保留候选、二进制SHA、邻置DLL及validation结果。

本批D1完成，不等于V3全部完成；D2完整残留裁决、D3同合同preflight和D4有限P1仍待。
公开Environment新增的冷事实与相应header/三交付验收属于本路线V4必需项，尚未签PASS。
本批统一Compute+Graph非测试实现62,483行/268文件。新增的是冷条件适配，不是算子数学。

## D3 同合同成本预检已定界

新有限peer不是旧Recorder：`compute/detail/vulkan_plan.h::prepare_plan`是已安装的
Vulkan专属冷接缝，将显式typed片段计划编译成普通Program。Graph编译也调用同一接缝。
以已验真实Qwen3.5-2B BF16 FFN和DeepSeek混合hash-MoE完整片段为固定primary/holdout，
显式计划端保留完整节点、常量、端口、view与workload，准备后销毁计划；两端均创建普通
Invocation并绑定同一输入cycle，submit→wait→release拥有结果lease均在计时内。

两端同算法/precision/encoding/转换/融合、cold资源规划、独立slot与native终态，
相同失败/取消/输出不可改写合同；初始化/首次原库lazy warm另列实际时间和常驻资源。
无caller-destination差异，无新增薄弱语义的benchmark-only输出。比较分类SAME_CONTRACT：
Graph前端对显式原生片段计划，验证低成本重复执行；不声称provider WIN、通用engine性能
或整模型推理。默认选择的原算法、状态、能力门均保持，不为测试添加产品后门。

只增一个独立maintainer bench target，继承原fixture/nanobench/paired reader/统一仪器；
不改旧AA raw及公共d5_common/统计核心。新完成服务不同于旧Recorder，A/A必须重新
对该新合同固定唯一8process（Graph/Graph与native-plan/native-plan各双执行/分配顺序），
预检不过则不启动n15。通过后一次n15四strata与placement，5%门/原controls；无n31。
先verify-only核对输出/两cycle/资源，再冻结candidate/toolchain/source ABI并运行。
此处只授权bench接线和preflight，不提前给P1 PASS；无新增产品算法。

## D2 有限残留裁决

[逐域表](../../../../evidence/correctness/vulkan-v3-residual/decisions.json)核对当前native
lowering、锁定原ggml Vulkan和已注册IREE完整程序及相应公开测试，不按库名判断。
12组残留分别属于字节/整数/NaN-RNE存储、低feature域、完整严格数学、checked
索引/发布/重复更新、wide prefix或用户自带SPIR-V/资源机制；没有已证明完整冗余域。
固定模型数学端口已成熟承载；所有可用ggml/IREE默认都在native compatibility前绑定。
保留的source在重叠域不是另一个可热搜的候选；仅独有完整能力仍保留，不伪称性能WIN。

原Relu/Gelu可运行正常有限数值并不等于完整NaN/Inf/尾部合同；原512-lane kernel也
不能替128-lane设备域。当前只审已锁定source/API/program，未扩大operator目录或
研究自研优化。无完整同合同第二承载时按用户既定原则不造竞赛；未来成熟完整替代
资格化即触发定向删除，不因当前保留授予SOTA。SIMD及下层资源没有改动。

D3 verify-only 已执行：两完整片段、两端输出/两cycle/资源预算PASS，实际dispatch/
managed-region数量及cold/warm常驻字节逐端完全相同；Graph对显式native-plan。
原完整算法/精度/接口仍稳定，下一步仅该合同的唯一A/A资格及适用n15，预算不重置旧比较。

primary拥有服务A/A在 `4c284c4f` 唯一8process PASS。进入P1前为既定placement要求
添加中性前缀的独立artifact，primary源码/EXE不改；placement首次单独8process资格，
失败不得换layout重测。固定正式样本primary/placement×双执行/分配四strata×n15，
两完整片段共同采样；原公共controls源码原样构建，n15正反controls与原统计/机器policy。
全过程一次，约25分钟上限，不升31。P1主统计仍逐轮paired比的进程中位，原op/s侧
中位仅绝对时间/floor诊断；不得把其比值导入主裁决。只有两artifact AA及原controls
均资格才签正式NI；未资格或测量未完保持明确INCONCLUSIVE，不假签产品退化。

neutral placement AA八process结束：完整数值/资源通过，但同native-plan自身的
DeepSeek分配因素偏差6.4987%越5%门。primary PASS不替placement签资格；n15不启动。
[具体工程取舍](vulkan-v3-performance-engineering-decision.md)保存固定结果、源等价和
未证明原因。本性能条件未解决不关闭V3/goal，但V4平台/三交付仍可独立推进。
