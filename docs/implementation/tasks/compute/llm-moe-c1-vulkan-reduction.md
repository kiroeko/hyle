# C1-Y34：Vulkan 成熟归约与概率接入

## SCOPED / AUDITED

- owner `/root`；baseline `3cd981b318f16e82f975848b976e085d7954c344`。
- 当前三个 owner：`compute_provider_ggml_vulkan`、`compute_route_vulkan`、`graph_route_vulkan`，
  保持 C1 AUDITED，仅对本卡有限 FIX 重新定界。CPU Q2 的 BLOCKED 不变。
- 需求：C1/F04 的真实 head128 L2、router/attention 概率；正交任意轴归约与广播mask。
  Sum/Mean/Softmax 已有成熟实现但尚未绑定，不能凭既有 HYLE shader 继续默认自写。
- 普通入口、共同数值/空域/NaN/Inf/±0、F32累加、half最终一次RNE、Graph结果/关闭/失败
  合同不变。继续使用现有 Program/region/Run、原厂内部资源及HYLE片段边界。
- 来源仅核对干净固定 ggml `456172ec733a135778adcd32d00e576a58232e45` 的行归约、概率
  API/shader/执行分派及当前HYLE适配；[源码与反例](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-reduction/source-audit.json)
  是审计输入，不是GPU或性能验收。

已定位两项具体约束。SumRows/Mean的XYZ分派补齐行数，而shader无尾行guard；沿已有Scan
接入方式分成至多三个完整行片段，原厂Concat组合，不发出多余行。长行Softmax的large2/3
各有一处 `max(max_val, vals[tid+s])`，每轮丢失先前累计峰值；128 lane、仅lane64为1000时
源码等价循环返回0，正确值1000。宽32769、末列峰值可触发该分块位置；修正为累积shared值，
保留原厂三阶段算法、调度与scratch。

| FIX | 固定范围 |
|---|---|
| Y34-A | F32 dense typed region接入最后轴keepdim Sum/Mean及普通Softmax。Sum/Mean尾行安全分片与必要输出shape/owner恢复；不自行实现归约核。Softmax无mask或同shape additive mask进入原厂程序，additive前的scale独立F32求值，禁止融合改变已冻结舍入顺序。 |
| Y34-B | 普通route冷期适配任意axes/keepdim、三浮点、rank≤8、合法byte-stride/View：必要Cast/Copy/不复制的permute/reshape；Sum/Mean使用F32行矩阵，Softmax恢复原轴。Boolean mask在缩放后Select为负无穷，复用既有Fill/Binary/Select；additive mask按实际广播域适配。half仅最终一次RNE，所有额外转换/复制/暂存实际计费。 |
| Y34-C | 冷期匹配subgroup/workgroup/grid/F32容量及长行原厂scratch界限；空域、rank0归约、Min/Max、LogSoftmax和设备不适用域保留现有合格路径。Softmax描述符输入/输出对齐走现有适配。只修large2/3累计max两行，使用现有精确patch机制、来源hash和依赖锁条目；不改pin、公共构建/selector/性能工具。 |
| Y34-D | 独立FP64参考、原数值门、真实managed路径证明；覆盖0/1、任意/负/多轴、rank8、half/stride、row511/512/513/262145、长行16384/16385/32769及末块极值、三种mask/scale0或负/NaN/Inf、additive中间舍入。Direct/Graph/Recorder、published中间值/分叉、双槽/变输入/旧结果/close/quota及已有OOM/drain回归。 |

## 成本、验证与停止

P0：Sum/Mean与原版Softmax算术仍为O(N)，尾行最多三个原厂片段；Sum分片的Concat可能保留
原厂临时存储。长行Softmax仍由原厂三个阶段、两份每分块F32统计数组执行。非末轴、half或
广播mask可产生F32整张暂存、转换与恢复；Boolean mask需要既有缩放/Fill/Select，不用省检查
或遗漏适配成本获胜。准备匹配与绑定一次，执行不搜索/计时。没有本批性能赢家声明。

- 当前先执行W1新增反例与三个受影响factor；再同候选W1 static/shared、Air/Mini适用回归及
  source contract/公开头。Windows DLL邻exe、PATH隔离、禁弹窗；Air验证层，Mini按已有角色。
- W1/Air实际本地2B张量的L2/概率片段进入source/static/shared，独立公式、重复运行、资源归零。
  原Y31/Y33片段不能代签这次新接线；适用完整三交付仍在C1内完成。
- 依赖锁内容变化可能扩展selector工程门；保留完整selected/skipped理由，未运行不写PASS。
  未变且身份匹配的证据复用；不得按文档提交重建，不反复执行同候选已通过的门。
- low-parallel2、重型串行；复用活动缓存，归档后及时清理本批重复producer和过期私有源码副本。
- 本批不跑P1。所有C1成熟接入稳定后，依原固定bundle完成等责性能和优选/淘汰；Min/Max和
  稳定LogSoftmax保留为已具名的语义缺口候选，不从此次借用推断自有实现胜出或删除。
- 来源调查到上述具名差异停止；产品纠正最多2轮，独立consumer编译纠正最多2轮。
  新反例超预算则按协议重新定界或BLOCKED，禁止放宽门限、用fallback假冒成熟路径通过。

审计裁决：有限FIX固定，产品尚未实施，适用验证未运行；C1整包、共同矩阵与性能未签收。
C1全部适用验收完成后通知用户，并在C2前等待提问。
