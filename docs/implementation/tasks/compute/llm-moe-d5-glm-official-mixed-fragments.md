# D5-F40：GLM 官方混合权重片段接线

<!-- MODULE REVIEW: AUDITED -->

## SCOPED → AUDITED

baseline `c40fccf2`，Windows canonical。当前owner为AUDITED的`compute_ops`，归属D5共享
模型consumer；不重开D1–D4，不修改已关闭的路线产品。F38/F39已签CUDA dense/grouped FP8
真实入口，剩余consumer不能继续用“官方一概拒绝”跳过已具备的权重角色。

审计依据为`tests/compute_models/d5_glm_remaining_facts.json`固定官方header、原模型程序，
以及现有四路线GGUF组合。GLM-5.3与Flash前三个dense层I12288均FP8；DSA版indexer的wq/wk
为FP8、score为BF16，Flash indexer全部BF16。两者MLA的qa/qb/kva/o使用FP8角色；Flash
KDA的o另为BF16。Flash MLA kv_b为BF16，DSA kv_b为FP8，禁止因为形状相似就当成同一格式。

## 有限FIX

1. 共用consumer层提供已接入FP8动态投影的编码权重、dtype边界和独立double oracle，明确
   BF16输入/输出舍入；不复制Provider、添加产品数学或用Q8/BF16权重代替FP8。
2. 两个官方dense FFN、DSA官方FP8 indexer、Flash混合FP8/BF16 MLA及稀疏cache反馈批量接线。
   真实hidden/query-rank/intermediate不缩小；T/cached-token容量按D5约定。
3. 复用F39完整官方MoE用于Flash NextN，连接EH、注意力、残差、norm、MoE与实际状态反馈；
   各角色读取固定manifest。全词表head/最终完整画像依旧需要已有独立consumer的真实接合。
4. 先完成该批共同接线再构建CUDA Direct/Graph；投影局部沿用BF16 1ULP，混合整链沿用
   已登记BF16误差门；cache/状态/selected ID按既有门。API/算法未改变不重跑F39全套P1。

组合审计发现的有限consumer纠正：原GLM Flash `swiglu_limit=10`对gate作上界截断、up作
双边截断；F38/F39共享FP8夹具漏接该截断，旧输入未达到阈值，不能外推越界输入。
本卡FIX2包含用已接入成熟Clamp/Copy/Concat/激活补足此分支，以及BF16阈值两侧独立数值检查；
不修改产品数学。旧收据作为其已运行输入域的事实保留，完整画像计数未因此增加过。

## 验收、预算与停止

一个共享consumer实现、一批普通CUDA Direct/Graph入口、复用现有build并-j2/测试-j1。
13个来源事实检查、适用consumer、docs/closure；只改测试/文档时不构建无关Provider或跑P1。
若现有Provider不能表达真实布局/权重转换，保留精确缺口、另定有限产品接线，不在consumer
内用host数学补洞。DSA kv_b的FP8吸收/转换先按原模型证明缓存边界，不能每次重放解码全部权重。
所有未验region、其它路线、三交付/P1保持未验；本批通过不自动增加完整画像格。

## 当前检查点：接线完成，组合数值未通过

`5a298870` 已统一构建并执行 CUDA Direct/Graph；两入口各6组中各5组通过。失败限于新接的
Flash MLA／NextN组合数值，原门和失败均保留。独立局部重算已排查KB/VB投影和注意力本段，
目前支持低精度逐层舍入传播方向，尚不能代替完整片段验收。
[收据与有界定位](../../../../evidence/correctness/llm-moe-d5-real-fragments/glm-official-mixed-checkpoint/README.md)。
当前继续FIX4中最早数值分歧定位，不扩建性能仪器、不重跑已通过的整批；ledger仍140/176。
