# D5：Vulkan A6 资格阻塞与停止记录

<!-- MODULE REVIEW: BLOCKED -->

- baseline：`d24360ca4fee884d1d765ffea840f54267f1f07e`；Windows canonical。
- owner：`graph`（D5聚合）、`compute_route_vulkan`、`graph_route_vulkan`。
  从AUDITED进入BLOCKED；承接[完整D5](llm-moe-d5-real-fragment-acceptance.md)，不取消其MUST。
- 产品、消费者、公共仪器FIX为零；原产品`9273bc71`未改，正确性154通过/20拒绝/2失败不变。
- 范围：只裁决已固定的[原生片段性能批次](llm-moe-d5-portable-fragment-performance.md)。
  CPU、Apple和CUDA代表片段均已合格，Vulkan有限恢复预算结束。

## 具体阻塞

用户授权在现有桌面占用和波动下测试；不要求进一步关闭应用。Vulkan中断恢复的8个A/A进程均完成，
Graph/Graph与Direct/Direct自比较都超过原门：单进程吞吐偏差最大22.45%，因素中位偏差也超过5%。
已有guard阻止产品采样，正式样本0；[完整收据](../../../../evidence/performance/graph-vulkan-d5-fragments-recovery-preflight/README.md)。
因此适用A6为BLOCKED、聚合INCONCLUSIVE；没有合格证据判产品性能FAIL，也不能写PASS。

前次完整n15及部分n31[继续保留](../../../../evidence/performance/graph-vulkan-d5-fragments-interrupted/README.md)，
缺少最终资格记录，不能补写为完整P1。没有合并两次样本、改5%门或继续重试。
普通桌面事实无法单独证明偏差来自哪一层；未据此修改路由、资源管理或Provider数学。

## 恢复条件与独立尾项

blocker owner为本批验证维护者。只有以下具名条件之一成立，才按BLOCKED→SCOPED重新固定有限预算：

1. 针对已记录自比较偏差，有具体测量修正或可说明的执行环境变化，原同拓扑、双顺序A/A通过；
   修正先单独定界，不能直接再跑完整样本，不以“用户再关闭程序”作为默认恢复方案。
2. 用户明确修改该工程性能验收要求；原A/A失败和未完成P1仍如实保留，不追认旧数据PASS。

重复原命令、文档更新或已经通过的CPU/Apple/CUDA结果都不解除此阻塞。
总目标仍有20格冷拒绝、Vulkan Q8两格数值失败、成熟覆盖残余裁决和D6终审；它们保持原需求，
不借性能阻塞缩减目标。其它owner的源码审计/已有文档收尾独立进行，本记录不授权新增产品数学。

## 已完成与验证

[CUDA本次P1](../../../../evidence/performance/graph-cuda-d5-fragments-p1/README.md)在n15完成：
8进程A/A、60产品进程、30公共控制、严格manifest重放，两个形状PARITY/5%非劣PASS。
CPU/Apple原合格证据和四路线A5复用，未重跑。GPU测试均已退出，已立即通知用户。
本批仅证据/文档/状态更新，selector选择docs/closure；不触发产品构建或再次P1。
统一LOC脚本本轮统计Compute+Graph为**60,373实现行 / 241文件**，不含测试，产品无变化。

selector的docs/closure通过，closure以baseline核对AUDITED→BLOCKED合法转移；
既有D5 ledger结构检查仍为154/20/2/0，`git diff --check`通过。
已清理归档后的旧失败/中断consumer构建约641MiB，当前合格构建、恢复构建、SDK和模型保留。
