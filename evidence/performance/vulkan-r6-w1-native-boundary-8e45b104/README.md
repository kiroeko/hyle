# R6 Vulkan Windows 原生边界成本

产品候选 `8e45b10423686a0753aaa9d461c4ef3b117412c5`，测量快照
`3894b72e3b4fbbda9457e2c31730e48bd0d43c49`。按
[冻结合同](../../../docs/implementation/tasks/compute/vulkan-r6-performance-contract.md)，在对应
Direct/Graph 正式 A/A 通过后，分别完成 primary 制品的执行正反序 × 分配正反序四层，各 15 个
独立进程；每进程含完整 strict-F32 FFN M1/M16。两个入口共 120 个进程，无扩样或新 placement。

## 结果与解释

W1-WIN engineering / RTX 4070、driver 610.62；caller CPU 2、process mask 0x14。计时前的
全输出独立 double oracle、输入/制品/参考 shader hash、machine gate 与进程收集均通过。
裸 Vulkan 使用相同常驻数学链、recorded commands、checked submit/fence wait，但没有完整
HYLE admission、取消/quarantine、共享 Context 并发服务或 Graph generated-result lease。
因此分类始终为 **NATURAL_ROUTE / NOT_COMPARABLE**，不产生非劣门或算法排名。

| 入口 / shape | 公开入口中位延迟范围（µs） | 裸原生中位延迟范围（µs） | 成对额外延迟中位数范围（µs） |
|---|---:|---:|---:|
| Direct M1 | 1276.438–1349.036 | 1277.253–1350.475 | −8.311–5.553 |
| Direct M16 | 4540.707–4797.774 | 4559.548–4814.868 | −36.229–18.402 |
| Graph M1 | 1264.590–1283.994 | 1249.859–1277.988 | −3.995–8.393 |
| Graph M16 | 4489.136–4590.947 | 4464.684–4592.001 | −8.800–7.361 |

每格是四个独立 order-stratum 中位数的最小至最大范围，不是合并置信区间。成对差额先在同一
进程内相减再取中位数，因此不必等于两列中位数之差。正负差值、原始区间和顺序依赖全部保留，
不据此声称零开销或某入口更快，不将额外服务责任归为 shader 算法损失。完整源码成本账单见合同。
正式同合同 P1 已单独裁决；这里不重复采样或更改其统计结果。

## 原始证据

- [compute-summary.json](compute-summary.json)、[graph-summary.json](graph-summary.json)：原始
  字节，全部八个分层及 raw hash。
- [identity.json](identity.json)：产品、测量快照、机器 profile、归档身份与分层摘要。
- [raw.zip](raw.zip)：120 份进程原始日志、两份 oracle、输入/前后机器观测、制品身份、各入口
  对应的合格 A/A 记录、独立 ggml shader/来源身份、build cache/map 和任务启动器。
  共 155 个具名文件，另含逐项 archive-manifest；全部条目 SHA-256 已校验。
- raw.zip SHA-256：`4a11b5e79d5497fe9030152ad6fe5b885d3a9c3448c4b3d2d58f9aa4fd9c812f`。

只签本机、本候选与两个已声明完整片段；没有测量任意型号、格式或生产模型吞吐。
Air 原生诊断、最终 ASan/交付/selector 和 R6 收口仍由主卡分别验收。
