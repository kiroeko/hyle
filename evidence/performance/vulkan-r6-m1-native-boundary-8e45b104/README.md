# R6 Vulkan Air 原生边界成本

产品候选 `8e45b10423686a0753aaa9d461c4ef3b117412c5`，Apple 测量快照
`a41d43d531365a1fb8617d60113b37924e9a9864`。在对应正式 A/A 已通过之后，按
[冻结合同](../../../docs/implementation/tasks/compute/vulkan-r6-performance-contract.md)串行完成
Direct/Graph：每入口 primary 制品的执行正反序 × 分配正反序四层，各 15 个独立进程，
每进程包含完整 strict-F32 FFN M1/M16。共 120 个进程，无扩样或额外 placement。

## 结果与边界

Air Mac16,12 / Apple M4、M1-MAC engineering、interactive QoS；没有硬 P-core pin 或 Formal
资格声明。实际 MoltenVK 1.4.1、loader 1.4.350、device API 1.2.334；两入口的全输出独立 double
oracle、前后机器 gate、输入/制品/参考 shader 身份与收集均通过。

| 入口 / shape | 公开入口中位延迟范围（µs） | 裸原生中位延迟范围（µs） | 成对额外延迟中位数范围（µs） |
|---|---:|---:|---:|
| Direct M1 | 5347.880–5382.711 | 5355.039–5387.931 | −9.430–−3.191 |
| Direct M16 | 25967.281–26336.582 | 26246.719–26917.900 | −553.651–−121.438 |
| Graph M1 | 5355.613–5399.860 | 5353.892–5401.318 | −1.750–−1.163 |
| Graph M16 | 26281.209–27548.209 | 26595.745–27337.343 | −389.371–432.296 |

每格是四个独立顺序分层中位数的范围，不是合并 CI。成对差额先按同一进程相减再取中位数，
不必等于两列中位数之差。保留全部正负差值与区间；不同入口分别串行运行，不能跨入口比较
绝对延迟。这里没有逐阶段计时，不能把全路径差额单独归因于 host 包装、锁或 driver/cache。

裸 Vulkan 执行相同常驻数学链和 checked submit/fence，但不承担完整 admission、取消/
quarantine、共享 Context 并发及 Graph generated-result lease。分类始终是
**NATURAL_ROUTE / NOT_COMPARABLE**，不签 NI、严格等速或算法输赢。负差不证明负的包装成本，
也不授权新增实现或采样。四份正式同合同 P1 的原始裁决继续独立保留。

## 原始记录

[compute-summary.json](compute-summary.json)、[graph-summary.json](graph-summary.json)保留原始
字节、全部分层和 raw hash；[identity.json](identity.json)记录候选、profile 与归档身份。
[raw.zip](raw.zip)保留 120 份进程日志、两份 oracle、输入/制品/前后机器记录、对应合格 A/A、
独立 ggml SPIR-V/来源身份、build cache/map 和任务启动器；共 155 个具名文件，另含
archive-manifest。Windows 接收后重新校验全部条目和 archive SHA-256，通过。

raw.zip SHA-256：`a661d8aa776a908549db456d9e40f5bcc5a69330a397b9d4f78001f8ea433d59`。

只签该 M4/MoltenVK/版本和两个声明片段；未知设备、格式及生产模型吞吐不外推。
最终交付/selector 与路线收口仍由 R6 主卡分别裁决。
