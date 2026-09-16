# CUDA vendor 内容身份与消费前校验

## Closure card / CLOSED

- baseline：`bf685ea3b2d3317d67802c4d1436c9bf1f440267`；owner 为
  `compute_provider_marlin`、`compute_provider_cuda`，仅重开依赖锁与构建 seam。
- trigger：R5 完整源码打包的全 vendor 审计拒绝 Marlin 与 SGLang，工作树及两 vendor Git blob
  均未改变。不是新算法/性能失败，不能绕过源码交付检查。
- 主旅程：完整源码 archive 的内容校验；正交旅程：CUDA ON 消费真实 vendor 时必须先验证内容，
  CUDA OFF/core-only 不要求这两份源码。
- 继承：两 owner 的旧 terminal/P1 保存在
  [baseline](../../../../evidence/correctness/cuda-vendor-content-audit-v1/inherited-baseline.json)。
  R4 的运算代码、默认规则、布局、完成语义、三交付执行和限定性能证据不重写。
- 预算：15 分钟具名排序/接线审计，两个有限 FIX；一次完整源码审计与负向校验，重型串行。
- 非目标：不换 pin/算法、不改哈希算法、不重写公共 registry/selector、不重跑 GPU P1。
- 验证：中央 CMake 正规范摘要与独立 Python 复算、Windows/Unix 完整源码审计；修改任一 vendor
  的验证副本必须在它的 enabled consumer 配置前被拒绝；OFF 保持无依赖；适用 selector 与 docs/closure。
- 性能：P0，只有配置期文件校验；原 C++/CUDA blob 完全相同，继承 R4 P1，不进行新计时。
- stop：锁摘要、消费前校验和真实源码交付通过即 CLOSED；不延伸为全库重审。

## 固定 FIX

| ID | 缺口 | 允许动作 |
|---|---|---|
| V1 | 新增两 vendor 的锁摘要与既有 CMake 规范不同 | 复核逐文件 blob，使用现有函数生成正确摘要；不改 vendor |
| V2 | 两个 CUDA owning factor 没有消费前树校验 | 各自 enabled 分支调用现有 vendor audit；OFF 不检查 |

七问：成熟来源/pin 不变；真实 source consumer 失败；复用已有 audit；冷配置 P0；纯 C++ 无新 FFI；
三交付运行字节继承且补完整 source 检查；普通用户获得早期、明确的内容身份错误。

## 审计裁决

15 分钟内已定位，不扩大调查。两个旧摘要均可由 Windows `sorted(Path)` 的大小写不敏感
文件排序精确复现；既有 `_hyle_dependency_tree_sha256` 使用大小写敏感的路径排序。独立 Python
使用 `relative.as_posix()` 排序，与中央 CMake 输出完全相同。全部 vendor blob 从 R4 最终候选至
baseline 没有变化，错误在锁的 manifest 排序，不能改全库哈希规则迁就这两条错误记录。

V1 FIX：Marlin 正规范摘要为 `fd3ab217a3de85cdcbbe81c6beb88d9d837b21b34fd0ee36920b6f93fc13eb25`；
SGLang 为 `9e60796cdbdce35e3e8eefca7bc29eed3533afc249992aa27fdfcd74f44f02a0`。
旧 evidence 中的 adapted tree 值保留为历史事实，本卡明确更正其 manifest 身份，不改写已测 blob。

V2 FIX：两 owning CMake 的 `HYLE_COMPUTE_CUDA_ENABLED` 分支在 `enable_language(CUDA)` 和
消费头文件前调用现有 `hyle_audit_in_tree_dependencies(<owned vendor>)`。因子未激活时不检查，
不让 CPU/Apple/core 裁剪消费者依赖 CUDA vendor。无需更改公共工具实现或产品运行路径。

## 不可变候选与已执行证据

[Windows/Air 原始记录](../../../../evidence/correctness/cuda-vendor-content-audit-v1/README.md)
已证明正确 manifest、完整源码全 vendor audit，以及两 owning enabled consumer 对变更副本的
早期拒绝。CUDA 运算 blob 与 R4 相同。有限 V1/V2 的不可变产品候选为
`cbedb837791da15392019299f2652d9d3d70ec9d`；其后的平台工具纠正不改变这两个 FIX。
[Windows 最终 selector](../../../../evidence/correctness/apple-r5-windows-selector/README.md)
已完成：15 个三交付门 PASS、93 个身份匹配门 REUSED、7 个有原因的 SKIPPED，聚合 PASS。
复用记录逐项核对 candidate 与原日志摘要，未运行项没有改写为 PASS。
按本卡 stop 进入 terminal 裁决，不重跑既有 GPU P1 或外推未测架构。

## Terminal / 七门裁决

候选固定为上列 `cbedb837`，V1/V2 完成；本轮 CLOSED。后续 R5/R6 产品施工不能反向改变
这份已核对的 vendor 身份或扩展本卡范围。

| 工程门 | 裁决与证据 |
|---|---|
| 第三方路径 | PASS：原 pin、逐文件 blob 与许可证不变；规范摘要独立复算，enabled 消费前校验接通 |
| 功能完整 | PASS：完整源码 archive 和两个被篡改验证副本覆盖本轮正/负真实消费旅程 |
| 内部复用 | PASS：复用现有中央树审计，不复制哈希算法，不修改 CUDA 运行实现 |
| 性能 | PASS / P0：仅配置期线性文件摘要；没有热路径成本。R4 P1 资格和方向按继承记录原样保留 |
| 交付边界 | PASS：没有新 API/ABI/FFI；错误在 owning consumer 配置前明确报告 |
| 构建交付 | PASS：Windows/Air source audit、实际三交付及最终 selected receipts；OFF 无新增 vendor 责任 |
| 易用与文档 | PASS：依赖锁与消费入口一致，本卡解释旧摘要更正与原始证据的关系 |

性能限制仍为 R4 实测 RTX 4070 / SM89 与对应 driver/toolkit 组合；不签其它架构 SOTA。
CPU decode 的批准例外不属于本卡，保持不变。重开条件为实际 vendor 内容/pin/许可证变更、
已复现的审计绕过或新的 source/package 消费失败；单纯文档提交或验证副本变化不触发重测。
