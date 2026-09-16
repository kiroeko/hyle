# R5 A5：Apple 原生生命周期检查点

归属 [R5 主卡](../../../docs/implementation/tasks/compute/apple-v1-four-route-foundation.md) A2/A5，
状态仍为 AUDITED。本次不修改产品 C++；在既有 Air 原生构建上增量编译一个测试文件并重新链接
两个测试程序，14 个 GTest 全部通过，新增 5 个。实际结果见 [原始 GTest 输出](LastTest.log)、
[CTest](lifecycle-native-tests.log) 与 [源码/二进制身份](receipt.json)。

## 真实覆盖

| 用例 | 原生边界与结果 |
|---|---|
| 排队取消 + 丢弃 active result/Invocation | 输入 native owner 在执行中保持存活；close 等待排空；取消槽退役，关闭后 owner 释放 |
| 执行中取消 | Completion 在真实 native drain 前仍 pending，无可读结果；完成后返回 Cancelled；其它新槽继续正常计算 |
| 晚到 native 异常 | 真 GPU square 前缀已编码后，通过 MLX 自定义 Primitive 注入异常；活动及已排队调用失败，后续接纳和 close 报错 |
| 独立 Context | 一个 native owner 阻塞时另一个仍能进入独立 native 提交边界，两边排空成功 |
| 分配拒绝 | HYLE ResultSlot quota 为零、冷 callable 抛 bad_alloc、Metal maxBufferLength 原生拒绝；无部分 Program/结果发布，已有程序仍可执行 |

NativeBoundary 只存在于 [测试文件](../../../tests/compute_route_apple/lifecycle_test.cc)，
使用 MLX 原生 Primitive 扩展；前缀是真 Metal 运算，没有产品 hook 或替代 Provider。
超大 metadata 触发锁定 Metal allocator 的分配前 maxBufferLength 检查，不尝试耗尽机器内存。
原生容量错误保留 MLX 原始诊断；没有按错误文本改写产品分类。

## 身份与适用范围

Windows baseline `6f3ff7ca`；新增测试 SHA-256 与 Air 实际文件一致。产品 C++ 仍为 `d900a8ce`
以来相同输入；Air 既有 source 与完整 `cbedb837` archive 的 749 个产品/交付文件
[逐项比较](product-identity.json)仅差本路线未启用的两个 CUDA audit 调用及其锁摘要，
Apple 活动输入相同。配置见 [日志](lifecycle-configure.log)，构建见 [增量日志](lifecycle-build.log)。

本次无 sanitizer、无 P1；Mini、host ASan/UBSan/TSan、最终性能及七门收口仍待完成。
不注入 GPU hang/device loss，不把受控 native 异常等同于全部硬件故障覆盖。
