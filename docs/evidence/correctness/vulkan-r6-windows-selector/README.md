# R6 Windows：选定门的最终验收覆盖

产品候选 `8e45b10423686a0753aaa9d461c4ef3b117412c5`；原验证快照
`3894b72e3b4fbbda9457e2c31730e48bd0d43c49`。本目录汇总 **111 个选定门的实际覆盖**，
逐门来源、快照和 receipt 见 [identity.json](identity.json)。这是跨快照的工程覆盖汇总；
原整批 selector result 仍为 `BLOCKED`，没有改写为一次全量 `PASS`。

## 三项证据来源

| 来源 | 门数 | 结果 / 限制 |
|---|---:|---|
| 原批次 | 84 | 原快照已通过的门，未重跑 |
| 同快照续跑 | 26 | 只运行原结果中因前置失败而未执行的门；原 source、build、toolchain 和机器身份保持 |
| 修正 Sched 测试 | 1 | 测试快照 `756d62206dbfd412ee95d1446a4f6fe64b73a90d`：完整 251 项通过，定向 32 次通过；shipping 输入不变 |

原 `TaskScope.AggregatesFirstStatusFailureAndStillDrains` 错把 typed join 的值发布当作
scope 错误聚合已经完成。修正只让该测试按 scope 完成顺序提交第二个 child，保持原错误码、
消息与 drain 断言。Sched 产品实现没有修改；原失败 receipt、日志与 BLOCKED result 一并保留。
见[定向测试修正](../sched-first-error-test-order/README.md)。独立修正批次的 docs/closure/configure
也通过，但没有重复计入 111 个门。续跑的 `PASS` 仅指 26 门子集。

七项 shipping 投影（CMakeLists、cmake、include、src、third_party、scripts、samples）在原快照
与修正测试快照逐树相同；比较 proof 和双方原始 receipt 已归档。最终源码交付、原生 Vulkan 与
性能快照之间的继承边界另见[产品输入身份](../vulkan-r6-product-delivery/p0-unchanged.json)。

## 15 个普通产品的实际三交付

表内数字是各旅程实际通过的 CTest 用例数，列顺序对应 source/shared/static。
每个产品均有独立 configure、build、CTest 和 relocated consumer 完成标记；不是只检查 target
存在。header-only 产品按真实消费者核对，不由本表虚构独立产品二进制。

| 产品 | source | shared consumer | static consumer |
|---|---:|---:|---:|
| compute | 1 | 1 | 1 |
| container | 1 | 1 | 1 |
| core | 2 | 2 | 2 |
| error | 1 | 1 | 1 |
| graph | 1 | 1 | 1 |
| io | 1 | 1 | 1 |
| math | 1 | 1 | 1 |
| mem | 1 | 1 | 1 |
| obs | 1 | 1 | 1 |
| platform | 1 | 1 | 1 |
| process | 1 | 1 | 1 |
| sched | 1 | 1 | 1 |
| simd | 1 | 1 | 1 |
| time | 1 | 1 | 1 |
| util | 1 | 1 | 1 |

上述普通组件回归承担中央构建变化的影响范围；Vulkan 真实 FFN、自定义 SPIR-V、变化输入、
独立数值 oracle、两系统 source/static/shared 与隐藏 producer 的完整旅程，见
[最终 Vulkan 三交付](../vulkan-r6-final-delivery/README.md)。不得用此处默认 CPU 消费冒充
Vulkan 的设备或算法验收。

所有重型任务串行，build 并行度 2、CTest 1。Windows 启动前把运行 DLL 放在 EXE 目录；
禁止交互式错误弹窗。该尾项没有重新运行正式 P1，没有更改 CPU 例外或任何性能门。
原 plan 的 5 项 skipped reasons 保存在 identity 和原始结果中，未运行项不写 PASS。

## 原始记录与完整性

- [raw.zip](raw.zip)：原/续跑/修正三组 plan、result、receipt、原样日志、上下文与执行脚本。
- ZIP SHA-256：`0fc35ad737ac19fd47df6f4414f89ee053d0c3b120b43f5e5fcf3c862494d447`。
- manifest 记录 296 个条目的 SHA-256 与字节数，逐项复核通过；
  `archive-manifest.json` 作为额外条目保留在 ZIP 内。
- Windows CTest 的原始本地编码时间戳按 bytes 保留；只为提取 ASCII 的测试计数采用容错解码，
  没有因归档读取问题重跑测试或改写日志。
- [implementation-loc.json](implementation-loc.json)：统一 `scripts/loc-report.py` 在 clean
  `3335aa1b` 统计全部 Compute + Graph 实现 **36,765 行 / 129 个文件**，不含测试、bench、vendor。
  本目录为证据归档，没有产品实现变化。

本记录补齐 R6 预声明普通门；R6 的七门终局仍由路线卡和交付卡裁决。R7 尚未实施完成，
本记录不签整个四路线基础架构 goal 完成。
