# 可选 factor 的 TRACE 构建门修正

## Terminal tool record / CLOSED

- baseline：`14e78c45cd58c2836ffc6b3bcc0394fdb4fbc220`。
- owner：build helper；六个 TRACE 消费者由 [R7 当前卡](../compute/four-route-foundation-final-review.md)
  绑定，不新增 runtime component、不改 selector 或 closure/performance 核心。
- 可复现源码反例：`cmake/HyleModule.cmake` 禁止 `OPTIONAL_ABSENT TRACE`，又无条件要求
  中央 TRACE consumer 传 TRACE；Apple/Vulkan 四个路线在 OFF 分支必须 OPTIONAL_ABSENT。
  因而合法的可选跟踪消费者不能同时满足 enabled 和 absent 两种配置。
- 有限改动：仅在 OPTIONAL_ABSENT 时豁免“真实 target 遗漏 TRACE”的检查。仍拒绝 absent
  带 SOURCES/TRACE/TESTING；enabled 的中央声明、真实调用与 TRACE 参数必须一致。
  不修改 token scanner、依赖下载器、中央产品 roots 或 trace runtime 本身。
- 调查预算：上述条件与实际四个 OFF 分支一遍，之后停止；性能 P0 冷 configure 条件。
- 预声明验证：enabled 正例；enabled 遗漏 TRACE、未中央授权 TRACE、无真实 hook TRACE、
  未声明但真实调用 hook 四种拒绝；absent 中央 TRACE 正例以及 SOURCES/TRACE/TESTING 拒绝。
  检查 absent 无实现/private 链/Tracy 依赖，实际 R7 普通 OFF/ON source/static/shared 由主卡承担。
- 停止：有限配置门与主卡受影响交付通过后 CLOSED；不以测试需求扩大通用构建工具。

scope 已在 `ecfed35f` 固定；独立源码审计确认两条条件确实矛盾，实际四个 OPTIONAL_ABSENT 分支
无 implementation source，公开头不需要 trace hook。唯一 FIX B1 冻结为给中央遗漏 TRACE 检查增加
`NOT ARG_OPTIONAL_ABSENT` 条件；其它检查与依赖投影保持。配置测试是独立临时 consumer 或现有
构建合同测试的窄扩展，不为此执行性能仪器或改产品算法。B1 与 R7 F1 必须在同一组合候选验证。

## 终局（2026-09-17）

唯一 FIX B1 已完成，候选 `87ae583407cbb5ed13108c3837b020e6e343a3ca`。
[九个配置门](../../../../evidence/correctness/compute-graph-r7/b1-configure.json)全部通过；
[R7 终局](../compute/four-route-foundation-final-review.md)完成 Windows/Air/WSL 的 OFF/ON 三交付，
实际 absent/enabled 路线、消费者定义和单一 Tracy client 均有证据。scope、architecture、correctness、
delivery、usability_docs、red_team、engineering_performance 七门 PASS；P0 为冷 configure 条件，
不启动性能采样。命令与日志见 [R7 evidence](../../../../evidence/correctness/compute-graph-r7/README.md)。
本卡关闭，不扩大公共构建工具。仅在真实 enabled/absent TRACE 回归、依赖传播或公开交付合同变化时重开。
