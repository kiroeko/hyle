# R6 Vulkan 验证策略登记

<!-- PROJECT UNIT REVIEW: AUDITED -->

- unit / owner：build-package TOOL / HYLE maintainers；不新增产品 component。
- baseline：`05917b22`；产品 owner `compute_route_vulkan`、`graph_route_vulkan` 均已 AUDITED。
- 具体反例：实际 R6 worktree plan 将 19 个 `third_party/ggml_vulkan_shaders/` 文件列为
  unknown，尽管依赖锁和 owning factor 已登记来源；现有两个 Vulkan factor 没有 ON 验证配置，
  因而默认 OFF 不能执行原生 correctness。产品授权检查本身没有问题。
- 范围/预算：10 分钟策略源码审计，只使用既有 JSON 字段；固定下面两个 FIX。无需改 selector
  Python、执行器、状态规则、性能仪器或其它 vendor/平台的裁决。
- F1：把准确的 `third_party/ggml_vulkan_shaders/` 前缀映射到 `compute_route_vulkan`。
  未登记 vendor 继续 unknown；CLOSED/QUEUED owner 仍拒绝产品修改。
- F2：两个 Vulkan factor 配置 `HYLE_COMPUTE_VULKAN=ON`，声明 Windows/Linux/Darwin。
  这是可构建的原生 API 平台集合，不是设备资格集合。缺少编译器、loader、合格 device 的 ON
  correctness 不能悄悄降级为 OFF 或标 PASS；R6 卡分别记录真实 GPU 和环境限制。
- 验证：owner 拒绝回归、三 OS 的 Vulkan ON 计划、无关 factor 不引入 Vulkan、完整工具回归，
  再对实际 R6 worktree 生成计划。计划只证明选择，产品 GPU/交付/P1 仍由 R6 验收。
- P0：只有冷期元数据匹配，没有运行时或性能统计变化。stop：两项登记及有限工具验证通过后 CLOSED。

审计已确认 `vendored_input_owners`、`factor_configure_cache` 和
`factor_supported_systems` 正好表达上述责任，F1/F2 固定为 AUDITED FIX；不扩充公共工具机制。
