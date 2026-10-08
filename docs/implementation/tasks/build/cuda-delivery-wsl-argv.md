# CUDA 交付入口：WSL 原样参数

<!-- MODULE REVIEW: SCOPED -->

- baseline：`b3691265`；本批工具，产品/registry不变。
- trigger：Linux configure首次FAIL；`--`经默认shell重解析，将
  `-DHYLE_COMPONENTS=compute;graph;simd`拆成三个命令。原失败与日志保留在固定任务记录。
- 有限FIX：WSL launch改`--exec`，不经shell重解析；补真实WSL含分号参数反例。
- 预算/验收：一次修复、轻量行为测试与同ID一次configure纠正；通过项复用、失败预算不重置。
  不重跑Windows、不改数学/P1/公共工具。配置未通过前不继续build。
