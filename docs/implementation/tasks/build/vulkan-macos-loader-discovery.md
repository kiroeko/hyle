# C3 macOS Vulkan 普通 loader 发现修正

<!-- MODULE REVIEW: AUDITED -->

- owner：`compute_provider_ggml_vulkan`；baseline `c96c7c6eefb41f9aca71c92b4304814a731fe088`。
- trigger：[Air 普通 source Direct/Graph 实际 loader 失败](../../../../evidence/correctness/compute-unified-c3/air-vulkan-loader-failure/receipt.json)。
  新C3正常旅程缺陷，不重开platform通用DynLib或恢复旧terminal队列。
- 已审计事实：本机已安装 `/opt/homebrew/lib/libvulkan.1.dylib` 和标准位置MoltenVK ICD；
  bare `dlopen`不搜索此目录。只在诊断进程提供标准loader路径后，两个原数学消费者exit0。
  诊断不签普通入口PASS，也不是要求用户常态设置环境变量。

有限FIX：在Vulkan owner的冷默认loader发现中，保留bare系统解析优先；未显式传loader且
macOS解析失败时尝试Homebrew Apple Silicon `/opt/homebrew/lib/libvulkan.1.dylib` 和
Intel conventional `/usr/local/lib/libvulkan.1.dylib`。显式路径仍有权威，不回退覆盖用户选择。
不改全进程环境、不安装driver、不改变GPU/Provider优选，不给submit增加查找或额外配置。
不把生产机器绝对路径固化进installed SDK。

验证：Air原source objects只补owner目标并在无DYLD覆盖条件运行原两个消费者；source/shared/static
搬迁继续原driver剩余阶段。Windows已完成域不进入新fallback，不重编/重跑P1。
P0：冷Context最多三次系统打开；成功owner仍保活DynLib至native终态，无热开销。
预算：一次具名接缝审计、一个限定修正、一次适用三交付；失败记录保留，不换日志/clone重置预算。
模型/算子扩张、通用平台loader、公共治理和计时工具不改。修正资格后归档并回C3；不代签其余矩阵/H1–H3。
