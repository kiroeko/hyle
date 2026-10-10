# C3 macOS Vulkan 普通 loader 发现修正

<!-- MODULE REVIEW: CLOSED -->

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

产品candidate：`a1c22ac347583e8a9af3be3d8e8b55534afcd710`。默认冷发现已接入上述两位置，
显式loader override不进入回退；非macOS路径不执行新分支。数学、driver和全进程环境未改。
实际Air无DYLD覆盖的普通三交付尚待验证，诊断成功不代签。

Air普通source已实际PASS：明确`env -u DYLD_LIBRARY_PATH -u DYLD_FALLBACK_LIBRARY_PATH`运行
原Direct/Graph，2.42s/0.47s；[收据](../../../../evidence/correctness/compute-unified-c3/air-vulkan-partial.json)
保持初始FAIL与诊断独立。完整源码archive hash匹配后checksum增量更新，只编译一个owner TU
及必要链接，原shader no-work。shared/static恢复现有driver循环，未删除source root，尚未签PASS。

原shared/static循环现已exit0，四项SDK消费者也全部PASS（static Graph0.78s/Direct0.73s），
两SDK搬迁和实际HyleConfig日志已归档。普通入口未设置DYLD回退环境，driver/数学未改。
本卡适用source/shared/static门完成；actor暂保CANDIDATE等待C3统一终审/其余直接需求绑定，
不代签尚未完成的mixed平台矩阵或H1–H3。统一LOC当前62,508实现行，不含tests/third_party。

## 限定修复终局：CLOSED

本卡七门PASS，限定默认macOS冷loader发现：Air Vulkan与CPU+Vulkan均完成真实三交付数学，
无DYLD覆盖；完整runtime assets、包metadata与当前离线route/recipe检查已有独立收据。
显式路径不覆盖、无全局环境改写、非macOS不进入新分支，独立C3反证未发现新的本卡缺陷。
P0为最多三次冷打开，不增加热成本，不升级依赖/改数学/建新通用loader；当前docs/closure/diff通过。
原source loader FAIL与诊断保留，不代签WSL实GPU或H3。reopen仅限同域可复现默认发现/
owner保活/交付缺陷，或实际新平台部署需求；本卡停止。
