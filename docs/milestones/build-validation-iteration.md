# 构建系统持续回归与演进计划

> 状态：🚧 第一阶段实施中（2026-07-29）。本页是 Jira 式施工记录；稳定原则见
> [`../architecture/build-system.md`](../architecture/build-system.md)，用户入口见
> [`../guide/usage-guide.md`](../guide/usage-guide.md)。

## 背景与目标

build/package 的 target/component/三种交付架构已经严格收口，但 math、util-runtime、sched、io、
simd、compute、graph 仍会在逐模块验收中改变。持续门必须同时满足两点：任何当前模块改动都不能悄悄
破坏全图和交付；“all build 通过”又不能被误读为该模块功能/API/性能已经验收。

本任务不再做一次性的最终矩阵，而是建立可递增的四层回归：CPU baseline、distribution、
portability、feature/release。新增模块进入 manifest 后自动获得前三层的机械覆盖；模块严格收口时再
追加真实 API/硬件/ABI 证书。

## 第一阶段：当前图机械全覆盖

- [x] 三平台每提交 baseline 保持 `HYLE_COMPONENTS=all`，外部 feature 显式 OFF，构建/运行完整
  `hyle_tests`；它只说明当前 graph 正常。
- [x] shared/static package smoke 从 mem-only 提升为 all-component CPU SDK；consumer 从
  `Hyle_AVAILABLE_COMPONENTS` 动态检查全部 imported targets，并继续真跑 mem C/C++ probes；
  `samples/package/<module>/...` 的 focused projects 自动发现，不再维护中央模块清单。
- [x] complete-source 解压验证从 core-only 提升为 all-component CPU graph。
- [x] Linux AArch64 cross 从 core-only 提升为 all-component CPU graph；仍不冒充 target execution。
- [x] shipping-layout roots 集中为一个显式清单；只放已严格收口的模块，CTest 直接运行其解析闭包的
  全部测试，不再维护第二份正则模块清单。
- [x] CI job 增加明确超时；PR 保留三平台 baseline，昂贵 distribution 在 PR 跑 Linux、main/手动
  再跑 Windows/macOS。
- [x] Windows：shipping-layout 16/16；shared/static all-component package consumers 各 16/16；
  complete-source manifest/digest/legal/extracted all-component build 全通过。
- [x] WSL/GCC 14：component contracts、shared/static all-component consumers 各 16/16、native SIMD
  5/5，以及 all-component AArch64 cross 均通过。
- [x] Apple M4：native SIMD 3/3，shared/static all-component consumers 各 16/16；per-component
  target 形状消除了旧单一消费者人为产生的 duplicate archive warning。
- [x] package/product/source/component smoke 只删除显式 scratch 下自己拥有的子目录；Apple package
  fresh 重跑无旧 dylib RPATH 诊断，WSL component contract 同一 scratch 连跑两次通过。
- [x] error 交付 handoff：真正的 C-only owning-handle consumer 与 C++ `Status`/`Result<T>` consumer
  分别直接请求 `COMPONENTS error`，进入 shared/static package smoke；error-only configure contract
  同时锁住 GoogleTest 必须服从 `HYLE_DEPS_DIR`。Mac 复证进一步删除 `gtest_main` 已传递的冗余
  `gtest` 直连，error test link line 不再重复同一 archive。
- [ ] GitHub runner 在额度恢复后复证。

首次 AArch64 扩面虽然生成了全部产物，却暴露 preset 未启用 Werror，SIMD complex fallback 的未初始化
误报会被放行。将 cross 门提升为 Werror 后，根因是异常值数组采用 sparse lane 索引，GCC 无法证明
mask 与初始化相关；不能用全数组清零给正常热路付费。实现改为 compact exceptional list，在向量
store 前保存异常 lane/value，store 后按列表覆盖，既保留 in-place alias，也只遍历异常项。Windows
SIMD 主测试 379 项、WSL native SIMD 和 AArch64 GCC14 Werror 均通过。

## 第二阶段：模块验收随施工递增

每个模块通过严格收口时，必须在同一模块施工提交中完成以下 build-system handoff：

1. 新增 `samples/package/<module>/...` 的 source-independent focused project，调用常用 C++ API；有
   C ABI 时同时用纯 C translation unit 验证所有权、Status、callback 与链接语言。通用 all-target
   probe 只守 export 图，不能代替这一步；目录下每个 `CMakeLists.txt` 会自动进入 shared/static smoke。
2. 若测试插桩会改变正常 layout/热路径，把该模块 root 加入 shipping-layout 集并证明
   `HYLE_TEST_INSTRUMENTATION=OFF`；没有双形状就明确记录不适用。
3. 把适用的 ASan/UBSan/TSan、QEMU ISA 或真硬件门登记到 portability/release 层；cross compile 与
   真机证据分别记录。
4. 更新 package feature/license/runtime closure probe，以及 architecture/guide/C guide；不因 API
   尚未发布而维护虚假的兼容 shim。

验收后，普通实现变动只回归受影响 probe；出现 dependency/ABI/feature 变化才重开对应构建合同，
不重新跑与改动无关的所有手工矩阵。

## 第三阶段：周期和发布门

- [ ] 增加最低支持 CMake 3.25 configure/package consumer，防止日常新 runner 掩盖最低版本退化。
- [ ] Linux ASan+UBSan 全组件周期门；并发模块使用独立 TSan 门，避免把互斥 sanitizer 混装。
- [ ] SVE/RVV QEMU 脚本接入周期门；继续保留真机性能为独立签字项。
- [ ] TLS/HTTP2/ggml/Vulkan 的 SDK-enabled CPU 门；CUDA 只在有受控 NVIDIA runner 时真跑。
- [ ] release 候选同时验证三种交付形状、完整 license/feature manifest、可迁移前缀和所有已收口
  module-owned consumers。
- [ ] 正式发布并冻结 ABI 政策后，再加入上一个受支持版本到当前包的 C/C++ compatibility consumer；
  0.x 未发布施工期不提前锁死错误接口。

## CI 成本与失败分类

- PR：三平台 CPU baseline；Linux distribution/component/source/cross。main 或手动：补齐三平台
  distribution。nightly/release：sanitizer、可选 SDK、ISA/硬件矩阵。
- 不缓存跨平台/跨 toolchain build tree；只允许校验过版本的 dependency source/compiler cache。
- 编译/测试失败归代码；下载或 SDK 获取失败归依赖供应；runner 在任何 step 前被 billing/spending
  limit 拒绝归基础设施。三类必须在记录中分开，不能通过重跑或缩小组件伪造绿色。
- 当前 GitHub Actions 仍受组织额度/账单限制，run `30423424066` 在 runner 分配前失败；恢复后先跑
  `workflow_dispatch`，通过才把远端状态作为本阶段证据。

## 完成条件

- 任一新 component 只改 manifest/模块所有权目录，就自动进入 all baseline、install/source package
  和 AArch64 compile，不需要手工追加四份清单。
- 每个已严格收口模块都有真实安装后 API probe；package 的动态 component target probe 与 focused
  probes 全绿。
- Windows/Linux/macOS baseline 与 main distribution 稳定，Linux cross/min-CMake/sanitizer 周期门
  有清楚责任和超时，feature/hardware 门不会被 CPU-only 绿色替代。
- 文档只在 architecture 保存长期模型，本页保留实施状态；旧 closure 数字继续作为历史证据但不冒充
  当前矩阵。

## 第一阶段代码量（2026-07-29）

| 固定口径 | code |
|---|---:|
| 全库（含 docs/in-tree third_party） | 275,967 |
| 自实现（含测试） | 134,300 |
| 自实现（不含测试） | 92,149 |
| build/package 实现 | 5,590 |

build/package 相对上一轮 5,523 净增 67 行，来自已有 CMake/preset/CI 门的机械扩展和 Windows
complete-source 一键入口；新增的 installed consumer probe 和本任务文档按既定口径分别属于测试源码
与文档，不计入模块实现。

## Error 交付 handoff 代码量（2026-07-29）

| 固定口径 | code |
|---|---:|
| 全库（含 docs/in-tree third_party） | 277,135 |
| 自实现（含测试） | 135,036 |
| 自实现（不含测试） | 92,659 |
| 当前处理模块实现（error） | 1,434 |

error 产品实现未增长；本批新增的是 source-independent package probe、通用自动发现门和构建/文档
合同。按固定口径，probe 位于 `samples/`，CMake 位于 build system，因此后三项保持不变；全库相对
上一轮 276,996 增加 139 行。
