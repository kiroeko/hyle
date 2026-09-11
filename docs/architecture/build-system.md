# 构建与组装架构

本文定义 HYLE 的构建原则和机器真相边界。当前可选项、依赖身份和组件边分别由 CMake/JSON 合同
给出，不在本文复制易漂移的列表。用户操作见[三种交付](../user/distribution.md)，维护环境见
[setup](../developer/setup/README.md)。

## 目标

- 每个内部组件有明确源码、公开头、依赖、测试和状态 owner；
- 授权源码、静态 SDK、共享 SDK 使用同一依赖图和同名 C++ 产品入口；
- 内部为解环产生的小 factor 不增加普通用户的选择负担；
- 外部源码的版本、内容、许可证、在线/离线路径来自一份锁；
- 普通提交只验证真实影响，完整平台/特性矩阵按风险显式运行；
- 父工程可以裁剪和组装 HYLE，不依赖仓库目录顺序或某个固定 build tree。

构建成功只证明组装合同，不代替模块的功能、API、正确性和性能收口。

## 一份内部图，两种消费层

[`cmake/HyleComponents.cmake`](../../cmake/HyleComponents.cmake) 是 HYLE factor DAG 和产品 façade 的
唯一声明。每个 factor 条目拥有身份、是否可构建、`PUBLIC/PRIVATE/PRIVATE_BUILD` 直接边、公开头例外
以及构造所需 metadata；同一声明投影选择闭包、稳定拓扑序、真实 target link、头 owner、安装集合和
测试图。模块目录不得再手写一份 HYLE 内部依赖。

内部构建 target 名为 `hyle_<factor>`。安装包需要表达传递闭包时可导出 `_factor_*` detail target；
它不是用户合同。普通源码和安装消费者只使用 `hyle::<product>`，并通过 `HYLE_COMPONENTS` 或
`find_package(Hyle COMPONENTS ...)` 选择产品问题域。façade 是无源码、无独立状态的 `INTERFACE`
聚合，不重复拥有 API、算法或 verdict。

`HYLE_DEV_FACTORS` 只给顶层维护/CI 精确构建内部 factor；父工程嵌入和安装不得依赖它。公开头位于
另一产品命名空间的 factor 用中央 `SPLIT_PUBLIC_HEADERS` 一次登记，不能在安装、测试和 include audit
分别维护名单。

中央 resolver 必须拒绝未知依赖、指向不可构建 identity 的边和 cycle，并以 O(V+E) 得到稳定拓扑。
factor 身份通常只追加；合法更名需要一对一迁移任务和状态证据。实时 factor/product 数量及边只能从
中央声明查询，architecture 不维护快照表。

## Target 与传播

`hyle_add_module()` 只接收本 factor 的源码和外部实现 target，内部 `PUBLIC/PRIVATE` link 从中央图
投影。配置末尾再把实际 target link 与声明对账。header-only factor 仍是正式 target 和状态 owner，
但不制造虚构 binary journey。

横切 target 有独立职责：

- build contract 只传播全部 C++ factor 共同需要的语言/编译语义；
- testing 只进入确实包含 instrumentation seam 的测试闭包；
- trace 只进入声明消费 tracing 的 factor；没有 trace consumer 时不能悄悄取得 profiler runtime；
- vendor/provider target 是 build-only mechanism，不进入 HYLE semantic DAG 或产品目录。

目标 CPU 由 `HyleTargetPlatform.cmake` 在 compiler 识别后归一化，源码、ISA 和组件能力只读取该结果。
不能直接拿宿主 `CMAKE_SYSTEM_PROCESSOR` 推断 Visual Studio 多平台目标。公开 inline 所需 ISA 必须由
owner 作为 usage requirement 传播；compiled variant 的更高 ISA 不得提高普通 consumer TU 的 floor。

SIMD expert journey 是显式 opt-in 的开发 target；普通 SIMD façade 不泄漏 Highway/xsimd 类型或寄存器
ABI。多个模块可以共享锁定 provider headers，但这不自动形成 HYLE 模块依赖。

## 外部依赖

[`cmake/HyleDependencyLock.json`](../../cmake/HyleDependencyLock.json) 唯一登记依赖名、revision/archive、
内容 hash、许可证、owner 和投影信息。模块 CMake 只请求具名依赖，不自行写 URL/version。configure
按所选闭包取得依赖，未选 feature 不探测、不下载。

源码是跨 build 的资源：`HYLE_DEP_SOURCE_CACHE` 保存锁定 source；每个 build tree 的 `_deps`/
`HYLE_DEP_BUILD_DIR` 只保存 binary/subbuild 状态。首次在线配置可以填充 cache，后续 fresh build 复用；
离线机器搬运完整受管 cache，或通过根级 `hyle-offline-deps.cmake` 显式注入已审查源码。产品 source
cache、benchmark peer checkout 和编译产物是三类目录，不能混用。

依赖升级必须由 owner module 的真实影响触发：先改锁和许可证投影，再验证直接消费者、三交付和适用
性能。更新 vendor 不是“只改构建”；它会触发 owning factor 进入合法审计状态。

## Option 语义

选项只表达用户需要控制且能稳定解释的产品选择：

- 普通布尔开关使用 `ON/OFF`；
- 允许探测降级的系统能力使用 `AUTO/ON/OFF`，`ON` 缺失即失败、`OFF` 不探测；
- developer/test/benchmark 开关不得改变 shipping ABI；
- 后端只有一个实现时不保留空 selector；
- option 只在 owning 产品进入选择闭包时出现或生效。

新增 option 前必须证明它不是从平台、编译器、组件选择或 feature presence 可导出的重复状态。正式 CI/
交付要显式固定重要 feature 集，不能依赖宿主偶然探测结果。

## 三种 C++ 交付

三种形状共享产品名、头 owner、依赖闭包和语义：

1. **授权源码**：父工程设置产品集合后 `add_subdirectory()`；HYLE 不能接管父工程全局 flags、输出目录
   或包管理策略。
2. **静态 SDK**：安装 owned headers、factor archives、façade targets 和必要依赖 metadata；链接器按需
   抽取实现。
3. **共享 SDK**：安装相同 headers/façade 与同级 factor shared artifacts；visibility、runtime 依赖和
   relocation 必须由真实 consumer 验证。

这三种都是 C++ 接口。内部 `extern "C"`、C/汇编编译单元不构成公开 C ABI；Native FFI 只有满足
[准入策略](native-ffi-policy.md)的具名跨语言消费者后才建立。

源码包不复制联网依赖源码；全离线发布必须同时交付受管 source cache/注入目录和许可证资产。详细用户
命令与陷阱只在[distribution](../user/distribution.md)和[offline setup](../developer/setup/offline.md)
维护。

## 验证与 CI

[`validation-select.py`](../../scripts/validation-select.py) 根据两份 Git 快照选择 owner、直接消费者、公开
闭包、逐头和交付门，并机械拒绝未经 `AUDITED` 授权的产品提交。文档、状态、格式和测试维护不启动
产品全量构建或 P1；P1 只接受 closure card 明确绑定的不可变候选。

PR/main CI 运行 Linux change-impact 合同。Windows、macOS、AArch64 cross、完整 source/package 和可选
feature 矩阵由显式 full workflow 运行；sanitizer、真实 GPU 与 SVE/RVV 执行仍按模块风险和机器合同
调度，不能由一次 all-components compile 冒充。

横向 CLOSED 资格与发布 smoke 不使用手写产品名单，也不把仍为 `QUEUED` 的产品混入 verdict；它从
中央 factor/product 图与唯一 lifecycle registry 投影完整依赖闭包均为 `CLOSED` 的公开产品，并机械
验证这些产品的 factor union 没有漏掉任何 CLOSED owner。一般开发者仍可显式选择 `all` 构建当前整个
源码树，但这种构建不构成 CLOSED 产品资格证据。

每个 session 使用独占绝对 build/scratch 目录，不复用另一会话或另一工具链的 CMake cache。产品交付
使用 selected-component smoke 同时验证 source/static/shared 及 relocated consumer；factor 维护只跑
自己的 target、适用消费者和 card 预声明风险。

## 新 factor / product 接入

1. 先按能力和质因数方法确定 semantic owner；在 registry 以 `QUEUED` 登记 identity。
2. 在中央 manifest 追加 factor、真实边和必要 split headers；若用户确需新的独立问题域，再单独登记
   product façade。
3. 用 `hyle_add_module()` 列本地源码/外部依赖，增加 factor test；需要时才增加 benchmark。
4. 外部源码先进入 dependency lock、许可证和唯一 owner。
5. 增加自然 C++ source consumer 与适用 static/shared relocation smoke。
6. 按 closure card 验证图、headers、产品形状、离线/feature 风险；构建通过不等于模块关闭。

维护者不得因为新 factor 改写旧 identity 顺序、复制依赖表、向普通用户公开所有 factor，或顺手扩大
CI/P1。当前限制与活动状态只写 registry/record，不在本文追加阶段性“已完成列表”。
