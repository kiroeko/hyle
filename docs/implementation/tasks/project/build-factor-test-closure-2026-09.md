# Build/package：exact-factor 测试依赖闭包

<!-- PROJECT UNIT REVIEW: AUDITED -->

## Closure card

- unit / owner / kind：`build-package` / HYLE maintainers / `BUILD`；不进入产品模块 registry；
- baseline：`09cb5d1a`；
- trigger：`HYLE_DEV_FACTORS=graph_provider_onednn` + `HYLE_BUILD_TESTS=ON` 虽能构造产品 target，却因
  测试所需的 CPU backend/Native fallback target 未被选择，只注册 header test；模块 aggregate 是空 target，
  `ctest --no-tests=error -L ^graph_provider_onednn$ -LE headers` 可复现失败。相同风险存在于 exact
  `compute_provider_onednn` 的 CPU 执行测试；
- 正常任务：维护者只选择一个 factor，configure 应自动取得该 factor 测试明确需要、但产品本身不应依赖的
  夹具 factor，并仍只构建/运行 owner aggregate；`HYLE_BUILD_TESTS=OFF` 的产品闭包、安装和链接边不变；
- scope：在唯一中央 `HyleComponents.cmake` 增加 `TEST_DEPS` 元数据；测试开启时把它及其正常产品依赖纳入
  resolved build closure；为两个 oneDNN factor 登记精确测试依赖；文档说明产品依赖与测试夹具依赖差异；
- non-goals：把 CPU/Native 伪造成 oneDNN provider 产品依赖；改变 provider API/算法；增加 selector 私有
  factor-root 表；运行全库产品或性能测试；
- finite FIX：`BTD-01` 声明/校验 `TEST_DEPS`；`BTD-02` 测试开启时有界闭包，关闭时零影响；
  `BTD-03` two exact-factor configure/build/CTest；`BTD-04` 中央组件合同和三交付选择语义回归；
- performance：仅 configure-time O(components + edges)；产品运行时、artifact 与公开 API 零变化；
- stop：若需要产品链接边、第二份组件真相或普通构建默认增加 factor，则 `BLOCKED`；否则四项完成即
  `CLOSED`。

## AUDITED 结论

测试代码已经通过 `hyle_add_test(... DEPS ...)` 声明真实 target link，但产品 DAG 有意不包含这些测试
夹具。合理形状是在同一中央 component declaration 上增加正交 `TEST_DEPS`，只参与 test-enabled 的选择
闭包，不进入 `_HYLE_DEPS_*`、真实 module link verifier、安装 metadata 或 selector 的产品反向依赖图。
新增测试 factor 自身的普通产品依赖仍由既有拓扑闭包取得；固定点最多选择有限组件，不需要第二套 target
构造顺序。

## 结果

待实现和有限验证后填写；当前为 `AUDITED`。
