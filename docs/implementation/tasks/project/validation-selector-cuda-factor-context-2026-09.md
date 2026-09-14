# Validation selector：CUDA compiled factor 的验证上下文

<!-- PROJECT UNIT REVIEW: CLOSED -->

## 问题与边界

Compute v7 的公开 API 变更依法把已关闭的 `compute_backend_cuda` 纳入反向消费者闭包，但
`HYLE_DEV_FACTORS` 只选择组件，不改变默认关闭的 `HYLE_COMPUTE_CUDA` feature。选择器因此构造了
`OPTIONAL_ABSENT` 空 target，随后由 `ctest --no-tests=error` 正确拒绝；这不是 CUDA 产品测试失败。

本任务只为显式选中的 CUDA compiled factor 增加既有
`factor_configure_cache` 投影：`compute_backend_cuda -> HYLE_COMPUTE_CUDA=ON`。CUDA Toolkit 继续由用户
安装并由 CMake `find_package` 查找；普通 Compute/CPU 配置仍不探测 CUDA，选择 CUDA factor 而机器缺少
合格 SDK 时仍须明确失败。任务不修改 Compute/CUDA/Mem 产品、模块 registry、普通 feature 默认值、
性能合同或候选算法。

## Closure card

- unit / owner / kind：`build-package` / HYLE maintainers / `TOOL`；不进入产品 registry；
- baseline：`c968769d3814f27e047757b8f732d1bb0db3826d`；
- trigger：不可变 Compute v7 候选 `18970739c5e0909cf968fecba9b96c9ea3560dfc` 的 W1-WIN selector
  receipt 在 `factor:compute_backend_cuda` 发现零测试，而同机 CUDA 13.3 backend 已曾合法关闭；
- finite FIX：登记唯一 cache 映射；增加“由 Compute 公共头反向选中 CUDA”及“无关 factor 不启用
  CUDA”的工具回归；用当前工具从同一不可变产品候选重跑所选闭包；
- 验证：`tests/tooling/validation_selector_test.py`、不可变候选 selector、
  `python scripts/check-module-closure.py`、`python scripts/check-docs.py`、`git diff --check`；
- stop：若需要改变 CUDA 产品、普通默认值、组件图或跳过缺失测试，本任务 `BLOCKED`；否则上述有限
  FIX 与真实候选验证通过即 `CLOSED`。

## 结果

`compute_backend_cuda -> HYLE_COMPUTE_CUDA=ON` 已作为唯一 `factor_configure_cache` 投影登记；Compute
公开边界变更会自动把该参数传给反向选中的 CUDA factor，无关 factor 不会得到它。工具回归 44/44
通过。

不可变产品候选 `18970739c5e0909cf968fecba9b96c9ea3560dfc` 已在同一 Windows/MSVC + CUDA
13.3 上由当前选择器回放：CUDA configure、build 和真实 factor test 通过。该次完整闭包随后在
`graph_provider_onednn` 的旧 fallback 测试判据停止；这不是 CUDA gate 失败，旧判据已由独立 Graph
维护提交 `09cb5d1a` 修正。普通 Compute/CPU 配置仍不探测 CUDA，机器缺少 Toolkit 时也没有降级假绿。

本任务没有修改 CUDA/Compute 产品、feature 默认值或模块 registry，有限 FIX 全部完成，项目工具单元
`CLOSED`。
