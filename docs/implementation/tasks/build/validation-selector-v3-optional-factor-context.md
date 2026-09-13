# Validation selector v3：可选 compiled factor 的验证上下文

<!-- PROJECT UNIT REVIEW: CLOSED -->

## 问题与边界

SIMD v7 的公开 API 变更依法选中了直接/传递 compiled consumer
`graph_provider_onednn`，但 `HYLE_DEV_FACTORS` 只裁剪组件图，不负责改变默认关闭的
`HYLE_COMPUTE_ONEDNN` feature。结果是空的 `OPTIONAL_ABSENT` target 可以构建，而同名
CTest 没有注册；选择器随后以 `--no-tests=error` 正确拒绝，候选 P1 尚未启动。

本任务只让显式选中的可选 factor 获得其验证所需 CMake cache。映射保存在
`scripts/validation-policy.json`，不复制组件依赖图，不改变普通 configure 的 feature 默认值，
也不把“没有测试”放宽成 PASS。首个登记是两个 oneDNN factor 对
`HYLE_COMPUTE_ONEDNN=ON` 的共同需求。

## 实现与验证

- 选择器合并全部已选 factor 的 cache 需求，并拒绝非法 key、空值和冲突值；
- 同一命令、策略和候选仍进入 gate identity，因此 feature 上下文不能复用错误 receipt；
- fixture 证明 oneDNN factor 会激活 feature，普通 factor 不受影响；
- 同轮校准 Compute/Graph 拆分后已经失真的反向依赖 fixture，并移除不证明所有权正确性的脆弱
  factor/product 数量断言；逐文件 owner 检查继续承担中央图覆盖合同；
- SIMD 产品候选、公共 API、算法、性能合同和用户默认构建语义均未修改。

该修复属于独立 build/tooling 维护事实；它解除 SIMD v7 前置验证阻塞，不构成额外 P1
尝试，也不授权扩展选择器或组件治理范围。
