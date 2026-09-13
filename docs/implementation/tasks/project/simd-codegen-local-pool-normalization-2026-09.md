# SIMD codegen：AppleClang 本地常量池标签归一化

## 触发

SIMD v7 的同一不可变候选在两台 Apple Silicon 机器完成相同 NEON
correctness 回放。Mac mini 的 provider codegen 门通过；MacBook Air 使用的
AppleClang 21.0.0.21000334 把 direct/routed 两个相同函数中的常量池分别命名为
`LCPI26_0` 与 `LCPI27_0`，导致旧 checker 把函数局部序号误判为产品差异。
两边均为 202 条有效指令，mnemonic、operand shape、call、stack reference 与
relocation 类型完全相同。

这是公共验证工具缺陷，不是 SIMD 产品 reopen trigger。修复不得修改候选产品、
性能合同或性能样本。

## 有限修复

- 只对 relocation target 中精确匹配的 Mach-O/AppleClang 本地
  `LCPI<函数序号>_<槽号>` 去掉函数序号；
- 保留 relocation 类型与槽号，外部 symbol、branch edge 和其他 relocation
  identity 不变；
- 用 parser 单元测试证明函数序号不参与比较，而槽号、relocation 类型和外部
  symbol 仍参与比较；
- 在原 MacBook Air object 上只重放 codegen checker，不重建候选、不重跑 P1。

## 验证

结果为 `PASS`：

- Windows 本地 parser regression：`3/3 PASS`；
- 原 MacBook Air Release/NEON object 不重建，修正后的 checker 报告
  `15 direct/routed pairs` 全部零 route instruction，另有 `6` 个 bounded
  peer comparison；
- Mac mini 的旧 checker 已在同一候选上通过，说明修复没有掩盖一条共同的
  产品差异；
- `git diff --check`、closure checker 与 docs checker 通过。

没有产品源码、产品构建参数或性能输入发生变化；产品 candidate 继续是
`c033ca40d3f119912927c52954079d8c01d6ec12`，正式 P1 没有重跑。
