# SIMD-17：交付与收口

- ticket state：`ACTIVE`
- kind：candidate / verification / closure
- budget：1 工程日；串行机器 4 小时
- depends on：SIMD-16

冻结不可变 candidate，验证 source embedding、installed static/shared、offline Highway+xsimd、fixed/exact/fat、
三平台适用 correctness、受影响 consumer 与冻结 P1 groups。检查 387/387 ledger 零漏项/零 `PENDING`、公开
C++ 易用性、Native FFI 清洁负空间、architecture/user/developer/implementation 分流、依赖/license、红队、
六问、四项 LOC 与 terminal receipt。未运行的适用 gate 必须是 `BLOCKED/INCONCLUSIVE`，不得写成 `PASS`。

## Candidate 前验证输入

- ledger/provider manifest/generated routes：387 operation、422 scope、零 `PENDING`，机械检查 `PASS`；
- W1-WIN MSVC 19.50、W1-WSL GCC 14.2、M1-MAC AppleClang 21 release：五个 SIMD test artifact
  均为 5/5 `PASS`；
- W1-WSL GCC 14.2 ASan 与 UBSan：均为 5/5 `PASS`。ASan 暴露的 VQSort debug-network threshold
  mismatch 已按 upstream `HWY_IS_DEBUG_BUILD` authority 修复并完成三平台 release 回归，见
  `evidence/simd-17-vqsort-sanitizer-fix.json`；TSan 对本轮纯 compile-time route、无新增共享可变状态的
  改动为 `NOT_APPLICABLE`；
- locked/offline Highway+xsimd 依赖的 positive/negative contract `PASS`，integrated VQSort source hash 为
  `425200b173ce47f74fd51c33857cdc43d5fd6ebce7152b79d7f6a51af3cbce0e`；
- W1-WIN 当前 P1：普通 row/dtype 六个产品 row 的 95% lower bound 全部高于 0.95；Highway 对 xsimd 的
  tanh/GELU 四个 row 全部 material `LOSE`，因此产品按已冻结例外选择 xsimd。收据为
  `evidence/performance/simd-17-w1-win-p1.json`。

上述是冻结 candidate 之前的 finding/fix 输入，不能代替不可变提交回放。下一步只允许：创建 candidate、
回放 source/static/shared 与适用 correctness、完成文档/红队/六问/LOC，并形成 terminal receipt。
