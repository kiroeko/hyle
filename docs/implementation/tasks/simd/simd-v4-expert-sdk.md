# SIMD v4：统一 expert provider SDK

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只修复 Highway/xsimd expert 能力在三种 C++ 交付之间断裂的问题。上一轮
> [`simd-v3`](simd-v3-provider-routing.md) 的 ordinary API、387-operation 迁移账本、422 个 route scope、
> 实现选择和 P1 性能结论不在本轮重做。

## Reopen fact

- trigger ID：`SIMD-V4-DELIVERY-01`；
- baseline：`38dd0be938a19d7989e80daeb859139ebad14e52`；
- 可复现事实：源码消费者可经 `hyle_target_simd()` / `hyle_add_fat_kernel()` 使用锁定 Highway/xsimd，
  但 installed static/shared SDK 明确不导出 expert target、helper 或 vendor headers。需要 HYLE ordinary
  API 尚未包装的 provider 能力时，三种受支持交付没有同一条用户路线；
- 击中的旧结论：公开交付形状发生变化，且真实 owner 明确要求 static/shared C++ 用户获得与源码用户
  相同的 provider-header 开发能力；这满足模块方法的 delivery/API-usability reopen 条件；
- 受影响范围：只包括 expert header capability 的 CMake target、fixed/fat helper、锁定第三方开发资产、
  package relocation/offline、示例和文档。

## Closure card

- component / owner / round / kind：`simd` / SIMD + build/package / `simd-v4-expert-sdk` / `REOPEN`；
- 真实 consumer task：C++23 用户在 HYLE 没有 ordinary 语义 API 的场景中编写一个 pointer/POD kernel，
  直接使用锁定 Highway 或 xsimd；同一 CMakeLists 必须能消费源码、installed static 或 installed shared
  HYLE，并能选择固定 ISA 或生成 HYLE 管理的 fat kernel；
- 范围：三种交付共同的 `hyle::simd_expert` build target；安装时自包含的精确 Highway/xsimd header
  资产；可搬迁的 `hyle_target_simd()` / `hyle_add_fat_kernel()`；selected-component、离线、许可证和
  source/static/shared consumer 验证；architecture/user/build 文档；
- 非目标：把 Highway/xsimd 全部函数改名为 `hyle::simd::*`；改变 ordinary SIMD API、387/422 route、
  kernel 算法、运行时 dispatch 或性能；承载必须编译上游 `.cc` 的 thread pool/VQSort/image/hash 产品；
  兼容 `hyle::simd_source_expert` 旧拼写；Native FFI/C API、稳定跨工具链 ABI、GPU/tensor/模型策略；
- 直接依赖：`core`、`platform`，以及中央锁定 Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c`
  与 xsimd `3b3151ff6655967c4d783d0534dde68fbb7b258f`；
- 直接消费者：独立 expert kernel fixture；普通 `hyle::simd` 用户不自动继承 vendor include/define；
- tier：`P0 / closure-correctness-v1`。本轮不改变运行时代码，性能要求是 fixed/exact 最终编译仍为直接
  provider code、fat path 仍只有既有 batch-boundary selection，不增加运行时 wrapper、allocation、lock
  或 metadata；timing comparison `NOT_APPLICABLE`；
- 总预算：90 分钟；实现 45 分钟、验证 35 分钟、红队与记录 10 分钟；不新增性能实验；最多一次产品
  修复重跑；
- machines：`W1-WIN` 为本地完整门，`W1-WSL` 与 `M1-MAC` 验证可用时运行相同的 source/static/shared
  expert consumer；不可达平台如实 `BLOCKED`，不能写成 `PASS`；
- strong peers：锁定 Highway/xsimd 本身；本轮只借用其原生 header API，不建立第三套 register peer；
- correctness/delivery gates：固定 ISA 编译与运行、fat kernel 编译与运行、vendor 类型不进入 HYLE
  binary ABI、source/static/shared、relocation、selected component、offline cache、许可证、负向缺资产；
- 预期输出：三种交付使用同一 target/helper/示例；ordinary 与 expert 成本隔离；完整 owner/升级边界；
- 停止条件：三种 journey 全通过且没有运行时产品漂移则关闭；若可搬迁 installed helper 无法在预算内
  重建准确 ISA/fat contract，进入 `BLOCKED`，不退回 source-only 假完整。

## 当前状态

`REOPENED`。下一合法状态是 `SCOPED`；冻结卡经状态检查后再执行 capability audit，产品代码尚未修改。

