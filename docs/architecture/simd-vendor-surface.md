# SIMD vendor ownership

本文只描述长期 ownership 和升级原则，不承担操作覆盖率、性能排名或任务状态。当前 product state 见
[`module-status.json`](../implementation/module-status.json)，本轮版本与施工事实见
[`SIMD/Math epic`](../implementation/tasks/simd/simd-epic-v1.md)。

## 两种依赖必须分开说

HYLE 的公开 SIMD 头不泄漏 Highway/xsimd 类型，正常产品构建也不从外部 checkout link 这两个库；但这
不等于产品实现“零第三方”。部分 checked-in kernel 是带来源、修改和许可证的 integrated-source port：

| origin | HYLE product paths | ownership |
|---|---|---|
| Highway | `src/simd/math/math_hyle-inl.h`、Highway-named transcendental TUs、`src/simd/bitpack/**` 的 port、`src/simd/sort/vqsort_hyle/**` 与 target TUs | HYLE owns public semantics/selection/delivery；Highway is mechanism origin；HYLE owns adaptation/local delta |
| xsimd | `src/simd/math/math_xsimd_hyle-inl.h`、shared body 与 xsimd-named target TUs | HYLE owns public accuracy/API/delivery；xsimd is mechanism origin；HYLE owns D-tag adaptation/local delta |
| simdutf | container UTF-8 implementation，不属于 `hyle::simd` target | container owns semantics/delivery；vendored simdutf owns implementation mechanism |

中央 [`HyleDependencyLock.json`](../../cmake/HyleDependencyLock.json) 是 integrated origin、archive、内容 hash
和许可证投影的唯一机器账本。文件名或 namespace 已改为 HYLE 并不会抹掉第三方来源。

## 公共边界

- HYLE owns `types/vec`、Runtime/exact/native/fat-kernel、pointer/span batch API、accuracy/layout/alias/error
  contract 与 Platform/build/package integration。
- vendor 类型、target macro 和 allocator 不进入 HYLE 公开签名或持久化格式。
- internal direct wrapper 可以使用成熟算法，只要不改变公开合同，并且 source/static/shared 与 exact/native
  交付完整。
- integrated port 只有在 HYLE 合同无法由稳定 wrapper 满足时保留；每次升级都要重新记录 upstream
  identity、local delta、许可证和内容 hash。
- HYLE 自实现只因真实 semantic/ISA/complexity/consumer gap 存在，不因 benchmark 某行获胜而复制。

## 升级规则

1. 先冻结官方 commit，不追逐浮动 HEAD；同一 capability 最多两个 strong peer。
2. 先比较语义、accuracy、layout、alias、异常与 ISA，再比较性能。
3. 只验证受影响 capability：广 correctness + 少量正常 consumer sentinel；不重跑全部操作笛卡尔积。
4. direct wrapper 可行时优先减少长期 port delta；若 port 必须存在，只移植与当前 origin family 有关的
   upstream 修复。
5. 任何不同 wire format、稳定性、NaN、dispatch 或资源合同的结果都标记 `NOT_COMPARABLE`。
6. 内容确定后由 CMake integrated-source hash 工具机械重算，禁止手写一个“看起来更新”的 hash。

## 明确负空间

Benchmark checkout 是 peer，不是安装产品依赖；Native FFI、稳定跨工具链 C++ ABI、per-ISA DLL、tensor/
model operator 和 GPU backend 不因 vendor 升级进入 SIMD。阶段性“哪个 op 曾赢”的记录只留在
implementation/evidence，不进入本架构页。
