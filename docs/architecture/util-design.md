# util 架构：小而杂，但边界必须清楚

`util` 不是一个算法大包，而是七类彼此独立、被多模块复用或直接提供给用户的基础能力。维护时先按能力族定位，不要把所有东西塞进同一个“工具箱”抽象。

```text
                         HYLE util
┌──────────────────────────────────────────────────────────────────────┐
│  compile-time / header-only                                          │
│                                                                      │
│  RefCounted<T> ── Ref<T>       ScopeGuard<Fn>       type_name<T>     │
│       │                Registry<Ptr, Args...>         type_hash<T>    │
│       │                       │                                      │
│  SpinLock + cpu_relax         │               Rng (xoshiro256++)     │
│       │                       │                                      │
├───────┼───────────────────────┼──────────────────────────────────────┤
│  compiled runtime                                                    │
│       │            demangle()            C ABI type-erased bridge    │
│       │                                      (c_util.cc)             │
└───────┼───────────────────────────────────────────┬──────────────────┘
        │                                           │
        ▼                                           ▼
 mem / container / sched / graph             include/hyle/c/util.h
 native C++ consumers                        C / FFI consumers
```

## 两条用户线路

```text
C++ 同工具链/源码用户
  include/hyle/util/*.h
  ├─ 模板、RAII、编译期类型信息
  └─ 热路径全部 inline

C / 跨编译器 / FFI 用户
  include/hyle/c/util.h
  ├─ POD: RNG state
  ├─ opaque: Ref / ScopeGuard / Registry / SpinLock
  ├─ callback + ctx: destroy / cleanup / factory / visitor / shuffle
  └─ buffer-fill: demangle / registry help
```

C ABI 不是逐个照抄 C++ 类型，而是按 `c-abi-design.md` 做能力擦除：

| C++ 能力 | C ABI 形状 |
|---|---|
| `Rng` | 4×`uint64_t` POD；可复制、可持久化、可 jump |
| `Ref<T>` | `void*` payload + destroy callback 的 opaque strong handle |
| `ScopeGuard<Fn>` | callback+ctx 的 opaque guard |
| `Registry<Ptr, Args...>` | key/help 自有副本 + factory(void*, args) |
| `SpinLock` | opaque lock；`cpu_relax` 留 header-inline |
| `demangle` | snprintf 式 buffer-fill |
| `type_name<T>` / `type_hash<T>` | 有意仅 C++；其能力依赖编译器静态类型替换，运行时 `void*` 无法恢复 |

## 依赖和性能边界

- `Registry` 的表依赖 `container::SwissTable`；其注册、查询和创建选择均受 mutex 保护。creator 在解锁后调用，允许重入；visitor 在锁内调用，禁止重入同一 registry。
- `SpinLock` 被 mem、container、sched、graph 消费，只适合极短、低争用临界区。它不是 fiber-aware，也不承诺在高争用下优于 OS mutex。
- `Ref<T>` 是侵入式共享所有权和 ABI 形状选择，不承诺每个编译器都比 `shared_ptr` 拷贝更快。
- `Rng` 非密码学。可复现并行流用确定 seed，再按 worker `jump()`；分组可先 `long_jump()`。
- `ScopeGuard` 的 cleanup 必须不抛异常；C callback 同样不得让异常穿过 C 边界。

## 维护闸门

1. 公开 C++ 面新增能力时，按 `c-abi-authoring-guide.md` 重新判断是否需要 POD、opaque 或 callback 擦除。
2. C 头每个 `hyle_util_*` 入口必须在纯 C smoke 和 `c-util.md` 中出现。
3. RNG 算法或状态布局变化必须更新固定序列测试；这是可复现契约。
4. Ref/Registry/SpinLock 的并发改动必须过 TSan；内存和 callback 生命周期改动必须过 ASan/UBSan。
5. 性能结论分平台写。尤其 SpinLock 和引用计数的代码生成差异很大，禁止拿单平台数字作无条件结论。
