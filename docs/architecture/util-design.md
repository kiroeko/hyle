# util 架构：先分层，再谈“工具箱”

> 本页维护 util 的长期边界。用户选型见 [`../guide/util.md`](../guide/util.md)，C/FFI
> 映射见 [`../guide/c-util.md`](../guide/c-util.md)，性能证据只见
> [`../performance/benchmarks-util.md`](../performance/benchmarks-util.md)。

`util` 不是一个可以随手塞 helper 的杂物目录。它有两个依赖层，必须分开冻结：

```text
P0 util-core headers（本轮审查单元）
  ScopeGuard / Ref / Rng / type_name / SpinLock + cpu_relax
       │ 只借 C++ 标准库与 core 宏头；无 HYLE binary target
       ├────────► mem / sched / graph 直接 include
       └────────► 普通 C++ 用户

P4 util-runtime（container 收口后再审）
  demangle + Registry + type-erased C adapter
       │ Registry 公开形状依赖 container::string_ref / SwissTable
       └────────► hyle_util binary + include/hyle/c/util.h
```

`Registry` 虽然是模板头，也不是 util-core：它直接 include container，并使编译出的
`hyle_util` 依赖 container/mem。反过来，container 不依赖 util-core，也不链接
`hyle_util`；这正是避免 `container ↔ util` 环的 seam。`demangle` 自身不需要 container，
但当前与 Registry/C adapter 同属一个 runtime target；是否拆 target 留完整 util 审查裁决，
不能在 container 形状未冻结时靠 target 数量先拍答案。

## util-core 能力与成熟形状

| 能力 | 正常场景 | 形状与参考 | 明确负空间 |
|---|---|---|---|
| `ScopeGuard` / `HYLE_DEFER` | 临时 rollback、C API cleanup、多个早退路径 | Abseil Cleanup / Folly ScopeGuard：栈内、move-only、一次执行、可 dismiss/invoke | 优先写拥有型 RAII；不提供异常穿越析构 |
| `RefCounted<T>` + `Ref<T>` | 对象内能承担计数、需要单指针侵入式共享所有权或定制 destroy | Boost/Folly intrusive ownership：relaxed incref，release + final acquire；支持 const、派生→基类转换/赋值 | 无 weak ref；有环/观察者语义选 `shared_ptr/weak_ptr` 或领域句柄 |
| `Rng` | 仿真、采样、测试、负载均衡等快速可复现伪随机 | 与 Folly 默认同族的 xoshiro256++；满足标准 URBG，Lemire 无偏整数 range，jump 拆流 | 不是 CSPRNG；不提供 token/key/nonce 安全性 |
| `type_name/hash` | 编译期诊断、进程内类型标签 | Boost.Core/ctti 的 pretty-function probe；全 constexpr | 不可持久化，不是跨编译器/plugin/wire ID |
| `SpinLock` / `cpu_relax` | 几个指令、低争用且不能睡眠的临界区/自旋循环 | oneTBB 紧凑 TTAS + 指数 backoff；Folly 同样把 compact lock 与显式 padding 分开 | 不公平、非 fiber-aware；分配/rehash/I/O/高争用必须用 mutex/调度原语 |

## 复杂度与布局合同

- `ScopeGuard` 构造/销毁、`Ref` copy/drop、`SpinLock` 无争用、`Rng next_*` 都是 O(1)；
  `Rng::discard(n)` 是 O(n)，固定大跨度拆流用 O(256) 的 `jump/long_jump`。
- `Rng::shuffle(n)` 是 Fisher–Yates O(n)；bounded integer 期望 O(1)，拒绝采样保证无偏。
- `type_name/hash` 在常量求值完成；hash 是 O(编译器类型名长度)，不属于运行期热循环。
- 普通 `SpinLock` 必须保持紧凑。若某个 owner 用 profile 证明伪共享，由 owner 显式
  `alignas(core::kDestructiveInterferenceSize)`，不能让所有嵌入对象无条件多付 64/128 bytes。
- `Ref<T>` 保持一个指针；计数位于对象中。它的产品价值是所有权/布局，不承诺每个平台
  的原子 copy 都快过 `shared_ptr`。

## 依赖和采用闸门

当前直接生产消费者只有 `SpinLock/cpu_relax`：mem 的 FiberStackPool、sched 的等待/调度
细节和 graph 的首错槽。审查时必须看临界区实际工作，
不能因为类型名叫 SpinLock 就默认正确。mem LeakTracker 曾把可能 allocate/rehash 的
`unordered_map` 操作放在 SpinLock 下；container ConcurrentVector 又会在写锁中分配 segment
并运行任意用户构造函数。本轮两处均改为 `std::mutex`，container 因而不再反向 include
util-core。sched/graph 消费者随各自模块收口继续裁决。

Rng/Ref/ScopeGuard/type-name 当前主要是用户能力。没有内部消费者不等于应删除，也不能为了
提高“采用率”硬塞进不匹配路径；它们分别对应成熟库长期提供的独立基础能力。

## C ABI 边界

util-core 本身是同工具链 C++ header seam，没有独立 binary ABI。现有 `hyle/c/util.h` 属于
util-runtime：RNG 以 POD state+batch 映射，Ref/ScopeGuard/SpinLock/Registry 以 opaque
handle+callback 擦除，type-name/hash 有意仅 C++。该 C 面仍使用历史私有
`hyle_util_last_error`，必须在 container→完整 util 波次迁到共享 error/`hyle_status` 后才可
取得严格证书；本轮不得把“纯 C smoke 能编”误写成 C ABI 已冻结。

## 维护闸门

1. 新 helper 先证明至少一个真实用户场景，再判断归 util-core、已有模块还是不做。
2. util-core 公开头必须逐头 self-contained 编译；禁止 include container/platform runtime 头。
3. RNG 算法/状态布局变化必须更新固定序列；默认线程种子变化不得改变显式 seed 的可复现流。
4. Ref/SpinLock 并发修改过 TSan；布局修改在三工具链 `sizeof` 和下游 owner 上复验。
5. benchmark 只验证同合同实现与适用边界，不反向决定 API，也不以高争用 spin 获胜为目标。
