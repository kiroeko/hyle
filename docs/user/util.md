# util：什么情况选什么

> **当前边界（2026-08-27）：**本页前半的 `util_core` 五项能力已完成 v1 收口；后半的
> Registry/demangle 是尚待重审的 `util` runtime 既有使用合同。util 当前只提供 C++ 产品面，
> 适用于 C++23 source/static/shared 三种交付，不描述 C API 或跨语言调用层。

`util` 提供几种彼此独立的基础能力，不存在一个需要整体 include 的“万能工具箱”。C++ 用户
按场景包含 `hyle/util/<capability>.h`，并链接可安装的 header-only target
`hyle::util_core`；只有 `demangle`、Registry 等 runtime 能力需要改为链接 `hyle::util`。

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS util_core)
target_link_libraries(my_target PRIVATE hyle::util_core)
```

源码嵌入时同样设置 `HYLE_COMPONENTS=util_core` 后链接该 target；不要靠手写 include path
绕过组件与安装包合同。

## 先看选择表

| 需求 | 首选 | 不要这样用 |
|---|---|---|
| 已经有明确资源类型 | 该类型自己的 RAII / `unique_ptr` | 为了少写一行给每个对象套 ScopeGuard |
| 临时 rollback、多个早退共享一段 cleanup | `ScopeGuard` 或 `HYLE_DEFER` | cleanup 抛异常；guard 捕获比自己活得短的对象 |
| 单指针、对象内计数的共享所有权 | `RefCounted<T>` + `Ref<T>` | 需要 weak observer/打破环时仍强行用 Ref |
| 可复现仿真、采样、测试、shuffle | 显式 seed 的 `Rng` | 用于 token、key、nonce 或任何安全材料 |
| 每线程普通非安全随机流 | `thread_rng()` | 保存其地址跨线程使用；依赖默认流可复现 |
| 极短、低争用临界区 | `SpinLock` + `std::lock_guard` | 锁内分配、rehash、I/O、等待或在 fiber 中长期持有 |
| 编译期静态类型名/进程内标签 | `type_name<T>()` / `type_hash<T>()` | 存数据库、写 wire format、跨编译器/plugin 比较 |
| 运行期 mangled symbol | `demangle()` | 把它当热路径或假定绝不发生内部临时分配 |
| 启动期注册、factory 不会失败或 NULL 已是完整语义 | `Registry<T*, Args...>` | 需要诊断时仍让 NULL 同时表示 missing、初始化失败和成功空值 |
| 启动期注册、backend 初始化可能失败 | `Registry<Result<T*>, Args...>` | 用旁路 thread-local/ctx 保存 error，或丢掉设备/驱动失败原因 |
| Registry 初始化完成后的高频 dispatch | `auto view = registry.freeze()`，缓存 `FrozenView` | 每次 dispatch 后继续注册，或让 view 活过 registry |

## ScopeGuard：只处理临时控制流

```cpp
#include "hyle/util/scope_guard.h"

FILE* file = std::fopen(path, "rb");
if (file == nullptr) return Status::NotFound("open failed");
hyle::util::ScopeGuard close{[&]() noexcept { std::fclose(file); }};

// 提前完成同一 cleanup；析构不会再执行。
close.invoke();
```

`dismiss()` 取消动作，`invoke()` 立即执行且至多一次，`active()` 只用于本地控制流。guard 是
`[[nodiscard]]`：必须绑定名字；临时对象会在当前语句末尾立刻执行。多个简单 cleanup 可用
`HYLE_DEFER(...)`，按声明逆序执行。能写成拥有型 RAII 类时，仍优先 RAII 类。显式
`invoke()` 会先消费 guard，再把 callback 异常传给调用者；析构期间 callback 抛异常则按
ScopeGuard/Folly/Abseil 同类合同终止进程，因此析构 cleanup 通常应写成 `noexcept`。

构造 guard 时，若存储 callable 失败，原始 callable 会立即执行，已经取得的资源不会因构造异常
泄漏。转移一个 callable move 可能抛异常的 guard 时，若 callable 可复制则使用 copy fallback；
只有目标构造成功后源 guard 才失效，失败时 cleanup 仍由源对象持有。不可复制且 move 可能抛异常
的 callable 会在编译期被拒绝；不要用“先 dismiss 再 move”绕过 ownership 合同。

## Ref：侵入式共享所有权

```cpp
#include "hyle/util/ref_counted.h"

struct Buffer : hyle::util::RefCounted<Buffer> {
  explicit Buffer(std::size_t size) : size(size) {}
  std::size_t size;
};

auto writable = hyle::util::make_ref<Buffer>(4096);
hyle::util::Ref<const Buffer> read_only = writable;
auto base = hyle::util::static_pointer_cast<const Buffer>(writable);
```

新对象从计数 1 开始：通常用 `make_ref<T>()`。已有“调用方拥有的 +1”用 `adopt_ref(ptr)`；
只有借用裸指针并要新增一份所有权时才用 `retain_ref(ptr)`。三者不能凭感觉互换。

`Ref<T>` 只有一个指针，没有独立 control block；计数在对象里。若对象不是普通 `delete` 释放，
在派生类实现 `destroy()` 并允许 `RefCounted<Derived>` 调用。`use_count/is_unique` 是并发快照，
不能拿来证明“接下来仍唯一”。需要 weak reference、环检测、aliasing constructor 或对象不能
嵌计数时，使用 `std::shared_ptr/weak_ptr` 或更贴领域的 handle。

`Ref<Base>`/`Ref<Derived>` 可直接按对象地址比较、排序和放入 `std::unordered_*`；排序使用标准
指针全序，即使两个对象无关也适合关联容器，不会为了比较临时 retain/release。
`static_pointer_cast`、`dynamic_pointer_cast`、`const_pointer_cast`
均有 lvalue/rvalue 形状：成功的 rvalue cast 转移现有 ownership，不做额外原子往返；失败的
dynamic rvalue cast 保留源引用，避免把一次探测变成意外释放。

普通用户应让 `Ref` 自己配对单份 retain/release，不直接手调 `incref()/decref()`。单份热路与
Boost/Folly 一样以前置条件换取最短原子指令：调用者已持有活引用且计数不会溢出。确实需要
一次转移多份引用时才用带数量的 overload；它会检查零、上溢和下溢。

## Rng：快、可复现、非安全

```cpp
#include <algorithm>
#include <random>
#include "hyle/util/random.h"

hyle::util::Rng rng{12345};              // 显式 seed：可复现
auto id = rng.range_u64(100, 999);        // 无偏闭区间
std::shuffle(items.begin(), items.end(), rng); // 标准 URBG，可直接组合
// 热路径 shuffle 直接走 HYLE 的 Lemire bounded draw：
rng.shuffle(items.begin(), items.end());
std::normal_distribution<double> normal;
double sample = normal(rng);
```

`next_float/next_double` 返回 `[0,1)`；整数 `range_u32/u64(lo, hi)` 是闭区间；浮点
`range_double(lo, hi)` 是半开区间。`discard(n)` 逐项前进，复杂度 O(n)。并行可复现流应显式
seed，然后用 `jump()`（2^128）或 `long_jump()`（2^192）拆分；不要让 worker 共享同一个 Rng。

`thread_rng()` 每线程首次使用时从系统熵、进程 salt 和唯一序列播种，之后无锁；它不承诺
复现，也仍然不是 CSPRNG。安全随机数必须使用 OS/vetted crypto 提供的 secure-random API。

## SpinLock：默认紧凑，适用面很窄

```cpp
#include <mutex>
#include "hyle/util/spin_lock.h"

hyle::util::SpinLock lock;
{
  std::lock_guard guard(lock);
  ++tiny_counter; // 几个指令
}
```

它是紧凑 TTAS 锁，争用慢路指数 pause 后让出线程。紧凑是普通对象的正确默认；只有 profile
证明相邻成员伪共享时，owner 才应显式 cache-line align/pad。SpinLock 不公平、不递归、不是
fiber-aware。临界区可能分配、rehash、调用未知 callback、等待或发生高争用时选
`std::mutex`；fiber 等待选 sched 的同步原语。`is_locked()` 只是诊断快照，不是所有权证明。

`cpu_relax()` 只是一条自旋 hint，不建立同步关系；必须与 atomic 条件配合。

## type_name 与 demangle

`type_name<T>()` 返回编译器静态字符串的 view，`type_hash<T>()` 是其 constexpr FNV-1a。
二者只在同一编译器构建内稳定；编译器版本、flags 或类型拼写变化都可能改变结果。跨模块
持久 ID 应使用显式 schema/UUID，不要用 type hash。

`demangle(mangled, out, cap)` 是 snprintf 风格：返回完整所需长度，`cap>0` 时总是 NUL 终止；
返回值大于等于 cap 表示已截断。它服务日志/backtrace 等冷路径，在 Itanium ABI 下底层
demangler 必然可能临时分配。

## Registry：注册阶段可变，服务阶段冻结

```cpp
#include "hyle/util/registry.h"

using EngineRegistry =
    hyle::util::Registry<hyle::Result<Engine*>, const Config&>;
EngineRegistry engines;

HYLE_RETURN_IF_ERROR(engines.reserve(16));
HYLE_RETURN_IF_ERROR(engines.register_creator(
    "cpu", &make_cpu_engine, hyle::util::RegistryPriority::kFallback,
    "portable CPU engine"));
HYLE_RETURN_IF_ERROR(engines.register_creator(
    "cuda", &make_cuda_engine, hyle::util::RegistryPriority::kPreferred,
    "CUDA engine"));

// 初始化结束：单向、幂等、O(1)、零分配。
auto factories = engines.freeze();
HYLE_ASSIGN_OR_RETURN(Engine * engine, factories.create("cuda", config));
```

Registry 会复制并拥有 key/help；调用者无需保留传入字符串。相同 key 的更高 priority 覆盖，
更低 priority 是成功的 no-op，相同 priority 返回 `AlreadyExists`，因此链接/静态初始化顺序不会
悄悄决定实现。`reserve()` 只用于启动期减少 rehash；注册和 reserve 在 freeze 后返回
`FailedPrecondition`。

返回值应贴合 factory 的真实失败模型。`Registry<T*>` 是 infallible/nullable 形状，missing 与
creator 返回 NULL 都得到 NULL。`Registry<Result<T*>>` 是推荐的 fallible backend 形状：missing
得到 `NotFound`，设备、驱动、配置或资源错误由 creator 原样返回。两者共用同一张表、相同
lookup value 和复杂度；Result 不是额外的 per-entry 元数据。默认 creator 也支持
`Result<Ref<T>>` 等 owning value。

可变阶段的查询受 mutex 保护。creator 和 `for_each` visitor 都在锁外运行，可以重入同一个
Registry；可变阶段枚举会为时间点快照分配一次，冻结阶段则无锁、无分配。`Registry::create()`
在冻结后只多一次生命周期 acquire load，适合普通调用；dispatch 热路径缓存 `FrozenView`，直接
查不可变 Swiss 索引。两种形状都是 O(key length) hash + expected O(1) probes，creator 自身
成本另计。

Registry 不提供 unregister：已经创建对象、函数指针和并发 reader 的代码/数据生命周期无法由
一个名字表安全推断。需要动态装卸 shared library、依赖排序或在途调用排空时，应由 module
manager 管理 quiescence，再销毁整个 Registry。Registry 必须活过所有 `FrozenView` 和在途操作；
析构前由调用方让使用者静止。

## 分层与证据入口

header-only util-core 作为独立 `util_core` 组件交付，当前 v1 收口记录见
[`util-core-v1`](../implementation/tasks/util/util-core-v1-closure.md)；Registry 和 demangle 属于仍为
`QUEUED` 的 `util` runtime component。`1b148ef`/`70fa4e8` 只是在旧代码树上得到的历史输入，
不代表当前 runtime 已恢复绿证；唯一当前状态见
[`module status`](../implementation/status.md)。分层仍然保留：它避免低层 mem/container 为一个
运行时 Registry 反向链接 `hyle_util`。
