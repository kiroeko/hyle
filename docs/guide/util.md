# util：什么情况选什么

`util` 提供几种彼此独立的基础能力，不存在一个需要整体 include 的“万能工具箱”。C++ 用户
按场景包含 `hyle/util/<capability>.h`；只有 `demangle`、Registry 和 C/FFI 面需要链接
`hyle::util`。纯 C 的精确映射见 [`c-util.md`](./c-util.md)。

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
`HYLE_DEFER(...)`，按声明逆序执行。能写成拥有型 RAII 类时，仍优先 RAII 类。

## Ref：侵入式共享所有权

```cpp
#include "hyle/util/ref_counted.h"

struct Buffer : hyle::util::RefCounted<Buffer> {
  explicit Buffer(std::size_t size) : size(size) {}
  std::size_t size;
};

auto writable = hyle::util::make_ref<Buffer>(4096);
hyle::util::Ref<const Buffer> read_only = writable;
```

新对象从计数 1 开始：通常用 `make_ref<T>()`。已有“调用方拥有的 +1”用 `adopt_ref(ptr)`；
只有借用裸指针并要新增一份所有权时才用 `retain_ref(ptr)`。三者不能凭感觉互换。

`Ref<T>` 只有一个指针，没有独立 control block；计数在对象里。若对象不是普通 `delete` 释放，
在派生类实现 `destroy()` 并允许 `RefCounted<Derived>` 调用。`use_count/is_unique` 是并发快照，
不能拿来证明“接下来仍唯一”。需要 weak reference、环检测、aliasing constructor 或对象不能
嵌计数时，使用 `std::shared_ptr/weak_ptr` 或更贴领域的 handle。

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

## 当前分层状态

本页的 header-only util-core 已按当前标准严格收口。Registry 与完整 C ABI 依赖尚未严格
收口的 container，将在 container 之后统一复审；现有接口可测试使用，但不能据此推断其
发布 ABI 已经冻结。这个分层避免低层 mem/container 为一个运行时 Registry 反向链接
`hyle_util`。
