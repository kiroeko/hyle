# util：什么情况选什么

> **当前边界（2026-09-02）：**`util_core` v2 与 `util` runtime 均为 `CLOSED`；当前 Registry lookup/create
> 合同和 demangle owning/buffer 形状的签证状态只看中央
> [`module status`](../implementation/status.md)。
> util 当前只提供 C++ 产品面，
> 适用于 C++23 source/static/shared 三种交付，不描述 C API 或跨语言调用层。

`util` 提供几种彼此独立的基础能力，不存在一个需要整体 include 的“万能工具箱”。C++ 用户
按场景包含 `hyle/util/<capability>.h`，构建时统一选择 Util 产品 façade：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS util)
target_link_libraries(my_target PRIVATE hyle::util)
```

源码嵌入时设置 `HYLE_COMPONENTS=util` 并链接 `hyle::util`；不要靠手写 include path 绕过产品与
安装包合同。内部仍把 header-only `util_core` 与 compiled Registry/symbol factor 分开，使 Mem/Sched
能精确复用 leaf primitive 而不产生反向环；这些内部 target 不进入用户 component catalog。

## 先看选择表

| 需求 | 首选 | 不要这样用 |
|---|---|---|
| 已经有明确资源类型 | 该类型自己的 RAII / `unique_ptr` | 为了少写一行给每个对象套 ScopeGuard |
| 临时 rollback、多个早退共享一段 cleanup | `ScopeGuard` 或 `HYLE_DEFER` | cleanup 抛异常；guard 捕获比自己活得短的对象 |
| 单指针、对象内计数的共享所有权 | `RefCounted<T>` + `Ref<T>` | 需要 weak observer/打破环时仍强行用 Ref |
| 生命周期完全限制在一个线程的侵入式共享所有权 | `ThreadUnsafeRefCounted<T>` + 同一个 `Ref<T>` | owner 或借用裸指针可能跨线程时省掉 atomic |
| 同步函数只在本次调用期间执行任意 callable | 参数按值接收 `FunctionRef<Sig>` | 把 FunctionRef 存进对象、返回或塞进异步队列 |
| required 且要保留 pointer/rebinding 语义的借用 | `NotNull<T*>` | 以为它拥有对象或延长 lifetime |
| null 是正常值 / 获取可能失败但成功时非空 | `T*` / `Result<T&>` | 用 NotNull 隐藏 optional/failure 语义 |
| 可复现仿真、采样、测试、shuffle | 显式 seed 的 `Rng` | 用于 token、key、nonce 或任何安全材料 |
| 每线程普通非安全随机流 | `thread_rng()` | 保存其地址跨线程使用；依赖默认流可复现 |
| 极短、低争用临界区 | `SpinLock` + `std::lock_guard` | 锁内分配、rehash、I/O、等待或在 fiber 中长期持有 |
| 编译期静态类型名/受控构建诊断标签 | `type_name<T>()` / `type_hash<T>()` | 当唯一类型 ID、存数据库、写 wire format、跨编译器/plugin 比较 |
| 需要保留完整的运行期 mangled symbol | owning `std::string demangle(name/type_info)` | 手工先 sizing、分配，再把同一个 symbol demangle 第二次 |
| 已有固定日志字段/栈 buffer | buffer `demangle(name, out)` | 以为 caller buffer 让实现无分配、无锁或可在 signal handler 调用 |
| 启动期注册、factory 不会失败或 NULL 已是完整语义 | `Registry<T*, Args...>` | 需要诊断时仍让 NULL 同时表示 missing、初始化失败和成功空值 |
| 启动期注册、backend 初始化可能失败 | `Registry<Result<T*>, Args...>` | 用旁路 thread-local/ctx 保存 error，或丢掉设备/驱动失败原因 |
| 只想按名解析 factory/metadata，稍后再构造 | `registry.find(name)`；服务期用 `FrozenView::find` | 为取得 creator 做 O(n) 枚举，或被迫提前创建对象 |
| 现在就要按名构造 | `registry.create(name, args...)`；服务期用 `FrozenView::create` | 无视 nullable 与 `Result<T>` 的不同 missing/failure 语义 |
| Registry 初始化后的反复查询 | `auto view = registry.freeze()`，缓存 `FrozenView` | 在 kernel/op 热循环重复字符串 dispatch，或让 view 活过 registry |

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

HYLE 的运营失败通常用 `Status/Result` 表达，因此 failure-only cleanup 的自然写法是先建立 guard，
成功提交后 `dismiss()`；exception-only 的 `scope_fail/scope_success` 可直接选工具链可用的标准
`<scope>` 设施。文件、锁、内存等已有明确资源类型时，仍使用其 owning RAII；`ScopeGuard` 不替代
`unique_resource` 或领域 handle。

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
`make_ref<T>()` 走普通 C++ `new`，因此 OOM 和 `T` 的构造异常会抛出；它不是 `Result` factory。
需要无异常 OOM、显式 HYLE allocator 或领域初始化错误时，由 owning module 的 factory 分配/构造，
把失败转成 `Result<Ref<T>>`，成功后再 `adopt_ref`，并让自定义 `destroy()` 通过同一 allocator 释放。

`Ref<T>` 只有一个指针，没有独立 control block；计数在对象里。若对象不是普通 `delete` 释放，
在派生类实现 `destroy()` 并允许 `RefCounted<Derived>` 调用。`use_count/is_unique` 是并发快照，
不能拿来证明“接下来仍唯一”。需要 weak reference、环检测、aliasing constructor 或对象不能
嵌计数时，使用 `std::shared_ptr/weak_ptr` 或更贴领域的 handle。
自定义 `destroy()` 以及 duck-typed `incref/decref` 都不得抛异常；ownership release 是
`noexcept` 边界，抛出会终止进程。

`Ref<Base>`/`Ref<Derived>` 可直接按对象地址比较、排序和放入 `std::unordered_*`；排序使用标准
指针全序，即使两个对象无关也适合关联容器，不会为了比较临时 retain/release。
`static_pointer_cast`、`dynamic_pointer_cast`、`const_pointer_cast`
均有 lvalue/rvalue 形状：成功的 rvalue cast 转移现有 ownership，不做额外原子往返；失败的
dynamic rvalue cast 保留源引用，避免把一次探测变成意外释放。

普通用户应让 `Ref` 自己配对单份 retain/release，不直接手调 `incref()/decref()`。单份热路与
Boost/Folly 一样以前置条件换取最短原子指令：调用者已持有活引用且计数不会溢出。确实需要
一次转移多份引用时才用带数量的 overload；它会检查零、上溢和下溢。

如果一个对象的**完整引用生命周期**都在同一个线程，可以显式改用非原子 base；handle 和 factory
不变，不需要第二套 owner：

```cpp
struct PlanNode : hyle::util::ThreadUnsafeRefCounted<PlanNode> {
  // ...
};
auto node = hyle::util::make_ref<PlanNode>();
```

这是类型级并发前置条件，不是运行时“快模式”。只要任一 `Ref<PlanNode>`、借用裸指针或最后释放
可能跨线程，就必须使用默认 `RefCounted<PlanNode>`；发布对象前把它转成只读也不会自动让非原子
计数变得线程安全。

## FunctionRef：只借用一次同步调用

```cpp
#include "hyle/util/function_ref.h"

void for_each_token(std::span<const int> tokens,
                    hyle::util::FunctionRef<void(int)> visit) {
  for (int token : tokens) visit(token);
}

for_each_token(tokens, [&](int token) { consume(token); });
```

`FunctionRef` 只有 callable/function 地址和 invoker 两个机器字，不复制 callable、不分配，也不拥有。
按值传参是预期形状。临时 lambda 会活到整个调用表达式结束，因此上例安全；被调用函数若保存或
返回 `visit`，表达式结束后就可能悬空。需要跨调用、异步、取消或排队时使用
`std::function`、标准库实现可用时的 `std::move_only_function`，或 owning module 已定义的 fn+ctx
record。

普通 signature 接受可抛 callable；`FunctionRef<R(Args...) noexcept>` 只接受 nothrow callable，并让
调用表达式保持 `noexcept`。FunctionRef 没有默认/NULL 状态：callback 真正可选时显式用两个入口、
optional owner，或一个语义清楚的 nullable fn+ctx，而不是让每次调用隐式猜测。

## NotNull：required pointer，不是 owner

```cpp
#include "hyle/util/borrow.h"

void execute(hyle::util::NotNull<Plan*> plan) {
  plan->run();
}

execute(&plan);  // raw pointer 自然构造；动态 NULL 会触发 invariant failure
```

只需要 required borrow 且无需 rebinding 时，`T&` 更简单；null 是正常语义时继续用 `T*`；操作可能
失败而成功结果必须非空时用 `Result<T&>`。`NotNull<T*>` 服务的是必须与 pointer API、opaque
`void*` context 或可重新绑定局部变量组合的 required borrow。它只在构造时检查动态值，随后与裸
指针同尺寸；不拥有、不 pin、不延长 referent lifetime，也不能把悬空 pointer 变安全。

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

`thread_rng()` 每线程首次使用时组合 best-effort `std::random_device`、地址 salt 和递增序列，之后
无锁。C++ 标准不保证 `random_device` 一定来自 OS 熵，这个 header-only convenience 也不保证跨
线程、跨 shared library 的 stream ID 唯一或区间不重叠；它只用于不要求复现的普通统计随机。
需要可复现或有意分区时使用显式 `Rng` + `jump/long_jump`；安全随机必须使用 OS/vetted crypto
提供的 secure-random API。

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

`cpu_relax()` 只是一条自旋 hint，不建立同步关系；必须与 atomic 条件配合。x86、Arm 和 RISC-V
分别发出平台 pause/yield 类 hint；unknown target 才安全退为空操作。它不代表 runtime ISA dispatch，
也不替 owner 决定何时 yield、park、wake 或取消。

## type_name 与 demangle

`type_name<T>()` 返回编译器静态字符串的 view，`type_hash<T>()` 是其 constexpr FNV-1a。
二者只适合作为受控构建内的诊断标签：hash 可能碰撞，本地/匿名类型在不同 TU 也可能得到含混拼写；
编译器版本、flags、plugin 或类型拼写变化都可能改变结果。正确性分派应能处理碰撞，跨模块持久 ID
应使用显式 schema/UUID，不要用 type hash。

需要完整 owning 文本时直接写 `std::string text = demangle(name)`；运行期类型可写
`demangle(typeid(object))`。它只执行一次平台 demangler，返回的字符串由调用方自己的 C++ runtime
拥有；standard string 分配失败会像其他 owning C++ convenience 一样抛异常。

`demangle(mangled, out, cap)` 是 snprintf 风格：返回完整所需长度，`cap>0` 时总是 NUL 终止；
返回值大于等于 cap 表示已截断。固定数组可直接写 `demangle(name, buffer)`，运行期类型可写
`demangle(typeid(object), buffer)`，无需手写 `.name()`/`sizeof`。输入可以与输出完全或部分重叠。

两种形状都只服务日志/backtrace 等冷路径：Itanium ABI 的 `__cxa_demangle` 会内部 malloc/free；
Windows DbgHelp 调用会串行化，超长 decorated symbol 还可能使用有界 temporary。caller-buffer 从不
接管这些 allocation，不代表函数本身零分配/无锁。尤其不要从 POSIX signal handler 或 Windows
exception handler 的受限上下文调用；需要 async-signal-safe crash capture 时先保存原始地址/符号，
回到正常执行上下文再反修饰。HYLE 的 mutex 只协调 HYLE 发起的 DbgHelp 调用；应用若还直接调用
DbgHelp，必须在应用层把所有 DbgHelp 使用统一串行化。

## Registry：注册阶段可变，服务阶段冻结

```cpp
#include "hyle/util/registry.h"

using EngineRegistry =
    hyle::util::Registry<hyle::Result<Engine*>, const Config&>;
EngineRegistry engines;

hyle::Result<Engine*> make_cpu_engine(const Config&) noexcept;
hyle::Result<Engine*> make_cuda_engine(const Config&) noexcept;

HYLE_RETURN_IF_ERROR(engines.reserve(16));
HYLE_RETURN_IF_ERROR(engines.register_creator(
    "cpu", &make_cpu_engine, hyle::util::RegistryPriority::kFallback,
    "portable CPU engine"));
HYLE_RETURN_IF_ERROR(engines.register_creator(
    "cuda", &make_cuda_engine, hyle::util::RegistryPriority::kPreferred,
    "CUDA engine"));

// 初始化结束：单向、幂等、O(1)、零分配。
auto factories = engines.freeze();

// 只解析一次；随后可在 owner 中缓存这个 typed function pointer。
auto cuda = factories.find("cuda");
if (!cuda) return hyle::Status::NotFound("engine factory not found");
auto make_selected_engine = cuda->creator;
HYLE_ASSIGN_OR_RETURN(Engine * engine, make_selected_engine(config));

// 不需要分开解析/缓存时，直接 lookup + invoke。
HYLE_ASSIGN_OR_RETURN(Engine * cpu, factories.create("cpu", config));
```

Registry 会复制并拥有 key/help；调用者无需保留传入字符串。相同 key 的更高 priority 覆盖，
更低 priority 是成功的 no-op，相同 priority 返回 `AlreadyExists`，因此链接/静态初始化顺序不会
悄悄决定实现。`reserve()` 只用于启动期减少 rehash；注册和 reserve 在 freeze 后返回
`FailedPrecondition`。

返回值应贴合 factory 的真实失败模型。`Registry<T*>` 是 infallible/nullable 形状，missing 与
creator 返回 NULL 都得到 NULL。`Registry<Result<T*>>` 是推荐的 fallible backend 形状：missing
得到 `NotFound`，设备、驱动、配置或资源错误由 creator 原样返回。两者共用同一张表、相同
lookup value 和复杂度；Result 不是额外的 per-entry 元数据。Registry 不替用户 `new` 对象；
factory 必须是 `noexcept` function pointer，`Result<Ref<T>>` 等 owning value 由 factory 显式构造，
OOM/设备/驱动失败都通过 `Result` 返回。

`find()` 不调用 factory，missing 返回 `std::nullopt`。它返回 `EntryView` 的值快照：creator 与 priority
被复制，key/help 是 Registry-owned 字节的借用 view；旧快照即使遇到更高 priority 替换也保持可读，
但不能活过 Registry。缓存 Creator 还要求提供这段代码的 executable/shared module 未被卸载；Registry
不会替 module owner 证明对象、线程和函数指针已经 quiescent。

启动登记必须显式执行并检查 `Status`。不要把“只靠全局构造器副作用的 registration TU”塞进
static archive：链接器可以直接丢弃该 TU，静态初始化也没有自然位置向应用返回 collision/OOM。
需要动态 shared-library backend 时，HYLE 的 `platform::DynLib` 只负责 loader handle。调用方自己的
module owner 必须完成 load、调用明确的 registration entry、排空创建对象/在途 reader、销毁
Registry，最后 unload；HYLE 当前没有可直接实例化的通用 `ModuleManager`，Registry 本身也不猜这些
生命周期。

可变阶段的查询受 mutex 保护。creator 和 `for_each` visitor 都在锁外运行，可以重入同一个
Registry；可变阶段枚举会为时间点快照分配一次，冻结阶段则无锁、无分配。`Registry::create()`
在冻结后只多一次生命周期 acquire load，适合普通调用；反复按名的 service lookup 可缓存
`FrozenView`，其 `find/create` 直接查不可变 Swiss 索引。两种形状都是 O(key length) hash + expected O(1)
probes，creator 自身成本另计。若 backend/op 已经能在初始化时解析成 typed handle 或 function pointer，应缓存解析结果，
不要在每个 kernel/token 内重复按字符串 dispatch。

Registry 不提供 unregister：已经创建对象、函数指针和并发 reader 的代码/数据生命周期无法由
一个名字表安全推断。需要动态装卸 shared library、依赖排序或在途调用排空时，应由调用方具名的
module owner 管理 quiescence，再销毁整个 Registry。Registry 必须活过所有 `FrozenView` 和在途操作；
析构前由调用方让使用者静止。

## 分层与证据入口

header-only util-core 仍作为独立内部 factor 构建和收口，但通过 `hyle::util` 产品 façade 交付；当前结论见
[`util-core v2`](../implementation/tasks/util/util-core-v2-capability-completion.md)。更早的
[`util-core-v1`](../implementation/tasks/util/util-core-v1-closure.md) 与
[`util-core capability replay`](../implementation/tasks/util/util-core-capability-replay-v1.md) 仅是历史输入；Registry
和 demangle 属于独立 runtime factor，当前结论见
[`util-runtime-v3`](../implementation/tasks/util/util-runtime-v3-rebuild.md)，更早的
[`util-runtime-v2`](../implementation/tasks/util/util-runtime-v2-closure.md) 只作历史输入。`1b148ef`/`70fa4e8`
只是在旧代码树上得到的历史输入，
不代表当前 runtime 证书；唯一当前状态见
[`module status`](../implementation/status.md)。分层仍然保留：它避免低层 mem/container 为一个
运行时 Registry 反向链接 runtime factor。用户只看到一个 Util 产品不改变这条内部依赖。
