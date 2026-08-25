# util 架构：两个依赖层，一个产品边界

> **Native FFI 状态校正（2026-08-25）**：HYLE 没有 native-C 客户，也没有已发布的 managed
> binding。本文关于 C/C++、C/FFI 或 C 用户的旧表述只记录待复审的 ABI 候选，不是当前跨语言
> 证书或兼容承诺；首发前可破坏性调整。先从真实 foreign-runtime 任务裁决 FFI 是否为 `N/A`；
> 只有保留 direct FFI 时，util-runtime 才必须通过 Java FFM + .NET P/Invoke、真实导出、handle/
> callback 生命周期与 batch 旅程后冻结。

> 本页维护长期设计。用户选型见 [`../guide/util.md`](../guide/util.md)，Native FFI 候选映射见
> [`../guide/c-util.md`](../guide/c-util.md)，性能证据只见
> [`../performance/benchmarks-util.md`](../performance/benchmarks-util.md)。

`util` 不是随手堆 helper 的杂物目录。它是一个产品模块，但按依赖切成两个可组装层：

```text
P0 hyle::util_core（INTERFACE，无 binary artifact）
   ScopeGuard / Ref / Rng / type_name / SpinLock / cpu_relax
                    │
                    ├──► mem / sched / graph
                    └──► C++ 用户

P4 hyle::util（hyle_util binary；严格收口波次 R2）
   demangle / Registry / C-shaped type-erased adapter（Native FFI 候选）
                    │
                    └──► error + mem + container + util_core
```

Registry 是模板头，但公开形状直接使用 container 的 `string_ref/SwissTable`，并借 mem allocator/
Region；它因此属于 util-runtime，不能反塞进 P0。container 不依赖 `hyle_util`，从而没有
`container ↔ util` binary 环。demangle 只有一个冷路径动态符号，本身不依赖 container；为它
新增 binary component 会增加发布和消费选择而没有形成独立能力族，因此与 runtime 共用 target。

## 能力与负空间

| 能力 | 正常消费者需求 | 成熟形状 | 明确不做 |
|---|---|---|---|
| `ScopeGuard` / `HYLE_DEFER` | rollback、C API cleanup、多个早退 | Abseil/Folly：栈内、move-only、一次执行、dismiss/invoke | 代替已有资源 RAII；析构期异常恢复 |
| `RefCounted<T>` + `Ref<T>` | 对象内计数、单指针侵入式共享所有权、定制 destroy | Boost/Folly/std 同形 retain/release、比较/hash/casts | weak ref、环检测、aliasing control block |
| `Rng` | 仿真、采样、shuffle、可复现并行流 | xoshiro256++ URBG、Lemire range、jump 拆流 | 密码学安全随机 |
| `type_name/hash` | 编译期诊断、进程内标签 | pretty-function constexpr probe | 持久/跨编译器/plugin/wire ID |
| `SpinLock` / `cpu_relax` | 几条指令的低争用临界区/atomic 自旋 | oneTBB/Folly 紧凑 TTAS/backoff | 高争用、阻塞、fiber-aware wait |
| `demangle` | 日志、backtrace、诊断 UI | caller buffer + required-size | 热路径无分配保证 |
| `Registry<Value, Args...>` | backend/plugin factory 在启动期组合，服务期按名创建 | raw nullable 或 `Result<T>` fallible factory、owned names、priority resolution、mutable→immutable publication | 动态库 unregister/unload、依赖求解、对象排空 |

没有内部消费者不等于应删除用户能力，也不能为了“采用率”把它硬塞进不合适的热路径。反过来，
新增 helper 必须先有真实消费者场景，并证明它不属于 core/container/mem/platform 等既有模块。

## Registry 状态机与数据形状

```text
mutable startup
  mutex-protected register/reserve/read
  creator + visitor outside lock; reentrant
  one Swiss index + owned key Region + cold metadata Region
             │ freeze(): O(1), zero allocation, idempotent
             ▼
immutable service
  the same Swiss index is release-published
  Registry reads: one acquire lifecycle load + expected-O(1) lookup
  cached FrozenView: direct expected-O(1) lookup
```

旧形状若在 freeze 时复制/重建第二张表，会让启动延迟、峰值内存和 OOM surface 随注册规模增长；
当前一表两阶段直接发布。lookup value 内联 creator，热 key 与冷 help/priority/Entry 分区，避免
dispatch 为诊断元数据触碰额外 cache line。这些是正常 init-heavy/read-heavy 消费形状的架构
优化，不是按 benchmark 的 key 数量或线程数特化。

Registry 拥有 key/help，priority 保证静态初始化/链接顺序不成为隐式策略：更高覆盖、更低忽略、
同级冲突报错。可变枚举在锁内复制时间点快照，锁外执行 callback；冻结枚举直接遍历不可变表。
`FrozenView` 不拥有表，Registry 必须活过 view 和所有在途 operation。没有 unregister 是安全
边界：module unload 需要追踪 created object、函数指针与并发 reader quiescence，应由独立 module
manager 负责，而不是让通用名字表猜测。

返回值形状决定失败合同，而不改变 Registry 的表布局。`Registry<T*>` 保留最短的 nullable
factory：missing 和 factory 返回 NULL 都是 NULL。会因设备、驱动、配置或资源失败的 factory 应
使用 `Registry<Result<T*>, Args...>`：missing 自动成为 `NotFound`，creator 的 `Status` 原样
传播。该策略不增加每项元数据，也不要求 benchmark-only API。

## 复杂度与布局合同

- ScopeGuard 操作、Ref copy/drop/cast、SpinLock 无争用和 RNG scalar draw 是 O(1)。
- RNG bounded draw 期望 O(1)，`discard(n)` O(n)，`jump/long_jump` 固定 O(256)，shuffle O(n)。
- type-name 在常量求值切片；hash 是 O(类型名长度) 的 constexpr 工作，不在运行期热路。
- Registry 查找是 O(|key|) hash + expected O(1) probes；register 另复制 O(|key|+|help|)，扩容摊销
  expected O(1)；枚举 O(n)。mutable 枚举临时空间 O(n)，frozen 枚举 O(1)；freeze 时间/额外
  空间 O(1)。最坏查找仍受恶意 collision 约束。
- `Ref<T>` 保持一个指针，计数在对象内；SpinLock 保持 1 byte。需要 cache-line padding 时由经
  profile 证明伪共享的 owner 显式添加，不能让所有实例无条件付费。
- SpinLock 成功 RMW 保留 acquire-release。历史 M4 真机证据表明机械削为 acquire-only 会造成
  material handoff regression；这是 ISA/codegen 合同，不以抽象模型的最弱合法序替代实测。
  具体点值与 freshness 只见 [`benchmarks-util`](../performance/benchmarks-util.md)。

## Native FFI 候选原则

`hyle/c/util.h` 是 util-runtime 现有 C-syntax schema 候选，使用 `hyle_status` 和
`hyle_allocator`，不发明模块私有错误槽或 allocator。HYLE 没有 C 产品或已发布 managed binding；
先从真实 foreign-runtime 任务决定整个候选面是保留、缩小还是 `N/A`。若保留，RNG 以 POD
state/batch 映射；Ref、Registry、SpinLock 以 opaque handle/callback 擦除；demangle 使用 caller
buffer，字符串采用 length-aware view。编译期能力不为 parity 导出。

当前候选账本有 47 个动态符号和 10 个 header-inline 便利入口；这只是实现盘点，不是兼容或
完整性承诺。候选 Registry creator 直接返回对象，
并接收可选 `out_status`：checked create 传入已初始化的 OK Status，nullable hot create 传 NULL，
让 fallible creator 跳过诊断构造；两条路不增加每项元数据。C++ 的 template/cast 语法与静态
type-name 有意不伪造。ScopeGuard 也不应机械变成 heap handle：它无法提供离开词法作用域自动
cleanup 的语义；foreign host 应使用本语言 defer/finally。这叫任务语义覆盖，而不是把每个 C++
函数名机械翻译成导出符号。

## 内部采用与依赖检查

mem 的短固定状态/指针转移、sched 等待细节和 graph 首错槽显式依赖 `hyle::util_core` 的
SpinLock/cpu_relax。会 allocate/rehash/执行未知用户代码的 mem LeakTracker 与 container
ConcurrentVector 使用 mutex。Registry 的 binary/C adapter 显式链接 error、mem、container 和
util_core；这些依赖同时由中央 component graph 与 target link 生成/验证。

下游审查仍按实际临界区和生命周期判断，不能把类型名当证据。sched 内锁下 move/construct
泛型 `T` 的路径仍由 sched 自己收口，不反向污染 util 结论。

## 维护闸门

1. util-core 公开头必须逐头 self-contained，并可由安装包单独请求；禁止 include runtime 头。
2. Registry 修改必须复验 raw nullable/`Result<T>`、mutable/frozen、reentrant callback、freeze
   race、allocator failure、TSan 和 installed shared/static C++ consumer；只有保留 Registry direct
   FFI 时，才追加候选 checked/nullable 形状及 Java/.NET handle/callback/lifetime 旅程。
3. RNG 算法/状态布局变化更新固定序列；默认 TLS seed 变化不能破坏显式 seed 的可复现流。
4. Ref/SpinLock 并发修改跑 TSan；布局修改在 MSVC/GCC/AppleClang 和真实下游 owner 上复验。
5. 先裁决真实 foreign-runtime 任务；无任务可记录 `FFI=N/A` 并删除候选。保留 direct FFI 时，
   动态符号、header-inline、纯 C smoke 与 guide 的机械对账只算 hygiene，还必须通过 Java FFM +
   .NET P/Invoke 的精确 shared-library smoke；错误统一走 error。
6. benchmark 只验证同合同实现与选择边界，不反向制造 API、常态元数据或 workload 特化。
