# util 架构：两个依赖层，一个产品边界

> **当前边界（2026-09-01）：**`util_core` 与 `util` 均为 `CLOSED`；施工/签证
> 状态只看中央
> [`module status`](../implementation/status.md)。本页描述当前设计，不把旧 benchmark 点值当作
> runtime 发布证书。当前只提供 C++23
> source/static/shared，Native FFI 为 `N/A`；BUILD-03 已删除旧 C schema、bridge、动态导出和 fixture。

> 用户当前选型见 [user/util](../user/util.md)，current util-core/runtime 与历史 util 证据见
> [util performance evidence](../implementation/evidence/performance/benchmarks-util.md)，施工历史见
> [util tasks](../implementation/tasks/util/)。

`util` 不是随手堆 helper 的杂物目录。它是一个产品模块，但按依赖切成两个可组装层：

```text
foundation：hyle_util_core（内部 INTERFACE factor，无 binary artifact）
   ScopeGuard / Ref / Rng / type_name / SpinLock / cpu_relax
                    │
                    ├──► mem / sched / graph
                    └──► hyle::util 产品 façade

runtime：hyle_util（内部 compiled factor；严格收口波次 R2）
   demangle / Registry
                    │
                    └──► core + error + mem + container
```

Registry 是模板头，但公开形状直接使用 container 的 `StringView/FlatHashMap`，并借 mem allocator/
Region；它因此属于 util-runtime，不能反塞进 foundation。container 不依赖 `hyle_util`，从而没有
`container ↔ util` binary 环。demangle 只有 buffer entry 和服务 header-inline owning convenience 的
internal sink 两个冷路径动态符号，本身不依赖 container；为它新增 binary component 会增加发布和
消费选择而没有形成独立能力族，因此与 runtime 共用 target。

## 能力与负空间

| 能力 | 正常消费者需求 | 成熟形状 | 明确不做 |
|---|---|---|---|
| `ScopeGuard` / `HYLE_DEFER` | rollback、C API cleanup、多个早退 | Abseil/Folly：栈内、move-only、一次执行、dismiss/invoke | 代替已有资源 RAII；析构期异常恢复 |
| `RefCounted<T>` + `Ref<T>` | 对象内计数、单指针侵入式共享所有权、定制 destroy | Boost/Folly/std 同形 retain/release、比较/hash/casts；普通 `make_ref` 可抛 | weak ref、环检测、aliasing control block；owner-specific fallible allocation factory |
| `Rng` | 仿真、采样、shuffle、可复现并行流 | xoshiro256++ URBG、Lemire range、显式 jump 拆流；TLS 仅 best-effort convenience | 密码学安全随机；默认 TLS 流的唯一/不重叠保证 |
| `type_name/hash` | 编译期诊断、受控构建内标签 | pretty-function constexpr probe；hash 允许碰撞 | 持久/唯一 correctness ID、跨编译器/plugin/wire ID |
| `SpinLock` / `cpu_relax` | 几条指令的低争用临界区/atomic 自旋 | oneTBB/Folly 紧凑 TTAS/backoff | 高争用、阻塞、fiber-aware wait |
| `demangle` | 日志、backtrace、诊断 UI | 完整 caller-owned `std::string`；或 caller buffer + required-size | 热路径、async-signal-safe、无内部 allocation/lock 保证 |
| `Registry<Value, Args...>` | backend/plugin factory 在启动期组合，按名解析 factory 或直接创建 | raw nullable 或 `Result<T>` fallible factory、owned names、lookup-only snapshot、priority resolution、mutable→immutable publication | 动态库 unregister/unload、依赖求解、对象排空 |

没有内部消费者不等于应删除用户能力，也不能为了“采用率”把它硬塞进不合适的热路径。反过来，
新增 helper 必须先有真实消费者场景，并证明它不属于 core/container/mem/platform 等既有模块。

## Registry 状态机与数据形状

```text
mutable startup
  mutex-protected register/reserve/read
  creator + visitor outside lock; reentrant
  one Swiss index + transactional owned entry storage Region
             │ freeze(): O(1), zero allocation, idempotent
             ▼
immutable service
  the same Swiss index is release-published
  Registry find/create: one acquire lifecycle load + expected-O(1) lookup
  cached FrozenView find/create: direct expected-O(1) lookup
```

旧形状若在 freeze 时复制/重建第二张表，会让启动延迟、峰值内存和 OOM surface 随注册规模增长；
当前一表两阶段直接发布。lookup value 内联 creator，所以 create 不读取 Entry 的冷 help/priority；
初次注册把 Entry、owned key 与 owned help 作为一个 Region allocation 提交，并在此前准备好唯一
仍可能分配的 Swiss capacity。这样 OOM 不会在单调 arena 留下不可达 key/help，普通 4--32 项
startup 也只需一个小 storage Region，而不是预留两个独立 4 KiB arena。这些是正常
init-heavy/read-heavy 消费形状的架构优化，不是按 benchmark 的 key 数量或线程数特化。

Registry 拥有 key/help，priority 保证显式组合顺序不成为隐式策略：更高覆盖、更低忽略、
同级冲突报错。可变枚举在锁内复制时间点快照，锁外执行 callback；冻结枚举直接遍历不可变表。
`find()` 返回 lookup-only `EntryView` value snapshot：creator/priority 被复制，key/help 借用 Registry-owned
storage。它使 backend owner 能在启动期解析并缓存 typed factory，而不必为了取得 callable 提前构造或
O(n) 枚举；没有新增表、entry 字段、allocation 或动态分发。
`FrozenView` 不拥有表，Registry 必须活过 view 和所有在途 operation。没有 unregister 是安全
边界：module unload 需要追踪 created object、函数指针与并发 reader quiescence，应由独立 module
manager 负责，而不是让通用名字表猜测。

返回值形状决定失败合同，而不改变 Registry 的表布局。`Registry<T*>` 保留最短的 nullable
factory：missing 和 factory 返回 NULL 都是 NULL。会因设备、驱动、配置或资源失败的 factory 应
使用 `Registry<Result<T*>, Args...>`：missing 自动成为 `NotFound`，creator 的 `Status` 原样
传播。该策略不增加每项元数据，也不要求 benchmark-only API。

creator 是 `noexcept` function pointer，而不是 captured/type-erased callable。运营失败必须进入
返回的 `Result`；需要状态时通过 Registry 的显式 `Args...` 或生命周期清楚的 owner 传递。旧
Clay 静态注册宏和 default-`new` creator 已删除：archive 中只有静态初始化副作用的 TU 可能被
dead-strip，而且构造器无法把 collision/OOM 可靠交还 startup owner。内建与动态 backend 都由
明确的启动代码调用 `register_creator()` 并处理 `Status`。

demangle 的 owning overload 是 header-inline `std::string` convenience。它通过一个窄 internal sink
让 caller TU 构造标准字符串，不把 DLL-owned STL layout/heap ownership交给调用方；sink 在 transient
platform result 有效期内同步执行。Itanium ABI 仍可能先由 `__cxa_demangle` malloc，再复制进
`std::string`；Windows DbgHelp 仍需要全局串行调用。buffer overload 是 `noexcept` 且支持输入/输出
重叠，但也不等于无内部 allocation，更不能在 signal handler 中调用。

## 复杂度与布局合同

- ScopeGuard 操作、Ref copy/drop/cast、SpinLock 无争用和 RNG scalar draw 是 O(1)。
- RNG bounded draw 期望 O(1)，`discard(n)` O(n)，`jump/long_jump` 固定 O(256)，shuffle O(n)。
- type-name 在常量求值切片；hash 是 O(类型名长度) 的 constexpr 工作，不在运行期热路。
- Registry `find/create` 是 O(|key|) hash + expected O(1) probes；register 另复制 O(|key|+|help|)，扩容摊销
  expected O(1)；枚举 O(n)。mutable 枚举临时空间 O(n)，frozen 枚举 O(1)；freeze 时间/额外
  空间 O(1)。最坏查找仍受恶意 collision 约束。
- `Ref<T>` 保持一个指针，计数在对象内；SpinLock 保持 1 byte。需要 cache-line padding 时由经
  profile 证明伪共享的 owner 显式添加，不能让所有实例无条件付费。
- SpinLock 成功 RMW 保留 acquire-release。历史 M4 真机证据表明机械削为 acquire-only 会造成
  material handoff regression；这是 ISA/codegen 合同，不以抽象模型的最弱合法序替代实测。
  当前 v1 的 W1-WIN 同合同工程门为非劣 PASS；M1-MAC 的超短手工诊断被执行顺序完全翻转，按仪器
  规则只能记为 inconclusive，不能驱动产品。具体身份、点值与 freshness 只见
  [`benchmarks-util`](../implementation/evidence/performance/benchmarks-util.md)。
- 两种 demangle 形状都是 O(input+output)，一次 public 调用只执行一次平台 demangler（Windows 因
  DbgHelp 不返回 required size，超长 decorated symbol 可能做有界容量重试）。owning shape 另有一个
  caller-owned string，buffer shape 不增加 API-owned storage。

## 交付与 Native FFI 边界

util 当前只交付 C++23 source/static/shared。BUILD-03 已删除早期 C-syntax schema、bridge、动态
导出和 fixture；Native FFI 为 `N/A`，没有 C ABI 兼容面。HYLE 模块直接使用 typed RNG、Ref、
Registry、SpinLock 和 demangle API。未来只有已命名的 foreign-runtime consumer 通过全局
[Native FFI 准入策略](native-ffi-policy.md)后，才按任务设计窄的 runtime 类型擦除边界；不会把
template/cast、ScopeGuard 或编译期 type-name 机械翻译成动态符号。

## 内部采用与依赖检查

mem 的短固定状态/指针转移、sched 等待细节和 graph 首错槽显式依赖 `hyle_util_core` factor 的
SpinLock/cpu_relax。会 allocate/rehash/执行未知用户代码的 mem LeakTracker 与 container
ConcurrentVector 使用 mutex。Registry runtime 显式链接 core、error、mem 和 container；
删除静态注册/default creator 后不再依赖 util_core。这些依赖同时由中央 component graph 与
target link 生成/验证。

下游审查仍按实际临界区和生命周期判断，不能把类型名当证据。sched 内锁下 move/construct
泛型 `T` 的路径仍由 sched 自己收口，不反向污染 util 结论。

## 维护闸门

1. util-core 公开头必须逐头 self-contained，并可通过 `HYLE_DEV_FACTORS=util_core` 独立构建；禁止
   include runtime 头。用户安装包只通过 `util` 产品请求和 `hyle::util` façade 取得两层能力。
   当前收口证据与 reopen triggers 见
   [`util-core-v1`](../implementation/tasks/util/util-core-v1-closure.md)。
2. Registry 修改必须复验 raw nullable/`Result<T>`、mutable/frozen、reentrant callback、freeze
   race、allocator failure、TSan 和 installed source/shared/static C++23 consumer。
3. RNG 算法/状态布局变化更新固定序列；默认 TLS seed 变化不能破坏显式 seed 的可复现流。
4. Ref/SpinLock 并发修改跑 TSan；布局修改在 MSVC/GCC/AppleClang 和真实下游 owner 上复验。
5. Native FFI 保持 `N/A`；若出现已命名的 foreign-runtime 任务，先按全局准入策略独立立项，
   不能在 util 修改中顺手恢复旧 C schema。
6. benchmark 只验证同合同实现与选择边界，不反向制造 API、常态元数据或 workload 特化。
7. demangle 修改必须同时复验 owning/buffer、plain/fallback/truncation/overlap、异常 cleanup、internal
   sink 的 shared-CRT ownership 边界以及 source/static/shared consumer；caller buffer 不得被描述为
   async-signal-safe。

新版 capability-family/public-surface 双向复检及当前 owner routes 分别见
[`util-core capability replay`](../implementation/tasks/util/util-core-capability-replay-v1.md) 与
[`util-runtime v2`](../implementation/tasks/util/util-runtime-v2-closure.md)。
