# util 纯 C ABI 使用指南

先用 [`util.md`](./util.md) 判断该选哪种能力；本页只说明纯 C/FFI 的准确形状。包含
`hyle/c/util.h`，请求 CMake component `util` 并链接 `hyle::util`。当前 ABI 有 **47 个动态
符号 + 10 个 header-inline 便利入口**；可失败的控制面统一返回 error 模块的 owning
`hyle_status`，不再有 util 私有的线程局部错误槽。

```c
hyle_status status = some_util_call(...);
if (!hyle_status_is_ok(status)) {
  char text[256];
  hyle_status_to_string(status, text, sizeof(text));
  fprintf(stderr, "%s\n", text);
}
hyle_status_free(&status); /* OK 也可安全释放 */
```

## 反修饰与自旋 hint

`hyle_util_demangle(&needed, symbol, out, cap)` 使用 snprintf 形状：成功时 `needed` 是不含 NUL
的完整长度，`cap > 0` 时总 NUL 终止；非 mangled 输入原样复制，`NULL` 输入视为空字符串。
它用于日志/backtrace 冷路径，底层 demangler 可能分配。

`hyle_util_cpu_relax()` 是 header-inline 的单次 ISA 自旋 hint，不建立同步关系。它必须配合
atomic 条件使用，不能当 sleep、yield 或 lock。

## 随机数

`hyle_util_rng` 是可复制的 256-bit xoshiro256++ POD 状态。先 seed；全零、空指针或非法范围
返回文档规定的 0/false 且不改状态。这是刻意的热路径 value-shaped 合同：在配置边界先用
`hyle_util_rng_is_valid`，不要为每个随机数构造 Status。

```c
hyle_util_rng rng;
if (!hyle_util_rng_seed(&rng, 12345)) {
  /* 配置错误：这里按应用自己的控制流退出。 */
}

uint64_t ids[1024];
if (!hyle_util_rng_fill_u64(&rng, ids, 1024)) {
  /* invalid state/buffer：这里按应用自己的控制流退出。 */
}

hyle_util_rng worker = hyle_util_rng_clone(&rng);
hyle_util_rng_discard(&worker, 100); /* 与连续取 100 个普通输出等价，O(n) */
hyle_util_rng_jump(&rng); /* 下一 worker 再 clone；2^128 间隔 */
```

scalar/range、`fill_u64/float/double`、`discard`、callback shuffle、`jump/long_jump` 与线程局部便利流均有
C 入口。batch 用于摊薄 FFI 调用。`hyle_util_thread_rng()` 返回当前线程借用指针，不能销毁、
不能跨线程；显式状态用于可复现流程。任何 token/key/nonce 必须改用 OS 或 vetted crypto CSPRNG。

## 类型擦除共享所有权

`hyle_util_ref_create(&out, payload, destroy, ctx, allocator)` 创建 strong count 为 1 的 opaque
handle，并复制 allocator；最终 release 同步调用 destroy 一次，再由同一 allocator 释放控制块。
普通系统分配用 `hyle_util_ref_create_system`。

```c
hyle_util_ref* ref = NULL;
HYLE_C_RETURN_IF_ERROR(
    hyle_util_ref_create_system(&ref, resource, close_resource, NULL));
hyle_util_ref* second = hyle_util_ref_retain(ref);
hyle_util_ref_release(second);
hyle_util_ref_release(ref);
```

每次成功 retain 必须配一次 release；最终 release 后所有线程都不得再访问 handle。callback 不得
让 C++ 异常越过 C 边界。`use_count/is_unique` 只是并发快照。C payload 本就是 `void*`，转换由
调用方类型系统负责，不需要复制 C++ 模板 cast API。

## Registry：按名字选择 factory

Registry 复制 key/help，并可使用调用方 allocator。字符串入口默认接收
`hyle_container_string_view`，因此支持非 NUL 结尾和 embedded NUL；常规 C 字符串可用 `_cstr`
inline 便利函数。

```c
static void* make_engine(hyle_status* out_status, void* config,
                         const void* options) {
  void* engine = application_make_engine(config, options);
  if (engine == NULL && out_status != NULL) {
    *out_status = hyle_status_unavailable("engine initialization failed");
  }
  return engine;
}

static hyle_status build_registry(hyle_util_registry** out, void* config) {
  hyle_util_registry* reg = NULL;
  hyle_status status = hyle_util_registry_create_system(&reg);
  if (!hyle_status_is_ok(status)) return status;

  status = hyle_util_registry_reserve(reg, 16);
  if (!hyle_status_is_ok(status)) goto fail;
  status = hyle_util_registry_register_cstr(
      reg, "fast", make_engine, config, HYLE_UTIL_REGISTRY_PREFERRED,
      "fast engine");
  if (!hyle_status_is_ok(status)) goto fail;
  status = hyle_util_registry_freeze(reg);
  if (!hyle_status_is_ok(status)) goto fail;

  *out = reg;
  return status;

fail:
  hyle_util_registry_destroy(reg);
  return status;
}

hyle_util_registry* reg = NULL;
hyle_status status = build_registry(&reg, config);
if (!hyle_status_is_ok(status)) return status;
void* engine = NULL;
status = hyle_util_registry_create_object_cstr(
    &engine, reg, "fast", &options); /* checked：保留完整诊断 */
if (!hyle_status_is_ok(status)) {
  hyle_util_registry_destroy(reg);
  return status;
}
/* 使用并按 factory 合同销毁 engine；所有调用排空后： */
hyle_util_registry_destroy(reg);
```

优先级是 fallback/default/preferred：高者覆盖，低者成功忽略，同级冲突返回
`HYLE_STATUS_ALREADY_EXISTS`。`reserve` 和注册只允许在 mutable 阶段；`freeze` 单向、幂等、
O(1)、零分配，此后查询是 lock-free expected O(1)。checked `create_object` 给 creator 传入
已初始化的 OK `out_status`，因此可区分 key 缺失、creator 失败和成功返回 NULL；失败时 creator
返回 NULL 并把 `*out_status` 替换为一个 owning error。`create_or_null` 给 creator 传 NULL status，
让热路径完全跳过诊断构造，并把 invalid/missing/failure/successful-NULL 都映成 NULL。

factory 和 visitor 都在 registry 锁外调用且可以重入。mutable 枚举先分配一个时间点快照；
frozen 枚举无锁、无分配。visitor 收到的 key/help 是借用 view，在 Registry 析构前稳定；
`out_completed` 区分完整遍历与 visitor 主动停止。Registry 不负责动态库 unload，也没有
unregister；销毁前必须排空所有调用者和 callback。

## SpinLock

FFI 可任选两种生命周期：

- heap：`hyle_util_spin_lock_create(..., allocator)` / `destroy`，系统 allocator 有 `_system`；
- caller storage：先读 `storage_size/alignment`，按要求分配后 `init` / `deinit`。

两对 API 不能混用；locked 实例拒绝 destroy/deinit。`lock/try_lock/unlock` 返回 Status，
`is_locked` 只是诊断快照。该接缝适合外语控制路径里的极短低争用临界区；可能阻塞、分配、I/O、
高争用或 fiber 等待时使用对应 mutex/fiber primitive。C++ 热路径直接用 inline `SpinLock`。

## 为什么没有 C ScopeGuard

C 没有可移植的词法析构。旧的 heap guard 要求每个出口手动 destroy，既没有提供“离开作用域
必执行”的安全性，还新增一次分配和一个可泄漏对象，因此已经删除。纯 C 使用单一
`goto cleanup` 出口；Rust/Go/C#/Java 等 FFI 使用语言自身的 RAII/defer/finally。

`type_name<T>` / `type_hash<T>` 也有意仅 C++：它们依赖编译器对模板参数 `T` 的静态拼写，
运行期 `void*` 无法恢复同一信息。C ABI 已覆盖相应的 RNG、共享所有权、Registry、demangle
和同步能力，没有用签名相似但语义错误的接口凑“全覆盖”。

## 常见坑

- 每个非 OK `hyle_status` 都是 owning 值；返回、move 或 free，不能只打印后丢弃。
- allocator、factory ctx、destroy ctx、Registry、线程局部 RNG 的生命周期由调用方保证；被更高
  priority 替换的 factory ctx 也必须活过已在途的调用。
- callback 不能让 C++ 异常越过 C 边界。Registry creator 只有收到非 NULL `out_status` 时才构造
  诊断；失败返回 NULL，成功保持 Status 为 OK。写入 error 后又返回对象违反合同且会泄漏对象。
- shuffle callback 不能重入并修改同一个 RNG；adapter 会先快照状态、完成交换，再写回状态。
- `create_or_null` 只适合 NULL 已是完整失败语义的热路径；需要诊断就用 checked 入口。
- freeze 后不能再注册；需要动态模块生命周期时建立独立 module manager，不要伪装成 Registry。
