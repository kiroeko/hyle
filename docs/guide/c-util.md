# util 纯 C ABI 使用指南

先用 [`util.md`](./util.md) 决定应选哪种能力；本页只说明它们怎样映射到纯 C/FFI。
头文件是 `hyle/c/util.h`。它只含 C 类型、opaque handle、函数指针和 header-inline 便利函数，可被 C 编译器及常见 FFI 直接消费。

当前 C 面属于依赖 container 的 util-runtime，不在本轮 util-core header seam 的严格证书内。
它仍以线程局部 `hyle_util_last_error()` 报失败；完整 util 收口时会迁到共享 error/
`hyle_status`，发布前不承诺保留当前签名。成功的可失败调用目前会清除旧错误。

当前面为 **44 个动态导出 + 2 个 header-inline 入口**。没有 api 表或版本门。

## 运行期反修饰与自旋 hint

| 入口 | 作用 |
|---|---|
| `hyle_util_last_error` | 返回当前线程最近一次 util C 边界错误；不要释放 |
| `hyle_util_demangle` | snprintf 语义反修饰；返回所需完整长度 |
| `hyle_util_cpu_relax` | header-inline 的单次 pause/yield hint；不是同步操作 |

```c
char name[256];
size_t needed = hyle_util_demangle(symbol, name, sizeof(name));
if (needed >= sizeof(name)) {
  /* name 已安全截断；按 needed+1 分配后可重试 */
}
```

## 随机数

`hyle_util_rng` 是可复制的 256-bit xoshiro256++ 状态，非密码学随机数。使用前必须 seed；全零状态非法。

| 入口 | 作用 |
|---|---|
| `hyle_util_rng_seed` / `hyle_util_rng_is_valid` | 初始化 / 校验状态 |
| `hyle_util_rng_next_u64` / `hyle_util_rng_next_u32` | 无符号整数 |
| `hyle_util_rng_next_bool` | 0 或 1 |
| `hyle_util_rng_next_float` / `hyle_util_rng_next_double` | `[0,1)` |
| `hyle_util_rng_range_u64` / `hyle_util_rng_range_u32` | 无偏闭区间 `[lo,hi]` |
| `hyle_util_rng_range_double` | 半开区间 `[lo,hi)`；边界须有限 |
| `hyle_util_rng_fill_u64` / `hyle_util_rng_fill_float` / `hyle_util_rng_fill_double` | DLL 内批量填充，摊薄 FFI 调用 |
| `hyle_util_rng_jump` / `hyle_util_rng_long_jump` | 前进 `2^128` / `2^192`，拆分并行子流 |
| `hyle_util_rng_shuffle` | 由 swap callback 完成 Fisher–Yates |
| `hyle_util_thread_rng` | 当前线程的非确定性便利实例；借用到线程结束 |
| `hyle_util_rng_clone` | header-inline 状态复制 |

```c
hyle_util_rng rng;
hyle_util_rng_seed(&rng, 12345);

uint64_t ids[1024];
if (!hyle_util_rng_fill_u64(&rng, ids, 1024)) {
  fprintf(stderr, "%s\n", hyle_util_last_error());
}

/* 每个 worker 一份确定子流 */
hyle_util_rng worker = hyle_util_rng_clone(&rng);
hyle_util_rng_jump(&rng);  /* 下一 worker 再 clone */
```

`thread_rng` 的 C POD 流与原生 C++ TLS 流相互独立；不要跨线程共享其地址。token、key、nonce 等安全材料必须使用操作系统 CSPRNG。

## 类型擦除共享所有权

| 入口 | 作用 |
|---|---|
| `hyle_util_ref_create` | payload + destroy callback，初始 strong count 为 1 |
| `hyle_util_ref_retain` / `hyle_util_ref_release` | 增减 strong count |
| `hyle_util_ref_get` | 借用 payload |
| `hyle_util_ref_use_count` / `hyle_util_ref_is_unique` | 诊断计数 / 是否唯一 |

```c
static void close_resource(void* object, void* ctx) {
  (void)ctx;
  resource_close(object);
}

hyle_util_ref* ref = hyle_util_ref_create(resource, close_resource, NULL);
hyle_util_ref* second = hyle_util_ref_retain(ref);
hyle_util_ref_release(second);
hyle_util_ref_release(ref); /* 同步调用 close_resource 一次 */
```

一次成功 retain 必须对应一次 release。最终 release 后，任何线程都不能再访问 handle。callback 不得抛 C++ 异常穿过 C 边界。

## 类型擦除 scope cleanup

| 入口 | 作用 |
|---|---|
| `hyle_util_scope_guard_create` | 保存 cleanup callback+ctx，初始 active |
| `hyle_util_scope_guard_run` | 提前执行一次并 dismiss |
| `hyle_util_scope_guard_dismiss` | 取消 cleanup |
| `hyle_util_scope_guard_is_active` | 查询 active |
| `hyle_util_scope_guard_destroy` | active 时执行一次，再销毁；NULL 安全 |

C 没有可移植的析构器，所以它不能复制 C++ “离开词法作用域自动执行”的语法；能力通过 opaque handle 可达，调用方应在自身语言的 `defer`/`finally` 中调用 destroy。

```c
hyle_util_scope_guard* cleanup =
    hyle_util_scope_guard_create(close_file_ctx, file);
if (cleanup == NULL) return 0;

/* ... */
hyle_util_scope_guard_destroy(cleanup); /* 正常/错误出口都调用 */
```

handle 非线程安全；callback 不得销毁自己的 guard。

## 运行期 registry

Registry 拥有 key/help 的副本；factory 返回值归调用方。所有注册、创建选择和查询均线程安全。

| 入口 | 作用 |
|---|---|
| `hyle_util_registry_create` / `hyle_util_registry_destroy` | 生命周期 |
| `hyle_util_registry_register` | factory+ctx 注册；高 priority 覆盖，低 priority 忽略，同 priority 报错 |
| `hyle_util_registry_has` / `hyle_util_registry_size` | 查询 |
| `hyle_util_registry_create_object` | 锁外调用选中的 factory |
| `hyle_util_registry_help` | snprintf 语义读取 help |
| `hyle_util_registry_for_each` | visitor 返回 0 停止后续 callback |

```c
static void* make_engine(void* factory_ctx, const void* args) {
  return engine_create(factory_ctx, args);
}

hyle_util_registry* reg = hyle_util_registry_create();
hyle_util_registry_register(reg, "fast", make_engine, config,
                            HYLE_UTIL_REGISTRY_PREFERRED, "fast engine");
void* engine = hyle_util_registry_create_object(reg, "fast", &options);
```

factory 在锁外调用，可以重入 registry；visitor 在 registry 锁内调用，不能重入同一个 registry。destroy 前必须让所有使用者静止。

## SpinLock

| 入口 | 作用 |
|---|---|
| `hyle_util_spin_lock_create` / `hyle_util_spin_lock_destroy` | opaque 生命周期；destroy 拒绝 locked handle |
| `hyle_util_spin_lock_lock` / `hyle_util_spin_lock_try_lock` / `hyle_util_spin_lock_unlock` | Lockable 操作 |
| `hyle_util_spin_lock_is_locked` | 诊断快照，不是所有权证明 |

这是给 FFI 控制路径的能力接缝。原生 C++ 热路径应直接用 header-inline `hyle::util::SpinLock`。它只适合几个指令的低争用临界区；高争用、可能阻塞或 fiber 内临界区使用相应 mutex/fiber-aware 原语。

## 有意仅 C++ 的能力

| 能力 | 原因 |
|---|---|
| `type_name<T>` / `type_hash<T>` | 依赖编译器对静态模板参数 `T` 的 pretty-function 替换；运行时 `void*` 没有可恢复的规范化类型信息。导出“调用方传字符串再哈希”会改变契约，不是假装补全 |
| `HYLE_DEFER` 语法 | C 没有可移植词法析构；cleanup 能力已由 C scope-guard handle 覆盖 |
| 原生侵入式 `Ref<T>` 布局 | 模板类型不冻结进 ABI；共享所有权能力由 type-erased C ref 覆盖 |
