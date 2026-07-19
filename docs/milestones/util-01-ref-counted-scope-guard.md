# util-01：ref_counted (Ref<T>) + scope_guard (HYLE_DEFER)

> **2026-07-19 收口复审**：本文件保留首次里程碑的实现史；当前合同与验证以
> [`util-module-closure.md`](./util-module-closure.md) 为准。复审新增批量计数
> 零/溢出/下溢 fail-fast、危险 CRTP 析构形状的编译期拒绝，以及 Ref/ScopeGuard
> 类型擦除 C ABI。

| 字段 | 值 |
|---|---|
| **阶段** | P3（util 簇，无 OS 依赖，可早做）|
| **模块** | util/（新建，header-only INTERFACE lib）|
| **Tier** | 0 |
| **LOC 估** | 1k |
| **骨架?** | 否 |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台 + WSL TSan；util 0→14 case）|

> **cloc 留档**：**生产 168 行**（ref_counted.h + scope_guard.h，header-only）+ **测试 186 行**。
> **clay 对比 391（ref_counted 317 + scope_guard 74）→ HYLE 168，不到一半**：现代化红利 + 故意删 footgun，**非 stub**——① C++23 concepts(`Refable`/`std::convertible_to`) 替 clay 每方法拖的 `enable_if` SFINAE 链；② CTAD 省掉 clay 的 `scope_guard()`/`finally()` 两个工厂函数；③ 删 clay 的 `num_refcounted_objects` 全局原子计数 + 隐式 `operator T*` + 一打冗余比较运算符。14 个测试(含 CRTP destroy 回归 + 8 线程 TSan)证明完整。

## clay API diff（ref_counted.h + scope_guard.h）

- **RefCounted<Sub>**：incref/decref(带 count)/is_unique/use_count 全覆盖。
- **Ref<T>**：构造/拷贝(share)/移动(steal)/upcast/reset/release/swap/`operator*`/`->`/`bool`/比较 全覆盖；工厂 make_ref(clay make_ref)/adopt_ref(clay take_ref)/retain_ref(clay make_ref_from) 全覆盖。
- **ScopeGuard + HYLE_DEFER**：clay scope_guard/finally/CLAY_DEFER 覆盖。

## 相对 clay 的改进（对抗自审驱动）

1. **修 clay 的 destroy() CRTP-dispatch bug**：clay decref 调**无限定** `destroy()`,名字解析到**基类** destroy——子类 override 永远不被调用。HYLE 用 `static_cast<Sub*>(this)->destroy()` 真正分发。**`CustomTeardown` 测试是这个 bug 的回归守卫**（clay 版会得 0 次自定义析构,HYLE 得 1 次）。
2. **decref 内存序优化（性能）**：clay 用 `acq_rel`（每次递减都付 acquire）。HYLE 用 **release + 仅末次 `atomic_thread_fence(acquire)`**（Boost/libc++ shared_ptr 经典实现）——非末次 decref 只付 release。WSL TSan 重验零 race。
3. **删隐式 `operator T*`**：clay 有,招致 `delete ref`/静默退化。HYLE 对齐 std::unique_ptr/shared_ptr,只给 get()。
4. **删全局 `num_refcounted_objects` 原子计数**：clay 每次构造/析构都 RMW 一个进程级原子（争用热点,perf-hostile）。HYLE 删之,泄漏靠 ASan 抓。
5. **转换/工厂显式化**：删 clay 隐式 `Ref(T*)`（adopt 还是 retain 在调用点说不清的 footgun,clay 自己注释都警告）→ 强制 `adopt_ref`/`retain_ref`/`make_ref` 三选一。

## 对抗式自审（六类）

- **① 正确性**：incref relaxed（新引用只能从已有引用形成）/ decref release+末次 acquire fence（教科书+优化版,TSan 验）；`~RefCounted` protected 非虚（禁基类指针 delete）；CRTP destroy 分发正确。
- **② 性能**：decref 省每次 acquire（见改进2）；正常 incref 仍是一条 relaxed
  RMW，零/溢出检查是不可预测误用的 fail-fast 护栏；Ref 一指针、无控制块。
- **③ 规范**：符合 conventions §6（intrusive Ref + relaxed/acq_rel + explicit 转换）、§8（ScopeGuard CTAD + DEFER）；无 std 拥有型出头。
- **④ clay 覆盖**：见上 diff，全覆盖 + 5 项改进。**简化登记**：ScopeGuard 只单向 `dismiss()`，未做 clay 的 `enable()` 重启用（defer 场景不需要,标注非缺口）。
- **⑤ 残留脚手架**：无。
- **⑥ 过度复杂**：Ref 比较运算符精简到 ==(Ref)/==(T*)/<=>（C++20 自动补 !=），未抄 clay 一打重载。

## 交付物

- `include/hyle/util/ref_counted.h`（RefCounted<Sub> CRTP + Ref<T> + make/adopt/retain_ref）
- `include/hyle/util/scope_guard.h`（ScopeGuard CTAD + HYLE_DEFER）
- `src/util/CMakeLists.txt`（INTERFACE lib）+ root CMake wire
- `tests/util/ref_counted_test.cc`（9）+ `scope_guard_test.cc`（5）

## 参考

- clay `utils/ref_counted.h`(317) / `sys/scope_guard.h`(74)。
- conventions §6（intrusive RefCounted/Ref,改进:explicit 转换）、§8（ScopeGuard/DEFER,CTAD 简化）、§11 速查表。
- Boost/libc++ shared_ptr 的 release+acquire-fence decref 内存序（性能改进 2 的出处）。
