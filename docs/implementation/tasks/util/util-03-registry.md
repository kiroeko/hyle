# util-03：registry 自注册 + DECLARE/DEFINE 宏

> **历史任务页**：本文件保留首次实现史，行数和测试数不是当前快照。2026-07-30 的当前合同
> 只见 [`util-module-closure.md`](util-module-closure.md)：Registry 现在拥有 key/help，
> creator/visitor 均在锁外可重入，同一 Swiss index 经 O(1) freeze 发布，并有 FrozenView 与
> 共享 Status/allocator 的类型擦除 C ABI。

| 字段 | 值 |
|---|---|
| **阶段** | P3（util 簇，完成 util 簇）|
| **模块** | util/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；util 23→29 case，含 death test）|

> **cloc**：**生产 94 行**（registry.h）+ **测试 92 行**。
> **clay 对比 341 → HYLE 94（不到 1/3）**:一套泛型 `Ptr` 模板替掉 clay 的 typed/ref/shared **三套宏** + **三个 std::unordered_map**(creator/priority/help) → 一个 SwissTable + 函数指针 Entry。功能等价(优先级覆盖/help/create/keys/has),更快(Entry 内联无堆),非 stub:6 测试含 death test。

## clay API diff + 四项改进（对抗自审/规范驱动）

- **覆盖**:register/has/create/keys(→for_each_key)/help/优先级(Fallback/Default/Preferred) 全覆盖。
- **改进①(§1 铁律)**:存储用 HYLE **SwissTable**,非 clay 的 **3 个 std::unordered_map**。
- **改进②(§4 ABI + 性能)**:Creator 是**函数指针**非 `std::function`——Entry `{fn-ptr, priority, help}` 变 trivially-copyable,**内联进 SwissTable**(无堆分配/类型擦除),调用无 std::function 间接。registry creator 是无捕获的类型构造器,不需要闭包。
- **改进③(§0 fail-fast + §5 no-throw)**:普通动态注册的同 key 同优先级冲突返回 `AlreadyExists`；静态 REGISTER 宏因没有错误接收者而 fail-fast，替 clay 的 `std::exit(1)`/`throw`。
- **改进④(§6)**:DEFINE 宏用 **Meyers function-local static**,替 clay 的 leaky `new` 单例。
- **改进⑤**:help 文本用 `type_name<T>()`(util-02),替 clay 的 demangle。
- **泛型化**:clay 有 REGISTRY/REGISTRY_REF/SHARED_REGISTRY 三套
  (unique_ptr/Ref/shared_ptr)；HYLE 一个 `Registry<Ptr, Args...>`。默认 creator
  按 raw pointer、拥有型指针、侵入式 `Ref` 分形状构造，不再假设
  `Ptr{new D}` 对所有智能指针都成立。

## 对抗式自审（六类）

- **① 正确性**:优先级 higher 覆盖/equal fail-fast/lower no-op 正确;Entry trivially-copyable 验证(函数指针+enum+string_ref 皆 trivial)故可进 SwissTable;create 未命中返回 `Ptr{}`(指针→nullptr)。
- **② 性能**:Entry 内联表中(无 std::function 堆+类型擦除);Creator 函数指针直调。比 clay 三 map + std::function 显著轻。
- **③ 规范**:四项改进全踩 §1/§4/§0/§6。
- **④ clay 覆盖**:全覆盖 + 三套收一套;**简化登记**:clay SetTerminate(测试切 throw)未做——HYLE 直接 EXPECT_DEATH 验 fail-fast,不需要可切换终止。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:比 clay 341 行精简到 94,三套宏/三 map 收敛。
- **键生命周期契约**：Registry 复制并拥有动态 key/help；返回 view 在 Registry 析构前稳定。
- **线程契约**：mutable register/create/has/help 同步；creator/visitor 在锁外，允许重入。
  mutable 枚举使用时间点快照；freeze 后查找/枚举无锁，FrozenView 直达同一不可变索引。

## 交付物

- `include/hyle/util/registry.h`（Registry<Ptr,Args...> + Registerer + DECLARE/DEFINE/REGISTER 宏）
- `tests/util/registry_test.cc`（6，含 SamePriorityCollisionAborts death test）

## 参考

- clay `utils/registry.h`(341)。conventions §9（Registry + DECLARE/DEFINE,改进:flat hashmap 换 unordered_map、禁 std::exit）、§1/§4/§6。
