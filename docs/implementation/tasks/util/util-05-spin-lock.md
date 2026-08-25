# util-05：spin_lock（TTAS + pause + padding）

> **2026-07-19 收口复审**：补齐 MSVC x86/ARM intrinsic 分叉、三平台基准和
> opaque C ABI。结论不是“总比 mutex 快”：它只适合极短低争用区，Linux
> 无争用与 Apple M4 八线程高争用都观察到 mutex 更快。完整数据见
> [`util-module-closure.md`](util-module-closure.md)。

| 字段 | 值 |
|---|---|
| **阶段** | P3（util 簇）|
| **模块** | util/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台 + WSL TSan；util 41→46 case）|

> **cloc**：**生产 36 行**（spin_lock.h，header-only）+ **测试 69 行**。
> **clay 对比 43 → HYLE 36**:相近体量,**功能更强**(TTAS + pause + 缓存行 padding,见下)。5 测试(含 8 线程互斥 + TSan)证明非 stub。

## 相对 clay 的改进（性能/正确性）

clay `spin_mutex_t` 是**裸 test-and-set**:每次自旋都 `test_and_set`(一次原子 RMW = 写),在争用时让缓存行在核间反复弹跳。HYLE `SpinLock`:

1. **TTAS(test-test-and-set)**:先用 relaxed `load` 自旋,只在锁看起来空时才发 `exchange` RMW——等待时在本地缓存自旋,不弹缓存行,owner release 时才重试。
2. **cpu_relax() pause 提示**:`_mm_pause`/`__builtin_ia32_pause`/ARM `yield`——省电、让出兄弟超线程、加速退出自旋。
3. **缓存行 padding**(`HYLE_ALIGNED`):flag 独占一行,不与邻近热变量伪共享。
4. **满足 Lockable**:lock/try_lock/unlock 直接配 `std::lock_guard`/`scoped_lock`——丢掉 clay 冗余的 `spin_lock_t` RAII 包装。

## 对抗式自审（六类）

- **① 正确性**:lock 用 exchange(acquire)/unlock 用 store(release)——release 同步到下个 acquire,临界区写对后续 owner 可见;8 线程各 +10万计数总和精确(无丢更新=互斥成立),TSan 验零 race;try_lock relaxed 预检 + exchange(acquire)。
- **② 性能**:TTAS 免争用时缓存行弹跳;pause 提示;padding 防伪共享。
- **③ 规范**:header-only;`std::atomic<bool>`(conventions:catomics→std::atomic + 自旋锁 padding);满足标准 Lockable。
- **④ clay 覆盖**:lock/try_lock/unlock 覆盖 + 4 项改进;clay spin_lock_t RAII → 用 std::lock_guard。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:无。
- **用途边界**:已注释——仅限**极短**临界区(几条指令);争用/长持有用真 mutex(自旋烧整个核);阻塞 OS 线程非 fiber(sched-05 fiber-aware sync 另做)。

## 交付物

- `include/hyle/util/spin_lock.h`（SpinLock + cpu_relax）
- `tests/util/spin_lock_test.cc`（5，含 8 线程互斥 + try_lock 排他）

## 参考

- clay `utils/spin_lock.h`(43)。conventions §11（catomics→std::atomic,留自旋锁 padding）。TTAS + pause 是教科书自旋锁优化。
