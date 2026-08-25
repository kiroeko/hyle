# util-04：random（xoshiro256++ PRNG）

> **2026-07-19 收口复审**：当前能力已扩为 state snapshot/restore、原生 u32
> Lemire、`long_jump`、极端有限 double range、进程唯一 TLS seed，并提供 POD
> + batch 的纯 C ABI。当前真相见
> [`util-module-closure.md`](util-module-closure.md)。

| 字段 | 值 |
|---|---|
| **阶段** | P3（util 簇）|
| **模块** | util/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；util 29→41 case）|

> **cloc**：**生产 99 行**（random.h，header-only）+ **测试 103 行**。
> **clay 对比 255（random.h 112 + random.cc 143）→ HYLE 99 header-only**:clay 是全局状态 + vtable api_t + 一堆 inline 包装;HYLE 一个 `Rng` 类封装,无全局/无 .cc。精简**且更强**(见下)。12 测试(含 160 万抽样均匀性、jump 发散、shuffle 排列)证明非 stub。

## 相对 clay 的改进（对抗自审/正确性驱动）

- **算法升级**:clay 用 **xoroshiro128+**(低位质量弱);HYLE 用 **xoshiro256++**——推荐的通用后继,高/低位均佳,同速度级。
- **无偏 range(真 bug 修复)**:clay `random_range = min + r % (max+1-min)`——**统计有偏**(除非范围整除 2^64)。HYLE 用 **Lemire 拒绝采样**(threshold = 2^64 % n,拒绝采样区),无偏。160 万抽样均匀性测试验证(16 桶各 ±5%)。
- **SplitMix64 seeding**:单 u64 种子扩散填 256 位状态,避免 xoshiro 的全零禁态(seed 0 也安全,测试 NotStuckOrZero 验);clay 把 seeding 丢给调用方。
- **jump()**:前进 2^128 步,给并行 worker 各一条不重叠流(sched 每 worker 一个)——测试验发散。
- **thread_rng()**：每线程一个 Rng、无锁；默认 seed 来自进程唯一原子序列和
  进程 salt，不再依赖可被 OS 复用的线程栈地址。

## clay API diff + 决策

- 覆盖:next(→next_u64)/bool/u32/double/float/range_*/shuffle/seed/jump。
- **shuffle 改进**:clay `shuffle(void*, element_size, n)`;HYLE `shuffle(n, swap_callback)`——调用方拥有存储,更灵活,无 void* memcpy。
- **批量能力**：原生 C++ 热循环可直接内联；C/FFI 面另提供 fill_u64/float/double，
  避免每个元素一次 ABI 调用。

## 对抗式自审（六类）

- **① 正确性**:xoshiro256++ 输出/状态更新照官方;Lemire threshold = (UINT64_MAX-n+1)%n = 2^64%n(数学验证正确);span==UINT64_MAX 特判(n=0 会除零)直接返回全宽;seed 0 经 SplitMix64 非全零。
- **② 性能**:全 header inline,next_u64 无分支;range 平均 <1 次拒绝(拒绝率 = (2^64%n)/2^64,极低);比 clay 的 `%` 略慢但无偏——正确性优先且代价微乎。
- **③ 规范**:header-only;无 std 拥有型;无 OS 依赖;FORCE_INLINE 热路径。
- **④ clay 覆盖**:全覆盖 + 4 项改进;next_n/shuffle 决策已记。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:无,单类封装。
- **非加密**:已注释——xoshiro 不抗预测,不可用于密钥/token(那是将来 crypto 模块的事)。
- **thread_rng 播种**：默认流保证同进程各次线程初始化不因栈复用而重复；需要
  可复现、可证明不重叠的并行流时仍应显式 seed，并按 worker `jump()`。

## 交付物

- `include/hyle/util/random.h`（Rng + thread_rng，header-only）
- `tests/util/random_test.cc`（12）

## 参考

- clay `sys/random.h`(112)。xoshiro256++（Blackman/Vigna 推荐通用 PRNG）。Lemire "nearly divisionless" 无偏有界随机。
