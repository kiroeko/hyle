# util-04：random（xoshiro256++ PRNG）

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
- **thread_rng()**:每线程一个 Rng(无锁、无实例),栈地址播种区分线程。

## clay API diff + 决策

- 覆盖:next(→next_u64)/bool/u32/double/float/range_*/shuffle/seed/jump。
- **shuffle 改进**:clay `shuffle(void*, element_size, n)`;HYLE `shuffle(n, swap_callback)`——调用方拥有存储,更灵活,无 void* memcpy。
- **简化登记**:clay `next_n(批量填数组)` 未单列——调用方循环 `next_u64()`(FORCE_INLINE 无开销)等价。

## 对抗式自审（六类）

- **① 正确性**:xoshiro256++ 输出/状态更新照官方;Lemire threshold = (UINT64_MAX-n+1)%n = 2^64%n(数学验证正确);span==UINT64_MAX 特判(n=0 会除零)直接返回全宽;seed 0 经 SplitMix64 非全零。
- **② 性能**:全 header inline,next_u64 无分支;range 平均 <1 次拒绝(拒绝率 = (2^64%n)/2^64,极低);比 clay 的 `%` 略慢但无偏——正确性优先且代价微乎。
- **③ 规范**:header-only;无 std 拥有型;无 OS 依赖;FORCE_INLINE 热路径。
- **④ clay 覆盖**:全覆盖 + 4 项改进;next_n/shuffle 决策已记。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:无,单类封装。
- **非加密**:已注释——xoshiro 不抗预测,不可用于密钥/token(那是将来 crypto 模块的事)。
- **thread_rng 播种**:栈地址低位可能相近,靠 SplitMix64 扩散放大差异;真要强独立性用显式 seed 或 jump()。已注释。

## 交付物

- `include/hyle/util/random.h`（Rng + thread_rng，header-only）
- `tests/util/random_test.cc`（12）

## 参考

- clay `sys/random.h`(112)。xoshiro256++（Blackman/Vigna 推荐通用 PRNG）。Lemire "nearly divisionless" 无偏有界随机。
