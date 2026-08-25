# container-02：XXH3 默认 hasher（历史施工步骤）

> **历史任务提示（2026-08-25）：**本页记录最初引入 XXH3 时的默认形状。当前 `hash_bytes`
> 已按长度组合 inline stream、baseline wyhash 与 runtime-gated wide XXH3；默认 string hash、
> typed hash 和显式 SipHash 安全策略见 [`container-design`](../../../architecture/container-design.md)。
> 现行认证只看 [`pre-release-readiness`](../../status.md)。

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 1k |
| **骨架?** | |
| **状态** | 🗃️ 当轮完成（Win x64 MSVC + Linux x64 GCC14.2 双平台，container 19→27 case）|

> **cloc 留档（`cloc hash.h hash.cc hash_test.cc`）**：**133 行** = 59 头 + 74(hash.cc 5 行 + 测试)。薄适配 + 强测试,非 stub。
> **当轮实现/自审**：① xxhash.h(7490 行,BSD-2)vendored,`XXH_INLINE_ALL` 仅在 hash.cc 内含 → **公开头 hash.h 经 grep 确认仅注释提及 XXH3、无 #include/符号**(ABI 干净 + 不拖编译)。② 当时小键(整型/枚举/指针)内联 splitmix64,大键(string_view/字节)走 XXH3。③ murmur 留编译期(本步不动)。
> **双平台抓修两处(真 build 实测暴露)**：① 测试用了 `__builtin_popcountll`(GCC-only)→ MSVC C3861 → 改 C++20 `std::popcount`(`<bit>`,可移植)。② `IntegralSpreadsLowBits` 测试**数学上错的**:期望 4096 整数产生 >4000 个不同的 12 位桶,但鸽笼原理下任何好混合器最多 ~2590(`4096·(1-1/e)`),只有 identity 哈希才到 4096——阈值反了。改测 splitmix64 的**双射性**(4096 整数→4096 个不同的完整哈希,零碰撞)+ 高位被填充(非 identity)。
> **硬指标达成**:string_view 对 1 万不同串零碰撞;整型双射零碰撞 + 高位扩散。

## 前置依赖

- 硬前置：`00-scaffolding`。被 `container-04`(SwissTable) 依赖。
- 当轮决策依据是旧版 hyle-scope/container-design 的 XXH3 与 H1/H2 讨论；现行 table-hash
  分层只看 [`container-design`](../../../architecture/container-design.md)，不依赖已漂移的文档行号。

## 目的

哈希是 hashmap、内容寻址、string_buf 驻留池的共同基元。本步当时把 **XXH3**(xxHash 的现代变体,SIMD 加速,业界最快通用哈希之一)接成 HYLE 默认运行期 hasher,并定下初版 `Hash<T>` 约定。

## 关键设计决策

### D1：XXH3 借入,封 .cc 不外露 xxhash 类型
- xxhash.h(7490 行,BSD-2)vendored 到 `third_party/xxhash/`。**用 `XXH_INLINE_ALL` 在 `hash.cc` 内部包含**,只导出 `hash_bytes(const void*, size_t, uint64 seed)->uint64`。公开头 `hash.h` **不含 xxhash.h**(ABI 干净 + 不让 7490 行进每个 TU 拖垮编译)。同 ggml/mimalloc 解耦原则。

### D2：小键内联混合,大键走 XXH3
- **整型/枚举/指针**：`Hash<T>` 内联 64 位 avalanche(splitmix64 式 finalizer),**不调 XXH3**——小键的函数调用开销 > 哈希本身。
- **string_view / 字节缓冲**：走 `hash_bytes`(XXH3),长键 XXH3 的吞吐优势压倒调用开销。
- `Hash<T>` 默认模板 + 针对 integral/pointer/string_view 的特化;用户类型可特化 `Hash<T>`。

### D3：murmur 留编译期
- 该轮运行期一律 XXH3。这个决定后来被长度分层的产品实现取代；clay 的 murmur **只在编译期 type_name 哈希**(constexpr 友好)保留——那是 container-06/type_name 的事,本步不动。

## 具体事务

- [ ] vendor `third_party/xxhash/xxhash.h` + LICENSE。
- [ ] `include/hyle/container/hash.h`：`hash_bytes` 声明(HYLE_CONTAINER_API)、`Hash<T>` 模板 + 整型/指针/string_view 特化、`Hashable` concept。
- [ ] `src/container/hash.cc`：`XXH_INLINE_ALL` + xxhash.h,实现 `hash_bytes` = `XXH3_64bits_withSeed`。
- [ ] container 模块从 INTERFACE 升为编译库(首个 .cc),加 HYLE_CONTAINER_API 三态宏。
- [ ] **不做**：SwissTable(container-04)、string_buf hash(container-06)、编译期 murmur。

## 验收标准

- **编译**：Win x64 + Linux x64;xxhash 不进公开头(grep 公开头无 xxhash)。
- **测试**：hash_bytes 确定性(同输入同输出)、雪崩(改 1 字节大幅变)、空输入不崩、seed 影响;Hash<int/ptr/string_view> 可用且分布合理(不同键极少碰撞,统计若干键无重复)。
- **硬指标**：`Hash<std::string_view>` 对 1 万条不同字符串零碰撞(64 位);整型 Hash 对连续整数不退化为恒等(高位有扩散)。
- **基准(留 bench 阶段)**：对标 std::hash / absl::Hash 吞吐。

## 参考

- xxHash（Yann Collet，BSD-2）XXH3。
- `hyle-scope.md` 行77；`container-design.md`。
