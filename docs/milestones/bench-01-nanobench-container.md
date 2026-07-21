# bench-01：nanobench 基础设施 + 容器基准对标 std

| 字段 | 值 |
|---|---|
| **阶段** | P3（验证设施，非 clay 吸纳——clay 无 bench）|
| **模块** | bench/（新建）|
| **状态** | ✅ 完成（WSL x64 GCC14.2 -O3 真实数据；nanobench vendored）|

> **背景**：`bench/` 目录此前**不存在**,但 root CMakeLists 引用 `add_subdirectory(bench)`(HYLE_BUILD_BENCH=ON 会配置失败),且 roadmap P3 门槛白纸黑字要求"hashmap 对拍 std + 基准"。本里程碑补上这个诚实缺口——用客观数据验证"超越 std"不是空话。
> **基础设施**：nanobench(charter 选定)单头 vendored 到 `third_party/nanobench/`(免 FetchContent 网络依赖,两平台可用);`bench/nanobench_impl.cc` 编译 impl 一次;`hyle_add_bench()` 已在 HyleModule.cmake。

## 真实基准数据（WSL Ubuntu24.04 / g++-14 / -O3 / 100k uint64 keys）

> ⚠️ **这是 bench-01 当时(mix64/prefetch/H1 优化*前*)的快照,作历史留档。当前权威值见 [`../performance/benchmarks-mem-container.md`](../performance/benchmarks-mem-container.md)**（container·hash / SwissTable 段；⏳ 待三平台刷新）。本节数字不代表当前性能。

**SwissTable vs std::unordered_map**（relative = std 基线,越高越快）:

| 操作 | SwissTable 相对 std | 绝对 ns/op (Swiss / std) |
|---|---|---|
| find-hit | **2.72x** | 3.08 / 8.38 |
| find-miss | **2.28x** | 5.11 / 11.64 |
| insert (fresh) | **6.10x** | 10.37 / 63.28 |
| erase (build+erase) | **4.11x** | 14.69 / 60.39 |

> SwissTable 四项全面碾压 std::unordered_map——SIMD 16 路 group 探测 + 平坦布局(无节点分配/指针追逐)兑现了"超越 std"的核心卖点。insert 6.1x 尤其显著(std 每插入一个节点一次 malloc;SwissTable 平摊 grow)。

**StringBuf intern vs std::unordered_map interner**（100k 流 / 2k 唯一,~50x 重复）:

| 版本 | 相对 std | ns/op |
|---|---|---|
| intern = lookup(add(s)) 初版 | **0.94x（慢 6%）** | 17.39 |
| intern = 单锁单 find（优化后）| **1.31x（快 31%）** | 12.90 |

> **benchmark 驱动优化的教科书案例**:初版 intern 比 std 慢——基准客观坐实了我对抗自审时标记的"intern=add+lookup 两次锁/两次 find/两次 hash"。修复:抽 `find_or_add_locked` 让 add/intern 共用,intern 一把锁内一次 find 完成(热路径=重复串直接返回)。重测 0.94x→1.31x,13 个测试 + WSL TSan 仍全过。

## 跨平台对照（Win x64 MSVC 19.50 -O2，同 100k）

| 操作 | SwissTable 相对 std (MSVC) | （对照 WSL）|
|---|---|---|
| find-hit | **3.30x** | 2.72x |
| find-miss | **2.27x** | 2.28x |
| insert | **4.77x** | 6.10x |
| erase | **4.75x** | 4.11x |
| StringBuf intern | **1.01x（持平）** | 1.31x |

> 两平台 SwissTable 全面超 std(MSVC 的 std::unordered_map 更慢→相对优势在 erase 更大)。**诚实差异:StringBuf intern 在 MSVC 仅持平 std**——MSVC 的 `std::mutex` 比 Linux futex 重,每次 intern 一把锁的开销吃掉了 SwissTable 的查找优势。结论:StringBuf 的价值在**稳定指针 + 引用计数 + 跨 DLL + 线程安全**,不是单线程吞吐;若单线程热路径要极致 intern,后续可出无锁/单线程变体(登记 bench-02/string_buf 优化)。

## 对抗式自审

- **基准公平性**:std 是 relative=1.0 基线;insert/erase 两边都付 fresh 构造+allocation(同等条件);doNotOptimizeAway 防 DCE;100k 规模够大摊销噪声(err% 多在 1-4%)。
- **诚实**:StringBuf 初版**慢于 std** 如实印出并记录,未藏;优化后才超。这正是用户要的"行数/性能远低于预期要解释"的反面——用数据说话。
- **已知局限**:① find-hit 的 SwissTable err% 8.1%(nanobench 提示 unstable,迭代少)——趋势可信(2.7x)但精确值有抖动,后续可加 minEpochIterations;② 只对标了 std,**未对标 absl::flat_hash_map / folly F14**(参考库已 clone,但接入它们的构建是独立工作)——roadmap P3 门槛的"对标 absl/F14"部分**登记 bench-02 待补**,不算完成;③ 单线程基准,StringBuf 的锁开销在多线程争用下的表现未测。
- **冷路径代价**:find_or_add_locked 新串路径 `insert 后再 find` 多一次 find(SwissTable::insert 返回 bool 非 slot 指针)。热路径(dup)无此代价。彻底省掉需 container-04 让 insert 返回 slot 指针——登记可议,非阻塞。

## 交付物

- `third_party/nanobench/nanobench.h`(vendored 单头)+ `bench/nanobench_impl.cc`
- `bench/CMakeLists.txt`(nanobench static lib,SYSTEM include 免 -Wall 噪声)
- `bench/container/{container_bench_main,swiss_table_bench,string_buf_bench}.cc`
- `src/container/string_buf.cc` 的 intern 优化(find_or_add_locked)

## 登记后续

- **bench-02**:对标 absl::flat_hash_map + folly F14(P3 门槛完整化);SwissTable find-hit 稳定性(minEpochIterations);多线程 StringBuf 争用基准。
