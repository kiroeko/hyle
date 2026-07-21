# container 实测铁证（导航 + 当前真相）

> 本页是 [performance 总账](./README.md) 的 **container 模块**入口。  
> **详表与优化史**（长）：[benchmarks-mem-container.md](./benchmarks-mem-container.md)（mem 已迁出，该文件以 container 为主）。  
> **模块收口**（正确性/C ABI，非性能全表）：[../milestones/container-module-closure.md](../milestones/container-module-closure.md)（2026-07-19）。  
> 队列/并发容器的 SOTA 对标多在 sched 消费路径 + `queue_sota_bench`；本页聚焦 hash / map / vec / list / str / path / fuzzy。

## 0. 数据新鲜度（先看这个）

| 面 | 证书状态 | 说明 |
|---|---|---|
| **StringBuf / string 收口表** | ✅ **2026-07-19 三平台** | Win / WSL / M4；lifecycle 落后已诚实登记 |
| **path** | ✅ 双平台 n=15 | Win+Lin；算法基线 + filesystem 分离 |
| **SwissTable map** | ⚠️ **2026-07 快照**（多 Win/WSL） | 总账有 ns 行；**非**本轮 mem 式三平台 sign 全刷 |
| **hash / mix64 / Hash&lt;T&gt;** | ⚠️ 历史 + 红队公平性更正 | ⏳ 待三平台 `bench-stat -n15` 重写表 |
| **SmallVector / IntrusiveList / CArray** | ⚠️ 主 Win + 部分补测 | IntrusiveList 平 boost 已测 |
| **fuzzy path match** | ⚠️ 主 Linux | O(N) scaling 证毕 |
| **并发队列** | 见 queue / sched 文 | 不在本页展开 |

**读法**：标 ✅ 的可当当前产品结论；标 ⚠️ 的方向大体可信，但**不要**当 2026-07-20 式三平台 sign 证书去对外硬报绝对 ns。

## 1. 总览判定（对真 SOTA，非只赢 std）

| 组件 | 真 SOTA 标尺 | 判定（当前文档共识） |
|---|---|---|
| **mix64 / Hash&lt;T&gt; 小 POD** | wyhash / abseil 形 | ✅ 赢或 SOTA 同款（整数 / 16B POD） |
| **hash_bytes ≤16B** | XXH3-small / wyhash | ➖ **inline mix**（不同算法）；非「输 XXH3 wrapper」 |
| **hash_bytes 中长串** | 裸 XXH3 / wyhash | ➖ 默认出线核有 seam；opt-in inline / best-of wyhash(&gt;128B) |
| **SwissTable** | absl::flat_hash_map | ✅ find/insert/erase/iterate **追平或反超**（iterate 组扫 2x absl） |
| **SmallVector** | absl::InlinedVector | ✅ 小 N churn **赢** 同款 SOTA |
| **CArray** | std::vector | ✅ growth 赢（mimalloc 质量）；read 平 |
| **IntrusiveList** | boost::intrusive::list | ✅ 赢 std；**平 boost** |
| **string_ref 搜索** | string_view / libc | ➖ 自研 SIMD 项平；子串/字符集 **转发 std**（刻意） |
| **StringBuf** | map+mutex / Absl 分片 peer | ⚠️ **Win 多平**；WSL/M4 lifecycle **输**；growth 赢；8t 距 Absl ~5% |
| **path 词法** | string_view 扫 / filesystem | ✅ 平算法基线；**碾** filesystem（零分配结构差） |
| **fuzzy** | 子序列 floor | ✅ O(N) 实证；bonus 付 2.35x 买排序质量 |

## 2. 非 SOTA / 冻结边界（诚实清单）

1. **StringBuf lifecycle（WSL/M4）**  
   intern / add-retain-remove / churn 相对同契约 peer 约 **0.73–0.86x**。  
   产品价值在 cached-hash growth、稳定 identity、并发接近强 peer、batch——**不是**「全生命周期常数三平台无条件赢」。旧 1.24x churn 证书已撤回。

2. **StringBuf 8-thread**  
   距 sharded Absl+mutex 约 **4–5%**（~0.95x），接近但非碾压。

3. **hash_bytes 默认 DLL 出线核**  
   与裸 XXH3 同算法时有固定 seam；短串热点用 `hash_bytes_inline` / 小 key 走 `Hash&lt;T&gt;` inline mix。

4. **string_ref 短输入 split/rfind/count**  
   4KB 点值可有 **0.83–0.96x**（DLL/dispatcher）；长输入摊平。

5. **hash / SwissTable 缺本轮三平台统一 sign 刷新**  
   方向（赢 absl）多次坐实；绝对 ns 与跨平台翻转未按 mem 纪律重跑。

6. **队列类**  
   能力已收口；性能叙事分散在 queue_sota_bench / sched——需要时单开，勿与 Swiss 表混谈。

## 3. 选型（不同场景不同 container 组成）

```text
整数 / 小 POD key 哈希     → Hash<T> / mix64（热路径）
长字节哈希                 → hash_bytes（>128B best-of wyhash；要消 seam → inline）
通用 hash map              → SwissTable（vs absl 已够强）
栈上小数组                 → SmallVector
堆动态数组                 → CArray（默认 SystemAllocator/mi）
侵入链表                   → IntrusiveList（= boost 形）
只读字符串视图 + 搜索      → string_ref（子串族借 std）
驻留串 / intern            → StringBuf（接受跨平台 lifecycle 税；要 growth/并发）
路径词法（无 syscall）     → path（零分配；勿用 filesystem 当算法对手）
quick-open 模糊路径        → fuzzy（O(N) top-K）
MPMC / stealing 队列       → bounded / sharded / WS deque（见 queue 文档）
```

## 4. 与 mem 的边界

- 容器**默认** `SystemAllocator`（平 mi）；注入 vtable 付多态税 → [benchmarks-mem.md](./benchmarks-mem.md)。  
- Swiss / StringBuf 内部用自有表与 slab，**不要**拿 MemoryArena 当通用 map 分配器。

## 5. 复跑（刷新 ⏳ 项时）

```bash
# hash
scripts/bench-stat.py <build>/bin/hyle_hash_bench -n 15 ...

# Swiss vs absl（需 -DHYLE_BENCH_ABSL）
scripts/bench-stat.py <build>/bin/hyle_swiss_vs_absl_bench -n 15 ...

# vec/list
scripts/bench-stat.py <build>/bin/hyle_container_misc_bench -n 15 ...

# string / StringBuf
scripts/bench-stat.py <build>/bin/hyle_string_ref_bench -n 15 ...
scripts/bench-stat.py <build>/bin/hyle_string_buf_bench -n 15 ...

# path
scripts/bench-stat.py <build>/bin/hyle_path_bench -n 15 ...
```

刷新后：改本页 §0 新鲜度 + 总账对应行；详表回填 [benchmarks-mem-container.md](./benchmarks-mem-container.md)。

## 6. 详表索引

| 主题 | 详节位置 |
|---|---|
| hash 公平性更正 + 长度表 | mem-container §container·hash |
| SwissTable vs absl | mem-container §container·map + README 总账 |
| SmallVector / list / CArray | mem-container §container·vec/list |
| string_ref / StringBuf 三平台收口 | mem-container §container·str |
| path | mem-container §container·path |
| fuzzy | mem-container §模糊路径 |
| util/obs 夹带 | 应看 util/obs 收口；mem-container「次要模块」勿当 container 主证书 |
