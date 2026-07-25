# container 实测铁证（导航 + 当前真相）

> 本页是 [performance 总账](./README.md) 的 **container 模块**入口。  
> **详表与优化史**（长）：[benchmarks-mem-container.md](./benchmarks-mem-container.md)（mem 已迁出，该文件以 container 为主）。  
> **模块收口**（正确性/C ABI，非性能全表）：[../milestones/container-module-closure.md](../milestones/container-module-closure.md)（2026-07-19）。  
> 队列/并发容器的 SOTA 对标多在 sched 消费路径 + `queue_sota_bench`；本页聚焦 hash / map / vec / str / path。IntrusiveList 已归 core，见 [`benchmarks-core`](./benchmarks-core.md)。

## 0. 数据新鲜度（先看这个）

| 面 | 证书状态 | 说明 |
|---|---|---|
| **StringBuf / string 收口表** | ✅ **2026-07-19 三平台** | Win / WSL / M4；lifecycle 落后已诚实登记 |
| **path** | ✅ 双平台 n=15 | Win+Lin；算法基线 + filesystem 分离 |
| **SwissTable map** | ⚠️ **2026-07 快照**（多 Win/WSL） | 总账有 ns 行；**非**本轮 mem 式三平台 sign 全刷 |
| **hash / mix64 / Hash&lt;T&gt;** | ⚠️ 历史 + 红队公平性更正 | ⏳ 待三平台 `bench-stat -n15` 重写表 |
| **SmallVector / CArray** | ⚠️ 主 Win + 部分补测 | owning vector 形状待本轮统一三平台刷新 |
| **ConcurrentVector** | ✅ Windows+WSL n=15；⏳ M4 | append/range iteration 同合同 SOTA；WSL `operator[]` 是已知缺口 |
| **并发队列** | 见 queue / sched 文 | 不在本页展开 |

**读法**：标 ✅ 的可当当前产品结论；标 ⚠️ 的方向大体可信，但**不要**当 2026-07-20 式三平台 sign 证书去对外硬报绝对 ns。

## 1. 总览判定（对真 SOTA，非只赢 std）

| 组件 | 真 SOTA 标尺 | 判定（当前文档共识） |
|---|---|---|
| **mix64 / Hash&lt;T&gt; 小 POD** | wyhash / abseil 形 | ✅ 赢或 SOTA 同款（整数 / 16B POD） |
| **hash_bytes ≤16B** | XXH3-small / wyhash | ➖ **inline mix**（不同算法）；非「输 XXH3 wrapper」 |
| **hash_bytes 中长串** | 裸 XXH3 / wyhash | ➖ 默认出线核有 seam；&gt;128B 选择 wyhash |
| **FlatHashMap** | absl::flat_hash_map / Folly F14 | ⚠️ 算法形状同档；迭代/API 与三平台强 peer 正在严格复审 |
| **SmallVector** | absl::InlinedVector | ✅ 小 N churn **赢** 同款 SOTA |
| **CArray** | std::vector | ✅ growth 赢（mimalloc 质量）；read 平 |
| **string_ref 搜索** | string_view / libc | ➖ 自研 SIMD 项平；子串/字符集 **转发 std**（刻意） |
| **StringBuf** | map+mutex / Absl 分片 peer | ⚠️ **Win 多平**；WSL/M4 lifecycle **输**；growth 赢；8t 距 Absl ~5% |
| **path 词法** | string_view 扫 / filesystem | ✅ 平算法基线；**碾** filesystem（零分配结构差） |
| **ConcurrentVector** | oneTBB `concurrent_vector` | ✅ Win/WSL：append、range iteration、并发 prefix 读同档或赢；⚠️ WSL `operator[]` 顺扫约 0.51x |

### ConcurrentVector：先校正合同，再谈速度（Windows/WSL Release，n=15）

旧实现把所有 writer 的元素构造放进一个 mutex；第一版并行改造又为“实时 size 只含
完成连续前缀”增加全局 ordered commit，8 writer 仅为 oneTBB 的 0.17x。该 reader
合同没有生产消费者，也是被单 writer+reader benchmark 诱导出来的错误形状，已撤销。

当前合同与 oneTBB 对齐：append 返回后其 index 可用，或 writers quiesce 后遍历；并发
append 时只能读取事先完成的 prefix。普通 fail-fast append 用一次 `fetch_add` 取号；
可恢复 `try_*` 在领取 index 前完成 allocation，因此失败不会进入 oneTBB 文档允许的
broken-vector 状态。两种错误模型不再互相征税。

Windows x64 / MSVC 19.50：

| 正常用法 | HYLE median | oneTBB median | n=15 判定 |
|---|---:|---:|---|
| 单 writer push 100k | **5.952 ns** | 7.688 ns | HYLE 1.29x；15/15 |
| 8 writer，预留后 append 200k | **32.255 ns** | 35.370 ns | HYLE 约 1.10x；13/15 |
| quiesce 后 range iteration | **0.371 ns** | 1.242 ns | HYLE 3.35x；15/15 |
| quiesce 后逐次 `operator[]` 顺扫 | **0.916 ns** | 1.250 ns | HYLE 1.36x；15/15 |
| 4 readers 读既有 prefix + 1 writer | **0.476 ns** | 0.849 ns | HYLE 1.78x；15/15；peer 绝对方差高 |

WSL / GCC 14.2：

| 正常用法 | HYLE median | oneTBB median | n=15 判定 |
|---|---:|---:|---|
| 单 writer push 100k | **6.299 ns** | 7.359 ns | HYLE 1.17x；15/15 |
| 8 writer，预留后 append 200k | **32.646 ns** | 34.182 ns | HYLE 约 1.05x；14/15 |
| quiesce 后 range iteration | **0.280 ns** | 0.388 ns | HYLE 1.39x；15/15 |
| quiesce 后逐次 `operator[]` 顺扫 | 0.840 ns | **0.427 ns** | **oneTBB 1.97x；15/15，未收口** |
| 4 readers 读既有 prefix + 1 writer | **0.468 ns** | 0.866 ns | HYLE 1.85x；15/15 |

WSL 的逐次索引差距与 GCC 对运行时 segment 定位的代码生成有关；它没有被包装成赢。
成熟容器的随机访问 iterator/range 形状已补齐：`++` 缓存 segment，因此正常顺序遍历
在两平台都领先；`operator[]` 继续作为真正随机索引的独立证据与后续优化项。M4 尚未
刷新，不能把本表外推为三平台证书。

## 2. 非 SOTA / 冻结边界（诚实清单）

1. **StringBuf lifecycle（WSL/M4）**  
   intern / add-retain-remove / churn 相对同契约 peer 约 **0.73–0.86x**。  
   产品价值在 cached-hash growth、稳定 identity、并发接近强 peer、batch——**不是**「全生命周期常数三平台无条件赢」。旧 1.24x churn 证书已撤回。

2. **StringBuf 8-thread**  
   距 sharded Absl+mutex 约 **4–5%**（~0.95x），接近但非碾压。

3. **hash_bytes 默认 DLL 出线核**  
   与裸 XXH3 同算法时有固定 seam；小 typed key 走 `Hash&lt;T&gt;` inline mix。旧
   `hash_bytes_inline` 只服务微基准且泄漏私有 vendor 头，已从公开面删除；较长字节串
   接受动态库边界，真实热点应缓存 hash 或批处理。

4. **string_ref 短输入 split/rfind/count**  
   4KB 点值可有 **0.83–0.96x**（DLL/dispatcher）；长输入摊平。

5. **hash / SwissTable 缺本轮三平台统一 sign 刷新**  
   方向（赢 absl）多次坐实；绝对 ns 与跨平台翻转未按 mem 纪律重跑。

6. **队列类**  
   能力已收口；性能叙事分散在 queue_sota_bench / sched——需要时单开，勿与 Swiss 表混谈。

## 3. 选型（不同场景不同 container 组成）

```text
整数 / 小 POD key 哈希     → Hash<T> / mix64（热路径）
长字节哈希                 → hash_bytes（>128B best-of wyhash）
通用 hash map              → FlatHashMap
栈上小数组                 → SmallVector
堆动态数组                 → CArray（默认 SystemAllocator/mi）
只读字符串视图 + 搜索      → string_ref（子串族借 std）
驻留串 / intern            → StringBuf（接受跨平台 lifecycle 税；要 growth/并发）
路径词法（无 syscall）     → path（零分配；勿用 filesystem 当算法对手）
MPMC / stealing 队列       → bounded / sharded / WS deque（见 queue 文档）
```

## 4. 与 mem 的边界

- 容器**默认** `SystemAllocator`（平 mi）；注入 vtable 付多态税 → [benchmarks-mem.md](./benchmarks-mem.md)。  
- Swiss / StringBuf 内部用自有表与 slab，**不要**拿 `SharedPool` / `SharedPoolCache` 当通用 map 分配器。

## 5. 复跑（刷新 ⏳ 项时）

```bash
# hash
scripts/bench-stat.py <build>/bin/hyle_hash_bench -n 15 ...

# Swiss vs absl（需 -DHYLE_BENCH_ABSL）
scripts/bench-stat.py <build>/bin/hyle_swiss_vs_absl_bench -n 15 ...

# vec
scripts/bench-stat.py <build>/bin/hyle_container_misc_bench -n 15 ...

# string / StringBuf
scripts/bench-stat.py <build>/bin/hyle_string_ref_bench -n 15 ...
scripts/bench-stat.py <build>/bin/hyle_string_buf_bench -n 15 ...

# path
scripts/bench-stat.py <build>/bin/hyle_path_bench -n 15 ...

# ConcurrentVector only（queue bench 其余 workload 不运行）
HYLE_QUEUE_BENCH_FILTER=concurrent-vector \
  scripts/bench-stat.py <build>/bin/hyle_queue_sota_bench -n 15 \
  --pair "HYLE ConcurrentVector" "tbb concurrent_vector"
```

定位单项时可再设 `HYLE_QUEUE_BENCH_VARIANT=hyle|tbb` 与
`HYLE_QUEUE_BENCH_CV_CASE=push|writers|index|iterate|read-write`；正式对比不设这两项。

刷新后：改本页 §0 新鲜度 + 总账对应行；详表回填 [benchmarks-mem-container.md](./benchmarks-mem-container.md)。

## 6. 详表索引

| 主题 | 详节位置 |
|---|---|
| hash 公平性更正 + 长度表 | mem-container §container·hash |
| SwissTable vs absl | mem-container §container·map + README 总账 |
| SmallVector / CArray | mem-container §container·vec |
| string_ref / StringBuf 三平台收口 | mem-container §container·str |
| path | mem-container §container·path |
| util/obs 夹带 | 应看 util/obs 收口；mem-container「次要模块」勿当 container 主证书 |
