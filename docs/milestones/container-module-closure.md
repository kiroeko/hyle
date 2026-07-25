# container 模块收口（2026-07-19）

> **2026-07-25 严格复审说明：** 本页保留 2026-07-19 的施工证据与当时点值，
> 不再代表当前接口或“实际使用导向”的最终证书。严格复审已经把 C++ 值容器从
> POD/显式 allocator 形状改为默认可用的 `SmallVector`、`FlatHashMap`，并把 C ABI
> 从私有 TLS last-error 改为共享 `hyle_status + out`；当前动态 C 面为 108 个声明。
> 新增的 [`container.md`](../guide/container.md) 负责告诉用户什么场景选择什么。
> `ConcurrentVector` 的多 writer 形状、真正的 typed `FlatHashSet`、map 迭代/便利接口及
> 同合同强 peer 性能证据仍在复审，因此 container **尚未获得 2026-07-25 严格证书**。
> 复审也删除了无生产消费者、属于 quick-open 应用策略的 fuzzy scorer/top-K 及其 C
> 导出；旧 container-06c 仍作为历史任务记录保留。
> 下文的 109 exports、124 callable、TLS error 和性能表只应按历史快照阅读。

## 结论

container 的功能、红队、稳定 C ABI、使用文档和既有性能面已收口。Windows x64、
Linux x64 和 macOS arm64 的最终验证命令与结果记录在本文末尾。这里的“收口”不
表示每个 API 都比每个平台的标准库快；它表示能力边界明确、真实落后项如实记录、
发布面可用，且没有已知 correctness/ABI P0/P1。

## 功能与下游

- CArray：记录 allocator/alignment，alias-safe growth，新增 fallible
  `try_reserve/try_push_back`，所有容量计算先检查 overflow。
- SmallVector：修正 allocator-aware move assignment、inline/heap 别名增长和
  overflow。
- SwissTable：const 查询/遍历、reserve/clear、alias insert、OOM 状态保持、move
  元数据完整；C ABI map 也使用 flat ctrl+slot Swiss probing，不是链式 facade。
- StringRef/StringAlgo/Path：null/range/overflow、严格 UTF-8、hex validation、
  O(N log K) top-K、路径边界；ASCII lower/upper 增加
  scalar/SSE4/AVX2/AVX512/NEON 动态 kernel。
- StringBuf：hash 不再作为 ID；generation+shard+slot 防 stale handle；缓存 hash
  消除扩容/删除的重复长串扫描；构造时一次选择 16/32/64 shard；短 Entry slab
  高水位复用；永久 intern 与可回收生命周期分离，四类 batch API 摊薄锁/ABI。
- Bounded/Sharded/WorkStealing/WsFifo：构造 OOM 产生可检测 invalid 状态，C
  create 不会返回半初始化 handle；Sharded consumer cursor 公平推进。
- ConcurrentVector：消除 capacity data race，限制 segment shift/address-space
  上界，OOM 返回 `SIZE_MAX`。
- sched/graph 已消费队列和 append-only 能力。IntrusiveList 已归 core，避免
  error/time 对 container 的反向依赖。

## 红队发现并修复

1. CArray/SmallVector/SwissTable 的自别名输入在 growth 后可能读悬空地址。
2. allocator/alignment 没有随 allocation 保存，跨 allocator reallocate/free
   可能错配。
3. SwissTable move constructor 漏 mask，极端 reserve 由 assert 终止而非失败。
4. StringBuf 用 hash 充当 identity，理论 collision 可别名；旧 ID 也可能在 slot
   重用后指向新串。
5. ConcurrentVector 极端 segment 计算会超位宽；capacity 读取存在并发不一致。
6. UTF-8 接受 overlong/surrogate，C shim 对 malformed 输入没有 last-error。
7. C top-K 使用全排序且 child-index 可溢出；改为定长最小堆 O(N log K)。
8. C hash map 最初 facade 每 collision node 一次分配，违背 flat Swiss 产品形状；
   已改成连续 ctrl/slot、tombstone、SIMD group probing。
9. C flat map 的第一版 raw slot stride 没向最大 key/value alignment 取整；
   `uint64_t -> char` 这类布局会让第二个 slot 起的 key 未对齐。stride 现已
   round-up，并用 pure C visitor 检查每个 key 地址。
10. 新 ASCII dispatcher 在 Windows 定义 TU 未见 dllexport 声明，库自身可构建但
   客户端 import lib 链接失败；通过让定义 TU 包含公共声明修复，并加入长输入测试。
11. StringBuf 低位分片会固定 SwissTable 的 H1 低位；若再 rotate，会把固定位送入
    H2 tag。现改为用 hash 中间位分片，原 hash 的低位 H1/高位 H2 都保留完整熵，
    同时消掉每次 lookup 的第二次 avalanche。
12. SwissTable 的普通 `reserve(size+1)` 看不见 tombstone 耗尽的 growth budget，
    `insert()` 仍可能在 noexcept 路径内 fatal rehash。新增可失败
    `try_prepare_insert()`，StringBuf 在插入前完成冷路径容量准备。
13. TSan 最初把 mimalloc 跨线程复用的旧/新 owner 报成 StringBuf shard race；
    根因是 shipping allocator 没被 sanitizer 插桩。TSan 构建现同时编译
    `mimalloc_static` 并启用 vendor 的 `MI_TSAN` page-flag 布局，188 项全绿。

## C ABI

- 头：`include/hyle/c/container.h`，实现：`src/container/c_container.cc`。
- 109/109 动态符号可从 Windows DLL 导出；另有 15 个 header-inline 便利入口。
- 纯 C 测试机械引用 124/124 callable names；C17 编译，未借 C++ 编译器。
- C++ 并发测试经同一 C ABI 覆盖 MPMC、sharded、owner/thief 与 append-only。
- 类型擦除覆盖 POD vector、POD flat map、pointer-payload queues、StringBuf。
  `SmallVector<T,N>` 的 N、typed inline hash 和非平凡 C++ 元素无法稳定类型擦除，
  有意保留 source surface，不导出模板实例。
- 所有 handle/内部 storage 都使用调用方注入 allocator；故障注入覆盖 handle OOM、
  内部 storage OOM、map growth、StringBuf intern、四类 queue 和 concurrent vector。
- 用户合同与全部入口见 [`c-container.md`](../guide/c-container.md)。

## 测试覆盖

- typed C++：hash oracle/collision/churn/move/OOM/alias；数组 inline/spill/move/OOM；
  字符串 null/UTF-8/path/long SIMD；StringBuf collision/lifecycle/stale ID/batch/
  concurrency；队列和 vector 并发。
- pure C：字符串/路径/StringBuf/vector/map/queue/vector、constant-hash 200-key
  collision、erase churn、map alias growth、malformed UTF-8、top-K order 和 OOM。
- sanitizer：ASan+UBSan 覆盖 typed+C；TSan 覆盖 C++/C ABI 并发面。
- ABI：严格 C17 include/compile、动态导出枚举、C++ 客户端 import-library link。

## 性能结论

2026-07-19 Release 三平台复测（nanobench，3 repetitions aggregate；比值
大于 1 表示 HYLE 更快）：

| 能力 | Windows | WSL GCC 14 | Apple M4 | 解释 |
|---|---:|---:|---:|---|
| Swiss hit/miss/insert/erase/iterate vs std map | 4.42/2.92/7.26/5.19/10.45x | — | — | flat layout + ctrl probing；insert/iterate 部分 epoch 方差高 |
| StringBuf single intern / mutex std peer | **0.994x** | **0.772x** | **0.934x** | Win 平；WSL/M4 输；动态入口+稳定存储合同 |
| `intern_many` / unlocked std floor | **0.990x** | **0.732x** | **1.045x** | Win 平、M4 小赢，WSL 输；不是全平台无条件 SOTA |
| 1KiB 长串增长 / flat-set floor | **5.36x Absl** | **1.71x Absl** | **1.67x std** | cached resident hash：输入字节只扫一次，rehash 搬定宽 slot |
| 8-thread / adaptive Absl peer | **0.948x** | **0.958x** | **0.956x vs std** | x86 距 Absl 约 4–5%；M4 未装 Absl，用同分片 std |
| 生命周期 / direct-ID+mutex peer | **0.990x** | **0.813x** | **0.785x** | Win 平；Linux/M4 的 generation+slab 常数仍可见 |
| 16-cycle churn / retained peer | **0.998x** | **0.861x** | **0.808x** | slab/slot 高水位复用；仍不虚报三平台赢 |
| UTF-8 validate+count vs simdutf | **1.004x** | — | — | 同档；validate-only 约 0.973x |
| hex encode/decode vs validated scalar | **1.32/1.23x** | — | — | 只证明 scalar reference，不声称 SIMD-codec SOTA |
| ASCII lower vs scalar branch | **1.52x** | — | — | runtime ISA kernel |
| path view ops vs filesystem | **26–58x** | — | — | 主要是零分配结构；对同形 string-view 扫描平 |

StringBuf 本轮追求的是**复杂度形状 + 强契约常数**，不是把每个平台的一个点值改成
绿色：`intern/add` 的预期复杂度为 O(|input|)，resident key 的 table growth/erase
不再重扫字符串；ID lookup/retain/remove 为预期 O(1)；`size()` 只扫描至多 64 个
构造期固定 shard，属于有硬上界的 O(1)；短 entry 的重复 churn 在达到高水位后不再
逐项向 allocator 申请。与之对应，Linux/M4 lifecycle 仍落同契约 peer
14–22%，已记录为冻结边界。

这些复杂度已经到达该合同的下界：新输入至少要读 Ω(|input|) 字节才能判定内容，
直接 ID 操作不可能优于 Ω(1)。因此本轮“超越 SOTA”落在 resident rehash/erase
不再乘字符串长度、稳定地址和回收合同不退化，而不是宣称存在比 O(1) 更低的
lookup 阶。后续常数优化必须用强 peer/profile 触发。

同一轮 4KB string-ref 的 split/rfind/count 出现 0.83–0.96x，而 64KB/1MB
count/rfind 回到 1.0–1.14x；这说明短输入 DLL/dispatcher seam 与仪器方差是实际
边界，不能继续写“container·str 全线赢”。实现仍保留：强制再做一层 wrapper
只会让 ABI 更慢；普通 C++ 热循环可使用 header-inline 视图操作，batch/长扫描由
动态 kernel 摊薄。

更完整历史矩阵和 SOTA 对手见
[`benchmarks-mem-container.md`](../performance/benchmarks-mem-container.md)。

## 有意边界与后续触发器

- hex 若未来成为 profiler 热点，应对拍 simdutf/专门 codec 后再决定独立动态
  kernel；当前证据不足以宣称专用 SIMD SOTA。
- StringBuf 已提供 batch add/retain/remove。后续只有同契约强对手显示持续落后或
  真实消费方证明新的瓶颈，才重开存储/分片参数，不按单个噪声点反复调参。
- 不把 arbitrary C struct 的 constructor/destructor 搬进 C ABI；需要非平凡元素
  的绑定层应以 opaque application object 指针作为 payload。
- 真正跨编译器/语言的全库 install/FFI 门仍属于 close-01，不由 container 单模块
  冒烟替代。

## 最终验证记录

| 环境 | 结果 |
|---|---|
| Windows x64 / MSVC 19.44 Release | typed+C 通过；124/124 test/guide callable coverage；109/109 DLL exports |
| WSL `/root/hyle` / GCC 14 Release | typed+C 通过；只在 native ext4 编译 |
| WSL ASan / UBSan | 两套 typed+C 均通过 |
| WSL TSan | 188/188 C++（含 C ABI 并发）+ pure C 通过，无 race；mimalloc 同步已纳入插桩 |
| macOS Apple arm64 / AppleClang | scalar+NEON library、typed+C 均通过 |

TSan 的 death-test 输出 GoogleTest 对多线程进程 `fork()` 的通用警告，但所有测试
正常完成。WSL 从未在 `/mnt/c` 构建；该路径只用于把源码同步到 `/root/hyle`。
