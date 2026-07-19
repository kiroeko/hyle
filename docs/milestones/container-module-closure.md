# container 模块收口（2026-07-19）

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
- StringBuf：hash 不再作为 ID；generation-tagged direct slot 防 stale handle，
  永久 intern 与可回收 add/retain/remove 分离，`intern_many` 一锁批处理。
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

## C ABI

- 头：`include/hyle/c/container.h`，实现：`src/container/c_container.cc`。
- 106/106 动态符号可从 Windows DLL 导出；另有 15 个 header-inline 便利入口。
- 纯 C 测试机械引用 121/121 callable names；C17 编译，未借 C++ 编译器。
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

Windows 2026-07-19 最终一次 Release 样本（nanobench，点值仅用于定位）：

| 能力 | 结果 | 解释 |
|---|---:|---|
| Swiss hit/miss/insert/erase/iterate vs std map | 4.42/2.92/7.26/5.19/10.45x | flat layout + ctrl probing；insert/iterate 部分 epoch 方差高 |
| StringBuf single intern vs mutex std peer | 0.91x | 单调用锁 + DLL seam，真实小输 |
| StringBuf `intern_many` vs unlocked std floor | 1.075x | batch 摊掉锁/接缝，吞吐路径 |
| StringBuf lifecycle vs same direct-ID+mutex | 0.75x | free/erase + API seam，非 SOTA 项 |
| UTF-8 validate+count vs simdutf | 1.004x | 同档；validate-only 约 0.973x |
| hex encode/decode vs validated scalar | 1.32/1.23x | 只证明 scalar reference，不声称 SIMD-codec SOTA |
| ASCII lower vs scalar branch | 1.52x | runtime ISA kernel |
| path view ops vs filesystem | 26–58x | 主要是零分配结构；对同形 string-view 扫描平 |

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
- StringBuf 单次 lifecycle 若成为消费方瓶颈，应提供 batch add/remove，而不是
  移除线程安全或伪装 benchmark。
- 不把 arbitrary C struct 的 constructor/destructor 搬进 C ABI；需要非平凡元素
  的绑定层应以 opaque application object 指针作为 payload。
- 真正跨编译器/语言的全库 install/FFI 门仍属于 close-01，不由 container 单模块
  冒烟替代。

## 最终验证记录

| 环境 | 结果 |
|---|---|
| Windows x64 / MSVC 19.50 Release | typed+C 通过；121/121 test/guide callable coverage；106/106 DLL exports |
| WSL `/root/hyle` / GCC 14 Release | typed+C 通过；只在 native ext4 编译 |
| WSL ASan / UBSan | 两套 typed+C 均通过 |
| WSL TSan | 184/184 C++（含 C ABI 并发）+ pure C 通过，无 race |
| macOS Apple arm64 / AppleClang | scalar+NEON library、typed+C 均通过 |

TSan 的 death-test 输出 GoogleTest 对多线程进程 `fork()` 的通用警告，但所有测试
正常完成。WSL 从未在 `/mnt/c` 构建；该路径只用于把源码同步到 `/root/hyle`。
