# container-09：ConcurrentVector 读优化并发向量

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台 + **WSL TSan 零 data race**；container +7 test）|

> **范围**：吸纳 clay `parallel/concurrent_vector.h`（145）。"插入一次、读多次、多线程"模式:并发读 **lock-free**(一次 atomic load + 索引计算),写者 SpinLock 串行。元素一旦插入永不移动,引用/索引终生有效(含跨 growth)。
> **cloc**：**生产 87 行**(concurrent_vector.h) + **测试 98 行**。
> **归属**:clay 列在 parallel/,但它不依赖 sched(仅 mutex 串行写),本质是容器,归 container 簇,P3 即可做(非 P5)。

## clay 五问题 → HYLE 五修复(对抗式发现)

| # | clay 问题 | HYLE 修复 |
|---|---|---|
| 1 | 读 lock-free 但靠 ctor 里**隐藏的 `reserve(65)`** 防外层 `vector<vector<T>>` 重分配(重分配会把 buffer 从并发读者脚下释放)——脆弱的承重不变量藏在一行 | **定长 C 数组 `T* segments_[48]`**:"外层永不重分配" 是**结构性**保证,非 reserve() |
| 2 | 每次 grow **拷贝整个旧 buffer** 进新的 + 旧的永不释放 → ~2x 活内存 + O(n)/grow | **分段(segmented)**:每段独立,grow 只分配一新段,**零拷贝**;内存 ~1x、插入 O(1) |
| 3 | T 必须 copyable(因 copy-on-grow) | 永不移动既有元素 → **T 可 move-only / 构造后不可移动** |
| 4 | 每次读 `CLAY_CHECK` 边界(热路径分支) | operator[] 不检查(std::vector 契约:调用方保证 < size())|
| 5 | `std::vector<std::vector<T>>`(违禁 std 垃圾容器)| 段由注入 `Allocator` 分配,无 std 拥有容器 |

## 设计要点

- **分段索引数学**:段 k 容量 `base<<k`(base=首段容量,2 幂)。元素 i:`m=i>>first_shift`,段 `k=bit_width(m+1)-1`,段内偏移 `i - base*(2^k-1)`。文档+真测(跨多段边界地址稳定)。
- **内存序(单一同步点)**:发布的 `size_`(写 release / 读 acquire)。写者**先**构造元素 + 存段指针,**再** release-store size_;读者**先** acquire-load size_,**再**碰 segments_。故对任何 index < acquired size,段指针与元素都全可见——非原子的 `segments_[]` 数组由 size_ 原子携带顺序(同 clay state 字技巧,但结构更干净)。
- **写者锁**:`hyle::util::SpinLock`(util-05,HYLE 原生)。临界区含分配——一般 spinlock+malloc 是反模式,但本结构契约即"写罕见",低争用下可接受;高写争用应换 mutex(已在自审标注)。

## 对抗式自审（六类）

- **① 正确性**:**WSL TSan 零 race**(并发3读者+1写者、4写者并发各5k 插入)是关键闸门;元素地址跨 growth 稳定真测;move-only(unique_ptr)真测;非平凡析构每元素恰一次(Counted live==0)真测;并发写索引唯一且覆盖 [0,total) 真测。
- **② 性能**:读 = 1 acquire load + bit_width(1 条 lzcnt 类)+ 索引,无锁无拷贝;插入 O(1) 无拷贝;内存 ~1x(vs clay 2x)。
- **③ 规范**:无 std 拥有容器(段走 Allocator);模板头(消费端实例化,不出 DLL ABI);SpinLock HYLE 原生;container→util 仅含 header-only SpinLock(无 link 环)。
- **④ clay 覆盖**:operator[]/size/emplace_back 全覆盖 + 五修复;clay `initial_capacity` honored(首段 bit_ceil)。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:分段是并发 vector 标准设计,非过度;索引数学有推导+测试佐证。
- **"比 std 快"**:N/A(std 无并发 vector)。**对标物是 clay 自身 + folly `ConcurrentVector`**——HYLE 在内存(1x vs 2x)、插入(O(1) vs O(n) copy)、类型(move-only)三维真优于 clay;folly 对标待 bench-02。

## 交付物

- `include/hyle/container/concurrent_vector.h`（分段存储 + lock-free 读 + SpinLock 写）
- `tests/container/concurrent_vector_test.cc`（7,含 2 个并发 + TSan）

## 参考

- clay `parallel/concurrent_vector.h`。folly `ConcurrentVector`(分段设计参照,只读未引入)。conventions §1(禁 std 垃圾容器)、§7(cache 友好)。util-05(SpinLock)、mem allocator。
