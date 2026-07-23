# platform 实测铁证

> 本页是 [performance 总账](./README.md) 的 **platform 模块**详节。  
> 源：`bench/platform/platform_bench.cc` → `hyle_platform_bench`。  
> 方法：`scripts/bench-stat.py`，信任 **[sign] 配对符号检验**。  
> **2026-07-20 重测**：Win MSVC 19.50 / Apple M4 AppleClang 21 均为 **n=15**；  
> WSL GCC 14.2 **n=5**（修 bench 墙钟后；n=5 时 sign p 门 0.01 常达不到，方向看 med ratio + 5/5 一致性）。  
> 旧 2026-06-30 双平台证书作废（源码 7 月大幅改过）。  
> sched 同步/fiber 不在本页 → [benchmarks-sched.md](./benchmarks-sched.md)。

## 墙钟陷阱（本轮血泪，流程已改）

| 段 | 问题 | 处置 |
|---|---|---|
| `park_timeout(0)` × 100k × 多 epoch | 每次真进内核（Win ~µs，Linux futex ETIMEDOUT ~数十 µs）；WSL 单遍曾 **70s+**，n=15 小时级 | kIters **100k→500**；文档只报数量级 |
| raw-wait 乒乓 `minEpochIterations(30)` | Linux ~20µs/round × 20k × 30 epoch → 单段数分钟 | minEpoch **30→5** |
| 双份阻塞乒乓（vs CV + vs raw） | 同形状跑两遍 | 保留（语义不同），接受 WSL n=5 |

**纪律**：OS 主导段必须先估 `iters × 单次 syscall × epoch × N`；亚 µs 算法段的 n=15 不能照搬到 futex 乒乓。

## 三平台总表（中位数 ns/op，除非注明）

| 操作 | Win n=15 | Lin n=5 | M4 n=15 | 判定（对标） |
|---|---:|---:|---:|---|
| **Parker 真阻塞乒乓** vs std CV | 341 / 638 → **1.75x WIN** | 22.5k / 23.5k → **1.05x 平** | 2053 / 2130 → **sign WIN 1.05x** | ⚠️ **平台翻转**：Win 大赢；Lin 平；M4 微赢 |
| **Parker** vs raw wait 地板 | 356 / 310 → TIE | 22.2k / 21.2k → **0.95x 平** | 2035 / 2062 → **sign 近平** | ➖ 三态机税在噪声/个位数 % |
| **Parker 快路径** unpark+park | 9.71（floor 4.68） | 10.64（floor 4.63） | 1.99（floor 0.90） | ➖ 只报绝对值；atomic **非公平对手** |
| **Thread spawn+join** vs std::thread | 43.4k / 44.4k → **TIE** | 33.56k / 33.56k → **TIE** | 10.6k / 10.7k → **TIE** | ➖ 平（OS 主导，证零封装税） |
| **directory_iterate** vs std::filesystem | 40.5k / 32.0k → **sign LOSE 0.93x** | 6.5k / 13.7k → **spread WIN 2.15x** | 16.5k / 18.4k → **WIN 1.12x** | ⚠️ **平台翻转**：Lin/M4 赢；Win sign 输 |
| **vm_reserve+commit+release** vs raw 一步 | 1144 / 877 → **sign LOSE 0.77x** | 1075 / 848 → **0.78x 输** | 846 / 332 → **LOSE 0.39x** | ❌ **契约差**：两步 vs 一步；见下 |
| **vm_protect** vs raw | 446 / 428 → **sign LOSE 0.96x** | 407 / 393 → **0.97x 微税** | 937 / 938 → **TIE** | ➖ 近平；小封装税 |
| **MappedFile** vs ifstream | 43.7k / 62.3k → **sign WIN 1.39x** | 12.7k / **8.7k → 0.68x 输** | 14.1k / 21.7k → **WIN 1.53x** | ⚠️ **平台翻转**：Win/M4 赢拷贝路径；**Linux 64KiB ifstream 更快** |
| **MappedFile** vs raw mmap | 43.7k / 42.0k → **sign LOSE 0.96x** | 12.7k / 12.2k → **0.96x 平** | 14.1k / 13.7k → **sign LOSE 0.97x** | ➖ 近地板，~3–4% 封装税 |
| **park_timeout 快路径** | 9.2 ns | 10.6 ns | 1.95 ns | ➖ 与 park 快路径同档 |
| **park_timeout(0) 轮询** | ~3740 ns | **~59k ns** | **3.7 ns** | ⚠️ **极端平台差**：Mac 近免费；Win µs；Linux ~60µs |

### 2026-07-23 `Result<MappedFile>` 成功路径复证

DynLib/MappedFile 错误接口迁入共享 `Status` 后，benchmark 已改为用户实际写法：
`auto mapped = MappedFile::open(...); if (mapped) mapped->data()`。Windows MSVC 19.50
n=5 得到 HYLE **43.37 µs**、raw mapping **41.82 µs**、ifstream **58.50 µs**；
配对中位比分别 **0.955x** 与 **1.327x**。它与上表旧 n=15 的
43.7/42.0/62.3 µs 落在同一统计带，未观察到 `Result` 成功态附加税，因此不改三平台
总判定。n=5 的符号检验最低 p=0.0625，本次只作“未回退”复证，不升级为新胜负证书。

WSL GCC 14 同形状 n=5 因完整 platform suite 在当前环境超过 10 分钟执行上限，未形成
可回收统计输出；旧 Linux n=5 数字继续保留但不冒充本轮重测。Linux 正确性与 UBSan
已通过，性能仍待在低噪声会话复跑；macOS 当前提交同样待远程复证。

### 2026-07-23 VM `Result/Status` 成功路径复证

VM/NUMA 删除 TLS last-error 后，benchmark 使用当前用户形状：reserve 取
`Result<void*>`，commit/protect/release 取 `Status`。Windows MSVC 19.50 的定向 smoke
run 得到一页 reserve+commit+release：raw **897.42 ns**、HYLE **1167.11 ns**（0.769x）；
RW↔RO protect pair：raw **438.51 ns**、HYLE **459.52 ns**（0.954x）。它们分别落在旧证书
1144/877 ns 与 446/428 ns 的同一带内，没有观察到结构化成功态带来的额外回退；两步
VM 的合同差和 protect 的约 5% 薄封装税仍按下节原判定。该单进程 run 只作接口迁移
回归哨兵，不替代上表 n=15 三平台证书；Linux/macOS 新接口仍待性能复证。

## 非 SOTA / 需诚实标价

### 1. `vm_reserve`+`vm_commit` 输 raw 一步分配（真，非噪声）

```text
raw Win:  VirtualAlloc(MEM_RESERVE|MEM_COMMIT)   // 1 syscall
HYLE:     vm_reserve(MEM_RESERVE) + vm_commit(MEM_COMMIT)  // 2 syscalls
raw POSIX: mmap(PROT_READ|WRITE)                 // 1
HYLE:      mmap(PROT_NONE) + mprotect/commit     // 2
```

**原因**：API 有意拆成「先占地址空间、再按需 commit」（fiber 栈 / 大 arena 模式），不是漏优化。  
**处理**：对标表保留但标 **by-design 两步 vs 一步**；需要一步的调用方用 `vm_alloc`（若有）或直接 raw。  
**不**用平台宏把两步合成一步伪装成赢——会破坏可移植语义。

### 2. Win `directory_iterate` sign 输 std（0.93x）

- Linux/M4 仍赢（d_type 免 stat + visitor 无分配）。  
- Win 底层都是 `FindNextFileW` 量级；HYLE 还付 UTF-16→UTF-8 visitor 转换。  
- 旧账“Win 平”被 n=15 sign 翻成微输——**绝对值接近，结构不占优**。  
- 不判“实现错误”，判 **Win 无 Linux 那种 d_type 优势**。

### 3. Parker vs CV 平台翻转（预期内）

- Win：CV = SRWLOCK+CONDITION_VARIABLE+谓词 → Parker 大赢。  
- Linux：glibc CV 已是薄 futex → 平。  
- M4：微赢。  
- **禁止**只用 Win 2.99x 写总账“全面赢”。

### 4. Parker 快路径不设“输赢分母”

atomic flag 无 unpark-before-park 安全；binary_semaphore 快路径仍 futex。只报 ~2–10 ns 绝对值。

### 5. `park_timeout(0)` 平台差巨大

同一语义：Mac ~4 ns，Win ~4 µs，Linux ~60 µs。  
这是 **OS 对 0-timeout wait 的实现差**，不是 HYLE 逻辑分叉 bug。热路径应用 **token 快路径**，避免靠 0-timeout 轮询。

### 6. Linux 上 MappedFile 对 64KiB ifstream **输 0.68x**

Win/M4 赢 ifstream（零拷贝优势）；Linux 上 med **12.7 vs 8.7 µs**（ifstream 更快）。  
对 raw mmap 仍 ~0.96x 平 → **不是 HYLE 封装烂**，是「小文件 open+mmap+fault+close」在 Linux 上不比缓冲 read 便宜。  
大文件/随机访问才是 MappedFile 的战场；本 bench 64KiB 全扫更偏 ifstream 甜点。总账勿写“MappedFile 无条件赢 ifstream”。

## 复跑

```bash
cmake --build <build> --target hyle_platform_bench
# 定向 smoke，避免运行无关的阻塞/线程段：
HYLE_PLATFORM_BENCH_ONLY=vm-cycle,vm-protect \
  <build>/bin/hyle_platform_bench
# OS 段建议 n=5~15；单遍先确认墙钟 <~60s
scripts/bench-stat.py <build>/bin/hyle_platform_bench -n 5 \
  --pair "std::condition_variable ping-pong" "HYLE Parker ping-pong" \
  --pair "std::thread spawn+join" "HYLE platform::Thread spawn+join" \
  --pair "std::filesystem::directory_iterator" "HYLE directory_iterate (visitor)" \
  --pair "raw OS (mmap+munmap / VirtualAlloc+Free)" "HYLE vm_reserve+commit+release" \
  --pair "raw wait_on_address+wake floor" "HYLE Parker ping-pong" \
  --pair "raw mprotect / VirtualProtect" "HYLE vm_protect" \
  --pair "std::ifstream read+sum (copies)" "HYLE MappedFile open+sum (zero-copy)" \
  --pair "raw mmap+sum floor" "HYLE MappedFile open+sum (zero-copy)"
```
