# container_intrusive 性能资格

> **状态：`PRODUCT LOCKED / STRUCTURAL PASS / INSTRUMENT LOCALLY CALIBRATED / FORMAL SOTA CERTIFICATE PENDING`。**
> 产品收口不再被证书基础设施阻塞；正式三平台数字签署作为独立性能基础设施任务继续。
> 2026-08-26 的 auto/safe 产品形状可以冻结；但加入双 phase observer、near-MDE control、双 fixed-floor
> budget、地址分配顺序 strata 及更严格 actual-object parser 后，所有较早 dirty 数字都与当前仪器
> identity 不同，统一降为 `HISTORICAL / UNQUALIFIED`。在当前工具完成新的完整资格轮之前，本页不宣称
> 当前 9 个产品 gate 已通过，更不宣称 clean/三平台 SOTA。Windows clean n=63 以及同 candidate 的
> Linux/macOS manifests 仍是正式硬门；旧 Core 内类型和旧 auto-only 数据也不自动继承。

稳定产品边界见 [container-intrusive design](../../../architecture/container-intrusive-design.md)，
本轮施工与消费者裁决见
[container_intrusive v2 closure](../../tasks/container_intrusive/container-intrusive-v2-closure.md)。

## 最小可证明命题

公开 target 是 `hyle::container_intrusive`，公开头是
`<hyle/container/intrusive_list.h>`。最终产品有两个明确的 node-retirement 合同：

- `IntrusiveListHook` / `IntrusiveList<T, &T::hook>`：auto-unlink；linked node 先销毁时，hook
  析构以 O(1) 从链中摘除；
- `SafeIntrusiveListHook` / `SafeIntrusiveList<T, &T::hook>`：safe-link；membership 必须由 list
  显式改变，linked node 析构是误用；Debug fail-fast，Release 正常未链接析构零指令。

二者均为 caller-owned、stable-address、member-hook、non-owning、零分配双向链表；hook 与 list
各只有两个 pointer，不存 count、owner、allocator、lock 或 dispatch，也不提供会把线性扫描伪装成
`sized_range` 的 `.size()`。benchmark 以 compile-time `sizeof/alignof` 同时锁住 HYLE auto/safe
以及各自 Boost peer 的 node layout/cache footprint。

强 peer 是同一 final executable 内的：

```text
boost::intrusive::list<
  Node,
  member_hook<Node, list_member_hook<link_mode<auto_unlink|safe_link>>, &Node::hook>,
  constant_time_size<false>>
```

Boost checkout 固定为 `df8eeb90846648464e3c8e378e312f0cec772e83`。`std::list` 会拥有并分配
内部节点，只保留为默认诊断的 allocating floor，不进入同合同结论。

## Workload 合同与复杂度

正式模式设置 `HYLE_CONTAINER_INTRUSIVE_BENCH_FORMAL_ONLY=1`，只运行 N=4096 的真实 aggregate：

| Policy / workload | 正常消费者形状 | epoch 内计费 | 复杂度目标 |
|---|---|---|---|
| auto + safe FIFO | completion/timer batch 入队并依次 drain | 全部 `push_back`、`front/pop_front`、checksum | 每节点 O(1)，整批 O(n) |
| auto + safe arbitrary cancel | 已知 node 以固定非顺序取消 | refill、node-to-iterator 等价恢复、remove/erase | 每节点 O(1)，整批 O(n) |
| auto + safe traversal | 已链接稳定节点只读扫描 | 每节点访问与 checksum | O(n) |
| auto + safe list-owner teardown | list owner 先结束，node storage 继续存在 | refill 与 list 析构时逐项 reset hook | O(n) |
| auto node retirement | node 先结束且仍 linked | raw storage 上 construct、link、固定顺序 node 析构及 auto-unlink | 每节点 O(1)，整批 O(n) |

数组/raw-storage 分配和 cancel order 构造在 epoch 外。node-retirement 使用无 live object 跨 callback
边界的 raw storage owner；不会让 `unique_ptr<T[]>` 长期指向已手工析构的数组。HYLE 与 Boost 的
构造、membership、checksum、析构及 lifetime 边界完全对称。

owner-teardown 与 node-retirement 不能合并：前者验证 list 析构清 membership，后者才验证 auto hook
定义性的 linked-node 析构路径。每个平台的 runner 都在采样前对实际 benchmark object fail closed：
HYLE/Boost × auto/safe 四个 owner-teardown callback 必须保留 O(n) clear 中的 next load 与至少双 pointer
reset-store shape，HYLE/Boost auto retirement 必须保留 linked-pointer test、load、至少四个 pointer store 与
O(n) retirement loop。不同 ABI 可能用零寄存器、成对 store 或其他等价 lowering；资格门不假装用一个
脆弱 parser 完整证明 store value。最终 linked/unlinked 状态由下面两个 opaque phase observations 变为
可观察语义，object gate 负责证明中间仍有应计费的 loop/store shape。

两类 lifetime workload 在独立的 noinline aggregate 中执行，并分别在 4096-node storage 的 linked
phase 与 teardown/retirement 后的 unlinked phase 调用一次位于独立 translation unit 的 opaque storage
observer；两侧合同对称，两个 observer 都不在 per-node loop 内，也不被解释成 batch 吞吐。CMake
显式关闭三个 benchmark target 的 IPO，runner 同时拒绝
`/GL`、`-flto` 或启用
`WholeProgramOptimization` 的构建。opaque call 阻止跨 epoch dead-store elimination，实际 object
audit 再确认编译器没有消去应计费的 clear/auto-unlink loop。

N=32、whole-list splice 和 allocating floor 不进入产品 gate。N=32 曾随 runtime-bound/链接位置翻转；
whole-list splice 每轮只做固定数量指针重链，是 O(1) 机制探针，不能用 N 倍 batch 标签制造吞吐结论。
公共 API 与 metadata 不由这些诊断反推。

渐进性质另由实现与专用 codegen gate签署：两端插入/弹出、已知节点 remove、`iterator_to`、单节点/
整链/range splice、append、swap 与 move assignment 都是 O(1)；clear/consume/traversal 是 O(n)。
MSVC/clang-cl/GCC/AppleClang 的严格 probe 锁 auto/safe hot operations 的 opcode 形状一致、safe 正常
析构零工作；纳秒 peer benchmark 不替代这项结构证据。

## 仪器资格

HYLE 与 peer 始终在同一个 executable 中。`HYLE_BENCH_RIVAL_FIRST=1` 只交换执行顺序；独立的
`HYLE_BENCH_PEER_ALLOCATION_FIRST=1` 只交换两侧 node array/raw-storage 的物理分配先后。正式产品
门完整运行 execution order × allocation order 的 2×2 strata，避免“先执行”和“先拿地址”两种效应
方向相反时互相抵消。它们不改变输入、lambda、layout 或计费边界。每个平台还运行独立
code-placement：Windows 从同一 compiler object 生成 `/FUNCTIONPADMIN:64` final image，要求
normalized object disassembly 相同且 PE `.text` 确实增大；Linux/macOS 对整个比较 TU 对称加入
`-falign-functions=64`，runner 要求两 target 的 source/defines/includes/flags 除该选项外完全相同且
object text 确实增大。任一 primary/placement × 2×2 execution/allocation stratum 未关闭，该平台不签。
正式构建只接受 Windows `Visual Studio 18 2026` 或 Linux/macOS `Ninja`，并从 target build metadata
验证 Release 优化与 `NDEBUG`，拒绝 `/Od`、`-O0/-Og` 等调试优化形状。`llvm-objdump` 对实际
executable/object 验证 Windows COFF x86-64、Linux ELF x86-64、macOS Mach-O arm64；工具自己的
resolved path、binary SHA-256 与 version output 也在采样前后冻结，不能用宿主 `platform.machine()`
冒充生成物架构或在审计后替换工具。

资格控制在独立 executable 中，避免改变产品函数 placement：

- A/A：两个名字执行相同 noinline 4096-node、四 pass aggregate；两个顺序都必须落入预声明 3%
  equivalence band；
- known delta：三个变体进入同一 runtime-bounded noinline scalar loop，每次迭代执行完全相同的普通
  load、volatile load 与累加；只把 trip count 从 64 份改为 near 的 67 份（理论多 4.6875%）或 large
  的 80 份（多 25%）。两个顺序必须把两者都正确判 `WIN`；near 的整段 CI 还必须落在“耗时慢
  3%–6%”对应的 throughput-ratio 窗口 `[1/1.06, 1/1.03)`；large 的整段 CI 必须落在预声明
  `[0.75,0.85]` 且比 near 更慢，不能让异常的大锤也冒充单调校准；
- timer floor：每 4096 logical operations 只有一个 compiler barrier，只记录绝对 floor，不作 peer
  胜负；
- phase-observer floor：每 4096 logical operations 固定调用两个 observer；与 timer floor 中较慢者都
  必须不超过所有产品 strata 中最快 median ns/op 的 1%，否则仪器没有资格解释产品差异；
- host：Windows runner 通过 `GetSystemCpuSetInformation` 拒绝非最高 efficiency-class affinity；
  Linux 要求显式 CPU mask；macOS 无可靠 hard affinity，正式 manifest 必须显式记录 interactive 或
  utility QoS；
- 统计：每个 shape/order 使用独立进程 paired samples；ratio 固定为
  `Boost op/s / HYLE op/s`，使用 98%+ distribution-free median CI，预声明 practical/NI margin 为
  3%。exact 五态与单边 non-inferiority 分开，不能把“不显著”写成等价。

2026-08-26 曾预声明并只跑 instrument-only n=31 的稀疏分支 control：nominal `1/32` 事件在两个
顺序实际都是约 `0.715` throughput ratio（慢约 40%），甚至比 nominal `1/4` 的约 `0.906` 更慢；
推断是周期分支预测成本主导。该轮因此 `FAIL CALIBRATION`，没有运行任何产品 benchmark，也没有把
频率常量解释成时间分辨率。原始诊断保留在
`C:\hyle-perf-evidence\container-intrusive-near-mde-calibration-n31-20260826`；当前 64/67/80
无稀疏分支 control 是一次新的预声明实验，不能与旧样本合并。

随后从零完成的 64/67/80 instrument-only n=31 校准通过：near 两顺序为
`0.955578 [0.954177,0.956851]` 与 `0.954764 [0.953586,0.958256]`，整段 CI 均在目标窗内；
large 两顺序约 `0.800`，A/A 两顺序均 `EQUIVALENT`。empty/two-observer 最大 raw floor 分别为
`0.000093447/0.000374011 ns/logical-op`。原始记录位于
`C:\hyle-perf-evidence\container-intrusive-tripcount-near-mde-calibration-n31-20260826`。这只认证
control 能看见真实的 near-MDE 差；因为没有运行产品，relative 1% floor budget 仍为 N/A，也不产生
任何产品性能结论。完整 runner 必须重新同时关闭 controls、floors 与产品门。

## 已作废的 Windows dirty 资格预检（2026-08-26，n=31）

> **`SUPERSEDED / HISTORICAL / UNQUALIFIED`。** 下列整轮来自更早的 harness；它没有当前的
> near-MDE control、双 floor budget、双 phase observation 和 allocation-order strata，且下列 SHA-256
> 与当前输入不相同。表格只保留“为什么后来继续红队仪器”的追溯价值，不能作为当前产品或当前
> benchmark runner 的绿色资格结果。

这是一轮预声明的新实验；没有合并或筛选此前样本。主机为 i9-13900K、Windows 11、MSVC
19.50.35728、x64 `/O2 /Ob2 /DNDEBUG`、static/header-only，逻辑 CPU 2（最高 efficiency class）、
above-normal。HYLE HEAD 仍是 `49512b7` 且 dirty；Boost clean。采样期间所有 input hashes 稳定。

旧仪器当时报告：A/A 两顺序 ratio median/CI 为 `0.9988 [0.9920,1.0046]` 与
`1.0010 [0.9958,1.0147]`，均 `EQUIVALENT`；1-load/4-load known delta 两顺序均正确判 `WIN` 且
单调；empty aggregate floor 最坏 `0.0000970 ns/logical-op`。primary/placement PE `.text` 为
92,163/117,955 bytes，object code 在 link padding 前相同。no-IPO 与实际 object lifetime-loop gate
也均报告 `PASS`。这些 gate 已随工具 identity 变化而失效。

下表每格为 `Boost/HYLE` paired ratio median 与 98%+ CI；`EQ`=`EQUIVALENT`，
`INC`=`INCONCLUSIVE`。exact 跨 `WIN/EQUIVALENT` 边界不妨碍独立 NI gate 在 CI 上界小于 1.03 时
诚实通过。

| Workload | Primary HYLE-first | Primary peer-first | Placement HYLE-first | Placement peer-first | Windows exact / NI |
|---|---|---|---|---|---|
| auto FIFO | `1.0141 [1.0083,1.0237] EQ` | `0.9948 [0.9659,1.0069] INC` | `0.9948 [0.9828,1.0050] EQ` | `0.9974 [0.9864,1.0085] EQ` | `INC / PASS` |
| auto arbitrary cancel | `0.9462 [0.9223,0.9636] WIN` | `0.9668 [0.9464,0.9846] INC` | `0.9401 [0.9226,0.9563] WIN` | `0.9622 [0.9466,0.9958] INC` | `INC / PASS` |
| auto traversal | `0.9987 [0.9956,1.0017] EQ` | `0.9996 [0.9952,1.0019] EQ` | `1.0011 [0.9981,1.0057] EQ` | `0.9987 [0.9962,1.0041] EQ` | `EQ / PASS` |
| auto list-owner teardown | `1.0017 [0.9949,1.0059] EQ` | `1.0009 [0.9964,1.0200] EQ` | `1.0006 [0.9937,1.0052] EQ` | `1.0016 [0.9978,1.0068] EQ` | `EQ / PASS` |
| auto linked-node retirement | `0.9605 [0.9542,0.9678] WIN` | `0.9414 [0.9266,0.9506] WIN` | `0.9657 [0.9534,0.9751] INC` | `0.9350 [0.9245,0.9468] WIN` | `INC / PASS` |
| safe FIFO | `0.9934 [0.9530,1.0015] INC` | `0.9882 [0.9515,1.0090] INC` | `0.9915 [0.9673,1.0055] INC` | `1.0041 [0.9767,1.0159] EQ` | `INC / PASS` |
| safe arbitrary cancel | `0.9602 [0.9173,0.9859] INC` | `0.9530 [0.9368,0.9621] WIN` | `0.9715 [0.9429,0.9893] INC` | `0.9475 [0.9290,0.9737] INC` | `INC / PASS` |
| safe traversal | `1.0012 [0.9962,1.0044] EQ` | `0.9991 [0.9909,1.0019] EQ` | `1.0007 [0.9978,1.0033] EQ` | `1.0013 [0.9916,1.0035] EQ` | `EQ / PASS` |
| safe list-owner teardown | `1.0011 [0.9968,1.0045] EQ` | `1.0028 [0.9941,1.0078] EQ` | `0.9983 [0.9955,1.0023] EQ` | `1.0008 [0.9938,1.0026] EQ` | `EQ / PASS` |

旧 summary 当时把 9/9 workload 的四个 Windows non-inferiority 门报告为通过；它不再说明当前
Windows dirty candidate 已取得任何资格，也不能升级成 clean/三平台 SOTA。

原始 logs、逐进程 records、summary、manifest 与 placement inspection 位于
`C:\hyle-perf-evidence\container-intrusive-auto-safe-frozen-diagnostic-n31-20260826`。
该轮旧产品/harness SHA-256 包括 header `ea86b691...`、benchmark `4720f664...`、observer
`4aaac76a...`、runner `2d4a133e...`，primary PE `128fab0b...`、placement PE `070dff27...`。
这些 dirty raw 不归档为发布证书，也不与当前工具的新样本合并。

此前 `...retirement-qualified-diagnostic-n31-20260826` 使用的是尚未加入 opaque observer 与
跨平台实际 object gate 的前一版 harness，已被本轮整体 supersede；两轮没有合并，也没有择取其中
更有利的 workload。它只保留作 DSE 风险被发现前的历史诊断。

此前资格 harness 的 n=15 整轮被 fail-closed 拒绝：known-delta 正确，但 A/A 两顺序 CI 上界
`1.0380/1.0339` 跨过 3%，另有一个 safe cancel placement 门功效不足。它没有被挑样本覆盖，保留在
`C:\hyle-perf-evidence\container-intrusive-auto-safe-retirement-qualified-diagnostic-n15-20260826`；
n=31 是明确提升功效后的全新完整实验。

## 真实消费者边界

time 已采用 safe list；但本页不把 TimerQueue 算法对拍冒充 container SOTA gate。现有 dirty sanity
显示 N=64/4096 的 arm/cancel/expire/churn 没有因 safe policy 迁移失控；真正的非退化机制证据是
auto/safe 完全相同的 footprint 与 hot-op codegen，以及 safe 析构在正常路径零工作。它是结构等价
推论加消费者 sanity，不是同一 binary 的旧-auto/新-safe 直接 replay。

TimerQueue N=1 相对 indexed heap、N=8 相对 Seastar 的差距属于 time 模块的小批量算法/仪器复审
backlog；不得改 list API、layout 或 metadata 来迎合。sched 的 waiter detach 后会复用 link 作锁外
wake chain，已裁决保留专用状态机；被拒绝的迁移诊断只记在 implementation task，不保留为长期
container benchmark。

## Clean 三平台证书

Windows 正式实验必须从不存在的 build/output 目录开始，以预声明 n=63 运行 primary/placement、
2×2 execution/allocation strata 及全部 controls：

```powershell
$intrusiveRoot = 'C:\bench-work\hyle-container-intrusive-v2-clean'
$llvmObjdump = 'C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\Llvm\x64\bin\llvm-objdump.exe'

python scripts/run-container-intrusive-bench-evidence.py `
  --build (Join-Path $intrusiveRoot 'build') `
  --boost-dir C:\public\project-hyle\boost `
  --output-dir (Join-Path $intrusiveRoot 'evidence') `
  --config Release --fresh-configure `
  --generator 'Visual Studio 18 2026' --architecture x64 `
  --placement --llvm-objdump $llvmObjdump `
  --affinity-mask 0x4 --priority-class above-normal `
  --runs 63 --equivalence-margin-pct 3 --require-clean
```

Linux GCC14 与 macOS AppleClang 最少各运行 n=15；若 A/A 或产品 CI 功效不足，保留整轮结果并以
预声明 n=31 开新实验，不合并、不追加到“刚好通过”。Linux 必须在原生 ext4 clone 上运行并选择该
主机自己的固定性能核；macOS 必须显式选择 QoS：

```bash
# Linux / GCC 14 / native ext4
CC=gcc-14 CXX=g++-14 python3 scripts/run-container-intrusive-bench-evidence.py \
  --build /root/bench-work/hyle-container-intrusive-v2/build \
  --boost-dir /root/vendor/boost \
  --output-dir /root/bench-work/hyle-container-intrusive-v2/evidence \
  --config Release --fresh-configure --generator Ninja \
  --placement --llvm-objdump /usr/lib/llvm-20/bin/llvm-objdump \
  --affinity-mask 0x4 --runs 15 --equivalence-margin-pct 3 --require-clean

# macOS / AppleClang / APFS
python3 scripts/run-container-intrusive-bench-evidence.py \
  --build /Users/kiro/bench-work/hyle-container-intrusive-v2/build \
  --boost-dir /Users/kiro/vendor/boost \
  --output-dir /Users/kiro/bench-work/hyle-container-intrusive-v2/evidence \
  --config Release --fresh-configure --generator Ninja --apple-qos interactive \
  --placement --llvm-objdump "$(xcrun --find llvm-objdump)" \
  --runs 15 --equivalence-margin-pct 3 --require-clean
```

`0x4` 只对已经核对 topology 的示例主机有效。三平台都要求 primary/placement、2×2
execution/allocation strata 与 controls；Windows 用 link padding，ELF/Mach-O 用对称 function
alignment。node allocation 在 epoch 外，故不把 heap allocator 吞吐伪装成 list 成本；独立 allocation
order 已承担 caller-owned object-address 风险矩阵的数据位置腿。

若 Linux/macOS 的 n=15 为 `INCONCLUSIVE`，n=31 runner 还会验证旧 manifest：structural/fixed-floor
必须已通过；A/A 只能是在 band 内但 CI 功效不足；known-delta 必须方向正确，near median 已处于
3%–6% 窗口且 large/near 单调；任何 product `FAIL` 均不得靠增样覆盖。三个 manifest 必须绑定同一 HYLE candidate
commit、Boost commit 和 product/harness input hashes；
每个平台的 `platform-summary.json`、raw logs、compiler/binary identity、execution controls、no-IPO
记录与实际 benchmark-object lifetime-loop audit 都完整，且 controls、structural gate 和 9 个 NI gate
全 `PASS` 后，才能在汇总记录中把 `cross_platform_gate_complete` 写成 true。每个 child 的 structured
telemetry 必须逐 run 精确绑定 formal/order/allocation/QoS；source/build/output/peer 在 Linux 必须是
native ext4、macOS 必须是 APFS；affinity/power/governor/topology 在采样前后必须稳定。正式 evidence
目录还归档实际 executables、objects、build metadata 与完整 lifetime disassembly，而不是只留不可
重放的绝对路径和 hash。Windows manifest 不能
代替 Linux/macOS，正确性构建也不能代替 peer performance。

## 历史边界与重开

2026-08 的旧三平台 n=15 发生在类型仍由 Core 持有、只有 auto-unlink 的时期；candidate
`a951f0a...`、harness `2707583...`。当时 Windows/Linux/M4 的 FIFO、cancel、traversal、auto-detach
通过旧 3% non-inferiority gate，并暴露 N=32 placement ghost。owner、target、API、safe policy、
harness 与 binary identity 均已变化，所以它只能标为 `HISTORICAL / UNQUALIFIED`。

正式签署后，只有真实 consumer 的 material regression、复杂度更优的成熟同合同 peer、算法/API/
layout/compiler/delivery shape 变化才重开。LOC、类型数量、单次点值、稳定亚纳秒差或 N=32 翻转
不能单独驱动产品形状。
