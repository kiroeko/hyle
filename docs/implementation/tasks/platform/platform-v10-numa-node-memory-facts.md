# Platform v10：NUMA 节点内存事实

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

> 本轮由冻结基础模块的真实推理消费者回放触发，最终从 `CANDIDATE` 终止为 `BLOCKED`。产品候选已经
> 实现并验证 NUMA node 容量事实；阻塞项仅是冻结 P1 holdout 的仪器资格化，不能被改写为产品性能失败。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`platform` / host NUMA topology facts /
  `platform-v10-numa-node-memory-facts` / `REOPEN`；baseline
  `5c529802cd2974b222856608dcaf4cdd0fcf240b`，tree
  `fe5e3841d5de46c39ce01b7dae6c0d14e2d53f38`；
- trigger ID：`PLATFORM-NUMA-NODE-MEMORY-FACT-2026-09-06`。锁定 vLLM
  `569adb5a9780f9c02d22a6b29826acf711512356` 的
  `vllm/utils/cpu_resource_utils.py:get_memory_node_info` 读取每个可见 NUMA node 的物理总量和当前可用量，
  `vllm/platforms/cpu.py:get_device_total_memory` 与 CPU worker 内存状态直接用于启动预算；锁定 SGLang
  `5bebe7a033ab57751fac3c9d8db219da62ca8c72` 的
  `python/sglang/srt/utils/numa_utils.py:init_threads_binding` 按 rank/node 绑定 CPU 与内存，并在 rank 多于
  node 时明确无法预先确定每 rank 可用内存。两条独立推理旅程共同证明节点容量/压力是通用 host
  planning fact，不是某个模型或 benchmark 的形状；
- HYLE 当前 `numa_nodes/numa_node_of_cpu/vm_*numa*` 能发现、选择和放置 node，
  `host_memory_info()` 只返回整机/进程约束，公开面没有任何 per-node memory query。消费者因此能选 node，
  却不能在分配模型权重、KV 或 worker arena 前对选定 node 做容量预算；这命中 Platform v9 的
  “named consumer reproduces a missing memory fact” reopen trigger；
- 权威 OS 机制存在但语义不应被抹平：Linux stable sysfs
  `/sys/devices/system/node/nodeX/meminfo` 给出 node distribution/utilization；Windows
  `GetNumaAvailableMemoryNodeEx` 给出 node 的 free+zero pages（多 node 时不含 standby）；macOS 没有公开
  multi-NUMA topology，HYLE 的逻辑 node 0 应组合现有 host snapshot。OS 不公开 total 时必须表达 unknown，
  不能从 available、型号、启动时采样或全机均分猜总量；cgroup 是进程级约束，也不能伪造为某一 node
  的物理容量；
- 受影响范围只含：NUMA node memory value shape/query、Windows/Linux/macOS 实现与 parser/system
  tests、Platform selected component 的 source/static/shared 交付、Platform architecture/user guide、
  一个直接 host-planning consumer journey，以及旧 Platform host P1 路线的有限 holdout；
- 明确不重开 `platform_thread`、`platform_fs`、Mem 或 Mem-VM：thread/rank policy、GPU-to-NUMA mapping、
  allocator reservation、model/KV sizing 与 cgroup budget composition 仍由各自 owner 决定。也不重做 CPU/
  cache/frequency、NUMA placement algorithm、Native FFI/C API、公共性能仪器或 Formal；
- 新 query 是启动/重规划控制路径，按 P0 验证每次至多一个 bounded OS/file snapshot、无缓存、无 allocation
  热路径状态；现有 host-planning P1 journey 只作一次未受影响 holdout，不为节点查询发明纳秒 benchmark；
- `CLOSED -> REOPENED` 前 registry snapshot：
  `C:\public\project-hyle\_validation\platform-v10-numa-node-memory\module-status-before-reopened.json`，
  SHA-256 `A3E938CDDE79B4A5CA09D2C37FFF30970ED3CDD1AEB17016F56DA4BA67748B00`。

## 初步停止边界

- Platform 只报告可由当前 OS 公开机制证明的 node 物理事实；它不替用户选 rank、node、利用率比例或
  allocation size；
- node id 必须来自同一 `numa_nodes()` 稀疏 id space；无效/不可见 node 返回结构化错误，不回落到 node 0；
- total 与 available 的 unknown/平台差异必须在类型中可见；零是合法观测值，不兼任 unknown sentinel；
- 审计若发现必须改变 placement、其它 Platform factor、Mem owner 或公共仪器，先登记相邻任务，不在本轮
  顺手扩 scope。

## Closure card（`SCOPED`）

- component / owner / round / kind：`platform` / host NUMA topology facts /
  `platform-v10-numa-node-memory-facts` / `REOPEN`；baseline
  `5c529802cd2974b222856608dcaf4cdd0fcf240b`；
- demand：`PLATFORM-NUMA-NODE-MEMORY-FACT-2026-09-06`；正常任务是在模型/KV/worker arena 分配前，
  枚举当前进程可用的稀疏 node id，读取所选 node 的物理容量/当前压力事实，再由上层结合全进程约束和
  利用率 policy 形成预算；查询也用于运行期低频重规划，但不是 allocation 热路径；
- 范围：一个自然、fallible、unknown-safe 的 per-node memory value/query；Windows/Linux/macOS 的公开
  OS 路线；无效/稀疏 node、零值、缺字段、解析溢出和并发冷查询；selected Platform 的 source/static/
  shared、直接 host planner、architecture/user guide 与内部复用审计；
- 非目标：rank/worker/GPU 选择、模型/KV/allocator budget、cgroup 到 node 的虚假分摊、reservation、
  placement/migration、memory distance/bandwidth、Platform-thread/fs、Mem/Mem-VM、Native FFI/C API、
  stable C++ ABI、公共仪器与 Formal；
- 依赖：仍为 `core`、`error`；直接 consumer 原型是 C++ host planner，未来 Mem/Compute/engine 只组合
  Platform fact，不产生 Platform 对上层的反向依赖；现有 `numa_nodes()` 是唯一 node-id authority；
- 语义档：单 node 的动态 snapshot；total 与 available 分开，不能证明的字段显式 absent；available 的
  OS 语义逐平台公开，不能冒充 reservation 或成功保证。暂不加入收集式 vector API：调用者已能遍历
  稳定 `numa_nodes()`，单项查询避免隐藏分配并自然处理某个 node 的失败；
- tier：`P1`。新增 query 本身为 P0 control path：Windows/macOS O(1) bounded native queries，Linux
  O(node meminfo bytes)，零常态缓存/metadata/后台工作；数值门只复用既有一次 host-planning workload，
  W1-WIN、5% material margin，验证未受影响 route，不把不同字段的 raw syscall 做苹果/橘子排名；
- 总调查预算 3 小时，其中性能最多 30 分钟；最多一轮产品实现、一次 Windows P1 holdout 和一次因
  correctness 失败触发的产品修复。没有可复现 material 问题时停止，不扩成 NUMA 百科；
- correctness machines：`W1-WIN`、`W1-WSL`、`M1-MAC`；P1 machine：`W1-WIN`；instrument：
  `closure-perf-v1`，P0：`closure-correctness-v1`；
- authoritative inputs：锁定 vLLM/SGLang journeys；Linux stable sysfs node meminfo 与 proc meminfo
  字段语义；Windows documented `GetNumaAvailableMemoryNodeEx`；Apple public host VM/sysctl 路线；
  vLLM 只作 consumer，不自动成为同合同性能 peer；
- 适用门：完整 capability/public-surface trace；API/usability freeze；parser + native system tests；
  concurrency/error/unknown；三平台 selected build；source/static/shared direct call；中央 component graph、
  selected headers、visibility/符号与 no-FFI；architecture/user guide；红队；P0 structural + 一次 P1 holdout；
- 停止：冻结的 `MUST` 全部 `COMPLETE/ROUTED`、未处置 material finding 为零且适用门通过则关闭；某 OS
  不公开 total 时 optional absent 是诚实能力，不是 blocker；若没有公开机制能返回任何 node memory fact，
  则该平台明确 `Unimplemented/Unavailable`，不得猜测。

`REOPENED -> SCOPED` 前 registry snapshot：
`C:\public\project-hyle\_validation\platform-v10-numa-node-memory\module-status-before-scoped.json`，
SHA-256 `2067C966DE0B4C66395AF5FE046ED5338D325759D48EC84C2D18FC6FA0EF9C98`。

## Capability / surface audit 与有限 FIX（`AUDITED`）

### 两遍能力与 owner 审计

| ID | 正常任务与语义边界 | 当前路线 | 覆盖/裁决 |
|---|---|---|---|
| `PLAT10-C01` | 枚举进程可用的稳定、稀疏 NUMA node id，并验证一个 id | `numa_nodes/numa_node_available` 已从 process placement view 建立唯一 id authority | `COMPLETE / KEEP` |
| `PLAT10-C02` | 对一个可用 node 读取当前物理总量（可证时）与可供新 host allocation 规划的动态量；失败、unknown、零值可区分 | 无公开 API；只有整机 `host_memory_info` | `MISSING / MUST / FIX` |
| `PLAT10-C03` | 把 node 物理事实与 cgroup/job/RLIMIT/整机 commit 等进程约束组合 | `host_memory_info().effective_available_bytes()` 已拥有全局约束 | `ROUTED`：consumer 组合；禁止把全局限额复制给每个 node |
| `PLAT10-C04` | node 选择、rank/GPU 映射、利用率、模型/KV/arena 大小与 reservation | vLLM/SGLang 各自 policy；HYLE placement owner 已独立 | `OUT_OF_SCOPE`：Sched/Compute/engine/Mem owner |
| `PLAT10-C05` | NUMA distance、跨 die bandwidth、page residency/migration 与 pressure feedback | 当前 watchlist 已保留真多节点证据条件 | `DEFER`：没有本轮第二旅程或真机 material 证据，不伪装成容量查询 |

自底向上 reverse trace 覆盖 `include/hyle/platform/{numa,host_memory}.h`、`src/platform/{numa,
host_memory}.cc`、Platform tests/CMake、selected header/package smoke、Mem 的 NUMA owner 与四组推理旅程。
现有 placement API 均有独立任务来源；没有 benchmark-only node query、第二套 node-id authority 或需要删除的
旧入口。新增能力仍属于 `platform`，不新建 factor，也不增加依赖边。

### 冻结 C++ API 与平台事实

```cpp
struct NumaNodeMemoryInfo {
  unsigned node = 0;
  std::optional<std::uint64_t> physical_total_bytes;
  std::uint64_t physical_available_bytes = 0;
};

Result<NumaNodeMemoryInfo> numa_node_memory_info(unsigned node) noexcept;
```

- `node` 必须属于同次进程生命周期的 `numa_nodes()` id space；错误 id 为 `OutOfRange`，不静默回退；
- 成功结果总有 `physical_available_bytes`；total 只有公开 OS 路线可证时才 present。若 total present，
  available 必须 clamp 到 total；零是合法压力状态；
- Linux：读取 stable sysfs `nodeN/meminfo`；total=`MemTotal`，available 是 `MemFree +
  Active(file) + Inactive(file) + SReclaimable` 的饱和、clamped planning estimate。该近似沿用锁定 vLLM
  的真实 CPU journey，但 guide 必须说明它没有 `/proc/meminfo:MemAvailable` 的 per-node watermark 修正；
- Windows：多 node 直接采用 `GetNumaAvailableMemoryNodeEx` 的 free+zero pages，total absent；单逻辑 node
  复用 `host_memory_info`，此时 total/available 都可证。不得把 performance counter、WMI 或启动时 free
  snapshot 当总容量；
- macOS：只有 HYLE 逻辑 node，复用 `host_memory_info`；任意其它 id 仍失败；
- 查询动态、fallible、不缓存，不是 reservation；重复查询可以变化。调用者若还受 cgroup/job/commit
  限制，必须另外读取 host snapshot，并保证全局 budget 只计一次。

不加入 `NumaCapability` bit：这是可直接尝试且可失败的事实查询，不是 placement primitive；单节点平台
通过 host fact 完整支持，伪造一个“native per-node” capability 反而会让用户写错分支。不加入 vector/
callback collector：稳定 node span + 单项 query 已完整、无隐藏 allocation，并允许逐 node 错误处理。

### 复用、成本与有限 FIX ledger

| FIX | 产品修改 | 复杂度/常态成本 | 验证 |
|---|---|---|---|
| `PLAT10-F01` | 增加冻结 value/query；复用现有 node-id、Error `Result/Status` 与 host single-node route | Windows multi O(1) 一次 native query；Linux O(meminfo bytes)；mac/single O(1) host query；其它热路零变化 | parser/native/invalid/zero/unknown tests |
| `PLAT10-F02` | Linux exact-prefix parser、checked KiB conversion、饱和求和和 required-field/DataLoss 处理 | 固定逐行 stack buffer；无 payload-size heap、无 node-count 嵌套扫描 | cross-platform fixture + WSL native sysfs |
| `PLAT10-F03` | architecture/user guide 加入选择规则、平台语义、cgroup/global-budget 陷阱和正常 C++ recipe | 文档不新增 runtime 成本 | docs/link/usability + hostile misuse review |
| `PLAT10-F04` | selected Platform source/static/shared direct consumer、header closure、visibility/symbol/no-FFI 与中央依赖复验 | 仍只 `PUBLIC core error`；无新 target/option/第三方 | component/package contracts，三平台 |
| `PLAT10-F05` | 新 query 签 P0 structural；旧 host-planning contract 只运行一次 frozen P1 holdout | 5% NI；不得新增 raw syscall microbench 或更改 peer/margin | `closure-correctness-v1` + `closure-perf-v1` |

以上五票是本轮全部授权。placement、distance/bandwidth、GPU affinity、Mem API、Platform-thread/fs、
新 CMake option、Native FFI 与公共仪器均不在 ledger；发现问题只能记录相邻 trigger。

### 七问预裁决

1. **第三方：**产品路径无第三方库；Windows SDK/Kernel32、Linux stable sysfs 和 Apple Mach/sysctl 是
   OS 合同。vLLM/SGLang 是 demand evidence，不进入链接或类型；无性能 peer 被伪造。
2. **功能：**四组 replay 中唯一未处置 Platform `MUST` 是 C02；ggml/llama 的 NUMA placement、vLLM
   容量/压力、SGLang rank binding 均能由现有 placement + 新 fact + 上层 policy 完整表达。
3. **内部复用：**node id 复用现有 NUMA authority，单节点事实复用 HostMemoryInfo，错误复用 Error；
   不反向依赖 Mem/Container/Sched，也不重复 cgroup resolver。verdict `PASS`。
4. **性能：**新路线是 P0 control path，timing peer comparison `NOT_APPLICABLE`；结构档最优且不改变
   allocation/placement 热路。旧 P1 host planner 仅作同合同 holdout，不把两种字段相互排名。
5. **FFI：**全局为 `N/A`；不增加 C header/schema/bridge、undecorated export、FFI option 或 layout。
6. **构建/导入：**仍在 `hyle_platform`；必须实测 source/static/shared 与 selected component，不新增开关。
7. **API/文档/红队：**单项 fallible query、optional total 与明确平台 available 语义形成自然 C++ shape；
   重点攻击 invalid sparse id、global limit 重复计算、Windows total 猜测和 Linux 近似冒充精确值。

`SCOPED -> AUDITED` 前 registry snapshot：
`C:\public\project-hyle\_validation\platform-v10-numa-node-memory\module-status-before-audited.json`，
SHA-256 `B2DD4B9A72671F32D03830513539AB1C108B9F6A7CCBDDCE75C289007556970D`。

## Product candidate（`CANDIDATE`）

- product commit/tree：`3c1ef2392d1f05c79edfe1c5d4ea5558bd917e3c` /
  `63b58d1051d86c4aecb1bf736ae84265e359a5f5`；公开 API、OS 路线、复杂度、测试消费者与文档在该树冻结；
- Linux 顺序解析 stable node meminfo；Windows 多 node 动态解析
  `GetNumaAvailableMemoryNodeEx`，单 node 与 macOS 复用 `host_memory_info()`。没有新增 cache、heap
  collection、常态 metadata、依赖边、target、option 或 FFI surface；
- candidate 前定向结果：W1-WIN MSVC 19.50、W1-WSL GCC 14.2、M1-MAC AppleClang 21 的 Release/WERROR
  selected Platform 编译、公开头和 native tests 全部通过；WSL 实际读取 node sysfs，macOS/Windows
  实际运行 logical-single-node route；
- Windows installed shared/static Platform SDK 的 focused consumer 均直接调用新 symbol 并运行通过；
  shared DLL export 是 C++ decorated symbol，未新增 C/Native FFI；candidate 后仍须完成最终交付记录、
  红队/P0 与冻结的 W1-WIN P1 holdout；
- `AUDITED -> CANDIDATE` 前态 registry snapshot：
  `C:\public\project-hyle\_validation\platform-v10-numa-node-memory\module-status-before-candidate.json`，
  SHA-256 `BDEA85E3518DDE903331A684810D57B2252B0B159FDF9E942FB99A0D586C5F9D`。

候选后若公开形状、node-id authority、可用量语义、OS 算法、依赖或正常路径成本变化，必须退回
`AUDITED`；测试、证据与 terminal 文档可以继续补齐，但不能静默重绑定产品身份。

## Verification manifest

全部验证绑定 product candidate `3c1ef2392d1f05c79edfe1c5d4ea5558bd917e3c` / tree
`63b58d1051d86c4aecb1bf736ae84265e359a5f5`，且没有候选后产品修改。

| risk / journey | verdict | evidence |
|---|---|---|
| API、parser、invalid/sparse id、unknown/zero/clamp | `PASS` | Windows Platform tests；exact Linux node/key/unit/overflow fixtures；每个 live node 的 native snapshot |
| concurrency / dynamic snapshot | `PASS` | 4 threads × 8 rounds 遍历可见稀疏 node；value 始终保留 node identity，present total 时 available 不越界 |
| W1-WIN | `PASS` | MSVC 19.50 Release/WERROR fresh selected Platform build、header isolation 与 native tests |
| W1-WSL | `PASS` | Ubuntu 24.04 / GCC 14.2 Release selected build、header/tests；实际读取 stable node sysfs |
| M1-MAC | `PASS` | AppleClang 21 arm64 Release selected build、header/tests；logical node 0 走真实 host snapshot |
| source/static/shared delivery | `PASS` | source selected target；Windows installed static/shared focused consumer 均直接调用新 C++ symbol；DLL 仅有 decorated C++ export，无 C/FFI surface |
| dependency/build | `PASS` | `hyle_platform` 仍只公开依赖 `core,error`；无新 target、option、第三方、反向依赖或 hidden collector allocation |
| usability/docs | `PASS` | architecture/user guide 明确 optional total、逐 OS available 语义、动态非 reservation 与全局 constraint 只组合一次 |
| red team | `PASS` | 攻击错误 node、Linux prefix/unit/overflow、Windows total 猜测、single-node fallback、global-limit 重复计费与多线程冷查询；无未处置 product finding |
| P0 node-query structure | `PASS` | Windows bounded native query；Linux O(node meminfo bytes) stack-first scan；single-node O(1) host route；旧热路径零新增状态/分支 |
| frozen P1 host-planning holdout | `INCONCLUSIVE` | 4/4 product NI 数值通过，但唯一 fresh bundle 的 5% known-delta control 未资格化，不能签 renewed P1 claim |

### P1 holdout 的准确解释

- contract：`platform-host-p1-v3`；W1-WIN；oneTBB
  `e00c68dcac0e83947c9b54f6d5304c0616c9639f`；15 个独立进程、两种执行顺序和 placement strata；
- 四个 same-contract product strata 的 5% non-inferiority 均为 `PASS`；median advantage ratio 为
  `1.020066--1.022953`，即点值约快 2.0%--2.3%。3 行 exact `PARITY`，1 行只因 CI 穿过等价带上界而
  `INCONCLUSIVE`；没有产品 loss；
- A/A、10% known delta、timer floor 与 placement control 均通过。5% known-delta 的两个 median
  `1.034243/1.042137` 落在预冻结 `1.025--1.075` 接受带内，但 99.26% CI 下界
  `0.993802/1.017608` 穿过下界，因此 control 为 `FAIL`，整包只能是 `DIAGNOSTIC`；
- raw bundle：
  `C:\public\project-hyle\_validation\platform-v10-numa-node-memory\perf-evidence`。contract-lock /
  manifest / summary SHA-256 分别为
  `2714905D56EA9CCED7132A5EA55FE44C2925AACEF45B14A774EEDE4EF987C420` /
  `54CE72F8D9E118530A597B9F1D3A95942CA8C93A76D266B4489D1F23AFEB7444` /
  `DEE64F9037346EBAA65D25C47A90BB48756FC9D940B71DBB11ABE6C2B7A4FBCE`；
- 冻结规则没有因 control failure 自动从 n=15 升到 n=31，且本任务只授权一次 holdout。不能在看到结果后
  改采样规则、换顺序、重跑或继承旧 candidate 的 receipt；这是独立公共仪器问题，不授权继续改
  Platform 产品。

## Closure decision

本轮从 `CANDIDATE` 终止为 **`BLOCKED`**。scope、architecture、correctness、delivery、usability/docs、
red-team 与新增 node query 的 P0 structural gate 全部 `PASS`；P1 engineering 为 `INCONCLUSIVE`。
产品没有被撤销：真实推理消费者缺失的 node 容量已经补齐，三平台与三种交付形状均通过；只是当前身份
不能获得新的 Platform-wide P1 签名。

- blocker：`PLATFORM-V10-PERF-CONTROL-RESOLUTION-01`——冻结 holdout 的产品 4/4 NI 通过，但唯一 bundle
  无法资格化 5% known-delta control；
- blocker owner：公共 `closure-perf-v1` 仪器/性能证据 owner，而不是 Platform capability owner；
- resume condition：另建有界 instrument task，在不查看或改写 Platform product 结果的前提下，预冻结
  known-delta 在 n=15 未分辨时的 n=31 escalation/判定语义；该规则经独立 qualification 后，新建
  Platform `RESUME` round 从 `SCOPED` 开始，只运行一次新合同下的 fresh W1-WIN holdout。不得改 API、
  OS 算法、peer、margin 或挑选执行顺序来恢复；
- known limitations：Windows 真多 node 路线只能公开 available，total 诚实 absent；Linux available 是
  没有全局 watermark 修正的 planning estimate；node snapshot 不是 reservation，也不吸收 cgroup/job/
  commit 全局约束；本轮没有 Windows 真多节点或 macOS 不存在的真 NUMA hardware coverage；
- reopen/resume triggers：上述精确仪器条件；或真实消费者证明 node-id/available 语义、correctness、
  security、OS route、依赖/交付缺陷，或具名需求证明 distance/bandwidth/residency 等 deferred family
  应归 Platform。

四项 LOC 绑定 candidate，`loc-report.py --module platform --commit 3c1ef239... --self-check --json`
为 `PASS`：全库代码 459,121；自实现包含测试 119,445；自实现不包含测试 75,504；Platform 实现 6,964。

<!-- MODULE CLOSURE RECEIPT: BLOCKED -->
