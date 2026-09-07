# 已冻结模块性能机会审计（2026-09）

<!-- PROJECT AUDIT: CLOSED -->

## 冻结任务卡

- owner / kind / demand：HYLE maintainers / `READ_ONLY_PROJECT_AUDIT` /
  `FROZEN-PERF-OPPORTUNITY-2026-09-07`，用户要求区分真实微输与顺序、布局、噪声及不等价比较。
- baseline / replay commit：`4f48e25f6590e0bb4c061f1bd38b00852b0dd248`。
  canonical checkout 为 `C:\public\project-hyle\hyle`；开始时工作树干净，`main`、本地
  `origin/main` 和远端 `refs/heads/main` 一致。
- 前置工具：selector v1 与格式证明修正 task 均已 `CLOSED`；上述提交的
  [CI](https://github.com/axiomates/hyle/actions/runs/34088377302) 为 success；实际执行
  `validation-select.py plan --base <baseline> --candidate HEAD` 返回 READY，空变更跳过 67 门。
  **这只确认已提交/推送的工具主体与轻量路线可用，不再认定最终锁定。** 用户随后提供独立只读复核：
  test CMake 误扩产品 seam、全库 P1 claim 作用域错误、native context 缺少真实 compiler/stdlib/flags
  强制约束，以及真实小 factor native 集成尚未执行。后续独立 selector 修复轮处理这三项和一次有限
  native 集成；本轮不改工具、不签 native PASS，既有性能审计不因此失效。
- 权威输入：[registry](../../module-status.json) 自动筛选
  `freeze_confirmation.marker == FROZEN`，得到 19 个 `CLOSED` 模块。只读取这些记录指向的
  当前 closure task、CURRENT 性能证据及其锁定合同、manifest、必要的机制与消费者来源。
  历史 rejected 数据只解释已失效命题，不补充当前性能格。
- 范围：一次证据清点、完整弱势格矩阵、同合同与机制审计、有界排序及至多一个重开建议。
  P0 的计时比较明确不适用；不把历史 P1 续签到当前 P0。
- 非目标：任何产品实现、API、layout、常态 metadata、registry/status 或 closure card 修改；
  util-runtime/sched/io 产品工作；compute/graph；四组引擎能力超集新一轮审计；工具或统计核心改造。
- 预算：本轮只读调查最多四小时、每个当前 bundle 一次提取、每个疑似机会一次已有源码/codegen
  机制核查；产品修改、构建、P1 qualification 和新增性能样本预算均为零。
  本卡不申请例外诊断额度，缺证据时记录具体 METHOD_GAP，不以补跑取得结论。
- 比较口径：`advantage_ratio > 1` 有利于 HYLE；5% reciprocal band 只决定原冻结 gate。
  任一顺序 median < 1、双顺序 < 1、CI 完全低于 1 或 CURRENT 的 PARITY/INCONCLUSIVE/已知
  微输都入矩阵。保留逐 order/placement 的 median、CI、单位、证据身份与限制，不池化方向。
- 裁决顺序：先 `NOT_COMPARABLE`，再检查顺序/placement/CI，随后判断稳定幅度、机制和消费。
  使用 `ORDER_OR_PLACEMENT`、`PERSISTENT_MICRO_LOSS`、`MATERIAL_LOSS`、
  `ALGORITHMIC_OPPORTUNITY`、`NO_ACTION`；证据完整度另列，不把缺证据伪装成平局。
- 排序：消费者热度 × 差距 × 证据可信度 ÷ 修复成本；定义量纲和未知值，禁止把无机制项目
  写成待实施 FIX。微输只有机制和真实 material 消费两把钥匙同时成立才推荐重开。
- 验收：19 模块全部有当前证据去向；所有命中弱势格有完整分类、同合同/控制/机制/消费判断；
  每个真实机会给出范围、置信度、reopen trigger；有限排序，至多一个建议模块及条件化 FIX；
  selector selected/skipped/reason、docs/closure/diff 均通过，产品与 registry diff 为空，提交推送。
- 适用门：证据 JSON/hash/引用和矩阵覆盖核验；`check-module-closure.py`、`check-docs.py`、
  `git diff --check`。无产品代码测试风险；C++ tests、sanitizer、交付和性能采样不适用。
- 停止：全部格完成分类（允许明确 METHOD_GAP）即关闭本 project audit；无两把钥匙可以推荐零个
  模块。不可把缺证据自动变成产品 blocker，不改变任何模块状态，不扩大审计或反复测量。

## 审计结果

本轮建议重开 **零个模块**，产品 FIX 集为空。三组“所有换向的中位点估计均输”完整保留，但其全部
置信区间仍跨 1；未获得可证明的 `PERSISTENT_MICRO_LOSS`、`MATERIAL_LOSS` 或成熟 peer 的
`ALGORITHMIC_OPPORTUNITY`。这不是用 5% gate 掩盖 1%–5% 差距，也不是证明差距必定来自噪声。

SIMD top-k 只有一个条件化调查项：通用入口与 small-k peer 的代码体积不同，有机制假设，尚无因果
证明或真实消费者 material profile。Concurrency 的 shared-lock 成本有明确语义用途，不能当作
无用开销删除。Error success 未找到热路径机制差异。

本文是基线审计事实，不是第二份当前模块状态或实施授权。用户提供的 selector 缺陷留给下一独立工具轮；
本轮没有用 selector 的 native receipt 作为任何产品验证或性能结论的依据。

### 证据覆盖与保留限制

审计基线 tree 为 `28a5ee8eca682298da1327f2ac5e29bdafa43a05`。自动筛选的 19 个 FROZEN 模块中，
4 个只签 P0；其余 15 个模块对应 13 份 CURRENT bundle、27 个 workload、140 个预声明 strata。

- [机器矩阵](../../../../evidence/frozen-performance-audit-2026-09/audit-data.json)：完整 registry 身份
  投影、current task/document hash、bundle/contract/peer/build/machine identity、逐层数字、分类与限制。
- [逐格表](../../../../evidence/frozen-performance-audit-2026-09/strata-matrix.md)：27 个 workload 的全部
  140 个计划格，缺失数字明确为未知，不补成 1；不是新样本。
- [原始 metadata 快照](../../../../evidence/frozen-performance-audit-2026-09/retained-metadata.zip)：
  8 份现存 bundle 各自原字节的 manifest、contract-lock、summary、controls，共 32 个 JSON；
  SHA-256 `c519c9e1c1af86b0c2886892e11229e2200f7995236ddf1cc83cf751cbb3192d`。
  ZIP 保持原始文件 hash，不是完整 raw observations/executable bundle，也不恢复 Formal 资格。

8 份现存 manifest 与 CURRENT 文档所列 SHA-256 一致，其 summary/contract-lock/controls 与 manifest
引用一致，canonical contract hash 一致。原 `QUALIFIED` 是继承的历史资格事实，本轮没有执行资格化。
Error、SIMD、Concurrent 所用现存 benchmark executable 的 hash 也与对应 manifest 核对一致；只读取
文件和已有 COFF 符号，未运行 executable。

| 模块 | 当前 round / 证据范围 | 本轮可读取程度 |
|---|---|---|
| core | core-v5 / disabled assertion 与 fatal hot/cold P0 | 当前 closure/P0；无 timing 格 |
| container_intrusive | intrusive-v4 / counted、forward、hlist、sequence | CURRENT 文档；原 bundle 缺失 |
| util_core | util-core-v4 / 3 种 FunctionRef traversal | CURRENT 文档；原 bundle 缺失 |
| error | error-v5 / 5 个 source consumer | CURRENT metadata hash 已核对 |
| obs | obs-v5 / owner、shared metrics | CURRENT metadata hash 已核对 |
| math | math-v4 / checked LCM、bit-ceil P0 | 当前 closure/P0；无 timing 格 |
| time | time-v5 / fixed-work interoperability P0 | 当前 closure/P0；不续签旧 clock P1 |
| timer_queue | timer-queue-v1 / scheduler lifecycle | CURRENT metadata hash 已核对 |
| platform | platform-v11 / complete host-planning snapshot | CURRENT metadata hash 已核对 |
| platform_thread | platform-v8 / Parker bytes、memory order、provenance P0 | 当前 closure/P0；不续签旧 Parker 点值 |
| platform_fs | platform-fs-v2 / warm-file lifecycle | CURRENT 文档；原 bundle 缺失 |
| mem_reclaim | mem-reclaim-v2 / immutable snapshot read | CURRENT metadata hash 已核对；同页旧 node-map 数字不迁给当前 Container |
| mem | mem-v6 / isolated erased heap lifecycle | CURRENT 文档；原 bundle 缺失；其余 family 为 P0 |
| mem_vm | mem-vm-v2 / model-loader lazy release | CURRENT 文档；原 archive 与 WSL 原目录缺失 |
| simd | simd-v6 / SwiGLU、RMSNorm、top-k | CURRENT 修正后 metadata hash 已核对 |
| container | container-v6 / weighted model-cache LRU | CURRENT single metadata hash 已核对 |
| container_concurrent | concurrent-v2 / immutable membership | CURRENT concurrent metadata hash 已核对 |
| container_queue | queue-v2 / unbounded 4P4C handoff | CURRENT concurrent metadata hash 已核对 |
| container_text | text-v2 / safe UTF conversion | CURRENT terminal single metadata hash 已核对 |

缺失的精确保留位置是 intrusive-v4 `...-58431ddb`、util-core-v4 `...439b8915-perf-evidence`、
platform-fs-v2 `...resume-evidence-r1`、mem-v6 `...host-perf-evidence-r2`、mem-vm-v2
`...hyle-mem-vm-perf-evidence-r1.tar.gz`；完整位置及期望 manifest hash 在机器矩阵中。对已知 Windows
保留根和 Mem-VM 指定 WSL 路径做了一次有界查找，未找到精确对象，停止追索；没有借历史 rejected bundle
补齐。它们是本审计的 `METHOD_GAP`，不自动改变产品状态。

140 格中，76 格有原 summary 全精度数字，24 格有当前文档按层标注的舍入 median；platform_fs 另有
8 个未标层的 median，保留原序列而不猜测 order 映射；32 个 intrusive 格只有 workload 的八层范围。
可读取的精确或舍入 CI 共 100 个，platform_fs 另有“8/8 CI 包含 1”的文档断言但没有界限。
**100 个已知 CI 中，上界低于 1 的格为零；intrusive 的逐层 CI 缺失，不能据此
宣称全库绝无单格稳定微输。** 其 forward 范围跨 1，仍足以排除该 workload 所有层 median 固定输。

source revision 与 product candidate 不同的 carry-forward 只采用当前 closure task 已说明的 factor
产品输入不变关系，例如 Container LRU 的 `111c28ea..3ba50461` 与 SIMD 的最终测试/文档修正。
不把完整仓库 hash 相同当作 factor 证据必要条件，也不假定任意跨提交证据都可继承。

### 全部弱势及 PARITY / INCONCLUSIVE 操作

`r` 为 HYLE 相对 peer 的 advantage ratio；表中范围只供浏览，裁决仍逐 order/placement，不池化。
`ORDER_OR_PLACEMENT` 按用户指定的“反转或 CI 跨 1”分类使用，表示现有证据不足以确认为产品损失，
不表示已经证明执行顺序或地址布局是原因。下表覆盖全部 15 个命中筛选的 workload。

| 模块 / 操作 | CURRENT r 范围 | median < 1 | CI / 控制事实 | 审计类别 |
|---|---:|---:|---|---|
| intrusive counted membership | 1.0118–1.0539 | 0/8 | exact INCONCLUSIVE；逐层 CI 缺失 | NO_ACTION + METHOD_GAP |
| intrusive forward arena | 0.9823–1.0089 | 数量未知；至少 1，少于 8 | 范围跨 1，逐层/控制原值缺失 | ORDER_OR_PLACEMENT + METHOD_GAP |
| util_core const borrowed | 1.00209–1.00594 | 0/4 | 4/4 CI 跨 1 | NO_ACTION + METHOD_GAP |
| util_core nontype free | 0.99156–1.00237 | **3/4** | primary 换向反转；4/4 CI 跨 1 | ORDER_OR_PLACEMENT + METHOD_GAP |
| error 64-node success | 0.992315–0.998603 | **4/4** | 两 placement 全部 CI 跨 1 | ORDER_OR_PLACEMENT |
| obs shared metrics | 1.004207–1.007760 | 0/4 | 2/4 CI 跨 1，另外两层小幅有利 | NO_ACTION |
| platform host planning | 1.015725–1.037168 | 0/4 | exact INCONCLUSIVE 不代表输 | NO_ACTION |
| platform_fs warm 4 MiB lifecycle | 0.987503–1.012026 | 5/8 | 文档记载换向/placement 反转、8/8 CI 跨 1 | ORDER_OR_PLACEMENT + METHOD_GAP |
| mem isolated heap lifecycle | 1.0068–1.0241 | 0/8 | 8/8 PARITY，点值均有利 | NO_ACTION + METHOD_GAP |
| mem_vm lazy release 64 | 1.0307–1.0467 | 0/4 | 2 PARITY + 2 INCONCLUSIVE，CI 均在 1 以上 | NO_ACTION + METHOD_GAP |
| simd split SwiGLU 11008 | 0.997435–1.002530 | 2/4 | 换向反转；4/4 CI 跨 1 | ORDER_OR_PLACEMENT |
| simd weighted RMSNorm 4096 | 1.001350–1.014199 | 0/4 | 4/4 CI 跨 1 | NO_ACTION |
| simd expert top-k 256/6 | 0.984152–0.990501 | **4/4** | 两 placement 全部 CI 跨 1 | ORDER_OR_PLACEMENT；条件观察 |
| container_text safe UTF conversion | 0.986317–1.010171 | 5/8 | 换向/placement 反转；7 CI 跨 1，另一层在 1 以上 | ORDER_OR_PLACEMENT |
| container_concurrent immutable membership | 0.977651–0.988241 | **2/2** | 双 CI 跨 1；合同没有 product placement 层 | ORDER_OR_PLACEMENT；外推合同限制见下 |

Util-core 当前文档的表实际有 **三个** free-function 负向点值；其正文写“两个”是文字计数差异。
本表依据 CURRENT 数字保留 3/4，不把它改写为产品缺陷，也未修改原证据页。

其余 12 个 workload 的 CURRENT median 全部大于 1、exact 全为 WIN，没有进入弱势筛选：
intrusive hlist（1.2206–1.2770）、sequence sort/merge（1.0881–1.1188）；util bound member
（1.24523–1.26437）；Error owning last failure、parallel first error、Join owning failure、registry borrow
（分别 1.2118–2.5097、1.4103–3.9143、1.5182–4.7840、1.7282–1.7515）；Obs owner metrics
（1.1053–1.1452）；timer lifecycle（1.1293–1.1898）；RCU snapshot read（1.3649–1.3812）；weighted LRU
（1.2223–1.2728）；unbounded handoff（1.2614–1.2888）。仍逐层保存在完整矩阵中，不把 Error
owning-failure 的最大顺序倍数宣传为稳定加速，也不把 intrusive hlist 相对旧 Sched baseline 的提升
宣传为领先成熟 peer。

### 所有换向仍输的三组

下面均为 n=15 的独立进程配对中位数与约 99.26% CI。负向幅度是吞吐下降 `100 × (1-r)`；延迟增长
应使用 `100 × (1/r-1)`，两者不能混报。

| 操作 | primary candidate-first | primary peer-first | placement candidate-first | placement peer-first |
|---|---|---|---|---|
| Error success | 0.998603 [0.980225, 1.014927] | 0.992315 [0.967282, 1.002550] | 0.998534 [0.980811, 1.015450] | 0.992954 [0.983568, 1.006976] |
| SIMD top-k | 0.990501 [0.966840, 1.022986] | 0.989142 [0.962797, 1.010234] | 0.984152 [0.954637, 1.008147] | 0.986975 [0.981475, 1.000915] |
| Concurrent membership | 0.988241 [0.961304, 1.025337] | 0.977651 [0.954083, 1.005561] | 未声明 | 未声明 |

名义吞吐差分别为 Error **0.14%–0.77%**、SIMD **0.95%–1.58%**、Concurrent **1.18% / 2.23%**。
不因小于 5% 省略这些数值，也不因全部点值同方向就把它们升格为稳定损失。

### 同合同、机制与消费审计

**SIMD top-k。** [公开实现](../../../../src/simd/sort/top_k.cc) 与
[peer adapter](../../../../bench/simd/inference_consumer_bench.cc) 都执行相同 release 参数检查、
disjoint caller output、NaN-last / tie-low-index 顺序、一次 noinline 调用；k=6 都为零分配的有界插入
O(nk)。没有发现成熟 peer 在这个同合同格上有更优复杂度。peer 是按 ds4 形状补齐 HYLE 语义的 adapter，
不是未改写 ds4 的实测跑分。

只读检查现存 MSVC object 再次得到 HYLE `top_k_f32` 的 `.text$mn` 长度 `0x738`（1848 bytes），
peer 为 `0x529`（1321 bytes），对象 hash 保存在机器矩阵。HYLE 自然入口同时承载 k=1 和大 k 的
heap/full-sort，额外分支/代码体积可能影响前端；**代码体积差是事实，导致约 1% 时间差仍是假设。**
不能由函数总大小推出执行指令数、I-cache miss 或分支 miss，更不能证明拆分会净获益。

锁定 ds4 `9ab705347c1775e7599ede7eb81a6255ec7dccb5` 的 `ds4.c:11583` 定义 `topk_desc`，
`:11618` 附近的 router normalization 实际消费它；这证明操作有用，不证明 HYLE 在该引擎中已集成或
占有多少时间。当前 benchmark 重复固定 sine 派生的 256 个分数；输入分布、缓存与预测器状态仍是形状
限制，未覆盖真实 router 分数序列。不得为重复序列增加布局、metadata 或 benchmark 专用 public API。

**Concurrent membership。** [实际计费循环](../../../../bench/container/product_v6_concurrent_bench.cc)
是 32768-entry immutable set、8 pinned workers、每 worker 4194304 次查询、50% hit；同 hash/equality、
system allocator、预留/填充/预热与 thread setup 均在计时外。计费为全部查询、checksum 与最早开始到
最晚结束的 wall-throughput interval，不是单条 contains 指令的独立延迟。

[HYLE contains/find](../../../../include/hyle/container/concurrent_hash_map.h) 进入/离开 shard shared
lock；[reader admission/removal](../../../../include/hyle/container/detail/adaptive_rw_lock.h) 正常
路线支付原子 RMW。它保护查找期间的并发 erase，相关 guard/erase 测试也已存在。锁定 oneTBB
`e00c68dcac0e83947c9b54f6d5304c0616c9639f` 的
`include/oneapi/tbb/detail/_concurrent_unordered_base.h:147/496/1082/1253/1259` 则显示 warmed find
沿 acquire-loaded links 查找，删除接口为 `unsafe_erase`，并发删除不受保护。

因此需要区分两个命题：冻结的 **immutable 查询任务**输入、线程和 observable 等价，原观察仍合法，
统计类别为 ORDER_OR_PLACEMENT；把这组数字外推为 **相同 concurrent-erase 保证**的 primitive 成本
比较则为 **NOT_COMPARABLE**。当前 closure 语义矩阵中的 contains/insert/erase 宽表述不能由这个纯只读
workload 全部证明，其宽泛解释作废。原数字不因此成为可去除锁的依据。这个审计范围限制没有改写
registry 或原冻结卡，也不是产品 reopen trigger。现有 generic active-registry provenance 没有给出
真实 HYLE immutable-only 热路径的生产频率。

**Error success。** 当前 source consumer 对齐了 64-node traversal、noinline 调用层次、Status/Result
生命周期和最终 observable；success 不分配。CURRENT 页面已认可的 retained codegen 显示 success node
与 evaluate 热路逐指令相同、OK 同为 one-word 编码；外层冷析构目标/布局不同。未发现多余热分支、
分配、同步或更差复杂度。故没有机制钥匙，不提出实现 FIX；没有读取或修改 compute/graph 产品实现。

**其余五个含负向格的操作。** intrusive forward 与 Boost 都使用 layout-matched caller-owned
one-link node、cached tail、相同显式 detach/callback，分配/输入准备在外，零内部同步，append O(1)、
整轮 O(n)；原始逐层证据缺失使更细归因停止。Util-core free-function 两侧是同函数体、两字 borrowed
view、同步生命周期、4096 次 erased invocation、无分配；当前三编译器 P0 记录未证明新增调用或字段。
SwiGLU 两侧为同检查、同 split arrays、同 exact AVX2 provider 的整行调用和稳定 sigmoid 算术；未找到
可解释反转点值的额外资源或复杂度。

Text 的[实现](../../../../src/container_text/string_algo.cc)与
[peer](../../../../bench/container/product_v6_bench.cc)同样支付 validation-with-errors、exact sizing、
endian-correct conversion 三次必要扫描，typed status/offset/count 和失败不写输出语义相同；无额外扫描
机制。名字中的 4096 是八字节计量单元，实际输入为 **256 × 128-byte records**，不是 4096 条生产记录。
platform_fs 两侧对同一 warmed file 执行相同 sharing/failure、complete read/checksum/close，buffer
allocation 在外；没有发现额外 syscall，warm-cache 形状也不能外推冷 model-load。五项均不列 FIX。

其他 PARITY/INCONCLUSIVE 行也没有隐藏劣势：Obs shared 数字全部有利，而且单 writer 测量不能证明多
writer throughput；host planning 对齐完整 value record、Result failure 与 scratch/syscall；Mem 对齐
同一 mimalloc binary、runtime-erased owner、alignment/failure 与全部 allocation/touch/free；Mem-VM
对齐 file lifecycle、64 advice、64 partial unmap、65 survivor teardown，HYLE 的 batch bookkeeping
已经是 O(R log R + F)，不是成熟 peer 拥有更好算法而漏改。P0 模块没有新增 timing 命题。

### 仪器分辨率与测试形状

A/A、5%/10% controls 与 timer-floor 的原 PASS 支持的是当时预冻结的门，不提供 1% 差异的检出保证。
例如 SIMD A/A 的两顺序 r 为 1.019028 和 1.001710，CI 分别为 [0.976717, 1.038744]、
[0.966159, 1.015486]；这个不确定度不能忽略。top-k 约 411 ns 的完整 row 明显高于约 0.209 ns 的
保守 floor，故也不能笼统说“低于计时器分辨率”。**绝对 floor 足够低与百分之一效应可辨认是两回事。**

当前文档的 top-k “约 4.3–7.9 ns”来自两侧绝对中位数之差，配对 ratio 则是另一估计量；本审计不把
前者当作后者的置信区间或已证明收益。Concurrent 是较长 wall interval，instrument 的微操作 floor
不等于该八线程吞吐形状的最小可辨差。Error A/A 也跨亚百分比尺度。没有新样本，因而没有补证
branch/cache 归因、输入形状泛化或真实端到端 materiality。

### 有限排序、reopen 与下一轮输入

排序只用于决定未来先看哪份消费者证据，不给未确证缺陷制造实施优先级。定义：热度 H 为 0（无具名
需求）、1（有具名真实操作）、2（profile 热点）、3（profile material）；差距 G 使用吞吐损失百分点；
证据权重 E 为 0（合同不适用）、0.25（点值与机制假设）、0.5（稳定区间与机制）、1（消费者因果验证）；
成本 C 是相对工程量，不是工期承诺。`H × G × E / C` 是透明的审计排序约定，不是统计概率或预计收益。

| 条件调查排名 | 模块 / 操作 | H / G / E / C | 分数 | 当前决定 |
|---|---|---|---:|---|
| 1（唯一） | simd / expert top-k 256/6 | 1 / 1.5848 / 0.25 / 2 | 0.1981 | EVIDENCE_WAIT；不重开、不实施 |

G 取当前最弱层的点估计只是保守排查排序，不池化效果。并发同步差异是必要语义成本，Error 无机制，
其他负向格反转或缺失证据；它们不构成“待实施优化清单”。合格产品机会数量仍为零。

SIMD 的当前 registry trigger 明确要求“real consumer profile proves ... material end-to-end impact”或
成熟 peer 更好复杂度；本轮两者均未满足。若未来出现该新事实，下一轮才能先建立 SCOPED/AUDITED：

1. SCOPED 绑定届时 registry 的 baseline、具名 consumer trace、任意 n/k 正常合同、现有两 placements
   与真实输入分布；只提出 **一个**内部 cold general-path separation 候选，预计限于
   `src/simd/sort/top_k.cc` 的私有函数组织。公开 API/layout/metadata 保持由消费者合同决定。
2. AUDITED 先证明 general-k 路由/分支确为该 consumer 的成本来源，且拆分没有增加 small-k 的常态调用
   成本；无法证明即空 FIX 关闭，不尝试第二种 API 或 benchmark 形状。当前没有批准这个 FIX。
3. 若两把钥匙成立才冻结有限实现与正确性风险（k=0/1/8/9/n、NaN/tie/invalid、任意 k、三编译器
   code shape）；验证范围由修复后的 selector 与 closure card 决定。只有新的 API/算法冻结候选才可能
   在独立获准的额度内 P1，本审计不授权任何采样。

眼前的下一工作必须先是 **selector 三处有限修复及一次小 factor native 端到端验证**：test CMake
仅 owner、factor/真实依赖输入 P1 claim、真实 compiler/stdlib/flags context。它属于下一工具任务；
工具完成本身也不满足任何产品 reopen trigger，不自动推进 Util/Sched/IO 或上述 SIMD 假设。

### 可重放的数据一致性检查

在仓库根执行以下只读 Python；它检查引用与覆盖，不重算统计、不启动 benchmark，也不把语义判断
伪装成机械证明。文档 hash 使用 baseline Git blob 避免 Windows checkout 换行差异。

```python
import hashlib, json, pathlib, subprocess, zipfile

root = pathlib.Path.cwd()
audit = root / "evidence/frozen-performance-audit-2026-09"
d = json.loads((audit / "audit-data.json").read_text(encoding="utf-8"))
sha = lambda value: hashlib.sha256(value).hexdigest()
blob = lambda path: subprocess.check_output(["git", "show", d["audit_baseline"] + ":" + path])
registry = json.loads(blob("docs/implementation/module-status.json"))
frozen = {m["module"] for m in registry["modules"] if m.get("freeze_confirmation", {}).get("marker") == "FROZEN"}
assert frozen == {m["module"] for m in d["frozen_modules"]}
assert sha(blob("docs/implementation/module-status.json")) == d["registry_git_blob_sha256"]
for module in d["frozen_modules"]:
    for path, digest in module["source_document_git_blob_sha256"].items():
        assert sha(blob(path)) == digest
assert sha((audit / "retained-metadata.zip").read_bytes()) == d["metadata_archive_sha256"]
with zipfile.ZipFile(audit / "retained-metadata.zip") as archive:
    assert len(archive.namelist()) == 32
    for bundle in d["bundles"]:
        if bundle["availability"] != "HASH_VERIFIED":
            continue
        for name, digest in bundle["files_sha256"].items():
            assert sha(archive.read(bundle["id"] + "/" + name)) == digest
        summary = json.loads(archive.read(bundle["id"] + "/summary.json"))
        lock = json.loads(archive.read(bundle["id"] + "/contract-lock.json"))
        canonical = json.dumps(lock["contract"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        assert sha(canonical) == summary["contract_sha256"] == bundle["contract_sha256"]
        for original in summary["workloads"]:
            row = next(w for w in d["workloads"] if w["bundle"] == bundle["id"] and w["id"] == original["id"])
            for source in original["strata"]:
                copied = next(s for s in row["strata"] if s["id"] == source["id"])
                assert all(copied[key] == value for key, value in source.items())
rows = d["workloads"]
assert len(rows) == 27 and sum(len(w["strata"]) for w in rows) == 140
for row in rows:
    assert {s["id"] for s in row["strata"]} == set(row["contract_workload"]["required_strata"])
assert sum(w["screening_included"] for w in rows) == 15
assert sum(w["all_medians_below_peer"] for w in rows) == 3
intervals = [s["estimate"]["confidence_interval"] for w in rows for s in w["strata"] if s["estimate"]["confidence_interval"]]
assert len(intervals) == 100 and not any(ci[1] < 1 for ci in intervals)
assert d["new_performance_samples"] == 0 and not d["recommended_reopens"] and not d["active_product_fixes"]
print("PASS: 19 modules; 27 workloads; 140 planned strata; 8 retained metadata bundles; 5 METHOD_GAP")
```

### 验证与停止记录

适用检查已执行：上述数据一致性检查 `PASS`（19 modules / 27 workloads / 140 planned strata）；
`python scripts/check-module-closure.py` 为 `PASS`（28 components / schema 1）；
`python scripts/check-docs.py` 为 `PASS`（454 Markdown files / 1230 relative links）。
提交前另执行 `git diff --cached --check`，并核对改动路径只有本 task、任务索引及三份审计证据文件。
本次审计工程单元新增产品实现代码为 **0 行**。

`python scripts/validation-select.py plan --base 4f48e25f6590e0bb4c061f1bd38b00852b0dd248
--candidate WORKTREE --output .scratch/frozen-perf-audit/selector-audit.json` 返回 READY，
selected factors 为空。完整 selected/skipped 与 reason 保存在该本地计划，可对提交后的 HEAD 重放。
这只验证文档计划；实际 docs/closure 命令直接运行，没有生成 native PASS receipt。

| 门 | 选择 | 理由 |
|---|---|---|
| docs / closure | SELECTED，2 门 | mandatory documentation / registry contract |
| configure | SKIPPED，1 门 | no product or factor test impact |
| factor correctness | SKIPPED，24 门 | outside the change and its required consumer boundary |
| headers | SKIPPED，24 门 | no public-header seam change for this factor |
| component contract | SKIPPED，1 门 | no global component/package seam change |
| delivery | SKIPPED，14 门 | no delivery boundary change for this product |
| P1 | SKIPPED，1 门 | no explicit frozen P1 card; sampling is never automatic；本卡采样预算也是零 |

合计 selected 2 / skipped 65。无源码或工具实现变化，因此不运行 C++ tests、selector 合同测试、
configure/build、sanitizer、交付回放或任何新性能采样；未运行门不记 PASS。registry 未变，`--previous`
状态迁移检查不适用。产品、registry、selector、benchmark/contract、当前模块 closure card 的 diff 均为空。

Terminal verdict：**CLOSED / READ_ONLY_AUDIT，含五份原始保留证据 METHOD_GAP；推荐重开 0，活动产品
FIX 0，新增样本 0。** 已到达有限分类与停止条件，不以缺证据扩大调查，不把条件观察变成实施授权。
下一 session 先处理独立 selector 修复轮；模块状态继续只以 registry 为准。
