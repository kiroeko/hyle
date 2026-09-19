# C1-AA-Q1：归一化同实现对照资格恢复

## SCOPED

- owner：`/root`；模块专属 `bench/graph_provider_onednn/normalization_preflight.py` 的验证任务。
  不新增产品 factor，不重开 SIMD、Sched、Platform 或公共性能统计核心。
- baseline：`64e1e169`，F1 产品为 `ae270af9`；Windows 是唯一产品真相。
- 当前授权来源：[oneDNN 活动卡](llm-moe-c1-onednn-normalization.md)的 AA04 资格阻塞。
- trigger：F1 的采样前 workload A/A 通过，n=15 的120个产品进程及30个公共控制进程完成，
  五项公共控制均通过；采样后 workload A/A 在256个比值中有一个1.21435，资格 FAIL。
- 证据：[完整失败与限定诊断](../../../../evidence/performance/onednn-normalization-f1-64e1e169-w1-win/README.md)。
  失败项是 oneDNN 对自身的 Direct LayerNorm 1×2048、primary、执行与分配均 peer-first；
  两个角色虽然保留候选/peer 标签，但此 A/A 模式实际都调用 oneDNN，不能误称 Native 输了。
  该进程内部配对区间也排除1，不能把它直接解释为一个 timer tick。

## 已固定事实

1. source、artifact、host 身份未漂移；Release `/O2`、同一 P 核和自然完成边界保持。
2. 所有 A/A 聚合及 variant/order/allocation 中位数仍在原5%带内；失败来自单个进程比值的
   10% gross guard。不能据此擅自删点、放宽门或把整轮改写成 PASS。
3. 唯一追加的有限诊断已经结束：同一 primary binary、oneDNN/oneDNN 的四种顺序/分配组合，
   受影响项依次为1.00731、0.99646、0.95003、1.01484。没有复现21%偏差；这不证明具体 OS、
   JIT 或代码/数据地址原因，也不替代原失败。无未使用的原样完整重跑额度。
4. F1 只是冷融合准入修复，没有 SIMD、Sched 或热状态改动。该资格异常不能授权修改那些产品。

## 有限调查与 FIX 准入

追加静态调查最多30分钟，只允许两条具名路径：

- 对照合同：核对本模块 gross guard、独立进程/固定因素中位数及公共工程门分别声明什么；
  解释单实例差异与进程中位数结论的关系，不能因已看到失败就修改统计目标。
- 实例差异：核对两个真实 oneDNN plan 的 JIT、资源/代码地址、预热和完成责任是否同义；
  若缺少失败实例的必要观测，就明确证据缺失，不通过反复运行等待再次失败。

先把具体反例、拟修的有限验证代码和验收写成 AUDITED ledger，再改模块验证代码。没有具体缺陷时
不得改公共仪器、性能阈值、产品 API、常态 metadata 或源算法。原始失败和全部观测始终保留。
本卡不是额外产品优化轮；oneDNN 主卡仍只剩原定第二轮性能纠正预算。

若确有可修的模块验证缺陷，验收包括：同义对照、故意固定偏差必须被拒绝、原异常必须可追溯、
制品/合同/机器身份与失败传播、全部过程有界；使用原统一性能仪器。任何恢复后的完整比较必须
先冻结新候选和合同，最多一次，不能追加原样重试。若方案要求改变公共仪器或放松原工程门，
先形成独立、可审查的具体方案，不能在这里顺手实施。

## 停止与恢复

当前仅 SCOPED，尚无验证代码 FIX，也没有性能 PASS。到调查预算后必须给出原因/证据缺口及
明确修复或 blocker；不能继续无界研究。主卡两个 owner 暂为 BLOCKED；Q1 形成受审资格恢复
方案后，主卡按 BLOCKED→SCOPED→AUDITED 重新绑定剩余 AA04，再冻结候选验证。
默认规则、Native 取舍和 C1 整包仍未完成；其他已授权且独立的 C1 工作不由此自动变成 BLOCKED。

## 有限源码复核与 Q01 冷观测

2026-09-19，基于 `c0aa5ae7` 的第一遍复核：两边均由同一个 `Problem` 提供数据、参数和
独立 FP64 oracle；`measure<Route>` 的 selected A/A 确实创建两个 oneDNN Route。两边各有
Context/Queue、immutable plan、invocation、输出及 workspace，均在同一 submit/wait 边界完成。
`Bench::compare` 使用共同 iteration count 和交错配对；父进程使用
`SetProcessAffinityMask`，不是只设置无法由子进程继承的线程亲和性。没有发现这几处语义或
拓扑不对等，也不据此宣布失败实例没有地址效应。

两个 oneDNN plan 会分别生成 JIT 制品；当前日志只绑定 EXE，没有失败实例的 JIT bytes、代码
地址、数据地址或逐段运行干扰事实。不能从源码恢复已退出的异常实例，不能把四次未复现写成根因。

先固定唯一冷观测 Q01：使用已核验的 F1 primary binary，`HYLE_BENCH_AA_ROUTE=selected`、
`HYLE_BENCH_VERIFY_ONLY=1` 和 vendor 自带 `ONEDNN_JIT_DUMP=1`，只执行一次独立进程。
保留原 oracle、绑定/完成检查，收集并散列实际 JIT 制品，核对同 workload 两个实例是否生成相同
代码。沿既有 P 核策略、DLL邻置和禁弹窗设置；输出只在有界临时目录，核验归档后清理。
该动作没有 `Bench::compare` 或计时样本，不改产品/benchmark/公共仪器，不重签旧 P1；它最多
排除本次冷观测中的代码生成差异，不能重建历史实例地址或证明其性能稳定。

## Q1 调查裁决：BLOCKED，停止追加观测

上述一次冷进程退出0，16个 workload 的数值/绑定/完成检查通过。实际生成32个 JIT 文件，
按 benchmark 的逐 workload 构造次序及 vendor 加锁递增的 `unique_id` 对应16对；每对字节及
SHA256完全相同，包括历史异常所对应的 Direct LayerNorm 1×2048。约805KB原始制品、执行脚本、
日志和逐对哈希见[Q1记录](../../../../evidence/performance/onednn-normalization-aa-q1/README.md)。
这不证明历史失败实例也没有地址、缓存或执行干扰，也没有新增任何计时或性能资格。

两条静态路径已检查完，在30分钟上限前停止：本模块单实例10% gross guard 与各固定因素5%
中位数门是不同的预声明要求，后者通过不能覆盖前者失败；两个独立实例的实际工作与完成边界
未发现不等价。历史失败实例缺少代码/数据地址及干扰记录，现有材料不足以定位那一次21%差异。
Q1 **BLOCKED**，没有可由具体反例授权的产品或仪器 FIX，不修改 SIMD/Sched、不放宽门、不原样
追加完整重跑。主卡两个 owner 的 BLOCKED、原始失败及默认规则未签署状态保持。

恢复需要先审计具名、有限的模块验证改动：保留每次 preflight 的独立原始记录与实例观测，避免
只留下最终检查日志；证明它不改变数学、完成、拓扑、热路径或统计门后，再冻结唯一新实验。
仅增加日志本身不能写作资格恢复。公共核心、产品常态 metadata 或放松阈值均不在这张卡授权内。
独立的 C1-Y ggml Vulkan 能力绑定继续推进，不以该单项阻塞扩大为整包停工。

## Q2：恢复所需的具名验证记录修复（SCOPED）

2026-09-19，baseline `8461925b`，owner `/root`。Q1的历史BLOCKED及21%异常保留；本轮只处理
其已写明的恢复条件，不重启已结束的根因调查。oneDNN两个owner按主卡重新进入SCOPED，F1产品
`ae270af9`不变。没有修改默认规则或授权额外产品优化轮。

具体缺口已由源码定位：模块preflight只把各child的合并文本打印到stdout；统一runner在前后两次
调用时均写`validators/normalization-workload-aa.log`。因此新实验需要模块自行保留每次完整
记录，不能只依赖最后一次公共validator日志。错误退出、timeout和解析失败也必须先留原字节。
现有benchmark只打印顺序标签，没有两个实际实例的地址记录；Q1退出后无法补回历史实例事实。

本轮有限范围：`bench/graph_provider_onednn/normalization_preflight.py`、同目录normalization
benchmark的冷诊断、对应合同的记录目录/受控环境及模块专属测试。公共runner/统计核心、产品
API/metadata、SIMD/Sched/oneDNN源算法与构建依赖不变；原16个workload、8个分层、P核策略、
配对方法、5%中位数带和10%单实例gross guard均保留。

拟定FIX仅三项：

1. 每次preflight独立目录，逐child原始stdout/stderr、退出/timeout、nonce/命令/受控环境、
   artifact/contract/source hash和最终裁决；先落盘再解析，第二次调用不能覆盖第一次。
   vendor原生JIT dump放在该child目录并散列；临时文件在证据核验归档后清理。
2. 只在A/A模式、两实例构造及正确性检查后、计时前，记录共享输入与各实例/output/workspace
   handle、实际Provider资格、进程/线程/CPU以及当前private executable地址范围；计时后记录
   同线程粗粒度wall/CPU事实。地址范围不冒称已经绑定每个JIT函数或证明具体干扰原因。
   复用原`run()`及`Bench::compare`，测量回调内不加查询、计数、输出或分配。
3. 原门保持的独立测试：既有gross反例与固定因素偏差必须拒绝；独立原始记录不会覆盖，
   异常/timeout/坏telemetry不能写PASS；验证实例完整性和代码/制品身份。两个真实verify-only
   进程覆盖primary oneDNN/oneDNN与placement Native/Native，不产生性能样本。

先完成AUDITED再改代码。实现/测试最多两次具名纠正；不调查新算法或添加公共仪器。上述改动
通过源码/测试与冷实际检查后，再依法固定CANDIDATE：唯一一次新实验沿原统一仪器、原16项/
全部分层与原n=15/31停止规则，没有原样重试额度。若资格再次失败，保留新记录并BLOCKED，
不循环等待好结果。日志增加本身不等于性能资格恢复，也不签默认规则或C1完成。

### Q2 AUDITED：固定实现与验收

SCOPED提交`882ecd0e`后，owner `/root`完成三项有限FIX审计。观察对象限定为benchmark自己的
实例与当前进程：handle保持opaque，只记值不解引用；Win32只查询private executable地址/
长度/保护属性，不读取页面、不挂钩JIT、不注入进程。JIT字节沿vendor已有dump采集，不能从
相同字节或地址范围直接推断哪个role的每个函数地址。Graph私有backing继续明确未观测。

| FIX | 固定改动 | 验收与边界 |
|---|---|---|
| Q2-A | preflight增加必填`--records`，每次原子创建独立nonce目录；每child独立cwd和二进制stdout/stderr文件，timeout/退出/解析失败的身份及原文先落盘；run及最终JSON使用LF、每文件hash | 第二次调用不得覆盖旧记录；失败时status保持FAIL。合同只把该目录绑定到统一output，并清除/限定诊断与vendor dump环境变量；公共validator最终日志仍按原机制工作 |
| Q2-B | 两个Route加冷`observe`，构造和oracle后记录实例/output/workspace handles、共享输入及已检查的Provider；只在显式A/A诊断开关下查询线程CPU/时间和private executable范围，计时后记录粗粒度wall/CPU差 | 现有两个`run()`、lambda、compare调用、问题数据和oracle保持逐字不变；打印发生在两实例均构造后，不插入两个构造之间。仅A/A打开vendor JIT dump，普通产品采样不打开；冷热成本不混记 |
| Q2-C | 抽出原A/A裁决原样复用，新增模块专属测试与两次verify-only实际观察 | 历史1.21435及固定因素偏差拒绝、对称输入通过、missing/duplicate marker和错误Provider/共用输出拒绝、非零退出/timeout原文留存、原记录身份不可覆盖。只验证validator判据，不伪造Provider或计时PASS |

不因诊断修改产品；`64e1e169..8461925b`的CPU Compute/Graph/Native/oneDNN产品子树没有变化。
新增诊断的编译/layout属于新的benchmark身份，必须重新资格化；不能复用旧EXE性能证书。
只增量编译两个bench并执行上述两个冷进程；适用Python测试、selector、docs/closure门必跑。
通过后才提交不可变候选和CANDIDATE登记，再以fresh build启动唯一原合同P1。Cold验收失败先
保留具体反例，最多两次纠正；资格再次失败按既定停止条件BLOCKED，不附加诊断重跑。

### Q2 实现与冷验收

模块已落实独立nonce目录、逐child原始字节/错误/身份、冷实例与地址范围、vendor JIT身份；
原门从原函数抽出原样复用，11项针对性测试通过。[两个真实冷进程](../../../../evidence/performance/onednn-normalization-aa-q2/README.md)
完成32个workload、64个实例、32份oneDNN JIT记录，没有`Bench::compare`或性能样本。
两个EXE的.text差仍在原4096字节内；只增量编译bench，CPU产品子树与已合格正确性/交付不变。

现有公共contract不接受任意vendor环境变量，因此只登记HYLE冷诊断开关；模块child自己限制
dump，普通采样拒绝dump。测试放在owner的bench目录，不修改公共selector/统计工具。冷验收
后仅纠正首次调用新建status的失败清理，新增第11项测试；所用helper和C++制品均未改变，
保留源码AST复用证明，不重复冷进程。数值、完成、热回调与统计门不变；这不是性能资格PASS。

### Q2 唯一实验停止与目录接线纠正（AUDITED）

运行HEAD `0e107a06dd12a549c8cf5354b0cff1dd6b9e97c6`，实现候选`07a69e60`。
[完整失败记录](../../../../evidence/performance/onednn-normalization-aa-q2/README.md#唯一恢复实验失败)
保留首轮16个A/A进程、256个比值、512个实例及原字节/JIT/身份。产品采样进程为0。
公共runner因validator新建源树未跟踪文件而停止；模块自身的原10% gross门也已FAIL：
oneDNN/oneDNN Direct LayerNorm 17×257，primary、peer先执行/candidate先分配，比值0.906388，
倒数1.103281。两项失败分别保留，目录修正不能撤销A/A失败或恢复性能资格。

接线缺陷已从现有公共`_expand_validator_command`确认：它只替换完整token，合同中的
`{output}/normalization-aa-runs`未展开，落到仓库字面目录。原记录已核验ZIP逐字节及SHA-256，
再以Windows原生命令将该唯一自建目录移到本任务scratch；原cwd/argv不篡改。

两个owner从CANDIDATE退回AUDITED，仅增加一个有限FIX **Q2-D**：合同传独立`{output}`参数，
模块`main`在此根下追加`normalization-aa-runs`；用实际公共expander与模块main的无性能回归
验证带空格输出路径、目录落点及失败记录。公共expander、C++、算法、观测、统计门均不改。
这是接线纠正，不是新实验预算；最多一次实现纠正，适用Python/selector/docs/closure门后
按已触发的停止条件BLOCKED。**不启动新native进程、冷检查、build或P1。**

当前记录能确认相同Provider、共享输入、独立输出/workspace及CPU2；粗粒度线程时间和代码
区域不能归因到两条arm，也不能证明某个JIT地址导致越线。不扩大根因调查、不裁决赢家或删
Native。独立C1剩余项继续，整包完成通知尚未触发。

### Q2 终局：BLOCKED

Q2-D修复为`7f625eb4e18e928f1a782be3ada169297d249f56`；
[接线结果](../../../../evidence/performance/onednn-normalization-aa-q2/path-fix-result.json)与
[12项测试](../../../../evidence/performance/onednn-normalization-aa-q2/path-fix-tests.txt)通过。
实际公共expander处理带空格输出路径后，模块将失败记录放到该output下独立nonce目录。
AST确认只改`main`，合同只有output token变化，C++/公共仪器未变。无构建、native benchmark、
冷检查或P1重试；selector只选docs/closure并保留118项跳过理由，两门通过。

本轮最终为：记录接线已修复；原实验A/A资格FAIL；产品性能INCONCLUSIVE，默认规则及Native
淘汰仍未裁决。两个oneDNN owner从AUDITED进入BLOCKED，owner `/root`，只阻塞这组性能资格。
恢复必须有独立证据指向具名的公平性/测量缺陷或相关环境变化，并先在单独有限卡审计纠正与
资格预算；若需要改变统计门或接受例外，须用户明确决策。仅补日志、路径已修复、换候选名或
等待机器空闲均不满足恢复条件，不隐式启动Q3或重置预算。

独立C1实现/验收继续；全部适用门未齐，不能通知C1完成或进入C2。统一脚本按完整Compute与
Graph（含各provider/backend/route）实现边界统计为48,294行/195文件，自检PASS；不含测试、
bench、第三方及文档。该数量没有被当作功能进度。
