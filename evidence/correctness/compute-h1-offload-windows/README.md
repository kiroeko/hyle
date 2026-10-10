# H1 offload：Windows 已完成验收与内存中断

产品候选为 `7eaf67b8a496ee19b609533980cd69e7efb5aa49`。本记录保存 H1-07至09新合同下
实际运行结果，不替代原合同的历史证据，也不签整包 CLOSED。

## 用户恢复后的当前结果

2026-10-10用户明确继续；同一产品候选的Windows source/shared/static三交付现均完成。
恢复只接已安装static SDK的消费者，未重编producer，也未重跑source/shared原数学全集。

| 新补验收 | 当前结果 | 证据 |
|---|---|---|
| installed static组合/Direct/contracts | PASS 3/3 | [receipt](static-consumers/receipt.json) |
| shared/static四个changed公开安装头分别独立编译 | PASS 8/8 | [receipt](installed-seams/receipt.json) |
| shared/static SDK真正改名搬迁后的组合消费者configure/build/run | PASS 2/2；原目录在验收时不存在，结束后恢复；无SDK副本 | [receipt](installed-seams/receipt.json) |
| 真实CUDA-only SDK拒绝CPU+CUDA消费者 | PASS预期configure拒绝，没有补下载CPU或偷偷fallback | [日志](final-seams/missing-composition-check.log) |
| 当前候选native迁移/计算物理重叠 | PASS；同device/context不同stream的16MiB H2D与两个成熟GEMM执行区间真实相交；不宣称性能赢家 | [分析](native-overlap/overlap-analysis.json)、[数值控制运行](final-seams/native-probe-math-check.log) |
| WSL当前候选 | 未运行，整包仍CANDIDATE | [归档](recovery-archive.json) |

static消费者批次单并行，观察到本轮进程private最大1.1GiB、物理可用最小42.51GiB；没有触发
16GiB物理/提交余量或12GiB本轮private观察门。它是每秒采样最大值，不冒充精确瞬时峰值。
安装头第一条cuda头命令漏传native Toolkit include，原C1083记录保留；只修task-private命令，
普通find_package消费者本来已得到该transitive include，无产品修改。通过后的同阶段终态与
失败历史均保留，不把旧错误行静默改为成功。

当前候选重新编译了小型native probe，并采一次仅CUDA的Nsight trace：8个成熟FP32 GEMM、
独立weight-slot DMA；两个交叠分别356.988us和379.741us，证明机制可物理重叠，不是收益资格。
同一EXE的未插桩数值控制运行也成功，trace不采需管理员权限的CPU调度，没有UAC/新P1。
原始小型trace/SQLite连同hash已归档，SDK/DLL/producer对象未复制入仓库。

下列各节保留首次运行、内存停止及只读检查时的历史边界；当前结果以本节和新receipt为准。

| 范围 | 实际结果 | 证据 |
|---|---|---|
| Windows source，单并行编译/测试 | PASS，5/5：普通 Graph、普通 Compute、CPU+CUDA组合、Direct权重迁移、组合contracts | [receipt](source/receipt.json)、[测试日志](source/source-test.log) |
| Windows installed shared SDK | PASS，3/3：组合、Direct权重迁移、组合contracts；统一stage将DLL置于exe目录 | [receipt](shared-sdk/receipt.json)、[组合](shared-sdk/joint-test.log)、[Direct](shared-sdk/direct-test.log)、[contracts](shared-sdk/contracts-test.log) |
| Windows static producer | 构建及安装 PASS；同一producer切换static，复用未变vendor对象 | [receipt](static-sdk/receipt.json) |
| Windows installed static SDK consumers | INTERRUPTED；组合configure中用户报告内存告警，尚无消费者测试结果 | [中断记录](interruption.json)、[configure尾部](static-sdk/joint-configure.log) |
| 四个公开源码头各按shared/static宏独立语法检查 | PASS，8/8；仅源码header语法，不是installed header/链接/搬迁验收 | [header receipt](source-header-receipt.json) |
| WSL新候选、安装头/资产/搬迁与新的native trace | 未运行或未裁决，不能写PASS | [归档索引](archive.json) |

source及shared运行覆盖普通CPU主存权重参与真实MatMul、CPU↔CUDA激活、有界staging、完整FP16
gated FFN、共享资源预算的strict/partial、超额回滚、alias去重、双槽/旧lease、准备失效/host-wait
拒绝，以及原迁移/取消/错误/关闭/owner丢弃合同。具体消费代码以
[候选文件hash](candidate-files.json)为准；通过边界仅上述Windows实际入口。

原静态编译曾触发预声明的12GiB可用物理内存保护，自己的编译进程树被停止，见
[首次保护记录](static-sdk/memory-interruption.json)。内存回升后使用相同对象续编，producer及安装
完成。随后用户再次报告Windows内存告警，本轮root Python PID75544及其子进程被停止；会话
终态exit1，停止后没有本轮编译或验证残留。静态消费者不能因producer通过而代签。

停止后的只读观察：可用物理内存约46.5GiB，提交内存约35.3/75.3GiB（46%），页面文件分配
11,842MiB、当时使用2,956MiB。它们不是告警瞬间峰值，不能据此否定用户告警或唯一归因。
当前driver只检查可用物理内存；后续恢复前需兼顾提交内存余量、编译进程峰值及机器上其他工作，
不能以物理余量单独证明安全。重型工作继续停止，等待用户明确继续。

[原task-private driver](validation-driver.py)保留实际运行入口；它是证据，不是新的公共仪器。
归档均为小型文本、receipt及hash；没有复制producer、SDK或CUDA对象，没有启动P1，也没有
以中断刷新性能/失败预算。
文本日志仅剔除行尾空白；索引同时保存原始与归档sha256，未改日志内容或结果。

## 停编译期间的启动保护补充

后续task-private入口增加物理可用16GiB、系统提交余量16GiB和本轮进程树private memory12GiB
三项观察门。系统提交页/限额使用Microsoft
[GetPerformanceInfo](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getperformanceinfo)
及[字段定义](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information)，
本轮进程树只统计当前runner及其实际子进程，不统计或结束用户程序。
每秒采样，触门或监控失败仅停止自己的stage并留证；它不是OS硬内存配额，也不保证峰值无超调。

[只读实测](memory-guard-readonly.json)只证明读取实际指标及缺少恢复标记时在启动子任务前拒绝；
未运行重型job或实测触门终止，不能代签这些资格。新
[driver](memory-guard-driver.py)保留独立版本，未覆盖前述实际验收所用旧driver。
新增`static-consumers`私有恢复入口只接着已安装static SDK，先核对四个changed公开头与产品
候选一致；不重复static producer、已通过source/shared或启动WSL。

只读API还取得本次开机历史提交峰值约178GiB；按Microsoft字段合同，CommitPeak是开机以来
历史最大值，没有发生时间。最近八小时资源耗尽System event2004未找到；这两项都不能还原
本次告警瞬间或证明其归因。当前重型停止约束继续有效，恢复标记只能在用户明确继续后使用。

## 已生成SDK的只读检查

[实际结果](sdk-readonly.json)确认shared/static各四个changed公开头与产品候选字节一致，共8项；
两种SDK的路线集合、内部recipe一致性及组合导出/Graph facade引用正确，导出的61个shared和
37个static库路径都存在。三份shared EXE及递归可达HYLE DLL共30个PE文件的直接导入可在exe
目录、Windows System32或Windows API set解析；没有运行EXE或加载CUDA数学。

本检查不解释CMake全部语义、不递归审查第三方DLL内部或loader-only加载，也不签installed
standalone header编译、static consumer链接/执行、搬迁、WSL或native overlap。上面已通过的
source/shared实际运行仍复用原receipt；恢复只补缺门，不重复producer。

用户再次询问40GiB编译占用后，只读核对确认当前无cl/nvcc/ptxas/cicc/ninja/cmake/ctest任务，
`wsl --list --running --quiet`无运行发行版，未为排查启动WSL。最近这批是共享producer转static的
HYLE库/SIMD及少量CUDA源码编译；原guard停止位置是SIMD
`math/transcendental_highway.cc`。该TU只编AVX2，但包含Highway、classic-MSVC限定的xsimd及
HYLE向量数学模板，启用/O2和/Ob2；不能称它单TU重复编译全部ISA。既有单cl约8.2GiB观察
不等于完整进程树峰值。历史receipt只有整机最小空闲量，无法证明40GiB归属；本轮没有重测
重型编译来填这个历史缺口。static producer已完成，剩下Windows三份installed consumer。
