# M2-B1 CPU 成熟接入证据

`source-audit.json` 记录精确锁定源与有限接入差异，不是执行/覆盖或性能 PASS。
活动范围和 C1–C6 见[当前任务](../../../docs/implementation/tasks/compute/llm-moe-m2-b1-cpu.md)。
`development-windows.json` 与 `development-windows-raw.zip` 是首个实现中检查点：Windows
MSVC Release shared、CPU-only、oneDNN-OFF，7个定向测试目标通过，含20项GoogleTest用例、
3个头自足检查及实际 executor consumer。原始构建失败/纠正与最终通过日志一并保留。
记录绑定产品/测试文件hash，不冒充不可变候选的完整 selector、三交付、跨平台或性能验收。
新成熟需求格仍为0/256，21行 CPU 完整门未完成；自研算子保持冻结。

`index-windows.json` 与 `index-windows-raw.zip` 保留后续 C1-I 标准库 Scan/TopK 检查点：
同一机器/配置的7个目标、28项GoogleTest用例通过，普通 Direct/Graph 都已实际绑定。含整数
溢出不发布输出、任意轴/奇数字节stride、稳定 NaN/ties、原始位模式保留、两输出与重放。
完整 selector、四环境、三交付及P1尚未运行，不以本记录关闭B1或增加完整覆盖格。

`layout-windows.json` 与 `layout-windows-raw.zip` 记录 C1-L 的后续开发检查点：同一配置的
7个目标、32项GoogleTest用例通过。包含任意轴 Gather 的四类型/非连续源与索引/越界输出保留，
以及普通 Graph 的 F32/F16/BF16 L2 原版组合、epsilon 位置、中间输出、双 slot 和重放分配检查。
原测试辅助函数编译错误与修正日志一并保存；没有数学内核修补、P1或冻结自研比较。
该证据仍不签完整B1、四环境/三交付/插桩或新增模型覆盖。

`broadcast-windows.json` 与 `broadcast-windows-raw.zip` 记录 C1-B 的7个开发目标、36项用例
执行通过：原版一般广播、任意轴集 Sum/Mean（rank6 63种轴集与三类型/keep-dim/两归约组成
756组差分），另含非连续输入、空归约/空输出与scalar。普通 Graph 分组均值/广播/方差是
单个原版 region，保留两请求和重放验证。计算仍使用原版 ggml，没有新增数学内核或性能采样。
范围与前述开发检查点相同，完整B1和其它适用验收仍未完成。

`remaining-numeric.json` 与 `remaining-numeric-raw.zip` 是直接调用锁定 ggml 的数值资格
反例，不是新增产品能力：确认 all-masked Softmax、ordered Scatter-add 组合与特殊值差异；
小尾值截断只记录事实，未代替适用误差门。该探针没有调用冻结 HYLE 算子或启动性能采样。
oneDNN 新能力范围方案仍待用户决定，原 Q2 状态和失败记录未改。

`indexed-mutation-windows.json` 与对应日志记录 C1-X 的成熟 Scatter-add 开发检查点：原版
ggml `get_rows_back`承载重复更新，公开Direct/Graph均选择`ggml-cpu`；F32/F16/BF16、任意
中间轴、多维/空indices、动态越界及旧结果保留通过7/7定向目标。原版`set_rows`对重叠目标
未定义，因此assign明确留作残差，F02.2不签PASS；跨环境、三交付、P1和CPU B1仍未关闭。
