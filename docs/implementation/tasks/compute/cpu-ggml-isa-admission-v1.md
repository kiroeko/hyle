# CPU ggml原版优化程序接入

<!-- MODULE REVIEW: CANDIDATE -->

- baseline：`12904dc99`，Windows canonical；普通CPU SDK取得/loader已独立关闭。
- owner：`compute_ggml_runtime`；本卡重新定界CPU有限范围后绑定当前record。
  旧F35的GPU/A6/D6剩余事实保留，不因本卡关闭CPU接缝而解除其它路线或其BLOCKED。
- trigger：真实默认CPU ggml将SSE42/AVX/AVX2/FMA/F16C/BMI2全部关闭，而Native为SIMD分派，
  因此不能据此给Native签WIN。旧M1安全ISA floor是可移植初始交付决策，不是已接优化程序的证据。
- consumer：普通CPU Direct/Graph和已有整region，在prepare绑定原版程序；不要求用户排序Provider/
  选ISA/提供timing，不提高consumer TU的编译floor，不拆whole-region。
- scope：原固定ggml CPU target的源码组成，以独立私有AVX2程序编译；保留现有基线程序。固定CPU/OS
  feature条件冷绑定，一套程序拥有自己的pool/plan/运行/释放；Host graph/POD元数据与workspace借用。
- non-goals：任何新数学、ggml升级、AVX512/AMX/ARM优化扩张、GPU修改、公共loader/性能工具重写。
- 调查预算：一次有界源码/接入探针，Windows已完成F32 norm→MatMul两次执行；仍待narrow/量化、
  资源和交付验证。最多两次具体根因纠正；不在本卡证明Native领先或运行逐算子P1。
- tier：P0，优化原程序的接入/ISA安全/资源生命周期；后续Native精确域裁决再固定适用P1。

## 审计与有限FIX

| ID | 原源码/接缝事实 | 固定改动与适用验收 |
|---|---|---|
| CI1 | ggml原CPU target拥有完整C/C++ kernel列表、头/依赖；原版同时支持基线与独立CPU backend | 从真实target的SOURCES/SOURCE_DIR/INCLUDE_DIRECTORIES读取同版组成，独立编译AVX2/FMA/F16C/BMI2；没有复制/修改数学源。固定x86_64 Windows/Linux，其它平台保留原路径 |
| CI2 | 原CpuExecution按slot持有fallible pool、plan和Mem workspace，整图compute和pool free必须属于同一程序 | 私有版本化整程序入口：init、pool创建、plan、整图run、pool释放；冷期绑定一次，run不查ISA/搜索/测量。复用Platform缓存CPU+OS事实；基线在不满足条件时可用，已启用且适用的优化程序缺失具名失败，不悄悄拿scalar参加比较 |
| CI3 | 两份程序使用同版ggml POD图/plan，但各自CPU查表与pool必须初始化/释放匹配 | F32/F16/BF16与既有GGUF真实链路、replay、不同slot、cancel、OOM/创建失败、close；不将F32探针外推到量化 |
| CI4 | 独立程序是runtime资产，source/static/shared须同样定位且不污染不相关owner | 复用已通过的owner runtime登记和package私有shim，消费者仍只调用同一staging helper；三交付/relocation、CPU OFF与基线条件隔离；不改通用staging/治理核心 |

Windows接入探针已实际通过，位于`.scratch/validation/cpu-ggml-isa-admission`：原数学源码只读，
新增module由原CPU target的完整源码组成，另接私有整图入口；主图由基线ggml构造，独立程序执行
完整norm→MatMul，再与基线核对。这里只证明F32 POD互操作方向可行，不签普通产品/完整Provider PASS。

## 预声明验证与停止

集中接完CI1–4再冻结一个候选，低并行编译一次；CPU runtime/provider/route/Graph直接consumer回归、
ISA冷条件/私有ABI拒绝、资源故障、真实量化片段和三交付串行验证。CMake组件/CPU OFF隔离检查，
selector记录selected/skipped及适用范围；docs/closure使用明确previous-ref。已有未改数学证据仅按
身份复用，不能复用基线的计时给优化程序签P1。

完成本卡仅表示成熟CPU优化程序有真实可用路径。随后按
[Native逐域裁决](cpu-native-provider-domain-decision.md)冻结有限重叠域比较和清退/路由FIX；
两者都完成才可通知CPU算子与路由收口。其它三路线暂停，SIMD基础保留。

## 交付根因纠正（1/2）

Windows源码交付的普通Direct/Graph F16片段通过；静态安装consumer发现私有shim错误地在
`bin`查找优化程序。CMake将`MODULE_LIBRARY`安装到`LIBRARY`目的地，即现有统一规则的`lib`；
DLL实际已安装到`lib`，不是编译或数学缺失。本包仅将shim的Windows路径改为实际`lib`位置，
消费者仍由staging helper复制到exe同目录，不改变通用安装/验证工具。该CMake路径纠正不影响
已通过的383项CPU GTest；只重验交付与对应文档/状态检查，不重复整批数学回归或P1。

## 当前已运行验收（尚未关闭）

- 产品候选：`bf4027e7`；安装路径纠正候选：`d0b774c5`，未改C++或上游数学。
- Windows CPU runtime、ggml Provider、普通CPU Direct及Graph共8/8 CTest目标通过，
  其中383项GTest；覆盖F32/F16/BF16、既有真实GGUF/lowp片段与资源/重放路径。
- Windows源码、静态SDK、动态SDK的独立普通消费者实际执行F16 MatMul和完整
  RMSNorm→MatMul Graph，均各重复两次通过；静态/动态消费者只传HYLE安装前缀，
  优化程序DLL由现有helper自动放到exe同目录。
- 待验：Linux优化程序及对应三交付、冷ISA/ABI失败与CPU OFF隔离。当前不能签CI1–4全PASS
  或关闭runtime；Native自研域仍未完成比较、删除和规则定案，本卡不签算法WIN。
- 完整日志/身份/统计见[当前证据](../../../../evidence/correctness/cpu-ggml-isa-admission-v1/README.md)。

## AUDITED绑定

CI1–4已经固定；Windows原版程序探针成功，没有改上游数学。当前runtime owner重新绑定本卡，
旧F35只保留历史与GPU待验事实，不重新施工。Platform缓存事实通过runtime的显式private依赖复用，
不扩充Platform API。AVX2 flags复用既有HyleIsa；私有bridge只用于一个完整CPU程序的pool/plan/run/free，
不是公共或跨路线逐kernel vtable。源/安装runtime登记通过本包私有shim，不修改通用helper。
公开CPU/Graph API与Provider顺序保持；赢的精确规则及自研删除仍由后续裁决卡固定。

## Candidate

- 固定候选：`129f36a9f0c5c31799728cc32afb8fc74f8d3eb8`。产品C++与`bf4027e7`相同，
  包路径已包含`d0b774c5`修正；后续GCC断言括号与运行资产声明仅修改测试接缝。
- API、原数学、layout与常态成本已固定。Windows383项GTest按相同产品C++身份复用；
  Linux以相同产品源、锁与完整同步清单完成51项GTest及三交付。没有用测试数替代范围矩阵。
- Windows及Linux静态/动态SDK实际搬移后重新发现、链接与运行均通过；检查后恢复原验证前缀。
- CPU OFF配置在故意无效的Torch root下通过，断言无CPU优化target、资产registry或活runtime。
  缺程序/错误ABI/缺必要回调的独立普通消费者分别得到预期失败。ISA前提的逐项删除与空事实
  由冷条件单元检查；没有冒称在物理非AVX2机器上测试过。
- 本卡P0只签固定程序接入、资源与交付，不签性能WIN。准备缓存加载/ISA事实并绑定整程序，
  invocation重放无探测/搜索/计时；只调用一次整图compute，pool/workspace按slot释放。
