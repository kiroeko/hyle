# CPU ggml原版优化程序接入

<!-- MODULE REVIEW: SCOPED -->

- baseline：`12904dc99`，Windows canonical；普通CPU SDK取得/loader已独立关闭。
- owner：`compute_ggml_runtime`，仍为AUDITED；本卡重新定界CPU有限范围后绑定当前record。
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
