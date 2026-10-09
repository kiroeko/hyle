# 下一阶段：统一计算配置、增量依赖图与混合执行

> 2026-10-09用户提出的新需求设计。baseline `ff59c88c1`；本页为设计/施工边界，
> 不声明以下新入口或能力已经实现，不恢复已完成goal或旧队列。
> 优先顺序：统一构建入口先施工；增量依赖图与CPU+GPU组合先完成有界设计再实施。

## 1. 三种图职责

| 层 | 责任 | 执行与数据owner |
|---|---|---|
| 增量依赖图（拟新增可选层） | 输入版本、依赖失效、按需求求值、可复用结果与更新事务 | 结果owner/缓存配额由本层管理，任务借Sched；计算节点可调用现有Compute/Graph或其它库 |
| 当前Compute Graph | typed DAG、合法region、数值/格式、资源/转换/依赖、可重复原生程序 | 各路线/Provider保留原生程序、私有资源与完成 |
| 混合执行计划（拟新增组合能力） | 已允许CPU+一个GPU路线的片段放置、跨路线边界、完整终态 | 每片段仍用原路线程序；边界借Mem/Compute完成，不造逐kernel万能执行器 |

普通LLM decode中，新token/状态使多数层都需重新计算；现有预编译、常量packing/共享、
显式状态重放与引擎KV/prefix缓存仍有实际用途。通用dirty图不是这些机制的替代。
图依赖是必须的，自动增量求值则用于跨次复用的稳定子任务；可服务图像编码不变、
权重准备更新、资产/几何处理等。KV可复用性、请求/模型语义由引擎决定，不靠比较tensor
地址或自动扫描payload推断。增量层不得给当前普通推理submit添加无关版本/缓存成本。

## 2. 统一配置入口：首先实施

### 2.1 用户只选择交付路线集合

拟新增唯一公开选择 `HYLE_COMPUTE_ROUTES=cpu|cuda|apple|vulkan`，支持显式集合如
`cpu;cuda`、`cpu;apple`、`cpu;vulkan`。它表示编译进SDK的能力，不表示运行时模型放置。
四个统一preset使用相同合同；保留HYLE通用产品选择、static/shared、安装路径与真正必要
的系统SDK/硬件编译目标参数，普通Compute+Graph preset代填产品和默认完整能力。

路线未选时不probe/download/build其native实现。路线已选时，HYLE按平台取得并配置
当前能力集合所必需的合格Provider；当前CPU x86使用oneDNN/ggml/ATen等，ARM按已验
合法承载，不要求用户排列库或另写依赖target。必要依赖缺失必须失败，不静默变成精简模式。

Torch/oneDNN/ggml等细粒度ON/OFF不再是普通产品配置入口；不能在普通完整CPU模式下
关闭必要Torch然后仍宣称完整算子覆盖。维护fixture的实现隔离属于内部developer机制，
不是第二套普通用户产品。可提供依赖cache/已下载archive/合格SDK root作为获取途径，
它们不能关闭能力。移除旧公开双轨开关，冲突旧配置给一次清楚迁移错误，不默默覆盖。

### 2.2 一致的能力与运行资产合同

source/static/shared具有同一能力集合，SDKmetadata记录路线、必要实现与版本/recipe。
安装consumer使用相同公开target；不拿HYLE_STATIC暗示第三方runtime全静态。
拟统一应用资产入口 `hyle_stage_runtime(app)`，按已链接的路线交付实际DLL/so/metallib，
复用现有runtime闭包与Apple staging；不新建下载器或第二个依赖登记表。系统driver仍
由运行机器提供，库不会代装。构建层不做运行时性能搜索，也不能仅凭构建路线集合
证明CPU/GPU组合计算已经成立。

### 2.3 有限施工与验收

C1统一route-set/preset及内部派生选择、移除普通provider禁用入口；C2统一SDK能力/
运行资产入口；C3一次候选批量核对四单路线及三种CPU+GPU构建组合、OFF隔离、
缺SDK/依赖诊断、source/static/shared搬迁与资产。数学/原热执行未变不重跑算子全目录
或P1；适用构建/交付必须实际验证，不能用文档或mixed编译PASS签mixed推理。
公共build改动必须单独绑定owner、SCOPED/AUDITED与有限FIX；不扩为全库重审。

## 3. 增量依赖图第一版设计边界

需求纵切是用户描述的参数化建模和既有异构资产/资源更新；实现前需要两条具名可运行
消费旅程及成熟来源对账，不能把模拟Provider当几何库接入或默认重新自写。

必须包含：显式DAG与稳定node身份、修改事务/环检查；输入版本或显式invalidate、
受影响依赖传播；请求root的必要dirty子图求值；结果owner/旧快照保活及有界缓存；
并发请求合并、cancel/失败、更新version防止迟到结果覆盖新版本；可诊断未命中原因。
输入身份/版本由调用方明确提供，不默认深比较大对象或hash大tensor。纯计算允许memo，
外部副作用不能因输入看似不变就跳过；可变资源必须显式声明依赖/版本与顺序。
输出等价/版本cutoff可作为有界扩展，不能在热路径盲比大结果。

不内置CAD几何kernel、特征业务语义、撤销重做、模型loader/KV策略；第一版先显式、
有限DAG，动态发现依赖、循环/固定点与持久化缓存须由实际consumer另证明。
复用Container/Mem/Sched/Error及当前计算程序，不复制调度器、timer或GPU资源系统。
成熟承载调查包含Blender depsgraph的失效/求值owner和oneTBB等依赖执行负空间，
不把领域实现整库搬入HYLE，也不把只有任务排序的库冒充完整增量求值。

## 4. Offload要分别描述存放位置和计算位置

| 模式 | 权重/状态存放 | 算子计算 | 构建能力 |
|---|---|---|---|
| 纯GPU | 主要GPU | GPU | 一个GPU路线 |
| host权重分批送GPU | host与有限GPU驻留 | GPU | 一个GPU路线＋现有host内存/传输，不因此要求CPU数学Provider |
| 真正CPU+GPU联合 | host/GPU各有明确所有者 | 部分片段CPU、部分GPU | `cpu;cuda`、`cpu;apple`或`cpu;vulkan`，以及待验的跨路线执行组合 |

`cpu_offload`名字不能推断CPU是否执行算子。Accelerate的cpu_offload将权重保存CPU、
按需转到execution device执行；ggml backend scheduler的多backend分图/复制是另一
类成熟接入依据。两者都不能仅凭名字算HYLE已有支持。

### 4.1 分工与计划

引擎负责模型加载、允许的层/专家/阶段放置约束、KV及驻留/eviction策略与循环；
HYLE负责通用片段资格、边界资源、实际copy/import/格式转换、依赖与完成、取消/关闭
和有界资源计划。以已固定路线内规则选Provider，组合计划只在冷期应用已验条件和
完整成本；不重新跑计时比赛，不让用户排序Provider。

第一阶段明确CPU+一个GPU路线；每region持有原路线Context/Program及独立运行槽。
边界记录源/目的storage、dtype/encoding/layout、必要转换、所有权与真正ready条件。
同源成熟runtime可完整承载混合region时优先保留其融合/资源调度；其它region经窄的
组合边界衔接，不为了HYLE逐算子调度拆成熟库。组合不强迫四路线统一Queue/vtable。

显式允许CPU计算才能选择CPU片段。纯GPU遇到不支持/容量不足仍明确拒绝，不能因为
SDK同时有CPU就自动fallback。接纳后失败不换设备重跑；并发槽、外部状态与旧结果
lease必须跨transfer真实终态保活，预算按CPU/GPU/暂存分别计费，unknown不当零。

### 4.2 CUDA/Vulkan与Apple的区别

CUDA/Vulkan先按真实buffer/copy/completion建立合法跨域边界，可规划异步传输与
预取但必须保留固定预算和成本。第一版可使用明确host staging，后续zero-copy/import
只有合格接口和必要同步证据才采用。Apple统一内存不等于MLX/CPU任意无成本共享：
验证实际owner、dtype/layout、可见性与stream完成后才能复用，否则明确转换/复制。

### 4.3 新能力资格

host权重模式实际分批/重用/预算/错误验证；联合模式各CPU+GPU组合都须一条真实
多片段消费者、相同原精度/编码输出、跨边界重放/旧结果/坏输入/OOM/cancel/close及
对应构建三交付。性能只有完整转换/上传下载/同步/融合损失成本，不拿裸kernel竞速。
不能以两个Context能创建、mixed SDK能编译或简单copy证明联合推理已支持。
模型容量不足的流式方案需实际验，不能把已跑微型分段外推成27B/2.4T全模型验收。

## 5. 当前合同与下一步

已收口四路线单路线实现/证据保持。用户提出的联合计算是下一阶段新增范围，
之前“模型CPU+GPU不在本轮范围”不能解释成未来禁止；也不能修改旧证据冒充已支持。
先实施统一配置C1–C3；并行推进这里只指独立设计工作，不自动启多agent或重型run。
增量图与混合执行各先收敛owner/API/生命周期/冷热边界、成熟承载和有限FIX，再改产品。
未创建新goal，本页不自动恢复旧goal/百分比/队列，也不创建空组件target。

## 参考依据

- [HYLE能力准入与复合机制](capability-completeness.md)
- [当前计算图](graph-design.md)与[冷路由](compute-routing-system.md)
- [Blender dependency graph](https://developer.blender.org/docs/features/core/depsgraph/)
- [oneTBB dependency flow graph](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.1-rev-1/elements/onetbb/source/flow_graph/dependency_flow_graph_example)
- [Accelerate dispatch/offload](https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.cpu_offload)
- [ggml backend接口](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/include/ggml-backend.h)：本轮URL读取失败；接口/示例另有官方主线原文，仅作待本地固定pin源码核对的调查入口，不签资格。
