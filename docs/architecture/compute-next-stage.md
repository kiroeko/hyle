# Compute/Graph：统一配置、冷路由与 CPU/GPU 组合设计

> 用户授权的新范围。C1/C2/R1/C3在已接受边界内完成；WSL Vulkan十二项实GPU运行保留未资格而不阻塞。
> CPU/GPU联合、异步、重叠和expert迁移按H1–H3独立实施验收，不借单路线资格代签。
> 2026-10-10 用户授权补齐 offload 基础边界：普通 CPU 存储与有界传输暂存分离、组合资源账单与预算、
> 准备后的传输/等待诊断；纳入原 H1–H3，不新增阶段。H1/H2已CLOSED；H3已完成Windows/Air的source、
> 原生重叠与shared交付，static/WSL可适用交付继续验收。
> 当前 owner/验收见[实施卡](../implementation/tasks/build/compute-unified-delivery.md)和 registry；
> [原设计卡](../implementation/tasks/build/compute-entry-design.md)保存入口统计与设计事实。
> 通用增量依赖图只登记为 DEFER 待办，本轮不实施、不增加组件或热路径成本。

## 1. 五项选择各有一个 owner

| 问题 | 决定者/时机 | 表达 | 不能据此推断 |
|---|---|---|---|
| SDK 编进哪些能力 | 构建配置 | 一个 HYLE_COMPUTE_ROUTES 集合及中央 component/依赖锁闭包 | 模型运行放置或全硬件 SOTA |
| 此任务允许在哪计算 | consumer，在 compile/prepare 前 | 单路线 Context，或明确 CPU+一个 GPU 的 native endpoints/片段约束 | SDK 含 CPU 就可偷偷执行 CPU |
| 数据在哪存放、何时 ready | Mem/native resource owner 和编译边界 | 实际 storage/encoding/ownership/completion | host 权重必须用 CPU 算术；统一内存无需同步 |
| 路线内谁实现 op/region | HYLE 冷规则 + Provider hard admission | [现有 matcher/decision table](compute-routing-system.md) | 用户排序或运行时计时比赛 |
| 图如何分片、转换和复用 | Graph 冷编译；engine 提供领域约束 | 保留 native 程序的合法完整 plan | 模型层数、KV/专家/缓存策略应下沉 |

普通用户只选择路线及真实输入，不另设置 Provider 排序、timing profile 或一个“启用混合推理”
布尔开关。联合任务由显式两个 endpoint 与片段放置表达，不复制第二套路由 IR。
已编译计划在执行期只取固定程序和运行状态；不探设备、搜索规则或改赢家。

## 2. 唯一普通构建入口

### 2.1 路线集合和默认

统一配置将原四路线开关及 CPU 的 oneDNN/Torch 开关替换为
`HYLE_COMPUTE_ROUTES`，接受非空路线集合 `cpu`、`cuda`、`apple`、`vulkan`。
单项和如 `cpu;cuda`、`cpu;apple`、`cpu;vulkan` 使用同一解析器。
统一小写、去重、固定 canonical 顺序；集合顺序没有 preference 含义。
未知项/显式空/AUTO/不支持的 OS-ABI 组合在 configure 失败。
省略时，Compute/Graph 产品保持完整 `cpu` 默认；无此产品闭包时不声明或探测计算依赖。

四个普通 `compute-cpu/cuda/apple/vulkan` preset 仅设置路线集合、产品和正常 Release 交付值；
tests/bench 默认关闭、build jobs 2。源码嵌入和 preset 最终都进入同一 CMake 合同，
平台 wrapper 只负责选择实际编译器，不再解释路线。Graph 已依赖 Compute，仍可单选 Graph 产品。

构建集合可包含多个宿主实际支持的 GPU 路线，供应用创建独立 native Context；
第一版联合执行仅资格化 CPU+一个 GPU，不以多路线 SDK 推断 GPU+GPU 联合支持。
CUDA-only 不拉 CPU 数学库；vendor 必需 host 控制代码不算 CPU Provider。

以下命令使用当前统一配置；C3已按明确接受的验证边界完成，不据此宣称H1–H3联合能力已交付：

```text
cmake --preset compute-cpu
cmake --preset compute-cuda
cmake --preset compute-apple
cmake --preset compute-vulkan
cmake -S . -B build-compute-mixed -DHYLE_COMPONENTS=graph "-DHYLE_COMPUTE_ROUTES=cpu;cuda"
```

### 2.2 完整路线自动闭合必要实现

| 路线 | 自动依赖责任 | 用户仍需提供的真实环境 |
|---|---|---|
| CPU | 对受支持 OS/ABI 自动接完整合格 ggml/ATen/格式程序；支持的 x86 接 oneDNN；ARM 不强开未资格 oneDNN | 受支持 C++ 工具链及完整受锁源码/CPU SDK 获取途径 |
| CUDA | 所需 cuBLAS/ggml/CCCL/原 SGLang/DeepSeek/FlashInfer/Torch 程序按平台和已准入条件接入 | 系统 Toolkit、标准 CUDA 架构目标及运行机器 driver/device |
| Apple | 合格 MLX/Metal/ggml-Metal 与原生资源 | macOS Apple Silicon、Metal 工具链/最低部署版本 |
| Vulkan | ggml-Vulkan/IREE 和必要 native 资源/编译制品 | 源码构建的 host shader 工具；运行时 Vulkan loader/device，macOS 为实际 MoltenVK 部署 |

能力完整度继续由共同模型/算子/格式目录和 native admission 约束；选路线不承诺非法 dtype、
任意 shape 或不支持硬件都可执行。合格平台上获取必要依赖失败必须诊断失败，不能静默得到 lite CPU。
不为未资格架构自动虚构缺失 Provider；未知硬件用合法成熟默认，缺真实能力明确拒绝。

普通产品不允许 Torch/oneDNN/ggml 等更细 ON/OFF。维护 fixture 使用内部隔离机制，
不导出第二套 custom/lite 用户产品。旧六个公开选择变量不保留兼容双轨：
发现旧 cache/父工程参数即给迁移错误，列出新集合示例；用户主动迁移后再配置。
由 owning target 派生的 `HYLE_COMPUTE_*` presence 宏保留，**宏不是用户开关**，不改变同一对象布局。

保留产品选择、HYLE static/shared/install，以及真正的工具链/标准硬件编译参数。
cache/root/archive/受审 source override 只改变获取路径，必须核对锁身份，不关闭能力。
共享 source cache 不共享跨工具链 binary build；不增加下载器/第二依赖注册表。

### 2.3 SDK、构建 recipe 与应用资产

source/static/shared 从同一中央图投影同名 `hyle::compute/graph`；安装 metadata 增补可查询的路线集合，
复用现有 features/lock/provider recipe。安装 consumer 读取实际 SDK，不用 producer 开关修改已编译库。
如传入路线需求，只能断言它是 SDK 能力子集，不能触发下载/重编译或伪造 presence。
HYLE_STATIC 只选择 HYLE linkage；必要 LibTorch 等动态 runtime 仍由闭包交付。

软件 recipe 包含相关版本/内容、compiler/ABI/linkage/flags、ISA 和 policy identity。
目前 CPU-only/mixed 的 ggml pin 不同；先保留合法成熟版本组合并明确身份，
不为“一个版本好看”未经资格升级 GPU，也不把 CPU-only 精确 WIN 复制到 mixed SDK。
Provider/版本/flags 改变后，旧 exact rule 只有所有实际条件都匹配才命中，否则合法成熟默认。

统一 `hyle_stage_runtime(app)`：沿**实际链接的 owner**收集 DLL/so/dylib/native assets，
组合既有 generic/Apple helpers 和 package registry；按 target 链接闭包而非猜测 C++ 调用来决定资产。
多路线 SDK 的聚合 target 可连入全部所选路线，届时资产也须完整；不虚构“只调用 CPU 就不需要其它 runtime”。
要求纯 CPU 产物隔离时选择单 CPU SDK；不增加每个 app 的 Provider 禁用菜单。
Windows 必须把所需 DLL 放 exe 同路径；支持 DLL 更新但 exe 未重链接的增量 staging。
Unix 用已审 loader/RPATH 策略并验证安装搬迁；Apple 带真实 mlx.metallib。
运行 driver/系统 loader 不代装，源码 host compiler 与二进制 runtime 要求分别说明。
离线模式复用受锁 cache/SDK archive 和现有禁网语义，缺件报具体路线/依赖/身份/获取方式。

## 3. 冷路由保留并校准

继续采用当前 `RouteEnvironment + WorkloadContext/ModelArtifactProfile + RouteSubject`：
实际 OS/ABI、ISA/SM/架构、driver API 与 release、runtime/toolkit、Provider 版本；
模型与制品 revision/digest；phase/region；operation 参数及**所有端口** dtype/encoding/shape/layout；
资源/并发/replay/完成条件。模型文件的 Q4/Q8 名称不能替代混合 tensor 合同。

两类合法默认必须区分：

1. 有完整资格的精确条件命中已验 best-of；普通用户无须知道该条件。
2. 其它合法条件使用预先固定的成熟工程默认，不假称测过的 WIN。

必要事实未知时不命中依赖它的规则；只有默认本身完整 admission 成功才可执行。
unknown 不当零，注册/数组顺序不当性能结论。没有一家能满足语义/资源则返回具名缺口，
不会因为同一 SDK 有 CPU 在 GPU-only 任务偷跑。

专家仍可接自定义 Provider/kernel/完整 region、限制所需实现或资源/replay。
这些是能力与集成控制，不是普通性能菜单；不删除真实 extension 为统一 API 造假的万能 façade。
cold diagnostics 说明 route/device、选中 rule/implementation/recipe、未命中的轴、具名 decline/fallback。
诊断信息复用现有 report owner，不给普通 submit 添字符串、规则扫描或全局 plan cache。

prepare 前的 transactional unsupported/resource decline 可以应用固定后备规则；
接纳/有副作用/创建 artifact 的 fatal error 不切换设备或重算。跨路线仍守同一错误边界。
单路线 R1 只做身份/入口/诊断一致性，不扩大重测全部模型或重排现有 winner。

## 4. 基础库应提供的 CPU/GPU 支持

### 4.1 先分存放与计算

| 任务 | 计算位置 | 所需编译能力 | HYLE 责任 |
|---|---|---|---|
| 纯 GPU | GPU | 一个 GPU 路线 | 原 native graph/program/资源 |
| host 保存权重，分批送 GPU | GPU | 一个 GPU 路线及 host 存储/传输 | 有界 staging、就绪/复用、资源和错误边界 |
| CPU 与 GPU 各执行片段 | CPU+GPU | `cpu` 加一个 GPU | native 片段组合、边界转换/等待、整体生命周期 |

[Accelerate cpu_offload](https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.cpu_offload)
的 CPU 参数存放与 GPU execution device 提供第一项区分依据；它不是 HYLE 推理引擎接入要求。

引擎负责模型文件/权重角色解释、允许的层/专家/阶段放置、KV/prefix 状态、驻留/eviction、
请求 batching 与 decode 循环。HYLE 接收已经解释的图、不可变制品身份、实际 descriptors、
允许的 endpoints、片段约束和预算，不认识“第 N 层必须 offload”或某请求的业务优先级。
HYLE 返回可执行资格/资源账单，让引擎做领域取舍。

### 4.2 llama.cpp 的实际参考方式

以固定源码和[本轮逐点对账](../implementation/tasks/build/compute-entry-design.md#llamacpp-源码对账)为依据：

| llama.cpp 机制 | HYLE 应借鉴的能力 | 保留在引擎的策略 |
|---|---|---|
| 层 device/buffer 分配、CPU MoE tensor override | 分开表达 storage 与执行位置；片段 required/preferred placement | 哪些层/专家权重留 CPU、模型 tensor-name/role 解释 |
| ggml supports_op/supports_buffer 与 op_offload 条件 | 原生 hard admission、预固定放置规则和完整边界成本 | 特定模型是否允许 GPU 执行 host 权重算子 |
| backend scheduler 的 split/reserve/compute | 保留成熟完整多 backend region 和私有资源，资格后复用 | 模型/batch/请求循环 |
| 层末 norm/l_last 显式 placement 避免来回 copy | region 和边界约束可压住碎片化；评价完整转移成本 | 哪个模型边界具有层末语义 |
| expert copy callback 依运行 IDs 只传使用到的 bank 范围 | 固定程序可以按输入 IDs 处理有界数据搬运；不重新挑 Provider | MoE cache 命中/eviction 和业务驻留策略 |
| async copy 不支持时同步两 backend 再阻塞 copy | 区分实际 async 和 host-wait fallback，报告并计费 | 是否接受该延迟/内存取舍 |

`--cpu-moe` 是权重 buffer 选择，不足以单凭名字保证每个相关 op 必定 CPU 执行；
实际还经过 scheduler/native offload qualification。HYLE 的 CPU-required 约束必须守住，
不能从“CPU buffer”推导或被 vendor 更高优先 backend 默默覆盖。
动态 expert IDs 属正常输入，数据选择不是 Provider 搜索；若 full region 已承载它，
HYLE 不再读取 ID、重写稀疏数学或复制模型 cache。

### 4.3 原生片段，而非万能执行器

当前 common CompileEnvironment 只承载同一 Device，Apple/Vulkan 另有原生 Program；
保留这些正常路线，不把一个 CPU Context 塞进去、删掉 guard 就称为联合执行。
拟在 Graph 增加**显式可选的粗粒度组合编译**，共用当前 typed DAG/语义和冷 matcher。
CPU+CUDA、CPU+Apple、CPU+Vulkan 使用各自 native endpoint 类型和局部公开头，
不使公共 Graph 头引入全部 GPU SDK。

组合编译器拥有 node coverage、canonical 边界、依赖/liveness 与 whole-plan 预算；
片段仍由原路线 compiler 生成 native Program/Executable，Provider 可吞一个 op、融合片段或完整 region。
借 [ggml backend scheduler](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/include/ggml-backend.h)
完整承载时，必须固定实际 backend/语义/资源规则及重复运行资格，普通用户不排序 backend 数组。
需检查其 split/allocate 是否随 shape/bind 重做；不能拿 managed replay 冒充零分配 prepared replay，
也不能把 reset 的失效资源交给旧 lease。不为 HYLE 全部亲自调度拆掉成熟融合。
其它片段只在粗边界组合，不能加逐 kernel vtable、统一 Queue 或第二套 scheduler/allocator。

候选入口形状示意（**尚非产品 API**）：

```cpp
// Route-specific header exposes only CPU and CUDA native types.
auto compiled = graph::cuda::compile_hybrid(
    graph, cpu_context, cuda_context, resources, options);
auto invocation = compiled->create_invocation();
invocation->bind_cpu(input_id, host_tensor);
auto result = invocation->run();  // first general bridge explicitly blocks to completion
```

Apple/Vulkan 对称表达任务，但保留 MLX array/Vulkan Tensor 原生绑定与结果类型。
`options` 传 region/node 放置、实际 workload、分域预算和明确 transfer 策略，不传 Provider 排名。
可共享 immutable compiled artifact；每次并发运行拥有独立 invocation、临时资源和输出 slot。
同步 `run()` 是简单使用与 correctness 基线，不是本轮组合能力的终点。
2026-10-09 用户确认：H1–H3 纳入有界异步传输/依赖、可重叠执行，以及引擎驱动的专家权重片段迁移。
各路线在实施前冻结原生 copy/compute 时间线、ready/completion 和进展 owner；复用成熟完整 region、
原生事件及已有 Sched，不另起后台 runtime。不能把在新线程上阻塞整条链当作实际 copy/compute overlap。
必要 host wait 要报告并计入完整成本；硬件/Provider 无合法重叠承载时明确说明该实际域，
不能仅因函数名 async 或返回 pending 就签 overlap。

### 4.4 边界资源是完整合同

每一条跨片段边必须记录源/目的 native domain、全部 port descriptor、实际物理 encoding/layout、
owner/borrow/lease、就绪事件、转换程序及 resource/transfer route。
相同物理格式才能 byte-copy；编码不同只能用合格转换或拒绝，不能 reinterpret 后称数值正确。
host staging、repack、上传/下载、同步、融合损失进入冷 P0/完整费用，不只比较 kernel。

| 组合 | 可复用基础 | 必须补齐/验证 |
|---|---|---|
| CPU+CUDA | CPU/CUDA native artifact、Mem resource、CUDA stream/copy/Completion、Graph 复用计划 | 跨 Device 粗边界；CPU↔CUDA ready/owner、pinned/pageable 合同、独立槽与终态 |
| CPU+Apple | CPU artifact、MLX array/stream/program、native owned results | 有 owner 的 host↔MLX 桥；统一内存访问/可见性/精度/layout 资格；CPU resource 不能直接冒充 MLX array |
| CPU+Vulkan | CPU artifact、Vulkan Tensor/Program/fence、显式 upload/download | staging 和整条链的保活/复用；VkDevice/queue family 必须相符；borrow_ready 不接受未处理的外部 pending semaphore |

第一版允许明确 copy 和 host 等待，不承诺 zero-copy。import/shared memory 只有 exact owner、
格式和同步条件被证明后才用。Apple 共享物理内存并不消除 MLX owner 或等待。

预算分 compiled、CPU、GPU、staging、每 invocation/private Provider 成本；未知不当零。
同一 alias 物理 backing 不重复当新分配，但其寿命/峰值必须覆盖所有使用者；driver 私有成本标明排除。
分批 host 权重以固定容量的 staging/device slot 承接，复用前等待上一使用者真正终态。
流式权重应作为显式受绑定输入/容量子片段，不能被当整个不可变 GPU 常量在 warmup 全部上传。
若 Provider 仅支持完整驻留/prepare packing，就报告不适用该流式旅程，优先其它合格成熟承载。

专家权重迁移由引擎提供实际 IDs/范围、目标存放/执行约束和驻留决定；HYLE 提供受预算的
数据选择/搬运、固定容量暂存与多槽复用、绑定更新、跨 native 时间线依赖和在途 owner 保活。
变更权重 payload/所选 expert 不重选 Provider；数值/物理格式/shape domain 或允许执行位置变化
必须按原合同重新准备。纯粹迁移和下一组权重准备可与独立已接纳计算重叠，但有读写依赖的
buffer 不得提前复用。成熟 region 管理这些机制时保留它，HYLE 不再重写稀疏数学或模型 cache。
专家命中策略、预测预取、淘汰和何时移到 CPU/GPU 属引擎；多 GPU 不纳入。

### 4.5 失败、取消和关闭

编译失败释放 partial artifact；只有显式无副作用 decline 允许冷期选择预声明合法后备。
运行期失败停止依赖后继、保留原生错误，已接纳 GPU 工作必须观测终态/保活，不重算到 CPU。
request_cancel 是协作停止：取消未接纳后继，不能承诺抢占既有 GPU kernel。
输出只在其成功 ready 后发布；旧 lease 防止 slot 覆盖。外部 mutable 状态由应用声明，
纯/effect 与 native 编译能力不匹配就拒绝，不能裁去 effect/control edge 假装完整。

组合 plan close 停止自己的新调用并排空自己的工作，不关闭应用借用的全部 Context 或其它 plan。
继承各路线已有 Context close/drain 的差异，不强制它们同名方法具有相同阻塞语义。
owner handle 可被制品保活；Mem resource/quota/executor backing 仍按明确借用合同存活至最后使用。
不因调用方丢弃结果或取消就在 GPU 未结束时释放 buffer。不同 invocation 对外部状态冲突仍需实际顺序。

### 4.6 资格与限制

三组合各需真实多数学片段的 typed consumer：推理子图和数据管线；完整正确性按原 dtype/encoding/状态，
跨边界重复运行、保留旧结果、坏输入、OOM、cancel、close 和 source/static/shared。
host 权重模式需真实分批容量/重用旅程，不能用设备枚举、两 Context 创建、mixed SDK compile、
简单 copy 或 mock Provider 冒充联合支持。
有限 consumer 不外推全部 27B/2.4T 模型容量/执行；同精度/所有必要边界成本才可称性能比较。
原单路线证据不因此失效或重跑，但也不自动签新组合。

异步/重叠额外验收：实际原生 copy/compute 进展与依赖证据、至少两个受限 slot 的
在途/更新/复用、expert IDs 每次改变但 Provider 固定、坏 ID/范围拒绝、OOM/cancel/close、
丢弃 completion/旧 lease 的终态保活。只用 callback/polling 名称或两个独立任务耗时相近
不能证明 overlap；若需计时，只对稳定候选的完整旅程有界运行，不扩大算子 P1。

### 4.7 Offload 基础机制补充合同

2026-10-10 用户授权把以下三项纳入当前七包 goal。引擎把显存预算、GPU 层数或比例转换为实际
片段放置、tensor 描述和资源约束；HYLE 接收这些约束，不解析模型层、决定 offload 比例或管理 KV。
H1 首先补齐并验证，H2/H3 在各自 native 域兑现相同任务；不复制 CUDA 类型或强造万能资源/Queue。

| 基础机制 | 必须提供的自然路线 | 验收边界 |
|---|---|---|
| 普通 CPU 存储与暂存分离 | CPU 计算可直接消费普通 host-accessible 存储；只有跨设备边界需要有界暂存；CUDA pinned backing 是传输域，不是所有 CPU 权重的必选存储 | CPU-resident 权重大于 staging 容量仍可参与真实混合数学；上传/下载、别名、格式、OOM、取消和在途复用有完整合同 |
| 组合资源账单与预算 | 复用现 Graph 的存储/工作区/生命周期报告及显式迁移池容量，按实际 CPU/device/staging 域汇总 compiled packing、每 invocation、全部在途槽及借用资源；分别标明已知值、保守上界、未知及排除项 | workspace 限额不冒充总显存限额；库内可管理资源按申明预算接纳，未知私有成本不能按零；外部权重/KV由引擎计入，实际共享 backing 不重复相加，旧结果保活仍占账单 |
| 准备后的传输/等待诊断 | 组合入口自然查询已准备依赖及迁移的实际路径，复用现 ReplayReport/CompileReport；区分设备排序、CPU等待、暂存/转换与合法重叠条件 | 查询与实际执行一致，准备前或改绑失效有明确错误；host-wait 拒绝策略涵盖组合新增边界，热执行不搜索/计时/重新选择 |

资源账单不承诺系统级显存预留或防止其他进程占用；driver/runtime 不可控成本明确列为未知或排除。
设备可用内存只是引擎决策时的快照。预算改变导致放置/格式/shape 域变化时重新准备相应计划；
相同域的 payload/expert 选择更新继续复用固定程序。完整模型的分配策略、缓存与集成验收归引擎。

具体源码反例、owner、H1-07至09及批量验收见
[H1 terminal](../implementation/tasks/compute/compute-h1-cpu-cuda-composition.md#h1-terminalcpucuda组合基础机制完成)。
三项已由CPU+CUDA的[H1终局](../implementation/tasks/compute/compute-h1-cpu-cuda-composition.md)与CPU+Apple的
[H2终局](../implementation/tasks/compute/compute-h2-cpu-apple-composition.md)完成适用验收；CPU+Vulkan按
[H3活动卡](../implementation/tasks/compute/compute-h3-cpu-vulkan-composition.md)补齐剩余交付。
旧候选或单路线证据只覆盖原合同，不代签这两个新组合。

恢复后先批量完成三项产品接线，API/算法固定再冻结一个新候选，集中验收受影响 source/static/shared
消费者。CUDA 重型编译改为 build1/test1，重型 run 串行，启动前确认内存余量；因内存暂停时不启动。
只补缺门和真正受影响项，同身份已通过证据复用；本次文档/状态更新不构建、不跑 P1、不重启 C3。

## 5. 增量依赖图只登记待办

`DEFER-GRAPH-INCREMENTAL` 的身份、触发和负空间见[设计卡](../implementation/tasks/build/compute-entry-design.md#明确待办)。
未来为参数化建模/资产更新提供可选通用版本与失效层，计算节点可调用当前 Compute/Graph。
当前 typed DAG 的编译复用不等于 dirty/result memo；AI decode 的 KV/prefix/模型状态仍由引擎解释。
本轮不扫描历史增量方案，不选库、不创建空组件，不给普通推理 submit 加 dirty metadata。

## 6. 实施顺序和停止

用户已启动实施：C1统一路线→C2 SDK/资产→R1冷绑定校准→C3批量交付，
随后 H1 CPU+CUDA→H2 CPU+Apple→H3 CPU+Vulkan；包括后续确认的异步重叠/专家搬运机制。
有限 FIX/owner/验收由各活动卡维护，设计卡不作为新的当前队列。
当前模块 terminal record 不回写，旧 goal/百分比/队列不恢复。
新组合提出的新需求不受旧“模型 CPU+GPU 本轮范围外”永久禁止，也不能把新设计当旧证据里的完成。
到达七包适用验收后停止，不自动扩算子、实现推理引擎或启动增量图。
