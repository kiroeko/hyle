# Compute：计算语义、原生执行与提供方

Compute 与 Graph 的环境、模型制品画像、真实端口/DAG 条件、确定性默认优选及 prepare/submit 分界由
[冷路由系统](compute-routing-system.md)统一定义；路线原生资源和执行差异仍见
[四路线](compute-routes.md)。

本文只定义目标设计，不签发实现或性能状态。当前接口见[用户指南](../user/compute.md)，当前组件状态
见 [registry](../implementation/module-status.json)，研究与未证明项见[六项复核](../implementation/tasks/compute/six-point-architecture-review.md)。
旧代码、类名、测试和 factor 划分都不是本次重构的前提；设计先由真实消费与原生 runtime 决定，再盘点迁移。

## 1. 产品承诺

### 1.1 原始需求与架构取舍

HYLE 是 C++ 基础加速库，不经济地自写全平台算子大全。Compute 提供直接计算与原生执行，Graph
把逻辑计算编译为低运行成本的程序。成熟实现提供广度，HYLE 在确有完整收益或能力缺口处渐进替换。
目标是同语义真实工作负载中的 SOTA；库名、下载成功、原型正确和“使用模板”都不能证明已达到目标。

模型格式、tokenizer、KV/prefix-cache 策略、batching、sampling、模型循环和 serving 属于引擎。
通用 tensor、quantization、kernel、资源/同步、编译和执行属于基础库；渲染/工程计算也可直接使用。

### 1.2 先证明接得上，再扩大适配

每个 Provider 按 **family + backend/runtime + 版本 + device/格式/数值合同** 准入，不按库名整体准入。
必须有真实入口、资源/线程 owner、准备/运行分界、错误/回收和交付依据。公开 API 优先；确需私有
kernel 时建立窄、版本锁定、许可证与升级责任明确的源码 factor，不暗中依赖未导出符号。

不能恢复的运营失败、无法实现的内存/同步要求会否决该候选；不能通过改名或假装成功绕过。
原型只签所测机制，正式适配和同层性能另验。当前不恢复 Native FFI 或稳定插件 ABI。

### 1.3 四条独立路线与明确非目标

CPU、CUDA、Apple MLX/Metal、Vulkan 是四条独立计算路线。用户选部署路线，构建取得相应实现；
准备期绑定实际 device、格式、数值与有限 shape，执行不重新选设备或 Provider。

**已收口单路线基础阶段未包含模型CPU+GPU联合推理**。2026-10-09用户新增的
统一配置/联合计算要求进入[下一阶段设计](compute-next-stage.md)，尚未实现或资格化；
旧收口不代签分层offload、CPU FFN/专家分担。纯CPU仍是完整路线，纯GPU遇到缺算子/
容量不足时选择同路线合格候选或明确拒绝，不能静默改用CPU模型kernel。

host 控制/metadata、冷权重准备、上传下载、映射、同步和回收仍保留；它们不是模型联合计算。
同一 GPU 上组合多个 Provider 也在范围内。HIP/ROCm 无设备和维护条件，当前不实现；将来可独立接入，
不预造空 target/stub。多 GPU 的模型切分和分布式控制属于引擎，不由本轮四路线推导为已支持。

### 1.4 架构速读

```text
独立推理引擎：模型/权重、KV/批处理、生成循环
  ├─ 单个计算 → Compute 准备/执行
  └─ 算子 DAG → Graph 编译 → 路线专属执行计划
两条路径都绑定所选路线的原生执行：CPU / CUDA / Apple MLX-Metal / Vulkan
实际计算由已验证的 HYLE 或第三方实现提供，Provider 负责接入。
```

算子描述“算什么”，属于 Compute 的公共语义，不是另一个调度产品；Graph 描述依赖并编译计划，
单个计算无需它。Compute 提供资源、准备、提交和完成能力。路线区分原生执行体系，Provider 区分
具体实现；同一路线可有多个 Provider，同一个库也可分别适配多个路线。它们是职责关系，不是每个
kernel 必须经过的一串虚调用。普通用户选路线和资源约束，HYLE 应用下文已验证的默认优选规则。

## 2. 两条用户路径与渐进替换

直接用户选择计算/原生 kernel、输入与路线，显式准备后执行；不依赖 HYLE Graph。
图用户提供模型无关 DAG，由 Graph 编译后重复调用。两者共享数值、资源和完成语义，不必共享执行器。
普通 C++ 用户不手工拼 target/storage/lane ID 数组；专家控制通过确有意义的显式选项展开。

Provider 是具体实现的适配层：准备时翻译计算，执行时提供已绑定的原生制品；不是另一套公共
`GgmlMatMul/CublasMatMul` 计算语言。三种接入均保留：

1. **原生 leaf/融合程序**：HYLE、vendor primitive 或受审源码 kernel 使用所选 native runtime。
2. **完整 Provider region**：把选定片段交给 ggml、MLX 等自己的计算图/执行制品。
3. **Provider 内扩展**：若它公开 custom Primitive/lowering，在其编译前插入 HYLE kernel，让它保留
   内部资源和依赖。不修改已编译黑盒，也不假定每个 Provider 都有这种扩展。

一个 HYLE kernel 更快，不意味着拆开成熟 region 后仍更快。开发时比较原版 region、包含 HYLE 扩展
的 region、独立 HYLE 程序的完整边界；layout/repack、copy、同步、融合损失和运行时开销一起计费。
无可分辨优势使用合格成熟路线，不为裁决微小点值改变公共 API。

### 2.4 HYLE 拥有计划与边界，Provider 拥有片段内执行

| 决策 | owner |
|---|---|
| 模型配置、请求/batch、KV 策略、允许的计算路线 | 引擎/调用方 |
| 逻辑图、region 边界、边界数据与依赖、并发约束、完整路线选择 | HYLE Graph |
| region 内部融合、kernel、私有 layout/allocator、native 提交 | 入选 Provider/路线实现 |
| 真正的线程与设备执行、其它进程争用 | OS/驱动 |

HYLE 可以委托整个图、block、融合片段或单 op；Provider 不得越过选定边界改变放置或资源访问。
不要求每个 kernel 回到 HYLE host 调度器。一个原生程序可只有一次外层固定调用，必要的跨片段协调
才保留。CPU worker、CUDA stream、MLX encoder、Vulkan command buffer 不模拟成同一种机器。

## 3. 公共事实与原生差异

Backend 是设备/内存/执行能力，Provider 是实际计算实现；二者不一一对应。完整对应关系、原生限制
和参考来源只在[四路线设计](compute-routes.md)维护，不把一个 family 的资格传播给全部 backend。

execution target、storage domain、execution lane 分别描述谁算、数据在哪、沿哪条时间线执行。
同 GPU 可有多个 lane/多种 storage；共享物理内存不等于已经同步，host storage 不赋予 CPU 计算资格。
这些是必须分清的事实，不强制成为普通用户的三套公开容器或固定对象布局。

### 3.2 Provider 的两种诚实运行模式

- **shared runtime**：真正借入所选资源/Queue/executor 来执行。
- **delegated runtime**：Provider 自己持有 allocator、线程、stream/context 或图；通过明确输入依赖、
  结果和 Completion 接入，不能接收一个 HYLE Queue 参数后实际忽略它。

直接 Compute 也能运行独立 runtime 的计算，不必为一个 MLX 算子依赖 HYLE Graph。共享 Queue 录入
入口与绑定独立提交 owner 的入口分别兑现真实合同，模式在准备时确定；不把区别塞进逐 op 热分支。
Provider runtime/单 op lowering 不反向依赖 Graph，Graph region 翻译放在对应组合层。

### 3.3 准备先验证资源和数值

候选在 prepare 前说明 ports/shape/layout/precision、resource/import、alias、线程/同步、输出 owner、
workspace/常量 packing、private memory、并发与失败限制。选定候选才创建重型制品；不能到接纳用户
工作后才发现格式或 storage 不兼容。实际运营错误不能伪装成可回退的能力拒绝。

prepare 可进行已选实现所需的 JIT、启发式 kernel 配置、constant packing 和受控 warmup；这些不放在
首 token 内。这不授权跨 Provider 试跑选优；第三方内部若有计时式 tactic 搜索，必须单独披露并在该
路线准入时裁决，不能仅因它发生在 prepare 就默认允许。
warmup 可能真执行，必须使用隔离输入/scratch/可变状态，不能推进用户 KV、RNG 或其它副作用。
无法隔离的计算不自动 warm/capture。纯状态实例化不得偷偷执行、重新搜索或 JIT。

### 3.4 Replay 成本合同独立于 runtime ownership

| 合同 | 重复执行允许的工作 |
|---|---|
| prepared | 参数更新、已准备调用/命令；HYLE 与 Provider 不建图、heap allocation、首次 JIT/packing 或 tactic 搜索 |
| managed | HYLE 计划固定；Provider 可按声明实例化执行节点、分配 metadata/结果/临时 storage、查询内部 cache |

managed 不豁免数值、内存、失败或完成合同，也不隐含未声明的首次编译/copy/sync。默认可选择完整成本
合格的两类；专家可要求 prepared-only，未知/不满足者拒绝。该要求不应让所有 Apple 用户默认丢失 MLX。
shared/delegated 不能推导 prepared/managed，`compile` 函数名也不能证明零分配。

### 3.5 共同合同与四套可独立特化的实现

共用必要的逻辑描述、错误/生命周期与边界信息；四条路线各自实现 compiler/lowering、Program、
run-state 和回收。这里是职责名，不固定旧类、vtable、候选搜索器、segment loop 或 CMake factor。
只复用同义且成本合适的实现；旧 API 和旧测试没有兼容要求。

- **构建期**：裁剪路线及第三方 backend、SDK、flags、runtime 依赖；不靠消费者宏改变公共对象布局。
- **准备期**：按已验证规则匹配实际 device、数值/格式/shape，绑定实现、资源和调用入口，不现场比较性能。
- **执行期**：调用固定程序，更新允许变化的数据。无 Provider 查表、设备查询、策略选择、隐藏建图。

静态可知的局部调用用模板/直接内联；运行时才知道的模型不能靠宏变为编译期常量。固定边界调用也
有成本，必须以同 Provider direct 对照证明，而非宣布字面“零指令”。完整 region 不应逐 kernel 经过
多层 HYLE 类型擦除。调试、profile 与 route registry 不进入未启用诊断的热路径。

### 3.6 输出 binding 与生成结果的 owner

预绑定输出适合 caller buffer 与固定地址程序；Provider 生成的输出则由真实 runtime owner 持有，
发布带所有权的结果。两者都支持，不强迫 MLX 输出再复制到 HYLE 影子 allocation。

逻辑 descriptor/residence 在准备时确定，实际 payload 可在运行时物化。固定 run-slot 可预留句柄
位置，managed 的实际 allocation 单独计费。Provider 内部中间值不强制导出；跨 region 只有真实
import/依赖能力成立才借用，否则显式 Copy/转换或拒绝。caller 指定固定目的地址时同样计入必要 Copy。

句柄可取得不等于结果成功就绪。host 读取等成功终态，设备消费用真实依赖；结果 owner 保留到异步
工作和全部合法借用结束。仍有借用时不得覆盖结果槽；要跨槽复用保留结果，显式保留/转移 owning result
或用另一个槽。失败不发布成功结果，取消不提前归还 allocation。

真实 allocation 可复用 Mem 的 typed resource/`Buffer::adopt`；未物化 array 或可变 wrapper 不冒充
稳定 allocation token。Graph 不为 Provider-owned payload 重复规划 backing；capture/严格模式须另证
地址稳定、导出同步和预算。不由这一机制引入任意数据依赖 output shape。

## 4. 路线与提供方选择

### 4.1 开发时验证 best-of，准备时应用规则

best-of 是 HYLE 维护者的研发责任，不是交给用户或其运行进程做的性能实验。默认路径固定为：

- **开发时**：按硬件/ISA、Provider 版本、数值/格式/shape、布局、资源和并发条件验证完整收益，形成
  有适用域与成熟基线的实现/融合规则；不把单个 benchmark 点外推到全部设备或尺寸。
- **交付时**：规则与对应实现一起编译、版本化和验证。规则可用模板、直接分支或私有有界决策表，
  不要求新增通用规则引擎、外部 profile 数据库或公共配置协议。
- **准备时**：按实际计算和设备匹配既定规则，检查合法性并绑定程序。静态已知的选择可编译期消除；
  模型加载后才知道的 shape 只匹配一次，不通过试跑重新确定赢家。
- **执行时**：调用绑定程序或绑定好的有限变体；不重新探测 ISA、遍历 Provider、测量或改变优选规则。

普通用户只提供计算、路线/设备和所需数值/资源约束，不排序 Provider、不提供 timing profile；指定
实现或裁剪依赖只是专家控制。Direct 与 Graph 都须使用对应粒度的默认规则，不能只让专家样例选到
胜出实现。Graph 的融合规则计入完整片段成本，不把逐 op 最快名单机械拼成整图最优。

规则先服从能力/数值/资源硬约束；已选实现缺席或条件不匹配时，只能用预定的合法基线或明确拒绝，
不能临时测量、隐藏转换/CPU fallback，或把运营错误当安全回退。未知硬件上的兼容基线不获 SOTA
资格；依赖升级影响的规则需定向复验。原始性能证据留给维护者，普通运行不加载 benchmark 资料。
本轮不新增用户侧计时 autotuner；profile 驱动的多计划竞赛不是默认能力或必须保留的旧实现。

#### 模型与软件身份也可决定计算方法

固定规则不限于单算子或 shape：允许匹配调用方显式提供的模型家族、模型 revision/配置身份、
prefill/decode 与片段语义，结合实际 GPU 型号/SM/架构、驱动、Toolkit/runtime、OS、Provider
版本及数值/量化/布局域，选择模型专用算法、融合片段或完整 region。模型信息来自 consumer，
不使 HYLE 承担 checkpoint 加载、模型脚本、采样或服务。没有模型注解仍可使用自然通用入口。

模型名称只是适用条件之一，不能代替实际图、数值、状态、资源与完成合同的合法性检查。
不同配置不得仅凭同名模型命中专用规则；未知/缺失条件使用已声明的合格通用基线或明确拒绝。
只有影响绑定合法性或已验证计算方法的事实进入规则与计划缓存身份，不为无关标签增加热路径成本。
模型配置、实现/规则版本或相关驱动/工具链条件变化使受影响计划失效，重新准备后才能运行。
规则在开发阶段验证，准备时匹配绑定，执行不重新查询模型/驱动、搜索或计时。

此项是架构要求，实际可用性以当前交付记录为准；实现须同时验证专用命中、未知模型/版本回退、
配置不匹配与缓存失效，不能只加模型字符串字段就宣称支持模型优化。

### 4.2 四路线的实现来源

Apple 与 NVIDIA 是主要优化目标；CPU 是完整基线路线；Vulkan 是正式跨厂商保底，不是实验。
Apple 上 MoltenVK 是 Vulkan portability，不替代 MLX/native Metal 性能资格。

- CPU 保留 HYLE SIMD/Native 的优势，按实际 ISA/shape 与 oneDNN/BLAS/ggml 等合格实现择优。
- CUDA 的普通 GEMM 优先资格化 cuBLASLt/cuBLAS，量化/attention/融合则按格式、硬件与完整成本
  选择 vendor、成熟 kernel 或 HYLE；“CUDA 默认”不是从零手写全部 CUDA 算子。
- Apple 的 MLX 已通过 Metal 使用 GPU；先保留其高效完整片段。HYLE Metal kernel 优先通过公开
  扩展进入相同 runtime，独立 native Metal 与 ggml-Metal 另作候选，不强制混用。
- Vulkan 使用真正的 SPIR-V/pipeline/资源同步，也可借 ggml-Vulkan 完整片段或受审源码 kernel；
  不模拟 CUDA stream capture。具体硬件能力/限制见[路线设计](compute-routes.md)。

### CUDA Q4 MatMul 的规范例

1. 用完整量化合同描述逻辑计算：activation、压缩 weights 和格式需要的 scales、zero-points/biases/
   offsets，明确 group、轴、舍入和累加。普通两输入 MatMul 不冒充所有量化格式。
2. 在选定 CUDA 路线内验证能力。GGUF Q4_0/Q4_K、Marlin W4A16、FP4 是不同合同；相同 4 bit 不保证
   可无损 repack。cuBLAS 也不自动支持任意 Q4 存储。
3. 匹配研发时已验证、且当前实际交付的实现规则：原生 kernel/vendor primitive、ggml-CUDA region
   或 HYLE 融合；不是在用户机器上逐个试跑候选。
   量化带来的额外 activation 舍入和 reduction 差异不能隐藏；严格数值策略不满足就拒绝。
4. 兼容的 packed constants 冷生成并按实际 owner 共享；output/scratch/locks 按并发槽独立。
   默认压缩权重常驻，kernel 内 tile 解码不是全权重浮点常驻。显式展开/再量化另需授权与完整内存计费。
5. 研发时把转换、融合、copy/sync/launch、准备和内存成本计入完整比较，并固化适用规则。模型准备
   时按这些规则构造实际计划；局部 kernel 赢但拆图后输的替换不进入默认规则，同-runtime 扩展也按
   完整收益验证。不是必须把 ggml kernel 拆成函数才能使用。
6. 绑定有限程序和运行参数后重复执行。没有合法 Q4 carrier 就明确不支持，不偷偷回 CPU 或全量展开。

## 5. Storage、同步与生命周期

Mem 提供 allocation/views/mapping/quota/lifetime primitive；Compute 解释 native resource、执行与
异步使用；Graph 提供 liveness/alias facts。metadata allocator、payload、第三方 heap、driver 内存
各有真实 owner，不混成一个“Context 内存”数值。释放使用原来的 owner/size/alignment。

不可变代码/权重可共享；可变 handle、scratch、命令/descriptor pool 和运行状态按 native 并发要求独占
或池化。只用于 host 编码的临时 metadata 可按合同提前释放；设备仍引用的数据、参数与状态必须留到
真实终态。资源和代码不能因 C++ wrapper 析构而提前释放或卸载。

接纳前可事务化拒绝；接纳后报错/drain，不在另一个 Provider 重跑。一个提交前缀的终态不得丢失；
事件记录失败也不等于没有接纳 GPU 工作。取消是请求停止，不是立即可回收，GPU 通常仍需排空。
关闭提供显式 drain，析构不隐藏无界等待或释放尚在使用的 native state。

同 lane/跨 lane 按原生语义建立依赖：CPU happens-before、CUDA event、Metal hazard/fence/event、
Vulkan execution + memory dependency 不是同一个实现。必要时显式且计费的 host wait，不隐含每 op
round-trip。普通 host、pinned、device-local、unified/shared 的可达性、coherence 与寿命分别验证。

## 6. 算子与 C++ 扩展

能力由 ggml/llama.cpp、SGLang、vLLM、ds4 的真实消费和独立成熟库共同驱动：tensor copy/layout、
cast/quantization、unary/binary/reduction、GEMV/GEMM、normalization/activation、embedding/RoPE、
attention 及 MoE 所需的通用计算。模型层和策略不下沉；当前阶段先完成基础运行机制，不填全目录。

typed custom schema 与实际 execution 是两种能力，都须向 source/static/shared C++ 用户开放。
显式、对象局部、owning 的实现配置声明 ports/数值/layout/alias/resource/effect/失败与 workspace，
在冷路径生成执行物；配置、native 资源与代码保活至终态。不要求用户包含 `src/` 私有头、手写第二套
Graph，或使用进程全局注册和字符串属性袋；跨 DSO identity 不使用静态对象地址。

### 6.1 算子接入按有限批次接线和验证

算子/region 接入的默认施工单位是由真实模型权重角色、格式、shape、路线和成熟 Provider 共同界定的
**有限批次**，不是单个算子。先固定该批需要的全部语义端口、Provider 原生片段、Direct/Graph
消费者和未覆盖格；再一次审计涉及的 owner，固定有限 FIX 与停止条件。优先接通成熟 Provider 的
完整 region，同一批中的缺口确定后才讨论 HYLE 数学。

批内先完成各路线的语义、资源、冷绑定、失败和重复执行接线；不得每接一个 op 就重新配置、构建、
运行完整 selector 或 P1。批末冻结不可变候选，按变更影响一次构建并运行适用 owner/consumer、
三交付和性能门。实际失败只定位并修正受影响的边界，再重跑受影响 gate；不通过重复全量验证求绿。
同一成熟 Provider 的同类 carrier 可以跨模型/profile 共用一组参数化接线和 oracle；批次按共享接缝
及可承担的机器资源定界，不要求每个模型、权重角色或算子各自产生一个构建候选。接线时允许静态
检查和能定位接口错误的最小运行检查，完整 owner suite、selector、三交付及 P1 留到批末集中执行。
真实模型覆盖优先按可复用的 Provider 接缝与权重角色横向组批，再把所涉及的模型/profile 列入同一
参数表；不为每个算子、region 或模型画像重复建立全量构建树和验收收据。一个批次应在审计时列清
覆盖格与结束条件：接线完成或得到精确拒绝后冻结候选，同一候选/路线的适用门只集中执行一次。
批次不能因等待验证而无限扩大；发现共享语义、资源或失败边界错误时，立即做有界定向检查和修复。
一个模型 profile 内的若干 region 或 MoE 子链只是批内接线检查点：允许有界定向测试尽早发现
真实接缝缺陷，但不因此创建新的全量构建/selector/P1 候选。能签署完整 profile/route/entry
验收格或明确独立路线能力时，才到该批的完整验证边界；检查点只产生 partial receipt。
未编译、未运行的接线始终是待验，不得写成 Provider 已可达或模型格 `RUN_PASS`。批次若遇到
具体上游拒绝或需要新数学的缺口，记录精确边界并按当前卡停止/重定界，不把批次无限扩张。

## 7. 构建和底层依赖

Compute 不依赖 Graph；图组合依赖 Compute。公共语义核不依赖 CUDA、Metal、Vulkan 或任何全套
Provider runtime。每条路线及每个不同 native runtime 的 adapter 独立拥有依赖/编译/安装资格。
Apple SDK 不探测 CUDA，也不因选择 Compute/Graph 固定拉入 HYLE CPU 算子和 oneDNN。

普通用户仍选 `hyle::compute/graph` 与真实 backend feature；内部 factor 只在依赖/维护确有必要时
拆。SDK 发现、第三方 backend 裁剪、nested FetchContent、flags/头/符号隔离遵守
[build system](build-system.md)，不继承旧 façade 的固定全因子列表。

底层使用 Error、Mem、Platform、SIMD、Sched 等已有正确能力；CPU 持久 executor 若已有自然公开组合，
不为架构图再造强制 bridge factor。只有具体消费缺陷才重开底层 owner，不在 Compute 复制同义机制。

## 8. 性能与验收边界

先审算法与成本结构，再测固定候选。分开证明：同 Provider/direct 的包装税；同精度/形状/资源/并发/
完成边界下完整片段的竞争力。decode/prefill 不混同，cold、host submit、最终完成、内存分别记录。
固定小输不能改名为误差；没有合格证据不声称胜出，无收益时保留成熟默认。

每条路线分别证明构建隔离、正确性、资源/线程寿命、部分接纳/晚到错误/关闭、实际组合、三交付和
完整性能。不能从 CPU 成功推出 GPU 成功，从 M4 推出 M5 矩阵加速，或从 NVIDIA Vulkan 推出 AMD/
MoltenVK 资格。有限研究、基础实现和 SOTA 验收是不同完成点；当前缺口只记在研究/活动任务，不在
架构页复制进度表。新增 op/Provider 能进入既定合同就局部演进，真实反例推翻共同语义时才重开架构。
