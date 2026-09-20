# M1-C：ggml CUDA 与 FlashInfer 原生接入

## Closure card

- owner：`/root`；baseline：`0f54dad4fd95714ef3a03f455f04dc746edccaea`，Windows main。
- 当前阶段：AUDITED；三个新身份已在`77caf875`登记QUEUED，六owner在`2673c5eb`完成SCOPED。
  本轮只授权下列C1–C6资源与成熟实现接入，不授权自研数学。
- 新增身份：`compute_provider_ggml_cuda`、`graph_provider_ggml_cuda`、`compute_provider_flashinfer`。
  分别拥有 Direct/native program、Graph region 接线与原版 attention 接入，不建立空 target。
- 既有 owner：`compute_ggml_runtime`的 CUDA vendor/资源补丁投影，`compute_route_cuda`与
  `graph_route_cuda`的普通工厂组装；按本卡新需求重定界。原生 backend 借用/错误机制先复用，
  只有具体缺口才将 `compute_backend_cuda`纳入，不预先重审整个 CUDA runtime。

上述三个既有owner按本卡明确的新后端/普通入口需求重开，原terminal record保留为已有范围的证据；
此动作不重审其旧数学。审计阶段仍没有新增target、依赖或公开头，不计接入完成。
- trigger：[M0清单](llm-moe-provider-integration-ledger.md)的 M1-C/Q-CUDA-API；现有 CUDA
  普通入口没有 ggml region 和 FlashInfer，ggml 公开 API 缺少外来 device buffer/stream 接缝。
- 先决：[M1-G](llm-moe-m1-ggml-runtime.md)与[M1-R](llm-moe-m1-route-binding.md)均 CLOSED。
  不恢复旧 C1，不修改旧 ggml CPU 或 oneDNN 的 BLOCKED 条件。
- 主纵切：本地 Qwen3.5-2B 所需 head256 prefill/decode；同 GPU 上原版 ggml 片段与成熟
  GEMM/attention 组合，显式状态输入输出、两个 invocation 与旧结果保活。这里只验 seam，
  F01–F19 的完整批量映射与模型旅程仍归 M2/M3。
- 正交纵切：无模型标签的普通 F32 数值 region，重复执行、独立请求与可恢复准备/运行失败。
- scope：原版程序的构建、资源、错误、调用、默认入口与三交付；保留成熟 region 内部融合。
- non-goals：自研算子/shader 扩张、调查、优化、比较或删除；数学 kernel 改写；模型加载；
  多 GPU/模型级 CPU+GPU 执行；公共治理与性能仪器重写。
- tier：P0，签资源与接入成本结构；M2 承担成熟默认路径的适用 P1，M5 前不比较冻结自研。
- 新接入反例C7：现有公共SDPA推导只接受同head，拒绝主纵切8/2 GQA；独立
  [语义与隔离卡](llm-moe-m1-cuda-gqa-semantics.md)移交`compute_ops`及native冷准入检查，
  不研究/修改自研attention数学，不改变C1–C6原生资源设计。
- 预算：本 seam 源码/设计总2小时，包含 Q-CUDA-API，不另叠加。前序只读输入从
  2026-09-19 21:43 UTC开始，22:37–22:45续查；保守计入已耗用60分钟，余60分钟。
  至2026-09-20 00:08 UTC，续查保守追加25分钟，累计85分钟，余35分钟。编译/已声明验证
  时间不冒称研究；每根因最多两次纠正，范围变化重新审计且不重置预算。
- 机器：Windows/WSL SM89 原生与三交付，不支持 CUDA 的平台验 OFF；重型串行、parallel2。
  Windows DLL 邻置，无 UAC、电源策略修改或远端源码回灌。
- stop：有限接入与适用门完成后关闭；源文件存在、空 target、枚举、copy-only 不构成验收。

## 已锁来源与具体接口事实

沿[M0源码记录](../../../../evidence/correctness/compute-llm-moe-m0/source-map.json)：
ggml `456172ec733a135778adcd32d00e576a58232e45`（MIT）；FlashInfer
`69ff11fc4954396d98326656dc85debd2223f637`（Apache-2.0）。不升级 pin，不把 Python/Torch 带入 SDK。

1. ggml 的 `include/ggml-cuda.h`提供 backend/buffer type，没有外来 GPU buffer/stream import。
   buffer context 会释放自有指针，不能仅改 `tensor.data`冒充合法零拷贝。
2. `common.cuh`的 CUDA context 持有 stream、cuBLAS/workspace、pool 和 graph；
   `ggml-cuda.cu::ggml_backend_cuda_graph_compute`区分初次执行与稳定 capture。
   不能冷准备时提前执行用户可变状态，也不能把 managed 路径称为零查询 prepared replay。
3. `CUDA_CHECK`可调用 abort，context/pool/buffer/graph/event 析构同样检查 CUDA 错误。
   仅最外层 catch 不够。capture counter 在 BeginCapture 前增加，失败必须退还计数并结束
   已取得的 capture，否则清理可能永久等待。修正限资源/失败边界，不改数学与选择启发式。
4. HYLE 已有 primary-context 借用、native Status 翻译与 admitted Queue/Completion，也有独立
   delegated region 合同。先复用这些机制，四路线不因此共享一个执行器。
5. FlashInfer 原版 `SingleDecodeWithKVCacheDispatched`与`SinglePrefillWithKVCacheDispatched`
   接受显式 stream/tmp、返回 cudaError_t；派发仍含 capability/occupancy/attribute 工作。
   Windows/WSL 可编译性与准备/重复执行合同尚待资格化，header 存在不是 PASS。
6. HYLE 已有 `ScaledDotProductAttention`的 layout/mask/scale/causal/accumulation 描述，
   先核对真实数值与 causal 对齐合同，不再建立同义 attention API。

## 有限 FIX（本轮冻结）

| ID | 准入与承载 | 允许修改 | 验收 |
|---|---|---|---|
| C1 | BORROW 原版依赖 | 一个 ggml vendor 源按需启用 CUDA；FlashInfer 原版 C++ 入口；复用 lock/cache/package/license | CUDA OFF 不获取/探测；三交付从同份 target/资源描述导出 |
| C2 | BORROW 算法、HYLE 资源边界 | ggml保留自有stream，通过两个device event连接已准入HYLE Queue；primary context、非 owning buffer、native error、部分回滚与无抛析构桥 | 不误 free 外来指针，不 abort 运营失败，capture 失败不遗留计数/活资源，不假报完成 |
| C3 | BORROW 原版 region | 每 invocation 保留 ggml program；Direct/Graph 共用必要 schema/资源映射，Graph 类型不下沉 Compute | 真实 norm/MatMul/状态片段、重复执行、两个请求、旧结果与放弃 completion；如实声明 managed/prepared 成本 |
| C4 | BORROW 原版 attention | WSL/Linux采用已资格化FlashInfer原版入口；Windows本pin原生接法失格，采用ggml原版attention；绑定既有公共描述 | 两平台均须完成head256 prefill/decode；布局、bottom-right causal、scale、FP32累加与workspace/stream；不同Provider不降低相同数值需求 |
| C5 | HYLE 冷规则与普通入口 | 组装已资格化成熟实现；使用 M1-R 模型/环境事实，保留真实图边界 | 未知条件不误命中，消费者不排序 Provider/提供 timing；同 GPU 正确完成与失败回收 |
| C6 | 交付与生命周期 | 适用测试、三交付、指南与有限证据归档 | 普通 Direct/Graph 实际执行，producer 隐藏与 prefix 搬移，DLL 邻置；未运行项不写 PASS |

## 能力、公开形状与资源合同

| 能力族 | 层级与当前覆盖 | 本轮承载与有限裁决 |
|---|---|---|
| CUDA原版region | 复合功能，MISSING | BORROW ggml完整backend graph；Compute拥有不含Graph类型的私有program/schema，Graph只翻译region边界；不拆上游内部融合/stream调度 |
| head256 GQA | 复合功能，PARTIAL（仅入口资格） | BORROW FlashInfer Linux与ggml Windows；复用`ScaledDotProductAttention`，mask/类型/layout未资格化的域明确decline |
| 原生buffer与完成 | 原子组合，PARTIAL | KEEP backend primary-context/resource/QueueSequence/Completion；FIX ggml的非owning wrapper与原生stream桥 |
| 普通Direct/Graph | 复合功能，PARTIAL | FIX现有`compute::cuda::create_context`与`graph::cuda::{create_context,compile}`的成熟Provider组合；调用方不排序库 |
| 模型/硬件规则 | 冷策略，ROUTED | KEEP M1-R `WorkloadContext`/CUDA Environment；绑定后不计时、竞速或搜索赢家，未知条件不冒用模型特例 |
| 全F01–F19映射/模型读取 | M2与应用边界，OUT_OF_SCOPE | DEFER批量lowering到M2；模型读取始终不进入SDK，不能因本卡的seam样例签64域 |

公开增量限三个expert工厂：`compute::ggml_cuda::provider()`、
`compute::flashinfer::provider()`与`graph::ggml_cuda::provider()`，头不暴露上游或CUDA SDK类型。
Graph工厂为同一计算实现附加现有region extension；Compute不依赖Graph。普通用户仍按现有
CUDA入口创建Context、prepare/compile、instantiate、submit和等待真实Completion。

- 每个ggml invocation独立持有backend、metadata graph、临时device buffer、cuBLAS/pool/cache
  与两个边界event；不可变prepared program只保存描述与冷绑定规则。不同invocation可并发；
  同一invocation按既有运行状态机串行，调用方不能提前释放借用的payload/allocator。
- ggml继续拥有自己的root与内部stream。submit在已准入HYLE Queue上record入口event，原生
  root等待后执行完整region，出口event回接HYLE stream，由既有Completion代表完整完成。
  不改cuBLAS的stream所有权，不另造Completion线程/事件循环；这两个device event成本计入接入。
- 外来device tensor使用明确非owning buffer wrapper，尺寸/对齐/provenance在HYLE边界验证；
  wrapper在invocation准备时建立，submit只重新绑定已验证地址，不释放输入/输出的外来指针。
  Graph内部值按liveness由invocation拥有；旧结果遵守既有result owner寿命。
- CUDA/cuBLAS/Driver运营错误保留原生数值，在adapter边界转Status。改写限失败/资源路径：
  部分构造RAII回滚、析构不抛、capture RAII平衡计数/EndCapture、失败时收束所有已接受原生
  stream。不能只catch最外层异常，也不能把原生错误化成成功的空Completion。
- 一旦运行已接受后失败，本invocation失效并保留到quiescence，后续submit明确失败；可从
  不可变program创建新invocation。放弃Completion仍走既有保活/回收责任，close不得提前free。
- ggml签`kSharedQueue/kManaged`、provider workspace、不可外部capture，保守声明可能阻塞host。
  原版首轮pool/handle分配、graph属性检查、稳定后的capture与cache维护如实记账；不在prepare
  偷跑用户状态来制造prepared资格。动态值必要转换计入完整region，常量不能热路径重打包。
- FlashInfer直接使用已准入HYLE stream、每invocation显式workspace与原版固定模板入口；签
  managed并披露原版capability/occupancy/attribute查询，不冒称零查询prepared。成功路径无
  runtime编译或计时。现有bottom-right causal合同见`docs/user/compute.md`；FP16 QK reduction
  不启用，BF16/FP16入口各自资格化。workspace大小/溢出冷验证。
- CUDA-OFF时两个新增Compute factor与Graph factor均absent，不获取依赖/探测SDK；FlashInfer
  本pin仅Linux CUDA准入，Windows不生成失格target。ggml启用CUDA仍复用一个vendor源，不带入
  CPU模型fallback。source/static/shared从同一target/资源/许可证声明投影。

## Q-CUDA-API资格裁决

[原始记录](../../../../evidence/correctness/compute-llm-moe-m1-cuda-admission/README.md)固定源码、
探针与全部失败。WSL CUDA13.3/GCC14.2、RTX4070 SM89实际运行原版header入口：head256，
Q/KV heads=8/2，prefill q=kv=17、decode q=1/kv=257，FP16/BF16各3次，独立double oracle通过。
每项单独持有stream/workspace；这不是两个同时在途请求、产品生命周期或三交付证明。

Windows MSVC14.50/CUDA13.3先因传统预处理器失败，使用标准预处理器后暴露GNU属性、
LLP64类型推导与`ushort`别名；有限等价兼容诊断后又暴露NVCC生成的host模板中裸`restrict`。
同根因两次纠正均未通过（改源限定符仍输出裸restrict；host宏又污染SDK的declspec）。
到此停止该Windows移植，不再扩编译补丁、不改数学、不将失格写成PASS。Windows本轮明确由
ggml成熟实现承载attention，仍须做相同head/类型/数值/资源验收；WSL入口资格不代签该结果。

## 七问的审计结论（待实现验证）

1. 版本/许可证按上方pins；ggml窄资源补丁独立锁定、build-local生成，FlashInfer Linux用原版
   header，诊断用Windows修改不进入产品。失格平台已有具名成熟接法，未证明域不fallback自研。
2. 主head256与正交F32纵切准入C1–C6；同GPU组合、状态与两请求在本卡验证，全模型覆盖仍在M2/M3。
3. 复用已审Compute/backend/error/mem与MetadataArena；不重开backend，不复制Queue/Completion。
4. P0分别列冷资源、每invocation存储与边界event、原生managed内部成本；无P1/冻结自研比较。
5. 仅自然C++三交付，vendor C入口私有，不形成HYLE C/FFI产品。
6. CUDA vendor与runtime库、FlashInfer header编译目标的锁/许可证/安装由各owner一处描述；
   三交付和OFF只有实际运行后才能判PASS。
7. 普通入口继续按路线创建；expert只表达显式组合。managed成本、借用payload和失败后重建
   invocation写入guide与公开工厂注释，不让用户拼第二套运行生命周期。

## 预声明验证与边界

- QUEUED/SCOPED/AUDITED 元数据提交只验 docs、closure（含中央 CMake 登记对账）与 diff。
  selector 因中央图文件保守列出的产品门保留计划，在实际产品候选固定后执行；阶段检查不是
  selector 全量 PASS，不把未运行 native/交付门写成通过，不在每次状态登记时重建全库。
- 先做有限 Q-CUDA-API 原版 C++ 资格 bundle；探针只签入口事实，不计 Provider 接入/模型覆盖。
  不靠复写 kernel、降低数值要求或改用更小 head 绕过失败。
- 真实 ggml region 与 FlashInfer prefill/decode，普通入口核对数值、状态、多 invocation、
  外来 buffer/stream 销毁顺序、prepare OOM、部分 enqueue/capture 失败与晚到完成。
- Windows/WSL 适用 native、host owner sanitizer 与三交付；CPU 插桩不代签 GPU driver。
  Mac 只签 CUDA-OFF。按影响复用未变消费者收据，不重跑已通过的全库门。
- SDK consumer 独立配置，隐藏 producer、搬移 prefix 后执行；可复用身份/flags 匹配的编译
  产物，避免反复冷编同一第三方库。完成后归档并清理本卡产物，保留缓存与本地模型。
- selector、docs/closure、统一 Compute+Graph LOC 按既有协议执行。本卡不解除冻结，也不签
  64域×四路线覆盖；下一层批量算子/片段接入仍由 M2 有限卡承载。

## 实施检查点（2026-09-20，仍为AUDITED）

FlashInfer原版已在`67f3651f`接入Linux普通CUDA Direct/Graph；`6ef8884d`仅修正nvcc生成
host实例的共享库可见性。原五项Provider测试、普通入口及source/static/shared已实际通过，
逐分配OOM与第二batch故障验证原生错误、accepted prefix保活和失败槽退役。
ASan/UBSan及环境修正边界见[接入收据](../../../../evidence/correctness/compute-llm-moe-m1-cuda-admission/README.md)。
无P1或冻结自研比较；新增签署模型覆盖格仍为0，不把本检查点当成整个C1–C6完成。

C2的ggml CUDA资源补丁正在实现/编译：原版stream、pool、fusion和capture由ggml保有，
HYLE只补外来buffer、原生错误、部分构造与capture失败回收。独立ggml默认关闭CUDA graph，
本路线显式启用其原版capture并关闭不在范围内的NCCL；首次未启用graph的构建在执行前停止，
原日志保留，不记验证通过。C3完整region、C4 Windows attention、其余平台与收口门尚未完成。

新增vendor目录的selector归属与新Provider平台事实由
[独立映射卡](../build/ggml-cuda-selector-owner-v1.md)登记并修正复现的共享CUDA开关冲突；不改门命令、授权或P1规则。
本检查点源码/设计保守追加15分钟，累计100分钟，余20分钟；编译/预声明验证不计成研究。

### ggml F32接线检查点（2026-09-20，仍为AUDITED）

C2原版资源纵切已在Windows/WSL实际运行；WSL还注入capture开始、encode与结束错误，验证
原生数值及不遗留capture计数。C3/C5实现复用同一Compute程序、每invocation资源和边界event，
Graph只附加region扩展，普通路线名单只保留一处组装。首域为dense rank-2 F32无参数RMSNorm
与同形Add；MatMul仍由已有cuBLAS承担同GPU串接，不新增数学kernel。

普通Direct、Graph完整片段、两个请求/重复执行/旧结果、失败槽退役和HYLE metadata逐分配
OOM已进入本轮测试。旧Direct测试写死四Provider的断言按当前平台名单迁移；不改其数学或
数值门。Graph提案按数据流合法收缩，不因无关cuBLAS节点的插入位置拆碎ggml片段。
该有限接法当前只提案整体合格集合；集合不能合法收缩时沿用单算子成熟路径，进一步片段
覆盖由M2批量lowering承载，不在此重写Graph planner。

本段续查/设计追加12分钟，累计112分钟，余8分钟；实现、编译与预声明验证单列，不重置预算。
C4 ggml Windows attention及剩余插桩/三交付/OFF仍待完成，新增签署模型覆盖格仍为0。

上述F32接线最终Windows/WSL各8项native/header门通过，原始失败与最终结果见
[检查点收据](../../../../evidence/correctness/compute-llm-moe-m1-cuda-admission/ggml-f32-admission.json)。
统一脚本统计Compute+Graph实现51,012行、217文件，不含测试；本记录不关闭M1-C。
