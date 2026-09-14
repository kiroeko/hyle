# Compute v10：owning custom execution

## Closure card

- component / owner：`compute`；baseline `a82ae7dc3facc4f8e1170d9b4c64a489cd48cbf0`。
- 承接未 terminal 的 [v9](compute-v9-shape-replay-substrate.md)：C90--C92/C94 的已实现结构保留，
  **C93 完整 qualified profile identity/匹配仍在本轮范围内，未被关闭或删掉**。
- trigger：[整体架构复核 A3](compute-graph-route-unification-audit.md#整体架构复核与合同修正)已经接受：
  只有 `Operation::custom` schema、私有 ProviderAccess 和一次性 native submit，安装用户不能把自己的
  owning kernel 放入同一 PreparedOperation/Graph 生命周期。
- 主纵切：推理用户提供未进 catalog 的 CPU/CUDA kernel，与成熟 Provider 共用 HYLE Graph；正交纵切：
  外部缓冲区变换和具名副作用资源，不把模型/KV/batch policy 放入 Compute。
- 范围：C100--C103 及继承的 C93；不增加算子、不新增 public Plan/factor/FFI/global registry，不改厂商内核。
- 依赖：复用 ComputeOps schema、Mem allocator/resource、Error、私有 Util-core RefCounted，以及已有
  Context→PreparedOperation→Invocation→Completion 保活链。Graph/Native/oneDNN 是直接回归消费者。
- 验证：Windows source-tree shared/static、逐头、direct/Graph 和 oneDNN 迁移回归；当前有限实现完成后
  用仅安装头/库的 CPU/CUDA 独立 consumer 验证生命周期、失败、并发与三交付。跨平台/sanitizer/基础 P1
  与最终 foundation candidate 一起执行，不能从 Windows 增量通过推导它们已通过。
- 性能：本次 P0，记录 owning 配置和 prepared/invocation 冷分配；内置 Provider 热回调、Queue admission、
  Completion 不增加分支/查询/分配。统一 P1 仍由基础卡在 immutable candidate 后一次启动。
- 预算：一次表示审计、一次实现纠正和终局独立红队；不修改公共仪器。改变 runtime owner、接纳后重跑、
  新全局注册或热态分配是停止条件；真实 backend seam 缺口先绑定其 owner，不能借本卡越权改 CLOSED 产品。

## 能力与形状审计

| 能力族 | 当前覆盖 / 动作 | 实现合同与有限 FIX |
|---|---|---|
| 描述计算语义 | COMPLETE / KEEP | OperationId/traits/infer/resource access 由 ComputeOps 唯一拥有 |
| 持有用户执行配置 | MISSING / C100 | `make_custom_provider(identity, backend, factory, allocator)` 持有可 move-only 的 factory，复制 identity 文本；失败不移动具名 factory |
| 冷查询与选中后编译 | MISSING / C101 | const factory 的 `propose` 返回结构事实与 route；`prepare` 返回 owning kernel 或明确 outcome，复用已选 resources，不在 propose 编译 |
| 不可变 kernel 与每次运行状态 | MISSING / C102 | `make_custom_kernel` 擦除 typed kernel；`Kernel::Invocation` 由冷 create 创建；hot enqueue 是一跳 typed trampoline，不新增虚函数层 |
| 普通用户完整旅程 | PARTIAL / C103 | 公开头、指南、direct/Graph、source/installed static/shared、CPU/CUDA；没有 src include 或手写 provider 函数表 |
| 完整 route profile identity | PARTIAL / C93 | 原 v9 的 build/device/domain/boundary 资格与匹配待实现，不用本次 owning identity 替代 |

这些是复合执行机制，HYLE 拥有生命周期适配，用户拥有算法；不把 kernel 算术移入 catalog。
用户只定义 C++ factory/kernel 类型，内部擦除表是模板实现细节，不公开 vendor context/region SPI。

### 表示与成本（实现前固定）

1. `Provider` 保持现有 inline payload 大小。增加冷 copy/move/destroy 行为，私有表可选 retain/release
   configuration hooks；内置 inline 配置不分配、不引用计数。custom factory 一块分配同时拥有配置、
   identity 文本与引用计数，Context 的 ProviderSlot 复制该 owner；销毁 Context 不早于全部执行持有者。
2. `KernelRequest/KernelPrepareRequest` 明确 Device、Operation、描述、精确资源和 opaque route；span
   仅在调用期借用。`ResourceRequirement` 公开既有冷约束的同一表示，内部使用它，不复制 matcher。
   本轮 custom factory 先声明 exact point；完整 bounded 资格仍由 G101/C93 的后续实现承担。
3. factory 的 propose/prepare 为 const、noexcept，可在多个 Context 并发调用；只允许冷准备，无执行
   tensor payload 或接纳外部工作。`KernelPreparation` 保留 prepared/unsupported/resource-decline/error
   四态，不从 Status code 猜安全回退。返回的部分 owning kernel 由 RAII 回收。
4. `CustomKernel` 是 move-only 的准备结果；调用方提供 workspace/runtime/persistent 事实，复用共同
   validator。typed kernel 及其独占 Invocation 按各自 allocator 创建/销毁，不用跨 DLL delete。
   prepare 以后 kernel 只读；可变 run state 在 Invocation，借用资源仍由消费者保活。
5. 本 factory 是 **shared HYLE Queue** 的 custom-kernel 路线；delegated runtime/region 的 vendor
   adapter 仍走现有 factor SPI，不把两种 owner 伪装成一个回调。`KernelCommandResult` 明确是否已接纳
   工作。enqueue 在已 admission 的 Queue 内记录，不再调用一次性 `submit_host/cuda::submit` 产生
   第二次 admission/Completion；使用 typed native handle 时遵守对应 backend 的 context/stream 合同。
6. 把已有 prepared callback 表移入安装的 detail erasure header，模板直接产出同一 callback 表；不是
   暴露整个 Provider SPI，也不加 warm adapter→virtual 的第二跳。不保证代码 DSO 自动保活：包含用户
   factory/kernel 类型代码的 image 必须存活至最后一个 Provider/执行持有者销毁。

自然旅程：定义 schema → owning factory → `Context::create` 配置 providers → 原有 `prepare` 或 Graph
compile → 每 lane 一个 Invocation → 重复 submit/Completion。没有第二套 Graph 或 custom 调度入口。

## 预声明反例与交付

- factory move-only、Provider copy/move/self-assignment、调用方原配置销毁、identity 字符串销毁；
- 无效 identity/backend、wrong device、配置/Context/kernel/Invocation 每次分配失败及安全恢复；
- exact 描述/resource/alias 错误，unsupported 与相同 Status code 的 fatal Error 不混淆；
- proposal/artifact runtime/workspace 承诺矛盾，partial kernel 销毁，成功以后不再 propose/prepare；
- kernel/Invocation 所有权、不同 Invocation 并发、接纳前拒绝与接纳后错误 Completion、调用方先销毁
  Provider/Context/PreparedOperation；CPU 实际计算和 Graph repeated replay，不只测计数；
- 仅 public 头的独立 consumer；shared factory/type 跨 image 构造并执行；CUDA native context/stream、
  多 lane、异步错误与保活必须真机验证，未运行保持未证明；
- docs/closure/previous-ref/diff 与 change-impact；不运行全库/P1，不升级外部库。

本卡已完成 scope 与上述表示审计，以继承的 `AUDITED` 活动阶段绑定新有限集合；不是 CLOSED 的重新
审查，也不重置/伪造 v9 的未完成项。七门与 foundation gate 均未签发。

## 2026-09-14 C100--C103：公开 owning execution 检查点

C100--C102 的上述表示已实现。`custom_kernel.h` 提供 typed factory/kernel，`custom_kernel.cc` 只负责冷
owner/结果适配；原 prepared callback 表移到安装的 detail 头，内置 Provider 与 custom 使用同一个
回调层。`ResourceRequirement` 公开既有约束表示，内部 matcher 复用它，不增加平行资源政策。

源码复核确认 Context 先销毁 provider context 再释放配置，PreparedOperation/Invocation/Completion
沿既有链保活；factory name/version 在移动 factory 前复制。Kernel 和 Invocation 各自保留或使用原
allocator 的释放边界。具名 factory/kernel 在无效参数或 allocation 失败时不被移动，部分结果按 RAII
清理；unsupported/resource-decline 与同码 fatal error 不混淆。

- 11 个 custom 单元用例覆盖 move-only、const copyable、over-alignment、复制/移动/自赋值、配置文本
  生命周期、全部冷分配失败与恢复、partial cleanup、错误分类、串行声明和独立 Invocation 并发。
  direct 与两节点 Graph 检查实际输出和重复重放不再 propose/prepare，不只检查回调次数。
- 公开 [独立消费工程](../../../../samples/package/graph/custom_kernel/README.md) 不包含 `src/`。CPU
  factory/type 位于单独 shared library；CUDA native kernel 位于独立 CUDA library，普通 C++23 factory
  使用 HYLE Queue 的原生 stream。CUDA 覆盖 257 元素尾块、双 Queue、两节点 exact capture/replay、配置
  保活，以及真实 launch 后注入错误的 Completion 和实际输出。
- 示例的 scale/bias 属于 Operation 数值语义，factory 只持有不改变结果的分块配置；CPU `std::fma` 与
  CUDA `__fmaf_rn` 同为一次舍入。静态 CPU consumer 曾因 factory 要求另一 image 的精确
  `system_resource()` 而失败，已按算法实际需求改为 host-accessible/alignment；未放宽产品 matcher。
- 串行 artifact 的 direct 用法在公开声明/guide 明确：同一 PreparedOperation 的全部 Invocation 须
  串行到 Completion 终态；direct 不额外增加共享锁。Graph 约束同一 Executable 内的并发，跨 Executable
  的外部借用资源仍由调用方同步。

本次实际 Windows Release 验证（MSVC 19.50；CUDA 13.3，RTX 4070）：

| 范围 | 缓存/命令定位 | 结果 |
|---|---|---|
| Compute/CPU/Graph source-tree shared/static 与逐头 | `C:/hyle-scratch/compute-route-v8-20260914`、`compute-route-v8-static-20260914`；`hyle-windows.ps1` Build/Test，`--parallel 2` | 各 7/7；补齐 11 个 custom 用例后 shared CPU/Native 定向 4/4、static CPU/Graph 定向 2/2 和 Compute 逐头 1/1 通过 |
| oneDNN direct/composition/Graph region 与逐头 | `C:/hyle-scratch/compute-route-onednn-v8-20260914` | 9/9 通过 |
| CUDA backend、late-failure 独立进程、Graph 与 CUDA 逐头 | `C:/hyle-scratch/compute-graph-v7-cuda`；CTest `--parallel 1` | 4/4 通过 |
| CPU/CUDA 普通源码消费 | `C:/hyle-scratch/custom-execution-v10-source-consumer`；`HYLE_SAMPLE_SOURCE` + `HYLE_SAMPLE_CUDA=ON` | 2/2 通过 |
| CPU/CUDA installed shared 消费 | `custom-execution-v10-shared-build` → `-shared-sdk` → `-shared-cuda-consumer`，均在 `C:/hyle-scratch/` | 2/2 通过 |
| CPU/CUDA installed static 消费 | `custom-execution-v10-static-build` → `-static-sdk` → `-static-consumer`；真实 `HYLE_STATIC=ON` | 2/2 通过 |

源码/安装样例的两种 kernel 在统一 fma 语义后均重新通过。CUDA 示例保持一个 HYLE runtime owner，
native kernel 可以跨 image；不承诺不同 DLL 各自静态链接 HYLE 后可互换 backend 私有资源身份。
这一限制、code image/allocator 保活、exact-only 和非 delegated 适用范围均写入 guide。

P0：Provider 仍为 table pointer + 6 pointers inline payload（本机 56 bytes）；内置配置不分配、不引用
计数。custom 配置一次冷分配，Context、typed kernel、typed Invocation 各有冷存储；复制 custom handle
有冷 retain/release。`operation.cc`、Graph warm execution 和 CPU/CUDA backend 产品代码本次未改，
custom warm enqueue 直接进入既有 prepared callback 的 typed trampoline，无额外虚函数层或 HYLE 热
分配/search/JIT。未运行 P1，不据源码结构签发最终性能胜负。

此检查点完成 Windows 的 C103 旅程，不签整个 foundation gate。Linux/macOS、适用 sanitizer、终局
独立红队和基础 P1 随完整 candidate 验证；C93 完整 profile、Graph bounded/variants、统一完整成本与
constant ownership 仍在既定范围内待实现。组件保持 `AUDITED`，不重开已 CLOSED 的 backend 或其它依赖。

### C93/G103：复用资源要求校验（实现前固定）

Graph 将 region 资源查询前移后，需要在尚未选择具体 storage 时拒绝 malformed proposal。给已有私有
`OperationPreparationAccess` 增加 `validate_requirement`，只转调 Compute 的同一 validator（alignment、
矛盾 flags、exact domain）；不新增公开 API，不使用一个伪造 resource 来绕过匹配，不在 Graph 复制规则。
由 Graph 的坏提案/fatal/不提前 prepare 反例和 Compute/CPU/Native/oneDNN 直接回归验证；warm callback、
Context/Queue/Prepared layout 和 backend 不变，P0 为常数次冷字段检查，仍不签 C93 完整 profile 完成。

已实现并通过：Windows static Compute/Graph 4/4，shared Compute/CPU/Native/oneDNN/Graph 与逐头
13/13，CUDA backend/Graph 4/4，CPU/CUDA source/installed static/installed shared 各 2/2。Graph 对
三类端口的坏 alignment/矛盾 flags/无效 exact domain 均在零 prepare 时拒绝；完整记录与剩余范围见
[Graph v10 本段结果](graph-v10-reviewed-foundation-contracts.md#g103region-冷资源-skeleton实现前固定)。

### C93：稳定实现键与临时准备 token 分离（实现前固定）

代码审计发现，现有 `ProviderOperationProposal::route` / `RegionProposal::route` 只是 Provider 在本次
propose→prepare 间解释的 opaque token，未承诺跨次发现稳定；不能把它直接拿来作性能资料的实现身份。
先补这一必要的身份分量，再接完整 profile。新增共用 `ProviderRouteKey`：默认 unknown；Provider 在
同一 build/configuration 下为同一实现给出稳定非零 key，不使用指针、发现序号或随机数。实现选择改变
且不能由完整语义/shape/layout/device/configuration 身份区分时必须换 key，或继续报告 unknown。

- Compute proposal 与 public owning KernelProposal 增加 `profile_key`，不替换临时 `route`；两者独立传递。
  私有 Provider 表升版，无旧版解释/别名。direct prepare/submit 行为不变，Prepared/Invocation 不存这个字段。
- Graph region 提案同样声明 key，并升 extension 版本；冷选择在临时 node-key 数组记录最终选中项，
  CompileReport 复制到 step。不得把候选指针/prepare token 当 key，不给 copy 伪造 Provider key；PlanStep
  与 Invocation 不增加 profile 元数据。
- Native exact 实现按 operation family 给稳定键，oneDNN direct BRGEMM 与 gated-MLP region 给各自稳定键；
  这些仍须与完整 Provider build/config、设备/驱动、语义、shape/domain、layout/precision、workload 和完成
  边界一起匹配。**有 key 不等于已有 qualified profile，也不用于本段排序**。
- 验证：unknown 不自动补键、临时 token 与 key 可不同、重选只报告胜出键、旧 SPI 拒绝；真实 Native/
  oneDNN/region/custom 与 Graph 数值、OOM/生命周期、逐头及三交付回放。P0 只增加冷字段复制，热回调不变。
  完整 profile 的输入形状、全维匹配和成本比较仍是 C93/G103 未完成部分，不能据此签完 foundation。

### 2026-09-15 C93 稳定实现键检查点

上述分量已实现：`ProviderRouteKey` 与临时 `route` 独立；Compute 私有表为 v9，Graph region extension
为 v12，不解释旧表。Native exact、oneDNN BRGEMM/gated-MLP 和 public owning factory 已接入。
Graph 在 compile 栈上的 SmallVector 保存最终 node→key，重选成功后才填入；报告复制后释放临时数组，
所以 `PlanState/PlanStep/PreparedState/Invocation` 均未为这个字段增加常驻成员或热路径逻辑。

- Windows static Compute/CPU/Native/Graph 与逐头 8/8；shared + oneDNN direct/composition/Graph/region
  与逐头 14/14；CUDA backend/late-failure/Graph/逐头 4/4。后补非连续 region 调度重排的 key 对应断言，
  在 static Graph 执行套件 174/174 再通过；没有因纯 fixture include/格式变更重跑全部平台。
- 新增 key 值类型、临时 token 改变而 key 不变、unknown 不补键、leaf/region 重选不泄漏被拒绝键的
  测试；真实 Native/oneDNN/MLP 断言键有效，跨域 copy 无键，非连续 region 重排保持正确 node 对应。
  原逐分配失败/生命周期/数值/版本失配回归继续通过。
- CPU/CUDA 普通样例现在显式声明并检查 `ProviderRouteKey{1}`，source/installed shared/installed static
  各 2/2 通过；仍不提供真实性能 profile。SDK 这条样例不包含 oneDNN，oneDNN 本次验证是上述源码构建；
  OD53/GP43 的终局完整安装/P1 义务未被它代替。
- shared producer 与 CPU/CUDA consumer 加载目录的 DLL 匹配：Compute SHA-256
  `9A0DAE2CFDCE74929149C6B5A636A374BB6982AFEE97977690E5EE841015CF55`，Graph
  `61E1382AEE00C50AB662D13AE67C0C35139A57EB787D26500102DD0166B80F0C`。
- P0：key 是一个 uint64 value，普通提案/报告各一次冷复制；Graph 临时键存储 `O(N)`、填充 `O(N)`，
  inline 16 个，较大图用已配置 allocator 一次分配。没有按 key 排序、隐式执行/调优或新 warm 元数据。
- selector 基线 `3c95186d` READY（82 项跳过），docs/closure/diff 通过；registry 与已 CLOSED backend
  产品代码未改。未运行本段 P1、Linux/Apple、sanitizer 或独立终局红队，不签发那些结果。
- 自实现不含测试：Compute **16,181**、Graph **10,784**，合计 **26,965**（同前一检查点口径，+45）。

这里只补足“选的是哪套实现”的稳定分量。C93 的完整 build/config/device/driver/domain/boundary/workload
匹配、G103 完整成本选择仍未完成；也没有以进程内实例编号或单个 key 代替完整资格。所有 foundation
剩余项保留，不进入算子扩充或终局收口。

### C93：完整只读 profile 匹配（实现前固定）

实现共同 `route_profile.h` 合同，作为 G103 完整计划选择的数据输入，而非 Provider 私有估值。
identity 分开保留 HYLE build、backend build/device/runtime/execution config、Provider build/config/
implementation、计算语义 boundary、shape domain、精确 tensor 描述、完整 physical route、workload
内容/仪器合同、latency/throughput/concurrency/determinism 与真实终态边界。稳定 fingerprint 由拥有该
事实的构建/设备/外部 profiler 提供；零是 unknown，不从地址、进程内实例号、显示名字或 POD padding
自动补全。匹配器只验证一致性，不能凭用户填写非零 ID 就宣称仪器或数据已科学资格化。

有限 FIX：无分配的结构化完整匹配与明确 mismatch reason；只读 span 查找；相同 key 的重复观测取
保守 interval 包络，不按顺序挑最小值；缺编译次数默认只比 replay，显式 amortization 缺 compile cost
保持 unknown，溢出/NaN/负数非法；合法 exact zero 与 unknown 区分。普通 direct prepare/warm 路径不
读 profile。此段只实现共同输入/匹配，Graph 的自动 key 生成、候选计划消费与有 incumbent 搜索须在
后续同一 C93/G103 范围完成；不能手填一组假 ID 给真实 Provider 签资格。

验证限 Compute 单元/逐头及既有 Graph/Provider 直接消费者；普通 source/static/shared 接入公共 matcher。
表驱动反例逐字段变更，涵盖 unknown、精确 tensor layout/format、顺序、生命周期借用、重复观测、非法
浮点/边界/并发与 amortization。P0 查有界线性只读扫描，P1/跨平台/独立红队仍与完整 foundation 一次做。
不改 Backend、公共构建/仪器核心、组件边或 CLOSED 依赖，只登记本 factor 的源/测试文件。

### 2026-09-15 C93 只读 profile 合同检查点

- `RouteProfileIdentity/RouteProfile`、`WorkloadIntent`、完整 matcher 与只读 lookup 已实现，复用现有
  BackendId/ProviderId/ProviderRouteKey/TensorDesc，不引入新 factor、cache、FFI 或设备查询。缺任一
  必需身份时保持 unknown；helper 只证明输入事实一致，不证明 profiler 的测量质量或事实真实性。
- 重复观测取保守区间包络，与输入顺序无关；latency/throughput、并发、确定性、内容/仪器合同及真正
  完成边界必须匹配。预计 replay 次数是查询摊销政策，不需要重新测同一 physical route；若改变了实际
  tactic/config/route，那些身份本身仍必须变化。默认不合并冷时间；明确摊销且缺 cold cost 时 unknown。
- 新增 10 个用例，Compute 共 44/44；逐字段反例含同家族不同 build/config/implementation、目标次序、
  空身份、同 shape/dtype/extent 的不同 strides/opaque encoding、各 workload 维度、Graph/operation
  terminal、NaN/Inf/负数/区间反转/加法溢出、exact zero、缺失与重复观测、借用结束后的值结果。测试中
  的 fingerprint 与时间仅为合同 fixture，不是任何真实 Provider 的资格或性能记录。
- Windows static `compute-route-v8-static-20260914` Compute/Graph/逐头 5/5；shared + oneDNN
  `compute-route-onednn-v8-20260914` Compute/CPU/Native/oneDNN/Graph 与逐头 15/15；CUDA cache
  `compute-graph-v7-cuda` Compute/Graph/Backend/late-failure/逐头 5/5。
- v10 custom sample 的 source/shared/static 各 CPU+CUDA 2/2；CPU 消费方显式调用安装的 matcher/lookup
  和 arithmetic validator，缺身份不能误报零成本。producer、SDK、consumer 的 CPU/CUDA 目录 Compute
  DLL SHA-256 相同：`8F9B626D49935BD60919108F5954EC4323A10AC104D68D0D5E78A1422EA6A2B1`。
  SDK 只重建实际改变的 Compute target，保留未改源码对应的其余制品；安装前缀始终显式指定 scratch。
- P0：单次匹配 `O(T + P + B)`（target、Provider、boundary tensor 数；tensor rank 有上界），lookup
  最坏 `O(K * (T + P + B))`，额外空间 `O(1)`、零分配。没有 cache/index、回调、时钟读取或新 warm
  字段；这是有限只读资料的接口实现，不把线性扫描声称为全局最优索引或端到端性能胜利。
- 当前 guide/样例说明同步；selector 基线 `75cdb4d2` READY、83 项有据跳过。CMake 只登记源/测试文件，
  registry、组件边、已 CLOSED 依赖和公共仪器未改。已有验证缓存重配触发了 SIMD 的重编译，但没有
  改 SIMD 源码或执行其性能测试。closure/docs/diff 通过，未运行 P1、Linux/Apple、sanitizer 或独立终局
  红队；这些仍随完整 foundation candidate 验证。
- 自实现不含测试：Compute **16,404**、Graph **10,978**，合计 **27,382**（+223）；全库自实现
  **122,508**，统一脚本 self-check 通过。C93 仍未签完：Provider/backend/Graph 的真实完整 key 生成与
  Graph candidate 消费尚未接入，不能让调用方手填假 key 代替；G103/G104 完整计划择优与 incumbent、
  G101/G102 bounded/variants、G106 聚合内存、架构探针和终局验证全部仍在范围内。
