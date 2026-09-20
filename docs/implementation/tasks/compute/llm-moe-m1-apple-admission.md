# M1-A：MLX 与原版 ggml Metal 接入

## Closure card

- owner：`/root`；baseline：`6217e5663acb6825a72cc8a1a2e3d82735d8ceee`，Windows main。
- 当前阶段：AUDITED；五 owner 在 `450809ba` 完成 SCOPED；新身份在 `cee2f293` 登记 QUEUED，
  只有身份，没有空 target 或产品实现。
- 本轮 owner：新 `compute_provider_ggml_metal`；既有 `compute_ggml_runtime`、
  `compute_route_apple`、`graph_route_apple`、`compute_ops`。
  新 provider 拥有独立 Metal program/schema/资源，不依赖 Graph；Graph region 翻译归现有 Apple owner。
- trigger：[M0 接入清单](llm-moe-provider-integration-ledger.md) M1-A 与 F10 原版递归片段。
  现有 Apple 路线只有整段 MLX 编译，尚无 ggml Metal 程序及跨原生 runtime 边界；
  公共 Operation 也没有 Gated DeltaNet 的显式多输出语义。
  此为新的 consumer/后端需求，依法重开上述 CLOSED owner，不重审旧数学或旧 terminal 范围。
- 先决：M1-G、M1-R、M1-C 已 CLOSED。CPU ggml/oneDNN 的 BLOCKED 条件不变。
- 主纵切：本地 Qwen3.5-2B 的原版 Gated DeltaNet recurrent region，显式非零初态、
  输出与新状态，重复调用、两个 invocation 交错；同 GPU 与 MLX 原版算术片段组合。
- 正交纵切：无模型标签的 F32 数值片段经普通 Direct/Graph 使用同一资源与错误边界；
  不把模型名写死成启用库的前提。
- scope：原版 Metal 的构建/资源/完成/失败、普通入口和必要语义适配、三种 C++ 交付。
  保留成熟 region 内部融合与调度，不能拆成逐 kernel 通用接口。
- non-goals：全部冻结自研算子的扩张、源码研究、优化、比较或删除；改写上游数学；
  全 F01–F19 批量映射、模型读取、推理引擎、公共治理或性能仪器改造。
- tier：P0，核对完整边界同步、转换、存储与重复执行成本。成熟默认路径的适用 P1 在 M2/M3；
  不以本 seam 样例签 64×4 覆盖，也不提前比较自研。
- 源码/设计总预算 120 分钟。登记与审计检查点累计 35 分钟；本次有限 native 资源与 region
  接口细化追加 35 分钟；资源故障边界与下述 UBSan 反例再保守追加 35 分钟，累计 105 分钟、
  余 15 分钟，不重置。实现、编译、预声明验证另记；
  每根因最多两次纠正。
- 机器：Air/Mini 原生正确性、生命周期、三交付；Windows/WSL Apple-OFF 与公共语义。
  重型工作跨机器串行、parallel2；Windows DLL 邻置，不改电源、不触发 UAC。
- stop：有限 FIX 与全部适用门通过后关闭；必要入口失格则写清具名成熟替代/实质阻塞，
  不以拷贝、枚举、只有 source mapping 或 expert probe 冒充完成。
- 构建平台事实按独立[selector 登记卡](../build/ggml-metal-selector-facts-v1.md)补入既有字段，
  不改变选择算法；独立 Metal factor 只探测 Metal 工具，不因测试该 factor 获取 MLX。
- 用户停止点：继续 M1–M4；**M5 开始前暂停汇报，等待用户明确继续**。

## 审计输入与待固定合同

- ggml 锁定 `456172ec733a135778adcd32d00e576a58232e45`（MIT）；
  MLX 沿既有 lock，不升级 pin。原版 ggml 有 Metal backend、独立 command buffer/event、
  Gated DeltaNet pipeline；声明存在尚不构成资源/数值资格。
- 现有 Apple Context 的 owned submission thread、FIFO admission、close/drain、
  Completion 与 owned result 保留。MLX array/stream 是本路线刻意的 native 类型，不推入共同层。
- MLX 与 ggml 各持原生程序。审计实际 buffer import、地址/布局、生命周期、同步和失败机制后，
  固定私有 bridge；共享物理内存不是无需同步的证据。必要搬移须明确记账，不能隐藏为零拷贝。
- 公共 Gated DeltaNet 只定义真实输入/输出、state orientation、类型/尺寸及推导，
  不增加 HYLE 数学实现。原版 fused/composed 接法由 provider 承载。
- 普通 `Context::prepare` 与 `graph::apple::compile` 自动绑定合格成熟程序；
  用户不排序 provider、不提交 timing profile。已有模型/设备/OS/MLX identity 沿用 M1-R。
- 三交付复用现有 dependency lock/cache、module-owned target、资源/许可证安装与路径定位。
  Apple OFF 不探测、获取或链接 Metal/MLX。只对本轮必要资源补丁负责，不顺手扩 vendor portability。

## 有限 FIX（审计已冻结）

| ID | 能力族/承载 | 必须固定的结果 |
|---|---|---|
| A1 | BORROW 构建与 native runtime | 单份 ggml source 的 Metal backend、原生库资源/许可证、OFF 与三交付形状 |
| A2 | FIX 资源边界 | native buffer owner/import、同步/失败/部分回滚、每 invocation 资源、输出保活 |
| A3 | BORROW 完整 recurrent region | 原版 Gated DeltaNet，明确状态朝向、衰减/beta 语义与支持域，普通 Direct 可达 |
| A4 | FIX 语义和普通 Graph | 必要公共 descriptor/inference；Graph 保留 MLX 与 ggml 原生 region，普通入口自然组合 |
| A5 | KEEP 已有运行合同 | FIFO worker、Completion、cancel/close 与旧结果 lease；不复制通用执行器或线程循环 |
| A6 | FIX 有界验证/交付 | 下列适用门、使用文档、成本账单与最终证据，未运行项保持未决 |

## 预声明验证

1. 元数据阶段只运行 docs、closure（带明确 previous ref）与 diff；selector 的产品门留给固定候选。
2. 共享语义：合法/非法 shape、dtype、输出/状态、整数溢出与输入不变；Windows/WSL/Air/Mini
   适用 compute_ops 及公共头门。其他路线尚未接 recurrent 时明确 unsupported，不能偷偷执行自研。
3. Air/Mini：原版 recurrent 对独立 double oracle，非零初态、多 token/单步、两个请求交错，
   普通 Direct/Graph + MLX 片段，重复运行/旧结果、无模型标签及不支持域诊断。
4. 资源风险：构造/分配失败、受控 native 失败、accepted-prefix drain、放弃 Completion、
   取消/close 与输出 lease；ASan/UBSan 只签适用 host owner，不代签 Metal driver。无须破坏机器的 GPU fault。
5. source/static/shared SDK：独立 consumer、producer 隐藏、prefix 搬移、普通 Direct/Graph，
   metallib/headers/许可证完整；Air 主验三交付，Mini 验原生及交付 portability。
6. Windows/WSL Apple-OFF 的配置与公共消费者；仅 Apple 改动不重跑已通过 CUDA 性能/三交付。
7. P0 原生调用账单、候选代码复核；不运行 P1 或冻结自研对比。统一脚本统计全部 Compute+Graph
   实现 LOC，不含测试/vendor。归档后删除本卡临时构建与验证副本，保留共享缓存/本地模型。

上述门当前均未新执行；已有 terminal 只在身份/合同匹配时复用，不签本轮新增接入。

## 审计结论与实现合同

源码输入为本机 locked ggml，以及 Air 共享缓存中的 MLX `1f8e74e3f12f`；
本地 2B config 确认 key/value head dimension 均 128、各 16 heads、FP32 recurrent state。
本段审计保守追加 15 分钟，总 35/120 分钟。未运行 probe 或性能测试。

1. `ggml.c::ggml_gated_delta_net` 与 `ggml-metal/kernels/gated_delta_net.metal`
   提供完整 token 循环和末态；不在 HYLE 重写此数学。原版首域为 F32、相同 Q/K/V heads、
   square state、D=32/64/128、正 token 数；本卡主验 D128/H16，正交验无模型标签的 D32。
   默认按合法能力域冷绑定此成熟实现；模型身份不充当普通功能开关。
2. 新 `GatedDeltaNet` 是纯、多输出语义：Q/K `[B,T,Hk,Dk]`，V `[B,T,Hv,Dv]`，
   log-decay 与 beta `[B,T,Hv,1]`，初态 `[B,Hv,Dv,Dk]`；输出 V 同形及新 FP32 状态。
   Q/K 已经由调用方选定是否归一化；先对 state 乘 exp(log-decay)，以 beta 更新 delta，
   再计算 Q 投影并乘 `1/sqrt(Dk)`。不隐含 L2 norm、sigmoid、外部状态写回或模型加载。
   公共语义允许 grouped query/key heads 与浮点 Q/K/V，原版首域不支持的条件冷拒绝；
   M2 才批量接原版 cast/repeat 等组合，不能把 descriptor 接受当成后端覆盖。
3. MLX `metal/allocator.cpp` 小数组可来自 Metal heap。普通 host-pointer remap 不能据此
   声称合格；私有 bridge 直接 retain 原 `MTLBuffer`，检查同 device、storage mode、offset/size，
   建立 ggml 非 owning wrapper。输入 owner 由 Invocation 保留；wrapper 只 release 自己的 native
   引用，绝不 free MLX payload。输出由 MLX allocator 正常分配，ggml 写入同一原生 buffer；
   packed scores/state 用两个有 owner 的 array view 返回，无 CPU 数学与 shadow payload copy。
4. Apple 专用内部 plan 按合法依赖划分 MLX compiled region 和 ggml whole region。
   Graph 只做边界翻译，Compute 不含 Graph 类型；现有纯 MLX 编译路径保留。
   每个 HYLE Invocation 独立持有 ggml metadata/backend/绑定，native program 不放入共同 Queue。
   复用原 submission thread、状态机、result lease。每个跨库边界完成前一 runtime 的 GPU 工作，
   再交给后一 runtime；此 managed 接法明确有 host wait，不能宣称 event-only 或 recorded replay。
   冷准备可在隔离的零输入/初态上触发原版 pipeline 编译，绝不执行用户 mutable state。
5. ggml `metal-context.m::ggml_metal_synchronize` 把失败保存在私有 `has_error`，
   public void synchronize 不能直接给本次 HYLE Completion verdict。有限资源 patch 导出
   checked 同步状态，保留原生错误数值，失败先 drain 再发布；失败实例不得复用。
   `metal-device.m` 的 buffer/encoder 部分构造与 nil pipeline 也纳入同一资源修复，
   不改 pipeline 选择、数学、融合启发式或 kernel。运行失败不转成成功空结果。
6. 构建只条件启用 ggml Metal；vendor C 符号保持私有，资源/锁/许可证沿单份声明导出。
   嵌入原版 Metal 源或已有 metallib 的最终选择按锁定 CMake 的真实行为固定；
   允许 cold runtime compilation，但不得在普通 hot replay 重编。无独立 HYLE shader。
7. P0 账单：每 invocation 的 ggml 元数据/原生 backend、每次必要 output allocation/绑定，
   每个跨库边界的 GPU completion wait；MLX 内部 fusion 不因 ggml region 被拆成逐 kernel。
   每个固定 region 的派发允许，禁止热路径搜索 Provider、计时或重新裁决规则。

能力裁决：A1/A2/A3 为 BORROW+必要资源 FIX（MISSING）；A4 为语义/Graph seam FIX（PARTIAL）；
A5 KEEP（已存在、须验证新 consumer）；A6 FIX。批量全覆盖 DEFER M2/M3，全部自研 REJECT 本轮。
上述公开形状、owner、lifetime、错误与常态成本已固定；若原生事实要求改变这些合同，先重新定界。

## 语义实施检查点（仍为 AUDITED）

公共 `GatedDeltaNet` 已有纯描述、六输入/两输出 traits、F16/BF16/F32 QKV 与 FP32 gates/state
的形状和类型检查；输入不变、状态轴向及 grouped heads 合同明确。没有新增数学实现。
Windows 静态 `compute_ops` 与 public-header 两门通过，含非方形 state、非整除 heads、错误 gate
宽度、零维度、类型与 arity、失败不部分修改输出。原始日志、运行脚本、selector 计划与源码 hash
见[语义收据](../../../../evidence/correctness/compute-llm-moe-m1-apple-admission/schema-windows.json)。

该收据只签公共语义，未签 Apple native、完整 Graph、host 插桩或三交付；其余 selector 门保持待执行。
Metal native import / checked completion 的原版资源适配及普通入口仍待实现。新增覆盖为 0/256，
没有 P1 或自研对比。统一脚本 `loc-report.py --self-check`：Compute+Graph 实现 **51,213 行 / 217 文件**，
不含测试/vendor，包含冻结存量。保留本卡小型 schema build 供后续增量验证，最终归档后清理。

## 普通入口实施检查点（2026-09-20，仍为 AUDITED）

原版 ggml Metal 已按路线条件构建，使用同份 locked ggml 和可复现的资源补丁；独立 Metal factor
不获取 MLX，Apple OFF 不探测 Metal 或获取 MLX。原版 Metal 的 `.m` 由 Objective-C 编译，
不把其 C 资源代码强制当 Objective-C++ 改写。Windows OFF 的测试登记沿既有 aggregate 机制修正。

普通 Direct `prepare(GatedDeltaNet{}, specs/examples)` 已接原版完整递归。普通 Graph 冷期划分
连续 MLX compiled region 与 whole recurrent region，只发布跨区/公共输出以保留 MLX 内部融合。
同一 submission owner 执行固定 region 顺序；各 Invocation 独立持有 ggml backend/metadata/import。
跨库 checked host wait、MLX GPU contiguous 整理 strided 输入，以及结果分配明确计入成本。
输出两个 MLX array view 共用一份 packed 原生 buffer，无 host 数学或 payload shadow copy。
冷环境身份增加 locked ggml revision。没有新增、研究、比较或删除任何冻结自研数学。

### 已运行

- Air Release/static/Werror：七门 PASS，包括 `compute_ops`、普通 Apple Direct/Graph、
  ggml Metal native 与对应公共头。主域 D128/H16、单步/17 token、独立 double oracle、非零初态、
  两请求交错、旧结果、重复调用；无模型标签 D32、strided MLX 输入；
  MLX → ggml → MLX → ggml → MLX 图及显式末态依赖；quota、放弃结果和取消/close。
- 首轮七门中六门通过；strided 测试的原生 `eval()` 后缺显式 ready wait，违反既有 bind 合同。
  只修正测试并重跑受影响 Direct factor PASS，其余六门复用相同产品身份。产品不放宽 ready 检查。
- Windows Apple OFF 配置与 `compute_ops` / 两 Apple route 公共头三门 PASS。
- selector 现有 suite 59/59 PASS，Windows 与 Darwin planning projection READY；后者只签选择事实，
  不冒充原生运行。独立工具卡结束；docs/closure/diff 通过。

完整命令、原始失败/修正记录、Windows 完整前向快照哈希及机器身份见
[native 检查点收据](../../../../evidence/correctness/compute-llm-moe-m1-apple-admission/native-air-checkpoint.json)。

### 尚未完成

A2 仍须完成分配/部分构造和 native failure 资格。有限资源复核已定位原版 pipeline cache 的
C++ map 分配异常可能越过持锁 Objective-C 函数，以及附属 owner 构造的回滚；纳入现有 A2，
先定向修正这些资源路径，不扩大到上游数学或全库审计。现有数值、quota 与 cancel 测试不代签此项。
随后仍有 Air host 插桩、source/static/shared 独立 consumer/搬移/导出资源门、Mini 原生与交付、
WSL OFF/公共语义，以及最后 P0/红队和证据汇总。没有用户决策阻塞，未进入 CANDIDATE/CLOSED。

M1 按实际接入验收估算约 **65%**；新增模型覆盖仍为 **0/256**，无 P1。统一脚本与 self-check：
Compute+Graph 实现 **51,892 行 / 222 文件**，不含测试/vendor，包含冻结存量。
Air 活动 build 约 99 MiB、源码约 35 MiB，保留供本卡增量验证；一次性 patch materialization
副本在收据归档后清理。M5 前停止点不变。

## A2 资源验证中的有限补充（2026-09-20）

原版 pipeline cache 的查询/插入分配失败现在由其 C++ collection 返回失败值，在 Objective-C
锁内不抛出；native pipeline、常量、event、context、device/library 的部分构造按 owner 回滚。
静态测试包含实际 patched upstream 源码，只替换测试进程中的 allocation/completion 边界。
首次 C++ 故障注入范围过宽，命中 Apple 私有 Metal driver 后留下锁；该 fixture 失格，不能
据此裁决产品。收窄到实际 collection 后，缓存失败/再入、导入回滚、编码失败、checked completion
保留原生码及失败实例退役通过；原始失败日志保留。没有改写 driver 或数学。

Air ASan 四项通过。UBSan 定位 `ggml.c::incr_ptr_aligned` 在 `ggml_graph_nbytes` 以零地址
计算容量时使用 `((char *)ptr + size)`，触发 null-pointer-offset UB。这是本轮真实
`MetadataArena::for_graph -> NativeProgram::create` 调用链中的资源缺陷；现有 AUDITED owner
`compute_ggml_runtime` 的 A2 定向修复仅将该偏移改为 `uintptr_t` 加法，保持原容量、对齐、
布局和正常成本，不另写 sizing 算法、不审查数学。补充 Windows/WSL 共同 runtime 资源门与
Air UBSan 复验；原版 Metal/MLX、Mini 和三交付仍按原预声明门。未运行项不签 PASS。

## 资源与 Air 三交付检查点（仍为 AUDITED）

上述一行 sizing 修正后，Air host ASan/UBSan 各四门通过；不代签 Metal driver/device，
macOS ASan 未启用泄漏检测，native wrapper 的失败回滚另有 live-owner 断言。
Air 普通 Direct/Graph 的 source/static/shared 三交付均通过，样例新增原版递归及
MLX → ggml → MLX 组合、显式非零初态和重复执行。安装消费者使用独立源码副本；搬移 prefix，
同时隐藏原 producer build 和产品源码后仍可配置/编译/运行。Metal 源内嵌、MLX 资源、许可和
private header / dynamic symbol 检查通过。已完成 producer build 自动清理，未累积 SDK 构建。

Windows 共同 runtime 原版 CPU 资源、公共语义和 Apple-OFF 头八门通过，DLL 邻置；
WSL 相同八门加 host UBSan 通过。Windows/Air 原始命令、失败修正和快照身份见
[资源/交付收据](../../../../evidence/correctness/compute-llm-moe-m1-apple-admission/resource-host-air-delivery.json)。
WSL 原始日志已收齐待最终归档；Mini 原生/三交付正在运行，候选与最后复核尚未签发。
没有新增自研数学或 P1，新增模型覆盖仍为 0/256。M1 接入进度估算约 **70%**，不以新增测试数量
换算覆盖；统一 LOC 仍为 **51,892 行 / 222 文件**（不含测试/vendor，含冻结存量）。

## Mini 与共享构建检查点（仍为 AUDITED）

Mini 原生七门和 source/static/shared 三交付全部通过，安装消费者同样搬移 prefix、隐藏
producer build 与产品源码。WSL 八门及 UBSan 的日志已归档；两个验证副本的 1,722 个
非文档输入与 Windows `b5ddc5ea25e058243f7b7093c26bf4ca2d23b02c` 相同。
见[跨环境收据](../../../../evidence/correctness/compute-llm-moe-m1-apple-admission/cross-native-delivery.json)。
两台 Mac 和 WSL 已完成的源码、build 与 SDK 副本已清理，保留原始日志、共享缓存和模型。

Windows selector 的共享构建先因未传入已安装 Vulkan SDK 环境失败；补齐环境后继续原有
selected gates。共同 compute、CPU/CUDA backend 及各自公共头通过；共同 runtime 的
CUDA 测试链接失败，原因是测试直接使用 `system_allocator` / `Status` 却未声明
`hyle_mem` / `hyle_error` 链接依赖，静态链接曾遮蔽该问题。只补测试 target 的两项依赖，
不修改产品依赖或计算实现。原始失败保留，仅重跑受影响测试并继续尚未执行的门。
最后 selected gates 与候选复核仍待通过，未进入 CANDIDATE/CLOSED，无 P1 或自研对比。
M1 接入进度约 **75%**；M1-V 尚未开始，新增模型覆盖仍为 0/256。M5 前暂停约定不变。

补齐链接后，此 CUDA fixture 实际运行暴露 `0xc0000094`。一次 Windows debug-event 与 linker
map 定位到 exe 内的 `ggml_time_us + 0x24`：测试静态嵌入的 ggml timer 未初始化，
`MetadataArena` 只初始化其 DLL 内的私有副本。普通 CUDA provider 已在冷 context 创建中调用
上游 `ggml_time_init()`，其三交付结果未失效。测试在自己的原生创建边界执行相同初始化；
不改变产品代码，不关闭断言或跳过 CUDA 执行。重跑该 factor 后再继续未执行的 selected 门。

共同 runtime 五门随后全部通过。剩余回归已通过 oneDNN、Compute 的 CPU/CUDA/Vulkan 路线
及 Graph 本体；`graph_provider_ggml_cuda` 的独立测试同样漏写直接使用的 `hyle_graph` 链接
依赖（25 个未解析符号均来自 GraphBuilder/Graph/Invocation/compile）。仅补该测试的 DEPS，
不改已关闭 Provider 的产品依赖；保留 41 个已通过门的精确候选结果，从失败门继续。
