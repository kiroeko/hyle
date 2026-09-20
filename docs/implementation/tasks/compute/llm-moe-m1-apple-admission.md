# M1-A：MLX 与原版 ggml Metal 接入

## Closure card

- owner：`/root`；baseline：`6217e5663acb6825a72cc8a1a2e3d82735d8ceee`，Windows main。
- 当前阶段：SCOPED；`compute_provider_ggml_metal` 在 `cee2f293` 登记 QUEUED，
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
- 源码/设计总预算 120 分钟。正式登记前输入已计 5 分钟；本次续查/设计保守追加 15 分钟，
  累计 20 分钟、余 100 分钟，不重置。实现、编译、预声明验证另记；每根因最多两次纠正。
- 机器：Air/Mini 原生正确性、生命周期、三交付；Windows/WSL Apple-OFF 与公共语义。
  重型工作跨机器串行、parallel2；Windows DLL 邻置，不改电源、不触发 UAC。
- stop：有限 FIX 与全部适用门通过后关闭；必要入口失格则写清具名成熟替代/实质阻塞，
  不以拷贝、枚举、只有 source mapping 或 expert probe 冒充完成。
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

## 有限审计清单（SCOPED，尚未授权产品修改）

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
