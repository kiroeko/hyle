# R6 Vulkan：严格 F32 完整片段与原生边界成本

本记录从属 [R6 主卡](vulkan-v1-four-route-foundation.md) V3/V6；baseline
`e408536690ea7fc76153f9013bf5aa300feada22`。先审源码和合同，再构造 benchmark；
当前未冻结最终候选，未运行 P1，不改变原 5% / n15→最多31 的预算。

## 强 peer 与有限源码裁决

独立只读 checkout `C:/public/project-hyle/ggml`，commit
`456172ec733a135778adcd32d00e576a58232e45`，HEAD 与工作树已核对。以下行号绑定该 pin：

- `ggml-vulkan.cpp:4502` 的非 cooperative 浮点 host recipe 在 ID3 填 16；这不等于 F32
  shader 的实际 BK。`:5463` 按矩阵的短边选 32/64/128；`:5535` 创建 F32 matvec 的 shared/subgroup/hybrid
  变体；`:8106` 为 NVIDIA 非 pre-Turing、N≤8192/K≥1024 选择较大的工作组。
- `vulkan-shaders/mul_mm.comp:172–177` 在 `DATA_A_F32` 分支直接 `#define BK 32`，
  ID3 的 specialization 只存在于其它分支；`:192` 的共享内存是
  `(BM+BN)*(BK/2+1)*sizeof(vec2)`。初次 P0 把 host recipe 的 16 误当作 F32 的实际 BK，
  `82c14c7e` 的 Windows 48 KiB 设备未暴露此错，Air 32 KiB 设备真实 validation 拒绝
  128 tile 的 34,816 bytes pipeline。V3 定向纠正：按实际 BK32 计共享内存，使 Air 选择
  合法 64 tile；保留上游 ID3 recipe 值（在 F32 中不生效）和原样 shader。此时尚未启动 P1。
- `mul_mm_funcs.glsl:34` 的 aligned load 省略边缘保护；ggml `:9531` 的选择还依赖其 buffer
  承载。HYLE ready native buffer 只保证公开 range，不能借用 ggml 的额外 padding 假设。
  V3 同项 FIX：仅在 M/N 都整 tile、K 整 tile、设备最小 storage offset alignment≥16 时
  使用原样 aligned 变体，其余保留边缘检查；不要求调用者补隐藏 padding。
- `ggml-vulkan.cpp:5341` 与 shader generator 的 strict FP32 fallback 提供同数值原始矩阵
  计算；cooperative/FP16 中间值不进入本次排名。matvec 的融合 flags 是 bias/scale；本次
  无 bias/scale 的 gate/up→SiLU→down 保留成熟融合 SiLU，未拆分其中的 sigmoid/multiply。
- 原 ggml runtime 没有本轮所需的 device/buffer native handoff，且每次 graph_compute 编码。
  已有承载裁决不重开。本轮比较受审原样 shader 完整片段，不能声称打败整个 ggml runtime。

- `prepare_plan` 的原常量值在 packing 后仍保留原 dense buffer，即使没有任何最终 dispatch
  再引用它；这会让完整 FFN 同时常驻两份权重。V3 同项资源 FIX：全部 lowering/packing 去重
  完成后，只释放最终 dispatch 不引用的常量 owner。仍被其它片段使用的原常量继续保活；
  不改变 caller Tensor 或返回结果。以 public owner/quota 和后续真实计算验证。

上述修正来自 P0 源码差异，在计时前纳入已有 V3，未增加算子/API/状态字段。API、资源、完成与
失败语义不变。只补与新合法分支相关的完整/尾部 tile 数值验证；最终制品再执行适用验证。

## 同合同比较与隔离

| 入口 | 候选 | 同合同参考 | 热态结束边界 |
|---|---|---|---|
| Direct | 公开 typed prepare + Recorder 组合完整 FFN，固定 caller-owned buffers | 独立原样 ggml SPIR-V/常量布局/push/specialization，经相同 Recorder 组合 | 原生 fence 成功，Completion 析构/保活交接完成 |
| Graph | 普通 typed DAG 冷 compile，独立 Invocation/owning output | benchmark 私有 native recipe 独立描述四个 dispatch，经相同 HYLE 原生运行槽/结果 owner | bind→submit→fence 成功→结果 lease 读取并释放→本次 owner 销毁 |

参考不调用 typed `prepare`、`lower_operation` 或 Graph compile 来生成数学链；SPIR-V 从独立
锁定 checkout 编译并记录输入/制品 hash。共用 HYLE native admission、完成与资源 owner 是显式
生命周期 adapter，两侧全部计费；这只验证成熟实现接入、规则与 Graph 编译非劣，不是 HYLE
算法胜利。Graph 的参考可以使用私有跨因子结构，不为 benchmark 新增公开入口。

Direct 与 Graph 各自比较，不将 caller-owned Recorder 与 owning Graph 直接排名。额外的裸
Vulkan recorded command + checked submit/fence wait 是 `NATURAL_ROUTE/NOT_COMPARABLE`，
只报告生命周期边界成本；它没有完整取消、abandon quarantine、结果 lease 或共享 Context 并发
服务。不得把这些额外责任全部算成 shader 或图调度损失。

## 完成边界与静态成本账单

源码锚点为当前 `src/compute_route_vulkan/`：`vulkan_recorder.cc` 的 `finish/submit`、
`vulkan_program.cc` 的 `create_invocation/bind/submit`、`vulkan_run.cc` 的
`submit_locked/observe/completion_vtable`，以及 `compute/detail/generated_result.h`。

| 阶段 | Direct 两侧 | Graph 两侧 |
|---|---|---|
| 数据/packing/编译/录制 | COLD，分别记录准备时间和实际 VkDeviceMemory quota peak | COLD，相同；记录可复用 storage blocks |
| admission | CHARGED：collect、queue mutex、ResetFences、QueueSubmit、pending/completion owners | CHARGED：另含 bind descriptor 更新、ResultSlot begin/publish、输出保留检查 |
| GPU 计算与可见性 | CHARGED：完整四 dispatch、shader barriers 与实际 fence wait | CHARGED：同样完整计算与完成边界 |
| terminal/owner 回收 | CHARGED：pending unlink、Completion 销毁、shared owners 释放 | CHARGED：同前，加结果 lease/ResultSlot 交接和本次结果销毁 |
| 返回后正常路径工作 | 无 deferred callback/worker/GPU 工作 | 无 deferred callback/worker/GPU 工作 |

正常 replay 不建 pipeline、device buffer、command/descriptor pool，不重新编码、不搜索规则。
host shared_ptr/ResultSlot 原子和对象局部互斥按真实路径计入；driver submit/wait 的内部资源成本
也计入。没有 HYLE worker、wake 队列或 OS polling 线程；异常 quarantine 不冒充正常热态成本。
Descriptor update 只在 Graph 两侧同样的 bind 中发生；Direct 两侧固定资源，轮换两个独立槽。
Host upload/download 作为同样的冷驻留/计时外 correctness 检查，不能只移出其中一侧；裸原生
诊断也使用相同 ready inputs/weights。冷 packing 不用热态重复 amortize 隐藏差异。

## 消费、数值与资格预算

沿已登记 vLLM `51918e252af4dfff381001f42b7ebe6a0454501b` 的 Llama-2-7b TP1 FFN
K4096/H11008/O4096，M1 primary、M16 holdout，有限 `PROVISIONAL_ARCHETYPE`。这里只采用形状；
全部 activation/weight/intermediate/output 为 F32，不把其它路线的 Q4/FP16 合同混入。
权重和输入用全索引整数混合生成，两组输入必须产生不同输出。完整输出先对独立 CPU double
oracle，F32 节点之间按公开输出格式取整；数值门在计时前固定为 `1e-5 + 1e-4*abs(expected)`，
至少半数输出拒绝全零。两侧逐元素比较另用 `1e-5 + 1e-5*abs(reference)`。

Windows RTX4070 与 Air M4/MoltenVK 分别签自己的工程域，driver/API/glslang/header/pin 必须由
实际执行制品输出，未知机器不外推。Windows 采用既有 native-runtime-pair 的 P-core process
budget 与 caller pin；macOS 采用实际 interactive QoS，不能声称硬锁 P 核。机器/电源条件继续
由统一工具核对，不修改全局电源策略或驱动。

同实现 workload A/A 先覆盖两侧实现、执行正反序、分配正反序和两种合法 placement；之后使用
统一 nanobench paired adapter。模块局部 validator 复用公共 parser/机器合同，不能改统计核心。
5% A/A factor-median bias / 10% gross process instability；失败先查成本/时序，不加 n 掩盖。
API/算法/普通交付稳定后的 Direct、Graph 各一次 P1，5% NI 与严格方向分开记；n15 仅合格 CI
未决时最多31，最多一次有源码与 consumer 双重证据的性能纠正。资格、oracle 或生命周期不等价
时在完整采样前停止，不能换一个容易通过的 peer 或缩小形状。
