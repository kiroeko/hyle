# Compute/Graph 六项架构复核

## 本轮合同

- baseline `624f81df`；2026-09-15 用户要求按路线隔离、调研/实验、Apple 组合、低成本渐进路由、
  多后端 Provider 融入和文档清理六项 review。旧 Compute/Graph 不构成设计约束。
- 只做源码/API/已有证据复核、有限研究程序和文档更新；不写产品、不更改 registry、SDK/依赖锁、
  公共治理工具，不恢复大 goal，不扩充 shipping 算子。模型 CPU+GPU 联合推理仍不在范围内。
- 预算 75 分钟；最多一个新 Apple 探针、一次 fixture 纠正。已有 CPU/CUDA/Vulkan 证据复用，不跑 P1
  或全库构建。不把“采用成熟实现”或“小代码通过”签为四条路线已经 SOTA。
- 新探针：MLX v0.32.2 的 GPU Q4 affine MatMul → 自定义 Metal SwiGLU → GPU Q4 MatMul，
  M=1/7、K=H=256，每形状三次变化输入。CPU 从实际 packed codes/scales/biases 独立解码并以 double
  累加作 oracle；逐值误差 `<= 1e-4 * max(1, abs(expected))`。保留第一份结果跨后续执行，检查不被
  覆盖；临时输入 owner 在提交后、最终同步前释放；非法 K 在调用验证阶段拒绝。不是故意 device-loss。
- 只签所测正常 lifetime、形状/参数拒绝与量化组合，不签任意外部 Metal buffer/Queue import、M5 NAX、
  并发、完整错误恢复、P1 或三交付。源码来自 Windows 完整 Git archive，Mac 仅验证。
- 文档清理：长期边界分别归 Compute/Graph/路线设计，删去架构中的重复任务表和旧实现细节；历史
  registry records/原始证据不伪造或抹除，失效计划退出当前入口并标明可从 Git 追溯。
- 验证：上述有限探针、change-impact selected docs/closure、`git diff --check`。
- 停止：逐项给出事实、问题、采用方案与尚未证明的资格；不以未完成的产品资格继续无限调研。

## 结果

2026-09-15：六项研究复核完成。**四路线独立原生实现 + 必要共同语义的方向成立；当前产品未实现完，
也没有证据签四路线 SOTA。** 本轮补的是架构决策、一个有限 Apple 探针和过时文档，不签 module
CLOSED、不解除 ggml CPU blocker、不改变 registry，也不自动恢复旧 goal。

### 六项裁决

| 用户问题 | 发现的实际问题 | 本轮采用的设计 / 状态 |
|---|---|---|
| 1. 路线切割 | `cmake/HyleComponents.cmake` 的 Compute/Graph façade 固定包含 CPU/native roots 及可选 oneDNN 身份；外层 CUDA OFF 不是完整四路隔离。路线独立却曾被“同一对象模型”约束 | 公共语义核、各 native compiler/Program/run-state 分开；产品与 vendor backend 两层按 feature 隔离。这是明确实现差距，本轮没改 CMake/产品 |
| 2. 全面调研/小实验/SOTA | 已有研究有真机但覆盖不等；不能从 CPU smoke、旧 Marlin、M4 或 NVIDIA Vulkan 外推所有资源/错误/性能资格 | 按下表分别记录源码与实际实验。四个引擎不是在每条路线都有实现；无源码的组合不虚构“参考通过”。功能机制可支撑设计，P1/正式资格仍欠 |
| 3. MLX/Metal | “Apple 没有推理加速硬件”和“MLX 与 Metal 是两个必须来回转换的栈”均不准确；老文档偏向固定 out-buffer/外部桥 | 优先 MLX 完整 region 与其内部 custom Metal/Primitive；支持 Provider owning result。新 Q4 FFN/lifetime 探针通过。M5 GPU NAX、独立 ANE、SDK/deployment/TF32 条件已分清 |
| 4. 渐进路由 | 宏不能消除未知模型、运行时 owner 或跨 region 成本；局部 kernel 赢不代表切图后赢 | 构建裁剪、冷选路、热固定入口；同时比较原版 region、内嵌 HYLE 扩展、独立原生融合。静态调用能内联，动态固定调用仍以 direct 对照测真实税，不签字面零成本 |
| 5. 多后端 Provider | ggml 有 Metal 而不是 MLX backend；MLX 自己有可选 CUDA backend。按库名做一个通用 adapter 会隐藏不同 owner | family × backend/runtime 分别准入；共享源码/格式知识，不强行共享 allocator/stream/可变 handle。MLX-Metal、ggml-Metal 是两个候选，不自动零拷贝互通 |
| 6. 文档 | Compute/Graph 架构混有任务队列、测试字段和阶段表；roadmap 要求同一对象；references 又要求独占全部 runtime、不准外露任何 native 类型 | 重写 Compute/Graph，独立四路线页；压缩总计划成历史索引，删失效进度/命令；同步 roadmap、build、references、ggml memory、指南与任务入口。历史 terminal cards/原始证据保留 |

本轮路线设计与源码来源见[四路线文档](../../../architecture/compute-routes.md)，完整参考 pins 见
[研究索引](compute-graph-route-unification-audit.md#本轮架构分析统一合同与独立执行)。不改变产品依赖。

### 实际证据矩阵

| 路线 | 已做小代码/真实消费 | 尚未证明，不能签通过 |
|---|---|---|
| CPU | 复核 `f6c45efd` public Sched + Native/oneDNN executor：Windows source/static/shared、真实 fan-out/嵌套/独立 Invocation/关闭回收 | Linux/Apple 对应旅程、最终并行 P1；CPU ggml 的 fallible threadpool/进程 abort blocker 没解除 |
| CUDA | 复核 original Marlin `1f25790b`，RTX 4070、M=1/7/64、K=N=1024、group128 的九次变化输入；HYLE shared SDK buffers/Queue/Completion，packed weights | 当前 vLLM/SGLang carrier、完整 Q4 FFN/capture/并发故障回收/三交付/P1；不是最新 Marlin 的 SOTA 资格 |
| Apple | 前轮 F32 MLX+Metal FFN 六次；本轮 Q4 affine/group64 FFN：M=1/7 各三次、独立解码 oracle、trace、临时输入释放、旧结果保留、非法 K 拒绝，全部 PASS | 任意外部 Metal buffer/Queue 互通、并发/异步故障/关闭 drain、完整 memory cost、M5 NAX、HYLE 正式 adapter/三交付/P1 |
| Vulkan | 复核 RTX 4070/Windows 两 compute pass + barrier：录制一次，三次变化输入 exact PASS | 其它厂商/MoltenVK、多 queue/family、validation layer/device-loss、完整推理片段/正式 adapter/三交付/P1 |

证据：[CPU](cpu-executor-foundation-probe.md)、[CUDA](cuda-compressed-provider-admission.md)、
[前轮 MLX/Vulkan](../../../../evidence/correctness/compute-four-route-admission/README.md)、
[本轮 Apple Q4/lifetime](../../../../evidence/correctness/compute-apple-route-review/README.md)。本轮只新增最后
一项；首次构建/运行成功，无 fixture 纠正，不因文档修改重跑已完成实验。

### 原生约束怎样影响方案

- CUDA：cuBLAS stream 设置会重置 user workspace；可变 handle、graph executable、captured pointers
  和 accepted prefix 的 owner 独立管理。原始 Marlin launcher 仍含冷配置，不能直接当终局 prepared
  provider。vLLM/SGLang 的 finite graph key/persistent buffer 是机制参考，Python/Torch 包装不是 C++ façade。
- Apple：MLX v0.32.2 NAX 要 Metal 4、SDK/deployment >=26.2 与相应 runtime/device；QMM 还有
  transpose/format/K/TF32 条件。该测试 M4/F32 不签 NAX。public custom kernel 能在同 GPU runtime
  内执行，但仍可能产生数组/allocator/cache 成本；array ready/callback 不单独证明错误终态。
- MLX C++ import：Python DLPack/MPS 支持在 `python/src/convert.cpp`，借 nanobind owner；不能照抄
  Python 文档宣布独立 C++ MTLBuffer/Queue 桥已成立。本轮选择已验证的同-runtime 扩展，没有这一外部桥
  也能实现局部 HYLE 替换；跨 runtime 使用必须另验或明示有成本的 copy/sync。
- Vulkan：ggml 的 graph compute 会逐次遍历/编码与分批提交，不等于录制一次。queue 顺序不替代
  memory dependency；descriptor/command pools、pending 状态、non-coherent flush/invalidate 和
  queue-family 所有权按 Vulkan 自己实现。shader 编译工具与 device 对 cooperative matrix、integer dot、
  subgroup 的支持须同时成立；“保底”不意味只保留最弱公分母 shader。新路线不模拟 CUDA capture。
- Build：MLX 有 CPU/Metal/CUDA、tests/examples/file-format、nested Metal-C++ 下载等默认项；后续
  adapter 必须复用共享源码缓存并固定实际 backend/部署合同。当前 `compute_sched` 仍 QUEUED；已有
  CPU public 组合成立，不因其被登记过就推导必须实现它，本轮也不擅自改 registry。

切割问题不是 oneDNN OFF 失效：`cmake/HyleComputeDependencies.cmake` 默认 OFF 确实跳过下载，
`src/compute_provider_onednn/CMakeLists.txt` 使用 OPTIONAL_ABSENT 不链接 oneDNN。问题是 façade
仍无条件带 CPU/native，而且还没有 Apple/Vulkan 的真实独立产品闭包；不能描述成所有平台都必下载
或链接 oneDNN。现有 oneDNN 专用配置是 SEQ/x64 ukernel，已由 HYLE 显式 executor 分片，不是
本轮发现它偷偷启动了全局线程池。

### 后续有限实现范围，不是恢复旧计划

1. 共同层只改必要语义/结果 owner/真实完成入口，直接 Compute 不依赖 Graph；路线选择闭包同时修正。
   逐项裁决旧实现复用、改写、删除，不写旧 API 兼容层，不把“四条路线”变成公共热态大 switch。
2. 分别落实 CPU persistent executor 程序、CUDA 正式 compressed carrier、MLX-Metal 同-runtime 扩展
   与 checked completion、Vulkan native program。每条先一个有意义的主纵切；不扩全 op 目录。
3. 固定实现后做实际失败/并发/三交付与两类公平性能（同实现 direct 税、同合同完整片段竞争），不拿
   本轮小程序代替。错误回收不成立的候选拒绝，无法证明硬预算的候选不能接受严格全内存预算。

尚欠的是具名 adapter/执行器/构建隔离与资格，不是未决定谁拥有输出/谁调度。方案允许厂商实现不同；
局部 API 不满足时拒绝候选或使用计费桥，不先强迫统一。仍不承诺“不会再改任何架构”，只有推翻共同
语义的实际反例才重开共同设计，避免因新的 op 或设备回到全部重做。

### 维护验证

`validation-select.py run --base 624f81df --candidate WORKTREE` 选择 docs/closure 并通过，94 项按影响面
跳过；`git diff --check` 通过。没有产品测试/P1 或 registry 迁移。研究程序本轮已按上表实际编译执行，
不将 selector 对 evidence-only 文件的跳过解释为那个实验没有运行。

文档清理以 baseline 的 Compute/Graph 设计与总计划为同口径，对比本轮三个文件加新增路线页；删除
失效任务队列、旧百分比、旧对象强制要求和重复架构，保留 terminal records 与 raw evidence。清理前
完整版本可从研究索引的 Git 链接恢复，没有删除产品代码或撤销旧实验事实。
