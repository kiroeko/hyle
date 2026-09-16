# R5 Apple：MLX Metal 原生 region 与 owning result

## Closure card

- 状态：`AUDITED`。身份在 `32aade77` 以 `QUEUED` 登记、`8c2f6962` 进入 SCOPED；
  以下 A1–A6 与 API/成本边界冻结，开始产品实施，尚无路线完成声明。
- owner：`compute_route_apple` 与 `graph_route_apple`；canonical baseline
  `553a7d6512f68f4b9a1ec9c253b6c4df04616fda`，保留已经 CLOSED 的 R1–R4。
- trigger：[四路线合同](four-route-foundation-refactor.md) R5 已获实施授权，目前没有 Apple 产品路径。
- 主 consumer：已登记 MLX C++ quantized linear/FFN 消费形状：常驻 affine Q4/group64 权重、
  变化 activation、decode/短 prefill；Direct 无 Graph 依赖，Graph 从 typed DAG 降成完整 MLX region。
- 正交 consumer：二维浮点矩阵变换与同 runtime 自定义 Metal 片段，两个运行槽、保留旧结果、关闭回收。
- 直接依赖：Compute 的 schema、Completion、ResultSlot/OwnedResult；Error/Mem；Platform Thread/Parker。
  Graph owner 额外消费 Graph 的公开只读结构。不给 MLX 套通用 Queue 或逐 kernel HYLE vtable。
- 范围：锁定 MLX Metal、路线原生 Program/Invocation/提交 owner、拥有原数组的结果、默认成熟规则、
  必要的 typed MatMul/Q4/GatedActivation lowering、同 runtime custom Metal、Graph、三交付及文档。
- non-goals：全算子目录、Python/Torch runtime、CPU 模型 fallback、外部任意 MTLBuffer/Queue 零拷贝桥、
  ANE/M5 NAX 资格、模型/KV/batching 策略、跨 GPU 模型执行、旧 API 兼容。
- 性能：P0 + P1，Air `Mac16,12` 的 M1-MAC profile；mini `Mac16,10` 仅 correctness/portability。
  原始 MLX 相同 region 是 direct 成本基线；最强适用成熟完整 peer 先采用 MLX compiled region，
  同合同与 NATURAL_ROUTE 分开。5% 非劣门与方向分开，不能用不同格式或线程/完成责任签非劣。
- 调查预算：60 分钟定向 source/build admission，最多一个锁定源码 native preflight、两次具名修正。
  只补当前 API、线程、完成和构建反例，不重翻历史或重新讨论四路线架构。
- 性能纠正预算：API/算法稳定后 Direct/Graph 各一次不可变候选 P1，n=15；只有资格成立而区间未决
  才最多到 n=31。最多一次有源码归因的机制纠正，重新冻结候选；不扩大 margin、不无限改测量姿势。
- 预声明验证：下文矩阵；重型串行、parallel 2，文档/状态不构建或重跑 P1。
- stop：新共同语义缺口先绑定精确 owner；原生完成/三交付不能兑现时形成具体 blocker；
  未运行适用门不签 PASS，未知设备不外推。其它模块的 CLOSED/BLOCKED 不自动解除。

## 能力审计与承载

| 能力族 | 层级 / 来源 | 当前覆盖 | 承载与动作 |
|---|---|---|---|
| MLX GPU arrays、compile、allocator、Metal kernels | 原子与成熟复合机制；MLX v0.32.2 C++ | HYLE MISSING，原生可用 | BORROW，保留整段融合与内部资源管理 |
| 独立 Direct region 与 typed 单算子 | 复合功能；MLX C++ 与普通数值管线 | MISSING | SELF 的窄提交/失败适配；计算 BORROW |
| Graph 降成完整 MLX region | 复合功能；typed FFN | MISSING | SELF，仅冷读取 DAG；不遍历热态原图 |
| 原生结果 owner、保留与完成 | 原子机制；MLX 与既有 R2 | 共同结果机制 COMPLETE，路线 MISSING | ROUTED/KEEP ResultSlot；SELF 原生完成 owner |
| 同 runtime HYLE Metal 扩展 | 原子扩展；MLX public fast::metal_kernel | 研究已证明、产品 MISSING | BORROW 接入点；有限自定义片段真实组合 |
| 固定线程与 checked drain | 复合生命周期；MLX thread-local compile/encoder | MISSING | SELF 路线 owner；复用 Platform Thread/Parker |
| 模型文件、量化策略、KV/batching | 应用策略 | OUT_OF_SCOPE | APPLICATION；不随 FFN 样例下沉 |

不机械拆出没有独立能力的 backend/provider 外壳。MLX runtime、计算适配与默认规则共同属于
`compute_route_apple`；只有 Graph lowering 需要另一因子。共同 Compute/Graph 不反向依赖 MLX。

## 审计冻结的有限 FIX

| ID | 问题与允许改动 | 必须验收 |
|---|---|---|
| A1 | MLX 与 nested sources 锁、CPU/CUDA/Python/文件格式关闭、真实路线构建及 package 投影 | Apple-only、OFF 无探测、共享 source cache/offline、source/static/shared 与许可证 |
| A2 | 原生 Context/Program/Invocation、固定提交线程、checked Completion、result lease | 两槽/变化输入、保留旧结果、并发接纳、丢弃 completion、close/drain、不提前释放 |
| A3 | 正常 Direct 入口、typed 三族与完整原生 region、默认 MLX 规则、同 runtime Metal 扩展 | 用户无 provider 排序/profile；准备暖机隔离；热态不重选、不首次 JIT；真正计算 |
| A4 | Graph 冷遍历/合法性、完整 pure 固定 shape region lowering 与 custom 扩展映射 | typed FFN、无影子输出 copy、非法 shape/资源显式拒绝、普通入口无需手写失败 glue |
| A5 | 路线风险验证与成本账单 | Oracle、边界、OOM/注入故障、晚到失败、取消/关闭、host 插桩与真 Metal |
| A6 | 用法、默认规则证据、三交付消费者、不可变候选 P1 | 每条路线自己的七门完整；未决事实不挪给 R7 首次接线 |

## API / usability 冻结

原生 MLX 类型只出现在 Apple 专属头与 target；通用 Compute/Graph 头不带 SDK。
Direct 的整个 region 可用自然 C++ callable 表达，接收 MLX arrays 与该 Context 的 stream；
typed 单算子准备与 Graph lowering 都应用固定的成熟 MLX 实现。无需另建一个 HYLE 私有 DAG DSL，
也不要求把 MLX region 拆成多个 host 提交。专家 custom Metal 仍沿同一 runtime 执行。

```text
Direct: apple::Context::create -> prepare(typed operation) / compile(C++ region)
        -> program.create_invocation -> bind ready inputs -> submit -> owning_result.wait
Graph:  GraphBuilder typed DAG -> graph::apple::compile(context, graph, constants)
        -> 同路线 invocation、submit 和 owning_result
```

- 固定提交线程拥有 stream、冷 compile/warmup 与所有 native eval/synchronize。它是路线的原生
  owner，不是共享万能执行器；复用 Platform 的线程/等待机制，普通用户不指定 worker。
- 单 Context 串行执行完整 region，多个槽可以排队且旧结果可跨槽保留；独立 Context 可并发。
  一个槽仍有 in-flight/结果借用时不覆盖。普通成功结果保有原始 MLX arrays，无 payload copy。
- bind 接受已经完成的 native 输入；原生 pending/外部 runtime 依赖没有被证明时明确拒绝，
  不偷偷 host wait 或从统一内存推导同步。输入不可在使用期间被其它持有者修改。
- 用户 region 是 pure GPU callable，不能捕获外部可变状态、CPU primitive 或另一 stream 的执行。
  冷准备检查可见 native graph 的 device/stream，并用独立输入暖机；custom 的真实隐蔽副作用是
  专家扩展合同责任，不默许替它 warmup。无可靠隔离的回调不进入这条 pure 编译路线。
- 接纳后即使 native 只执行了一部分，也保留 Completion、输入与输出到 checked drain。
  array availability 不是成功终态；失败不发布可读取 lease。取消不提前释放 GPU 使用中的资源。
- close 停止接纳、排空已接纳工作并 join owner；Context 析构有明确 drain 成本。
  OwnedResult/Completion 析构不隐式等待；Program/Invocation/结果可以比 Context 句柄长寿，关闭后拒绝新工作。
- managed 成本：固定 HYLE 计划/槽、每次一次外层提交、原生 MLX 的 arrays/metadata/result/temp 与
  cache-hit 成本、必要 mutex/原子/线程唤醒和一次 region checked synchronization 全部计入。
  不宣称零分配 recorded replay，不把 owner-thread 成本移出完整性能计时。
- 默认 MLX 成熟完整 region；HYLE custom 只在有完整收益证据的 device/格式/shape/SDK 域替换。
  保留冷 key 中实际硬件架构、OS/驱动 runtime、编译 SDK/部署与 MLX 版本事实；不在热态探测或计时。

## 具名源码事实与有限待证项

- 锁定 MLX v0.32.2 / `1f8e74e3f12f31365464a6867c6579f0e9b29d85`，MIT。
  本地 peer HEAD 更新，不切换它；按 pin 读源码。旧 Q4 探针只提供
  [格式/原生 lifetime 证据](../../../../evidence/correctness/compute-apple-route-review/README.md)。
- `mlx/compile.cpp:415` 为 thread-local cache，`mlx/backend/metal/device.cpp` 有 thread-local
  encoder。`CommandEncoder::synchronize` 在 waitUntilCompleted 后检查错误；array 的 ready 路径不能
  代替这个成功检查。由此选定固定 owner，避免在不确定 worker 上假设已经暖机。
- `mlx/transforms.cpp` 异常路径会尝试提交并排空 touched streams；HYLE 仍保留自己的已接纳工作到
  checked terminal，不能把该实现事实当任何部分提交均未发生。
- MLX CPU/Metal/CUDA 独立开关；Metal-C++ 26、json 3.11.3、fmt 12.1.0 是 nested dependencies，
  A1 纳入同一锁/缓存/许可证。选择 Metal JIT 时，源码嵌入库，冷暖机必须覆盖已编译 shape；
  若选预编译 metallib，必须另证明 relocation 资源发现，不能依赖维护者的绝对路径。
- 现有 Graph 公开只读结构足以降图；R2 affine Q4 定义是 `q * scale + bias`，不是整数 zero-point。
  不因 Apple 新 factor 便利而重开 Graph/Compute schema。

### 审计裁决

采用固定 owner 线程与 MLX `compile`/`async_eval`/checked `synchronize`，不增加一条未经证明的
caller-thread 快路径。Direct native arrays/callable 是有意的 Apple 专属 C++ 接口；它不要求
Provider 排序或 profile，也不向共同头泄漏 SDK。Graph 从公开只读结构降图，共同 owner 无需重开。

A1 选择 MLX_METAL_JIT=ON：shader 源码嵌入 native 库，支持可重定位 static/shared，冷准备承担
实际 JIT 与暖机。CPU/CUDA/Python/GGUF/safetensors/tests/examples 全部关闭。原生算法仍是锁定 MLX，
热态只有已暖形状；无 metallib 的维护者绝对路径依赖。此为待产品验证的构建选择，不是性能 PASS。

Apple 现已发布官方 [Metal-C++ Git 源码](https://github.com/apple/metal-cpp)，采用其 26 release
commit `f567ed836e4cbb85788c42115a2682bbe68097ee`；136 个 .hpp 与 MLX 所指定的官方 26 ZIP
逐字节一致。因此四个实际依赖都能使用现有 Git lock/cache 合同，无需重写公共依赖工具。
版本、archive/tree/license hash 与核对 receipt 见
[本轮 source admission](../../../../evidence/research/apple-r5-source-admission/README.md)。
不另开同内容的研究 native 构建；下一次真机编译直接验证 A1/A2 产品输入，旧 probe 不计产品完成。

## 七问与验证矩阵

1. 第三方：上列 pin/许可证与 A1 receipt；没有 CPU fallback，版本升级由本 owner 重审。
2. 功能：量化 FFN + 数值变换/custom 两种具名旅程；资源与失败 glue 由同一个路线提交 owner 承载。
3. 复用：Graph 只读接口、Compute ResultSlot/Completion、Platform Thread/Parker；Sched 通用 pool
   不保证同一 MLX 线程，因此不为形式复用而添加 pool/逐 kernel dispatch。
4. 成本：P0 先固定 whole-region/managed/线程账单，再同合同 P1；原生对照不同服务责任只签自然成本。
5. 边界：仅 C++；没有 C/Native FFI，不建立 Python 依赖或兼容旧接口。
6. 交付：完整源码、installed static/shared 都有真实 Direct/Graph 计算消费者，Apple-only 不取得 CPU/CUDA。
7. 易用：普通 typed op/DAG 与原生 region 分清，结果借用、ready 输入、close 成本与限制写在 API 附近。

| Gate | 适用风险 / 预声明证据 | 当前 verdict |
|---|---|---|
| scope | 两 owner、A1–A6、两个真实消费形状 | AUDITED |
| architecture | 依赖隔离、原生 full-region、正确输出 owner 与线程/完成源码审计已固定；产品回放待验 | AUDITED |
| correctness | Air + mini 真 Metal；Q4 独立解码 oracle、变化输入、非法域、两槽、lease、故障、并发/close | UNPROVEN |
| delivery | 两 Mac source/static/shared 与 relocation；Windows/WSL Apple OFF 无探测；public headers | UNPROVEN |
| usability_docs | 普通 Direct/Graph sample、默认规则边界、native custom、成本和误用 | UNPROVEN |
| red_team | 候选后的独立审阅视角；不用未授权 subagent，检查失败/并发/数值/交付反例 | UNPROVEN |
| engineering_performance | P0、Air 冻结候选的适用 P1；host ASan/UBSan/TSan 按实际可支持边界说明 | UNPROVEN |

MLX 自带 USE_TSAN 在 macOS 拒绝配置；须核对 HYLE host 插桩与 vendor 边界，不能把未跑 TSan 写 PASS，
也不能据此无声把整条并发检查列为不适用。故意 GPU hang/device loss 不注入，相关限制如实记录。
