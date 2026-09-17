# R5 Apple：MLX Metal 原生 region 与 owning result

## Closure card

- 状态：`CLOSED`。身份在 `32aade77` 以 `QUEUED` 登记、`8c2f6962` 进入 SCOPED；
  A1–A6 经 AUDITED 实施、不可变候选验证后完成下列七门；历史检查点按当时事实保留。
- candidate：`1636365bbe47779de24e84d7c7d1ed8088fe1af2`，tree
  `8e5c24179c53b203ba1509a8d5dff0b805548052`。后续文档/状态不改变产品候选或触发重测。
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
  A1 纳入同一锁/缓存/许可证。Metal JIT 的多数源码嵌入库，基础 kernel 仍有 metallib；
  冷暖机必须覆盖已编译 shape，并证明资源 relocation，不能依赖维护者的绝对路径。
- 现有 Graph 公开只读结构足以降图；R2 affine Q4 定义是 `q * scale + bias`，不是整数 zero-point。
  不因 Apple 新 factor 便利而重开 Graph/Compute schema。

### 审计裁决

采用固定 owner 线程与 MLX `compile`/`async_eval`/checked `synchronize`，不增加一条未经证明的
caller-thread 快路径。Direct native arrays/callable 是有意的 Apple 专属 C++ 接口；它不要求
Provider 排序或 profile，也不向共同头泄漏 SDK。Graph 从公开只读结构降图，共同 owner 无需重开。

A1 选择 MLX_METAL_JIT=ON：多数 shader 源码嵌入 native 库，冷准备承担实际 JIT 与暖机。
首次源码构建发现其少量基础 kernel 仍无条件生成 `mlx.metallib`，原“JIT 可免资源文件”判断撤回。
A1 因此同时保留该真实构建产物、安装与消费者邻置/relocation 验证；不能借默认绝对 build 路径过门。
CPU/CUDA/Python/GGUF/safetensors/tests/examples 全部关闭。原生算法仍是锁定 MLX，热态只有已暖形状。
两 Mac 原先只有 CLT，Metal CLI 的环境准备纳入本卡；不改 shader 算法或借 wheel 签源码交付。

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
| scope | 两 owner、A1–A6、量化 FFN 与数值/custom 两种真实消费形状 | PASS |
| architecture | 独立 MLX full-region、固定原生 owner；Graph 冷 lowering、原数组结果 | PASS |
| correctness | Air/Mini 真 Metal、15 项 native 用例；完整 Q4 oracle、失败/并发/关闭；Air host ASan/UBSan/TSan | PASS |
| delivery | 两 Mac 三交付/relocation；Windows/WSL OFF、逐头及 Windows selected batch 完成 | PASS |
| usability_docs | 普通 Direct/Graph sample、用户指南、默认规则与 native custom，真实三交付消费 | PASS |
| red_team | 候选后的独立源码审阅与已执行风险检查，没有新的产品 FIX | PASS |
| engineering_performance | P0、Air Direct/Graph 两 workload 原定 5% NI 全过；n15，严格方向全部 INCONCLUSIVE | PASS |

MLX 自带 USE_TSAN 在 macOS 拒绝配置；须核对 HYLE host 插桩与 vendor 边界，不能把未跑 TSan 写 PASS，
也不能据此无声把整条并发检查列为不适用。故意 GPU hang/device loss 不注入，相关限制如实记录。

## A1–A4 产品检查点（仍 AUDITED）

已实现固定原生 owner 与 Completion/result lease、typed Direct 三族、Graph 冷遍历与 dead-pure-node
清理、整段 MLX 编译及同 runtime custom Metal 映射。Graph 可在编译后释放，常量由 native Program
持有；重复执行不重新走 HYLE DAG 或 custom lowering。元数据驱动编译只需 shape/dtype，不要求
用户制造 dummy payload。MLX 生成的原始数组直接作为 owning result，不进行影子输出复制。

typed 域目前明确为 2-D F32 MatMul、unsigned affine Q4/group64、F32 SiLU 门控。Q4 的公开格式
是低 nibble 优先的 uint32 `[N,K/8]`，与其它 Q4 编码分开。Graph 接受固定 dense schema，拒绝符号、
资源效果和显式 control edges；不声称覆盖旧 Graph 的所有能力。native arrays 自身的 stride 处理仍由
MLX 承担，不把 Graph metadata 当作外部 buffer import。专家 native region 保留 MLX 更广的表达能力。

Context 冷创建保存实际 device/architecture、含 build 的 macOS runtime、编译 SDK/部署下限、
MLX pin 和 Metal compiler；没有把编译 SDK 当驱动版本。当前固定基线为成熟 MLX region，尚无
新 HYLE 替换优势或未测硬件的 SOTA 声明。

[检查点证据](../../../../evidence/correctness/apple-r5-foundation-checkpoint/README.md) 的 Air native
构建（WERROR）和 9 个 GTest 通过，Windows OFF 双头检查通过；只签这些局部结果。三交付、
普通 facade 接线、其余适用风险/插桩、mini 和最终 P1 继续按 A1–A6 执行，七门仍未完成。

## A5 生命周期检查点（仍 AUDITED）

新增 [5 项原生生命周期用例](../../../../evidence/correctness/apple-r5-lifecycle/README.md)，
Air 实际 14 个 GTest 全部通过：排队/执行中取消、丢弃后的保活、close 排空、晚到失败与 queued
失败、独立 Context 原生并发，以及 quota/bad_alloc/Metal 容量拒绝后的安全恢复。只使用测试内
MLX Primitive 边界，不给产品添加 fault hook；GPU 前缀是真 square，未用 mock Provider。

产品 C++、API 和正常成本没有变化，代码量仍为已统一核对的 Compute+Graph 33,815 行 / 116 文件
（`d900a8ce` 输入相同，不含测试、bench、third_party）。本检查点不签 Mini、host 插桩、最终 P1
或 R5 完成；其余预声明门继续保留未证明状态。

## A6 性能接线（尚未正式采样）

[有限比较合同](apple-r5-performance-contract.md)固定 Direct/Graph 对独立 MLX native expression
的同生命周期接入比较、裸 MLX NATURAL_ROUTE 成本、两个真实 FFN 原型、完整 oracle、正反序/
分配顺序/placement、A/A 与原 5% NI/n15→最多31预算。默认仍为相同成熟 MLX 实现；A/A 或 NI
不计为 HYLE 算法胜利。主卡适用验证与停止条件不变，P1 只在不可变候选上启动。

性能接线暴露的 benchmark CMake 误选由独立[工具卡](../build/validation-benchmark-build-inputs-v1.md)
按有限预算修正；不改变本卡原有产品交付与性能验收责任。

[完整 FFN 数值检查点](../../../../evidence/correctness/apple-r5-ffn-preflight/README.md)已通过四个
实际可执行文件的 M1/M16、变化输入、独立 packed-weight double oracle，以及 Graph 冷拒绝/恢复。
预检纠正了一项会放过全零结果的周期 fixture 缺陷；最终数据与 F32 容差在任何计时前固定。
这些是正确性证据，不是 A/A、P1、Mini 或路线完成声明。

## 不可变候选与剩余验收

普通 Direct/Graph API、typed 域、原生提交与资源所有权、默认成熟规则、source/static/shared 形状及
性能口径已停止变化。Air 原生与三交付检查点已执行；Windows broad 的已选三交付仍在串行运行。
WSL Apple OFF 已有实际隔离/逐头证据；Mini、host 插桩、最终 A/A/P1 和七门汇总继续验收，不将未执行项写成 PASS。
如候选暴露具体产品缺陷，按 A1–A6 退回 AUDITED 后修正，不在 CANDIDATE 下改动产品。

[候选后的独立审阅视角](../../../../evidence/correctness/apple-r5-candidate-review/README.md)已检查公开入口、
失败/取消保活、并发关闭、Graph 冷/热边界、数值与比较责任、资源交付及用法，没有产生新的产品 FIX。
该源码审阅不替代尚未运行的 host 插桩、Mini 和最终性能门；候选保持不变。

## 最终平台与插桩证据

候选不变。[Windows selected batch](../../../../evidence/correctness/apple-r5-windows-selector/README.md)
已完成全部适用门；[Mini](../../../../evidence/correctness/apple-r5-final-mini/README.md)的 15 项
native 正确性、两个逐头检查和完整三交付全部通过。[Air host 插桩](../../../../evidence/correctness/apple-r5-final-air/README.md)
分别实际运行 ASan、UBSan、TSan，覆盖 HYLE host 实现及测试；MLX vendor/Metal shader 未插桩，
不能将 host PASS 外推为 GPU sanitizer。Windows/Air/Mini 的证据与 WSL OFF 共同补齐本轮平台/
交付义务。最终 P1 与自然成本已按冻结合同串行执行完成，见下列终局。

## 终局（2026-09-17）

[最终性能证据](../../../../evidence/performance/apple-r5-final-1636365b-m1-mac/README.md)保留 Direct/Graph
的 A/A、八分层 n15、controls、oracle、validator、原始观察值及 manifest。两入口 M1/M16 全部通过
原定 5% 非劣门，没有追加 n31 或性能纠正。每一严格 CI 均跨过 1，方向仍为 INCONCLUSIVE；
仪器 practical PARITY 不改写为 HYLE 算法优势。同合同两侧使用相同原生 owner 和生命周期，
独立 MLX region 为成熟完整基线；已实际执行的 caller-thread 原生诊断保留 NATURAL_ROUTE /
NOT_COMPARABLE，不签 NI 或排名。

当前 typed 域与 pure 固定 shape Graph 已在普通 C++ 入口、两 Mac 三交付中走通。常量、运行槽、
原生结果与 Context 的寿命/失败/关闭均有实际验证；用户无需 Provider 排序或 timing profile。
host 插桩通过不外推到未插桩的 MLX/Metal shader。不故意注入 GPU hang/device loss；ready 输入、
pure callback、单 Context 串行 owner 和 managed MLX 分配成本仍是明确合同。无任意外部
MTLBuffer/Queue 零拷贝桥、CPU 模型 fallback、M5/NAX 资格或未知设备 SOTA 声明。

产品/性能候选仍为 `1636365b`；交付候选 `cbedb837` 到它的 shipping source diff 为空，身份见
Mini/Air 交付记录，未用文档提交触发新构建。两个 manifest 的 `hyle-perf.py verify` 均返回
QUALIFIED/CURRENT/EXACT、路径校验通过；实际 product gate 由同时归档的 selector/raw 数据证明。
方向已登记到 [Compute/Graph 总账](../../evidence/performance/benchmarks-compute-graph.md)。

统一 `loc-report.py`：本候选 Compute+Graph 全部实现 **33,815 行 / 116 文件**；2026-09-17 工作树
为 **36,753 行 / 129 文件**，包含尚未验收的 R6，均不含测试/bench/third_party。LOC 不折算完成度。
R5 完成；R6/R7 与 Mem 依赖尾项仍按各 owner 推进，整个 goal 尚未完成。

停止本卡产品施工。reopen 限可复现正确性/寿命/并发/交付缺陷、真实 consumer 新能力或公开合同
变化、采用依赖升级及合格同合同 material 回归，不凭未知硬件或微小点值重开。
