# R1：Compute/Graph 路线选择与交付隔离

<!-- PROJECT UNIT REVIEW: CLOSED -->

这是 build/package 横向施工卡，不增加模块身份或第二套 registry。
属于[四路线基础重构 R1](../compute/four-route-foundation-refactor.md)；2026-09-15 已获实施授权并完成定向审计。
2026-09-16 的不可变候选已完成具名 native 验证与七门；本 R1 横向施工卡 CLOSED，后续计算基础仍待 R2–R7。

## Closure card

- 实施 baseline：`ec142a92e00e27a73abbae4f040b6b361465e9a7`；首版规划 baseline 为 `98778b55`。
- trigger：[六项复核](../compute/six-point-architecture-review.md)确认 Compute/Graph façade 固定包含
  CPU backend/Native roots，不能交付不带 CPU 模型 kernel 的独立 GPU 产品。oneDNN OFF 当前有效，
  不是本轮要修的下载故障；未实现 Apple/Vulkan 也不通过空 target 冒充。
- normal consumer：源码或安装 SDK 用户只选择 `compute`/`graph` 产品，再选实际计算能力；例如
  Windows/Linux CUDA-only 不取得 Compute 的 CPU Native 模型 kernel。纯 CPU 用户不发现 CUDA SDK。
  必要 host 基础设施可以使用 SIMD 等底层能力，这不等于 CPU 参与模型计算；不为“零 CPU 依赖”复制
  或禁用这些能力。
- 默认实现合同：按[既定优选规则](../../../architecture/compute-design.md#41-开发时验证-best-of准备时应用规则)
  交付正常所需的已验收实现；禁用 Provider 属专家裁剪，不要求用户手选哪家更快。R1 管候选闭包，
  R2/R3–R6 才实现默认组装/规则绑定；不能以本卡只修构建为由遗漏这项产品交接。
- 正交 consumer：只使用 tensor/operation/扩展合同的无 backend 底座不冒称有执行设备；非 Compute
  产品保持原依赖。测试夹具可以取得 CPU，但不污染 tests-off SDK。
- scope：用户能力选择、中央 roots/真实 link/公开头/install/feature 投影、直接 package consumers，
  以及有限旧资产/新 C++ 使用形状映射。runtime API/算法改写属于 R2–R6。
- non-goals：修改 CUDA/CPU kernel、重写公共 resolver/package/selector 框架、改变无关产品默认值，
  升级依赖、下载系统 SDK、引入 HIP/模型 CPU+GPU 联合计算或 Native FFI，创建 MLX/Vulkan stub。
- 成本：P0，只有 configure/install 的依赖投影，不增加 runtime metadata、dispatch 或 symbols；P1
  语义上不适用于仅构建隔离改动。若为了过门必须改 runtime，停下归属到对应产品卡，不能沿本卡施工。
- 调查预算：现有 resolver/option/install/selector 的一次定向审计，最多 30 分钟；有限实现预算一个
  工作时段（最多 4 小时），包含一次有依据的纠正和适用验证。外部条件阻塞或超预算按既有协议裁决，
  不以扩大本卡或无限重测处理。构建最多 2 jobs，重型验证串行，不与用户其他构建争抢。
- 本轮固定 RI1–RI5；CPU/CUDA 的交付重开另由 [backend delivery 卡](../compute/compute-backends-v4-route-delivery.md)
  经过 SCOPED/AUDITED，不修改其运行时。

## 已核对的事实

| 位置 | 事实 / 影响 |
|---|---|
| `cmake/HyleComponents.cmake` 的 product declarations | Compute 与 Graph 固定列出 CPU/native、CUDA 及 oneDNN roots；真正 factor 构造/依赖仍由同一中央图管理 |
| `src/compute_backend_cpu/CMakeLists.txt`、`src/compute_provider_native/CMakeLists.txt` | CPU runtime 与 Native 算子都是无条件 compiled；Native 又取得 SIMD/fat kernels，不能用运行期“不调用”代替交付裁剪 |
| `cmake/HyleComputeDependencies.cmake`、两个 oneDNN factor | oneDNN 默认 OFF，跳过下载且有 OPTIONAL_ABSENT；ON 是 x64 SEQ ukernel，不是偷偷创建第二线程池 |
| CUDA factor / package registry | 已有 OFF/AUTO/ON 和 system SDK `FIND_DEPENDENCIES` 机制；先复用，不另造下载器或第二套开关 |
| `HyleComponents` 的 TEST_DEPS、`HyleConfig`/Package | 测试选择与产品 link 分离；source 和 installed 必须投影相同 feature 闭包；不能只修顶层 build 的列表 |
| 当前 registry | CPU/CUDA runtime 为 CLOSED，其余相关 owner 多为 AUDITED；ggml CPU BLOCKED、compute_sched QUEUED 保持，Mem v10 另有未结尾项 |

## 候选选择与有限 FIX

这是审计后固定的有限 FIX，不是已实施结论。只允许为下列消费能力修改相关机器合同，不能用本轮重新
设计所有 CMake 选项。新事实若需要改变范围，先按协议裁决。

| ID | 候选动作 | 验收 |
|---|---|---|
| RI1 | 提供 `HYLE_COMPUTE_CPU=ON/OFF`，ON 保留普通 CPU 产品，OFF 排除 CPU runtime/Native 和 CPU Provider roots。沿用 CUDA/oneDNN 实有选择，不引入单选路线枚举 | 选择表达是否交付 CPU 计算，而不是实际模型 offload；用户不需 HYLE_DEV_FACTORS；未选 Compute/Graph 不触发其依赖 |
| RI2 | 在中央声明/既有投影中表达路线及默认实现 roots；正常发行与专家裁剪明确，源码/头/install/features 同源 | 规则与真实实现闭包一致；不假启用未验收 Provider；OFF 不是 stub/隐藏 link；无 backend 模式明确无计算能力 |
| RI3 | CPU OFF + oneDNN ON 是矛盾配置，明确拒绝；oneDNN AUTO 在无 CPU 能力时关闭而不探测。CUDA OFF 不探测 Toolkit；CUDA-only 不带 Native | 正/负 configure、包内真实头/target/运行库闭包一致；CPU OFF 不禁止必要 host 控制/基础库调用 |
| RI4 | package 样例按实际 feature 选择正常旅程；无 backend/独立 CUDA 交付反例；给各路线默认组装样例留真实交接 | 三交付一致；不靠私有 factor/build tree；专家手写 Provider 列表不能冒充后续默认 best-of 验收 |
| RI5 | 定向测试和指南更新，形成旧资产保留/改写/删除 owner 清单 | 本卡只清选路/build 问题；不把 CUDA runtime 可用写成量化 provider 已完成，也不为计划创建不存在选项 |

拟新增 CPU 开关的默认 ON 是当前“未指定设备时仍有可用 CPU 计算”的产品选择，不是承诺兼容旧 API。
Apple/Vulkan 真正接入时根据各自 runtime 裁决其 feature 和默认规则；这里不先登记空 flags/targets。
多路线可同时构建，但一个具体程序只绑定所选计算路线；CUDA-only 配置显式 CPU OFF。
oneDNN 默认 OFF 只是已核对的现状；后续按 CPU 纵切资格与默认规则确定发行组合，不把这个历史默认
固定成用户必须手动选库的长期形状。默认改变和实际 Provider 的资格分别验收，不先假签可用性。

进入实现前须确认中央图是在 feature 判定前还是后解析，并对选中、缺席与 TEST_DEPS 分别验证。
选择结果不得反向修改公共头布局；如果改动某 CLOSED factor 的 CMake/公开交付合同构成产品变化，
先给该 owner 建立合法 reopen record。只动横向胶水也必须通过 selector，而非用“仅 CMake”绕过 owner。

## C++ 使用形状边界

只固定两个用户任务：直接计算的 prepare→run→checked completion/result；Graph 的 build→compile→
run-slot→repeat。具体路线创建真实资源，不要求 MLX 假借一个没有使用的 HYLE Queue。R2 再用可编译
样例固定新函数/类型；本卡不添加假 API 声明或测试专用公共入口。

普通源码消费者设置产品和真实 feature 后链接 `hyle::compute/graph`；installed 使用同名 target 和
包公布的 capability，不能在 consumer 侧定义一个宏让未编译的 backend“出现”。native 专家头只属于
真实已选路线。三交付语义一致，无 C 入口、无旧 ABI 兼容负担。

## 预声明验证与适用性

先由 change-impact selector 对最终 diff 给出真实 owner、直接消费者与 package 门；本轮不执行
all-components build、CI 或 P1。下表是实现后的门，尚未运行，不能从本次文档检查推导 PASS。

| Gate | 具名覆盖 |
|---|---|
| configure / component graph | CPU-only；无 backend；CUDA-only；CPU OFF + oneDNN ON 拒绝；AUTO/OFF；tests-on 的 CPU fixture 与 tests-off 产品分离；无关 core root |
| package | Compute/Graph source、installed static/shared 与 relocation；selected feature/headers/targets/dependencies 精确，OFF 不留下假能力 |
| 平台 | Windows + WSL 的 CPU 与真实 CUDA；macOS 的 CPU/无 backend 与无 CUDA 探测；Apple/Vulkan 实际 adapter 在 R5/R6，不在本卡伪造 |
| P0 / hygiene | 无产品 runtime 字节/算法变更；无额外 metadata/call；逐头/导出/许可证与 docs/closure/diff |

可复用现有入口（实现前按实际 diff 与新样例固定参数，不另建通用仪器）：

```text
python scripts/validation-select.py plan --base 98778b555f25acc172722efa7ab8a9911032c520 --candidate WORKTREE --output .scratch/validation/compute-route-isolation.json
cmake -DHYLE_SOURCE_DIR=C:/public/project-hyle/hyle -DHYLE_SMOKE_ROOT=C:/public/project-hyle/hyle/.scratch/compute-route-isolation -P cmake/HyleCiComponentContract.cmake
cmake -DHYLE_SOURCE_DIR=C:/public/project-hyle/hyle -DHYLE_SMOKE_ROOT=C:/public/project-hyle/hyle/.scratch/compute-route-isolation -DHYLE_SMOKE_COMPONENT=compute -P cmake/HyleCiSelectedComponentSmoke.cmake
cmake -DHYLE_SOURCE_DIR=C:/public/project-hyle/hyle -DHYLE_SMOKE_ROOT=C:/public/project-hyle/hyle/.scratch/compute-route-isolation -DHYLE_SMOKE_COMPONENT=graph -P cmake/HyleCiSelectedComponentSmoke.cmake
```

这些 smoke 会清理自己的子目录；执行前验证 resolved scratch 位于本任务专用目录，没有其他会话缓存。
WSL/macOS 用各自完整 canonical 源码副本和专用绝对路径，不原样运行 Windows 路径。平台、generator、
feature 与 job 数按真实环境固定；上面普通 smoke 命令不独自证明全部新 feature 组合。

## 七问与当前交接点

第三方版本不变；功能是路线交付隔离而非算子扩展；复用中央图/SDK/package 机制；仅 P0 构建成本；
FFI 仍不提供；三交付和易用性按上表验证。C++ runtime correctness/P1 不因本卡规划签新资格。
发现 runtime/API 修复需求时交对应 owner，不把正确性未知改名为本卡已解决。

## 实施审计（2026-09-15）

实施前为 **AUDITED**。本轮新 goal 已建立；不恢复旧 goal/百分比。范围与 RI1–RI5 固定后才改构建。

- `HyleComponents` 先算 roots/TEST_DEPS，顶层之后才发现 vendor/SDK。CPU 是确定的产品能力开关，
  在中央产品 roots 展开后、闭包遍历前裁剪；维护 factor 显式选择与产品 roots 分开。
- CPU OFF 删除 CPU backend/Native/两个 oneDNN roots，避免 TEST_DEPS 把 CPU 算子带回产品。
  oneDNN ON 矛盾诊断和 AUTO 禁用由现有 owning dependency 配置承担，不新增 SDK 探测器。
- 已有 OPTIONAL_ABSENT 的 factor 仍可供内部图检查，但不进入用户 façade、installed targets/headers。
  只调整既有投影的缺席分支，不重写 resolver、selector、closure 或性能仪器。
- CPU factor 发布存在性宏与 package feature；宏只控制 consumer 选择真实入口，不改变公共对象布局。
  CUDA ON 的 SDK/flags/runtime 维持既有合同；OFF/AUTO 缺席的 native 头与空 target 不交付。
- package 样例分开 CPU 计算、CUDA 既有资源 journey、无 backend 的逻辑描述验证。R1 的 CUDA
  fill/copy 仅证明当前资源交付，完整 CUDA 算子/Graph/custom execution 仍在 R4 验收。
- source/static/shared 使用同一显式 CPU/CUDA/oneDNN 集；保留现有 smoke 的默认 oneDNN 覆盖，
  允许卡内 CPU-only/no-backend/CUDA-only 的显式参数，不改公共工具选择策略。

| 当前绑定 owner | R1 有限范围 / 未完成责任 |
|---|---|
| `compute`、`graph` | 本卡接管产品 façade、样例、交付；旧 v15/v10 产品为待迁移 baseline，未签基础完成；R2 重新定界 runtime |
| `compute_provider_native`、`compute_provider_onednn`、`graph_provider_onednn` | 本卡接管选路/安装边界；现有 kernel/region 不改，R3 验收默认规则与完整运行 |
| `compute_backend_cpu`、`compute_backend_cuda` | 独立 backend delivery 卡，只因公开交付改变重开 |

`compute_ops`、Mem v10、ggml BLOCKED、compute_sched QUEUED 均保持原状态。R1 横向完成不关闭仍有
基础 MUST 未验收的 Compute/Graph/Provider；交接时将它们重新绑定后续包的有限卡。

旧资产与自然 C++ 形状按 [七包迁移表](../compute/four-route-foundation-refactor.md#旧代码去留与-owner)
逐项归属，无 API/算法移除。无产品 runtime 改动的 P0 通过条件为相对实施 baseline 的源码/头 diff 为空。

首个 implementation commit 为 `d4006bf7`。selector 的 broad configure 被无产品实现的 ggml 审计身份
阻塞；另立 [VI1 工具卡](validation-audit-identity-v1.md) 定向纠正分类，不改变 ggml 恢复条件。
组件合同必须使用仓库外的专用 `C:/hyle-scratch/r1-20260915-d4006bf7` 根，避免其共享源码缓存
违反既有 cache 位置合同；首次仓库内 scratch 的环境失败不签产品失败或 PASS。

`8158d74d` 的 Windows factor/header 与组件合同通过；源码交付发现 Visual Studio 的 DLL staging
跨目录顺序缺陷，交由独立 [staging v2 卡](build-runtime-staging-v2-vs-order.md) 定向修复，
不通过重复构建或手动 copy 掩盖失败，受影响交付仍待重验。

首版登记验证：`validation-select.py run --base 98778b55 --candidate WORKTREE` 选择 docs/closure，
结果 PASS，其余 94 项跳过或复用；`git diff --check` PASS。未构建、未运行 runtime 或 P1，registry
保持原样，不进行虚构模块关闭。

## 不可变候选与本包裁决（2026-09-16）

产品候选为 `37dc8b542d52361cfa7b6cc176dc65555941e48d`；完整平台结果、原始日志、命令与失败尝试见
[R1 证据](../../../../evidence/correctness/compute-route-isolation-v1/README.md)。实施授权 baseline 为
`e469747c`，产品源码不变比较仍以 `ec142a92` 为准。此前“待重验”均是当时记录，以下是当前结果。

| FIX | 实际结果 |
|---|---|
| RI1 | `HYLE_COMPUTE_CPU` 默认 ON；OFF 在产品闭包遍历前排除 CPU/Native/CPU oneDNN roots，维护 factor 选择保持独立 |
| RI2 | CPU usage requirement/package feature 同源；缺席 CUDA/oneDNN 不再导出假 target/native 头，未添加 Apple/Vulkan 占位能力 |
| RI3 | CPU OFF + oneDNN ON 明确拒绝；AUTO/OFF 与无关 root 不探测未选 SDK，CUDA-only 不带 CPU 计算实现 |
| RI4 | Compute/Graph 样例按真实能力执行；CPU、无 backend、CUDA 三类 source/static/shared/relocation 和自定义计算消费通过 |
| RI5 | 组件、包和 staging 定向反例通过；指南补明 native Linux CUDA、静态 archive 嵌入 shared factory 的 PIC 条件及 VS helper 调用目录 |

Windows selector **92 selected gates PASS、5 skipped**；显式平台验证 **38 cases PASS**（Windows 19、
WSL 12、Air 7）。selected 范围因公共 CMake/package seam 扩展，未额外启动 all-components、CI 或 P1。
skipped 保留原理由：ggml 空审计身份及无公共头、io_core/simd 未登记独立逐头套件、未声明自动 P1。
ggml 原 BLOCKED 与恢复条件不变。VI1 和 staging v2 有独立有限卡，不能把工具修复算作计算路线完成。

Linux 普通静态 SDK 的 EXE 消费通过；CPU custom shared factory 使用生产时启用 PIC 的静态 SDK。
原 non-PIC 失败和错误选中 Windows nvcc.exe 的失败均保留，没有改名为 PASS。纠正使用已有生产选项
和正确的 Linux SDK 环境，无产品 C++ 或默认 flags 改动。Visual Studio 首建问题以真实顺序修复，
六种 staging 合同与受影响真实交付重验通过。

| 最终门 | 当前 verdict / 依据 |
|---|---|
| scope | PASS：RI1–RI5 与 BD1/BD2 完成；runtime/默认优选/新路线仍归 R2–R6 |
| architecture | PASS：中央能力投影和路线隔离；无 stub、隐藏 fallback 或新增通用执行器 |
| correctness | PASS：候选的 selected factor/header/component 与具名平台消费；未将缺席能力签成计算可用 |
| delivery | PASS：三平台适用的源码、静态、动态、relocation、真实 CPU/CUDA custom execution |
| usability_docs | PASS：最终指南与证据链接通过 docs；AUDITED→CANDIDATE 的 registry 迁移通过显式 previous-ref 检查 |
| red_team | PASS：矛盾 feature、缺席依赖、首次/增量 staging、PIC 与 native SDK 反例均有真实结果 |
| engineering_performance | PASS（P0）：相对 baseline 的库产品源码/公共头 diff 为空，无算法、layout、常态分配/分发/同步增量；P1 不适用纯构建交付变化 |

实施/验证从 2026-09-15 22:18 开始，具名 native 验证最迟于 2026-09-16 01:34 完成，在四小时预算内。
候选后的证据、文档和状态写入不构成新产品候选，不重建、不重跑 P1。

统一脚本在候选上自检通过：Compute+Graph 全部实现 **31,135** 行（不含测试），全库自实现不含测试
**126,320**，全库代码 **689,555**。R1 未改变库实现行数；旧实现和文档不折算为新基础完成度。

本卡终局后停止 R1 产品修改。CPU/CUDA delivery owner 依次 CANDIDATE→CLOSED；共同 Compute、Graph
和三个现有 Provider 仍有基础 MUST，保持未完成，R2/R3 必须重新绑定已审计的有限卡后才改 runtime。
本卡届时只作为完成的交付证据，不授权继续旧队列。reopen 仅限可复现的路线/SDK 泄漏、正常包消费
回归或新的公开交付合同变化，不据此全库重审。

候选记录检查：`check-docs.py` PASS（323 Markdown、1236 相对链接）；
`check-module-closure.py --previous-ref 37dc8b54` PASS（39 components、schema 4）；
`git diff --check` PASS。selector 以 `37dc8b54` 对 WORKTREE 仅选择 docs/closure，94 项跳过/复用，
上述两个 selected gates 已执行；没有因证据/文档/状态变化重建或重跑 P1。

终局：候选/证据提交为 `4d87e963`，CPU/CUDA delivery 从该提交的 CANDIDATE 进入 CLOSED，
以其为最终状态迁移的 `--previous-ref`。归档 LOC JSON 在首次 staged diff 检查发现 CRLF 格式，
终局仅规范为 LF，解析后的全部数值不变；最终完整 diff 检查包含新证据。R1 有限 FIX 已完成，
停止本卡产品施工，按七包顺序进入 R2 的新审计卡。
