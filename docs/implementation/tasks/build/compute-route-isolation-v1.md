# R1：Compute/Graph 路线选择与交付隔离

<!-- PROJECT UNIT REVIEW: AUDITED -->

这是 build/package 横向施工卡，不增加模块身份或第二套 registry。
属于[四路线基础重构 R1](../compute/four-route-foundation-refactor.md)；2026-09-15 已获实施授权并完成定向审计，尚未签交付通过。

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

当前为 **AUDITED**。本轮新 goal 已建立；不恢复旧 goal/百分比。范围与 RI1–RI5 固定后才改构建。

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

首版登记验证：`validation-select.py run --base 98778b55 --candidate WORKTREE` 选择 docs/closure，
结果 PASS，其余 94 项跳过或复用；`git diff --check` PASS。未构建、未运行 runtime 或 P1，registry
保持原样，不进行虚构模块关闭。
