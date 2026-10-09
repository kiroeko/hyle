# Compute/Graph 构建、冷路由和组合执行设计轮

> 2026-10-09 用户授权的新独立 goal；baseline `17ff6555c7af0a8bd44d648209c40f53de758420`。
> Windows canonical 工作树启动时干净。此轮交付入口统计、设计与有限后续施工包，
> 不改产品源码、不重开 CLOSED 模块、不宣称新入口或联合计算已经交付。

## 范围与停止条件

正常旅程：用户选择 CPU/CUDA/Apple/Vulkan 能力集合，源码嵌入或消费已安装 C++ SDK，
由普通 Direct/Graph 入口冷绑定合格实现。新增需求是 CPU+一个 GPU 的资源/片段组合，
主纵切为引擎指定的推理子图，反偏置纵切为 CPU 数据准备→GPU 数值变换→CPU 归约。
HYLE 不解析模型、不制定层/专家/KV 放置，不扩算子目录。

本 goal 的四阶段如下；它们是设计交付，不是第二份 module 状态：

| 阶段 | 输出 | 当前结果 |
|---|---|---|
| I1 | 实际入口、owner/consumer、隐式能力变化与可复现统计 | 完成，74 项；见下表和 JSON |
| I2 | 唯一构建入口、完整依赖、SDK/运行资产、迁移与默认 | 设计完成；产品未实施 |
| I3 | 五种路由职责、native 组合、资源/终态和引擎边界 | 设计完成；三种组合未资格 |
| I4 | 固定有限施工包、反例复核与文档验证 | 完成；七个未启动实现包，静态门 PASS |

预算：一遍源码入口扫描、一遍针对发现问题的 peer/反例复核、一遍文档检查；
不恢复历史算子扫描/性能队列。新反例只定向核对，不循环推翻整体架构。
全部设计和适用静态门完成后提交推送并结束本 goal；后续实现按下述包重新绑定 owner。

## I1：实际入口统计

完整 [inventory.json](../../../../evidence/architecture/compute-entry-design/inventory.json) 保存每项的
baseline 源码行、文件 SHA256、受众、逻辑 owner、consumer、能力影响和设计裁决。
`build_system` 是共享 build seam 的逻辑归属，不是假造的 registry component；
聚合 Provider 家族也不替代实现前的实际 factor owner 审核。

| 入口族 | 数量 | 统计口径 |
|---|---:|---|
| 当前能力开关 | 6 | CPU/CUDA/Apple/Vulkan + oneDNN/Torch CPU |
| 产品/维护 factor 选择 | 2 | HYLE_COMPONENTS、HYLE_DEV_FACTORS |
| 交付开关 | 2 | HYLE_STATIC、HYLE_INSTALL |
| 依赖位置 | 6 | source cache、build dir、CPU SDK root/archive、CUDA archive/dir |
| 工具链/离线轴 | 7 | 既有标准编译目标/SDK/编译器/部署版本/离线注入族 |
| 维护配置 | 6 | tests、bench、werror、sanitizer、trace、SIMD exact target |
| 集成工作流 | 8 | 两个路线 preset、两宿主 wrapper、source/install、两 staging helper |
| 普通路线入口族 | 8 | 四 Direct + 四 Graph；CPU Graph 接 common compile |
| 共同语义和生命周期入口族 | 8 | operation、descriptor、profile、workload、prepare、compile、replay、owned result |
| 专家/扩展入口族 | 15 | 原生资源、Provider/kernel/region、硬约束、预算、显式 transfer 等 |
| 内部绑定接缝 | 6 | 四策略入口、ggml recipe、same-Device guard |
| **总计** | **74** | 不把重载/引用/别名/逐算子 schema/派生宏计为新入口 |

可复现命令（只读取固定 Git baseline，输出到指定文件，不配置/构建）：

```text
python evidence/architecture/compute-entry-design/inventory.py --output .scratch/compute-entry-design/inventory.json
```

### 有限发现与裁决

| ID | 源码事实 | 裁决 |
|---|---|---|
| E1 | 六个能力开关可独立组合；Torch CPU OFF 会撤回 ATen 与低精度承载 | FIX：普通路线完整能力由平台闭包派生；不给 Provider OFF 入口 |
| E2 | compute-cpu/cuda 有 preset；Apple/Vulkan 无对称普通 preset；手动 CUDA ON 不取消默认 CPU | FIX：一个 route-set，四 preset 只代填集合，不第二次解释路线 |
| E3 | generic staging 处理 DLL/loader，Apple helper 另管 metallib | FIX：一个应用 helper 按实际链接 owner 组合，复用原 registry |
| E4 | 单 CPU SDK 用较新 ggml，含 GPU 的 SDK 共用另一合格 pin；性能 recipe 条件随之变化 | KEEP+显式 identity：mixed SDK 不继承 CPU-only WIN；版本变化不能当同一候选 |
| E5 | common CompileEnvironment 的所有 Context 必须同一 Device；Apple/Vulkan native Program 另有类型 | KEEP：原单路线 guard；新增粗粒度组合 seam，不能只删除 guard |
| E6 | 模型画像/任意端口条件/版本/资源规则已可表达；submit 不重新搜索 | KEEP：原冷 matcher 和成熟默认；不重写一套路由 IR/autotuner |

核心源码： [component selection](../../../../cmake/HyleComponents.cmake)、
[ggml recipe](../../../../src/compute_ggml_runtime/CMakeLists.txt)、
[Graph compiler](../../../../src/graph/executable.cc)、
[native Apple Graph](../../../../include/hyle/graph/apple.h)、
[native Vulkan Graph](../../../../include/hyle/graph/vulkan.h)。
当前 native Apple/Vulkan 编译域还明确拒绝 symbolic/effect/control-edge 等图；
组合设计须尊重这些能力，不能从公共 IR 的存在推断所有路线承载全部语义。

## I2/I3：设计交付

长期设计只在 [compute-next-stage](../../../architecture/compute-next-stage.md) 维护，
继续复用 [当前冷路由合同](../../../architecture/compute-routing-system.md)。
本卡不复制第三方依赖清单、算子目录或第二套 component DAG。

有界 mature peer 核对：本地 source cache 中 lock pin
`ggml-456172ec733a-41b7e10fd424/include/ggml-backend.h:318–351` 及
`examples/simple/simple-backend.cpp:63–64,108,149–151`：scheduler 可以分图、reserve/alloc、
拷贝和计算，backend 数组顺序有优先语义，reset 使旧 tensor allocation 失效。
因此不能直接公开数组排序给普通用户，也不能直接使用 reset 做仍有 lease 的重放。
优先保留其完整 region，HYLE 包装固定资格/资源/终态；这是源码设计依据，尚无混合运行资格。
[固定上游 header](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/include/ggml-backend.h)。

[Accelerate cpu_offload](https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.cpu_offload)
明确区分 CPU 保存参数和 execution device 执行，用于反证“host 存权重必选 CPU 数学”的配置。
上述资料只支撑分工设计，不借框架能力代签 HYLE。

### llama.cpp 源码对账

用户明确要求参考后，读取官方 repository 的固定 commit
`e60eff95fda7b6e46bf402a8e9f1e86d88ab29c8`，仅四个有关文件。
[llama-reference.json](../../../../evidence/architecture/compute-entry-design/llama-reference.json)
记录 URL、完整 SHA256、逐行原文与设计观察；不是新产品依赖或混合运行证书。

| 固定源位置 | 真实行为 | 本轮设计结论 |
|---|---|---|
| common/arg.cpp:2763–2778,2960–2966 | cpu-moe/n-cpu-moe 写 tensor buffer override；op-offload 另有开关 | 存放与计算不合并成一个枚举；模型层数/角色/CLI 留引擎 |
| src/llama-model.cpp:1614–1626,1925–1928 | 按层配置 device/buffer；权重 usage 提供 locality 信息 | consumer 给 placement，库验证真实 resource；不能用模型名代替 tensor 合同 |
| src/llama-context.cpp:419,650,2750–2757 | CPU 可借 device host buffer；多 backend scheduler；层末/norm placement 抑制转移 | 复用成熟 region/native host backing，whole-plan 计入跨边界成本 |
| ggml/src/ggml-backend.cpp:1110–1122 | host 权重 op 仍可能按 supports_op/offload_op 移到更高优先 backend | CPU-required 必须固定，不能将 vendor 顺序当 HYLE 普通 ranking |
| src/llama-context.cpp:2655–2736 | 专家 ID 就绪后按使用区间传权重；model cache 由 context 持有 | 保留完整 MoE region；数据选择与冷 Provider 选择不同，不自写数学/cache |
| ggml/src/ggml-backend.cpp:651–670,1854–1893 | 同 layout copy；无 native async 时同步再阻塞；复用 buffer 需前序 event/完成 | host wait 和暂存寿命如实报告，不能把 async 名称当非阻塞保证 |

这些是入口到实际 scheduler 的源码链，不只读命令帮助。HYLE 仍须为自己的 lock pin、
native资源、语义/取消/lease 和三交付做后续资格，不能照抄 master 的热分图行为或继承性能证书。
静态完整 region/明确阻塞的组合 run 先收口；非阻塞通用桥须另有进展 owner，当前七包不暗造后台线程。

### 独立交叉复核

一遍重新从 consumer/反例检查设计，不以新接口示意证明产品支持：

| 反例 | 设计裁决 |
|---|---|
| 只选择 CUDA，但机器恰有 CPU/Torch | 构建不带 CPU 数学；运行不扩大允许 endpoints |
| host 权重＋GPU 数学，没有 CPU Provider | 合法的 GPU 路线旅程；host 存储不等于 CPU 数学 |
| SDK 含 Apple，但用户代码只调用 CPU | staging 以实际 target 闭包判断；聚合链接可需要全部资产，不靠扫描调用猜删除 |
| 同一模型内 Q4/BF16/F32 混合 | profile 只说明制品；每 port 的描述、物理编码和 native admission 才是数值合同 |
| CPU-only exact WIN 遇 mixed ggml/compiler/flags | recipe 不匹配则成熟合法默认；不移植未验 WIN 或禁掉正常 op |
| 明确 CPU-required region，但 ggml 愿意 GPU offload | 固定约束必须由 Provider 资格证明；不允许 vendor 顺序覆盖 |
| expert IDs 每轮改变 | 已绑定程序可处理输入数据；不重新选择 Provider，模型 cache/eviction 留引擎 |
| native async copy 实际退到阻塞；旧 output lease 未释放 | 报实际等待/费用；终态之前 backing 保活，slot 不覆盖；不能直接套 scheduler reset |
| 当前 Apple/Vulkan 不承载 effect/symbolic 子图 | 联合 compiler 拒绝或选择真实支持片段；不删除语义以换取执行 |

结论：没有新增两套路线事实/选择器，没有普通 Provider OFF/排名菜单，没有通用 Queue/逐 kernel
执行器或隐式进展线程；完整成熟 region 优先。同步桥/借用 backing/单 CPU+GPU 是明确首版限制。
未来 adapter 必须复用已有 checked 输入/错误边界，不能把 vendor assert 当正常坏输入诊断。

## I4：后续有限施工包（当前均未启动）

共享 build seam 没有独立 component；C1/C2 的实际 product owners 由 selector 和中央图确定，
相关 Compute/Graph/Provider owners 必须另有新卡、baseline、SCOPED→AUDITED 和有限 FIX 后才改产品。
新需求与公开交付变化是 reopen trigger；历史 CLOSED 卡不自动恢复。

| 包 | 顺序/归属 | 固定改动与验收边界 |
|---|---|---|
| C1 | 首先；shared build seam/所涉路线 owners | 实现 route-set 派生完整闭包、四 preset、旧六开关迁移错误；验证空/未知/重复/不支持平台、无路线产品不 probe、维护隔离；不改数学 |
| C2 | C1 后；package/assets/所涉 Provider owners | 统一 stage helper、SDK 路线/recipe 查询与 requirements； source/static/shared 同合同、DLL 同 exe、原生资产、安装搬迁/增量更新/离线缺件；不造下载器 |
| R1 | C2 后；compute/graph/四 route owners | 校准 build identity→native facts→普通冷诊断，消除默认菜单/未知版本误标 WIN；保留 expert 扩展和已有 matcher；检查单路线无 CPU 偷跑、hot 无重选 |
| C3 | C1/C2/R1 接线结束后一个候选 | 批量四单路线和 CPU+三 GPU SDK 的真实适用平台三交付/头/资产/OFF 检查；重用身份相同数学证据，recipe 变化只跑受影响接缝；mixed 编译不签混合运行 |
| H1 | 上述完成后；graph/compute/CPU/CUDA/ggml owners | CPU+CUDA 首个完整静态多片段旅程和 host 权重分批旅程；成熟跨 backend region 优先，外围组合仅粗粒度；真实数学/边界转换/生命周期/配额/取消/三交付；不首次扩目录 |
| H2 | H1 的边界合同稳定后；graph/CPU/Apple owners | CPU+MLX/Metal 粗粒度桥；先明确 host copy/等待，import 需 exact owner/可见性证据；复用完整 MLX/ggml-Metal region；MacBook Air 实际资格 |
| H3 | 同一边界合同；graph/CPU/Vulkan owners | CPU+Vulkan 粗粒度桥；实际设备 buffer/staging/fence；Win/Air 实 GPU，WSL 无 GPU 则 runtime 保持未资格；不以 llvmpipe 签 GPU |

C1/C2/R1 集中接线后才启动 C3，共享 dependency/cache/producer 工程。
H1–H3 各自接线结束，按一个不可变 candidate 批量验证，不逐个边界/算子造新 build。
固定低并行 build 2 / test 1；host/GPU 重型验证串行；源码只从 Windows 精确同步。
设计没有计时，因此不称新 SOTA；新数学或完整边界成本确受影响才预声明有预算 P1，
只有一家合格成熟 Provider 时按正确接入和 P0 成本收口，不无限追加性能姿势。

### 明确待办

`DEFER-GRAPH-INCREMENTAL`：用户已要求未来支持通用增量依赖图，本轮登记但不调研/实现。
未来先写单独设计/准入卡：稳定 node/input version、依赖失效、请求 root 求值、结果 owner/配额、
失败/取消/更新 generation；计算节点可借当前 Graph，调度借 Sched。
候选语义 owner 为 Graph 的独立可选依赖层，尚未定 factor/公开 API。
触发条件是用户启动该独立任务；不得创建空 target 或给当前 decode 热路径加 dirty metadata。
CAD 几何 kernel、建模业务和推理 KV/cache 策略不纳入；不改当前 57 个 CLOSED 记录。

## 本设计轮验收与终局

仅适用：inventory 可复现/anchor 检查、设计职责与负空间交叉复核、change-impact docs-only 计划、
check-docs、check-module-closure、git diff --check。产品编译、运行正确性、三交付、P1 未在本轮执行，
不为未来新能力写 PASS。原终局事实不被文档更新重测或改写。

终局固定以下命令：

```text
python evidence/architecture/compute-entry-design/inventory.py --output .scratch/compute-entry-design/inventory.json
python scripts/validation-select.py plan --base 17ff6555c --candidate WORKTREE --output .scratch/validation/compute-entry-design-plan.json
python scripts/check-module-closure.py
python scripts/check-docs.py
git diff --check
```

inventory 重新生成与归档 SHA256 相同：
`c525b4dc31d6a3c0f998dcf57c0632e714bfa48d8188ffbf1517dc03e8b84788`。
selector 仅选择 docs/closure，不启动 build 或 P1；实际 `run` 结果为 PASS：
docs=PASS、closure=PASS，132 项 skipped/reused 保留 selector 理由，product owners/授权问题为空。
结果保存在 `.scratch/validation/compute-entry-design-result.json`，执行命令：

```text
python scripts/validation-select.py run --base 17ff6555c --candidate WORKTREE --scratch .scratch/validation/compute-entry-design --output .scratch/validation/compute-entry-design-result.json
```

check-docs 实际通过（880 Markdown / 4207 relative links），check-module-closure 实际通过
（57 components / schema 4），git diff --check 通过。源码/公开头/CMake/preset/registry 与
baseline 的 diff 为空；本轮未启动产品构建或性能测试，不存在新运行能力的 PASS。

统一 `scripts/loc-report.py --module compute-graph --module-path <全部 compute*/graph* 根> --commit 17ff6555c --json --self-check`
统计结果见 [loc.json](../../../../evidence/architecture/compute-entry-design/loc.json)：**62,486 code lines / 268 files**，
包含 HYLE 公开接口/执行/资源/Provider 胶水和 src 内维护片段，不含测试/bench/第三方目录；
不是“自研数学算子行数”。本轮没有产品改动，数值保持。

**设计交付完成**：I1–I4 全部完成，74 项入口可复现，唯一配置/冷路由/基础库组合分工与
七个有限后续包已固定，增量依赖图已登记 DEFER。提交推送后结束本设计 goal，
不自动启动 C1 或增量图；全部七包产品实现及三组合运行资格仍未完成。
