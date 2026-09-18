# C1-Y：ggml Vulkan managed region 接入

## Closure card

- component / owner：`compute_provider_ggml_vulkan` / `/root`；原 CPU `compute_provider_ggml`
  的 BLOCKED 和恢复条件保留，本 identity 不承载 CPU/CUDA/Metal runtime。
- baseline：`d23a3dbf1c58a1e0a7675e8cc94e4b6626b4e578`；上位 [C1](llm-moe-c1-tensor-state.md) 与
  [唯一计划](llm-moe-coverage-completion-plan.md)。当前阶段以 registry 为准。
- trigger：用户要求先接成熟实现、核对缺口后才补 HYLE 算法；固定 ggml 的 Vulkan 算子/融合
  已能承担真实 C1 计算，但公开 API 缺同 VkDevice buffer bridge，context OOM 仍会 abort。
  [有界接入试验](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-region-admission.json)
  已证明窄桥接可执行原版 region，没有算法重写必要性。
- 主 consumer：普通 HYLE Vulkan Direct 与 Graph 的 RMSNorm/参数缩放、门控和 residual，使用
  本地2B的2048宽度、1/17/128行；稳定 topology/常量、变化输入、两独立运行槽、保留旧结果。
- 正交 consumer：F32 通用归一化/逐元素 region；同设备原生 buffer producer/consumer 与 HYLE
  已有 SPIR-V 程序组合，证明桥接不依赖模型类、checkpoint 格式或专用引擎。
- 分工：本 factor 拥有固定 ggml Vulkan runtime、窄补丁、受管 region 及其 native 资源边界；
  `compute_route_vulkan` 拥有自然 Direct 对象、结果与该 native device 的接入；
  `graph_route_vulkan` 保留 DAG/边界划分与普通编译入口。后两 owner 的新增 FIX 定界后才实施。
  HYLE 语义图不交给 ggml scheduler，provider 内部融合/临时资源管理予以保留。
- 非目标：更改 ggml 算子算法、引入 CPU 模型 fallback、GGUF/模型加载、四 backend 共用 executor、
  一次接完整算子目录、清理 SIMD、重写公共治理或性能仪器。
- 依赖方向：Vulkan route 消费 backend-qualified implementation factor，后者不得反向依赖
  route 或 Graph；私有 native bridge/region 描述不公开 ggml C 类型。精确 API/依赖/交付 target
  在 AUDITED 固定，不能从验证副本直接拷进产品或形成循环。
- 性能：P0 必做；managed encoding、私有分配、转换/同步、融合收益实际计费。只在 C1 完整候选
  固定后沿其等责 P1/5%门资格化；本卡不把原版 graph 写成无分配 recorded replay。
- 预算：本轮接入路径选择已结束，采用完整 managed region。下一轮仅核对下表四项具体接口，
  最多45分钟定界；代码纠正最多两轮。预算到点形成有限 FIX 或具名未决项，不重开全库研究。
- 预声明验证：W1 原生 Vulkan、Air/Mini MoltenVK；独立公式、真实权重片段、重复执行/双槽/
  旧结果；context/buffer/descriptor/submit OOM、部分接纳错误、关闭/取消/drain、native bridge
  并发和 owner 退役；validation、源码/静态/共享 C++ 消费及 relocation/offline、OFF无探测/
  下载、Windows DLL 邻置。适用项未运行不得签 PASS。

## 有限待审范围

| ID | 具体接入问题 | 必须固定的责任 |
|---|---|---|
| Y1 | 固定源码及窄补丁的 CMake 接入 | 单个 owner 入口引用 lock/cache/package registry；原厂 targets、路线选项/版本资格、source/static/shared 的私有链接与资源；原始 cache 不变、build-local 补丁可重复并逐文件核对；低并行；不复制 vendor 全部源码清单 |
| Y2 | ggml context 与 native device/buffer owner | context 创建 OOM 可恢复；部分初始化回滚；同 VkDevice/queue 借用与外部 token；共享 global 初始化和 queue 锁；配额/私有分配事实准确。只修接入和运行失败边界，不借故重写算子 |
| Y3 | 受管 region 的输入/状态/完成 | 冷构图/资格/绑定与 provider 原生运行分开；每槽 metadata/临时资源、pending owner、接纳部分工作后排空、取消/关闭；普通结果寿命与原生片段组合；热期不做跨 Provider 搜索/计时 |
| Y4 | 普通 Direct/Graph 及三交付 | 用户仍只选 Vulkan；首个完整 C1 region 的具名语义域/默认规则，保留 provider 融合；实际替代路径验证后再比较存量 HYLE 实现，不由实验探针签产品支持 |

这是 SCOPED 的具名待审清单，不是 AUDITED FIX。新 identity 已在 `2f65bad3` 登记 QUEUED，
当前进入 SCOPED，CMake 暂保留不可构建 identity。固定 API、failure、owner、常态成本与 CMake
描述后才能授权产品代码。现有 C1 T–X 的 AUDITED 不覆盖新 runtime 或新的公开资源合同。

## 当前证据与未完成

W1 的原版计算以及窄 native bridge 验证副本已通过三种形状各三次重放；最大绝对误差约
`7.33e-7`，scalar oracle 门为 `6e-6 * (1 + |expected|)`。这证明源算法与接入路径可用。
HYLE 产品接线、OOM/并发/完成合同、其他平台、三交付和完整性能均未验收；64行/256格不因此
增加 PASS，Compute/Graph 实现 LOC 仍以最后产品统计为准。

身份登记改变中央组件图，selector 因此选择 broad component/交付门；该计划保存在
`.scratch/validation/ggml-vulkan-queued.json`，当前只完成文档/状态门，未把选中门写成已运行。
它们随本接入轮稳定产品候选执行，不在每次 QUEUED/SCOPED/AUDITED 状态提交重复构建全库，
也不以已有 runtime 探针代签 HYLE 构建或 package 门。

统一 `scripts/loc-report.py --module compute_graph --commit 7c0e54aa --json --self-check` 沿原22个
实现路径统计为 **44,473行 / 178个实现文件**，self-check通过，包含 GLSL，不含 tests/bench/vendor；
[原始统计](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-admission-implementation-loc.json)。
本轮只新增来源实验、接入身份/卡与构建原则，产品实现行数未增加，不能算新算子覆盖。

## C1-Y 首批有限 FIX（AUDITED）

2026-09-18，owner `/root`，审计 baseline `ba5f0363`。先实施可以独立验证的源码/构建接入和
metadata owner；不等待整个 runtime 桥接完成才开始产品代码。Y2 的设备/队列/配额与 Y3/Y4
仍是本卡必须完成的后续项，尚未授权改 Vulkan 公开对象或运行语义；实现前另行固定对应 FIX。
这不是把范围缩为构建或将 Provider 接入签收。

| FIX | 冻结决定与改动 | 验证及边界 |
|---|---|---|
| Y01 | 新 factor 的 CMake owner 统一声明锁定 `ggml_vulkan_runtime` 456172ec、窄补丁和原厂 ggml/base/vulkan targets；CPU/其他 backend、测试/examples、OpenMP/native tuning 关闭；Vulkan 路线 PRIVATE 消费新 factor，factor PRIVATE 依赖 core/error/mem，不反向依赖 route/Graph | 新 pin 不更新旧 BLOCKED CPU ggml；原始 checkout/cache 必须保持 clean；OFF 不取 ggml、不探 SDK。复用原厂源码清单，shader 子进程串行、外层最多2并行 |
| Y02 | 完整认证原始源后生成 build-local patched source；补丁按精确原文件/结果 SHA 核对，拒绝漂移；构建/许可证/静态依赖导出由现有 package registry 投影 | 同一源码重配不重写未变文件；篡改源码/补丁失败；source/static/shared 编译及独立 consumer。SDK 的 host shader 工具仅 source build 使用；运行 loader/native bridge 最终交付随 Y2 后批接通，不把暂时的 vendor loader 链接写成最终 ABI |
| Y03 | 私有 move-only metadata arena：调用方 allocator 负责 arena，ggml context 负责自身小对象；修改固定源码的 `ggml_init` 使 malloc/arena OOM、大小溢出和外部 arena 对齐失败返回 null，失败回滚；C++ adapter 映射 Status | 不安装 ggml C 类型到 HYLE 公共 API；不覆盖全局 abort handler、不 longjmp；独立配额失败、move/析构/双 arena、溢出/非法对齐回归。其余 ggml 构图容量与 GPU 失败边界不由此代签 |
| Y04 | 使用新 factor 构建的真实 ggml Vulkan backend 运行已登记 RMSNorm→scale→SiLU→residual region，宽2048、行1/17/128、多次变化输入，独立 double oracle | 证明实际库/算法链和固定源补丁可用；这个内部接入用例不代签普通 Direct/Graph、双槽 GPU 并发、真实权重或三交付完整旅程 |

依赖/交付是原子接入机制，metadata arena 是私有资源 owner；模型装配仍由上层 consumer 负责。
`KEEP` 原版算子及融合、原有 Vulkan 路线和 SIMD；`FIX` 上述四项；`DEFER` 只表示按顺序等待
本卡后批的队列桥接/配额/受管完成与自然入口，并非移出 C1。构建和 arena 成本只发生在冷准备；
原版每次 graph 编码/内部管理成本仍计入后续 P0/P1，当前不声称 allocation-free。

当前覆盖：源码 pin/构建/package 与 arena 为 MISSING→实施中；成熟 region 算法已在 admission
证明可用；普通 HYLE 路径/原生生命周期为 MISSING。没有普通用户新增 API。三交付最终均通过
同一 factor/package registry；公开 route 对象不因本批改变。通用 build/selector/性能核心不改。
原有宽组件选择门继续保留待执行；本批实际执行对应 factor、OFF、补丁身份、static/shared 配置
构建和内部真实 GPU correctness。完整 provider 三交付/其他平台/运行失败验收未运行即未通过。
