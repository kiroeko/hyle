# C1-Y：ggml Vulkan managed region 接入

当前有限增量[Y34](llm-moe-c1-vulkan-reduction.md)：三个Vulkan owner已重新SCOPED/AUDITED，
成熟Sum/Mean/Softmax接入和具体尾行/长行缺陷适配已实施；产品`05581df8`通过W1 static/shared
136/135项，测试修正`11804be7`通过Air136、Mini135项（验证层专用用例1项跳过）。
`923895c5`另完成W1/Air真实2B归约/L2/概率六交付432例/1296次执行及三环境公开头。
selected工程余门和C1完整成本优选未完成；本轮仍AUDITED，不签C1完成。

当前有限增量Y33：三个 Vulkan owner 对
[LayerNorm 成熟接入](llm-moe-c1-vulkan-layernorm.md) 的候选`4d255a51`已完成W1/Air/Mini
适用回归及W1/Air本实际片段三交付；Mini验证层专用用例明确跳过，由W1/Air覆盖。
三个 owner 继续 C1 AUDITED；完成其余已登记 Provider 接入后再冻结完整 C1 性能候选。
Y32工程门已完成，CPU Q2 BLOCKED保持；本批不签性能赢家或C1完成。

此前有限增量Y30（2026-09-19，baseline `a302c004`，owner `/root`）：本Provider与Vulkan
route已对[原版Argsort整数padding修复](llm-moe-c1-tensor-state.md#c1-y30原版-argsort-的整数-padding-修复scoped--audited)
完成SCOPED/AUDITED并冻结Y30-A–C。只修已复现的大宽度遗漏及其冷资源准入；不更换排序实现、
pin、公共构建机制或CPU blocker。产品改动前的复现、预算及适用验收以该节为准。

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
- 非目标：除当前 Y22 具名比较语义适配外更改 ggml 算子算法、引入 CPU 模型 fallback、GGUF/模型加载、四 backend 共用 executor、
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

上表是进入 SCOPED 时的具名待审清单。新 identity 在 `2f65bad3` 登记 QUEUED，随后进入 SCOPED；
首批已审计的产品授权见下方 Y01–Y04，其余项仍需固定 API、failure、owner、常态成本后实施。
现有 C1 T–X 的 AUDITED 不覆盖新 runtime 或新的公开资源合同。

## 接入前证据

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

实现接入还需登记新 factor 的 selector 声明（owner 路径、Vulkan feature、适用平台），并使项目
原有 Vulkan 依赖入口接受直接选择该 factor；只是声明接线，不改 selector 核心。54项 selector
回归首跑有一条把 `compute_provider_ggml_vulkan` 的前缀误认成 dormant CPU identity；实际选择
列表没有 CPU ggml。仅将该测试断言改为解析完整 factor ID，53项原通过、该项修正后单独通过。

## 首批实现与验证记录（2026-09-18）

产品提交 `977057403fd1c7023b0c12d2996886b4aa582452`，审计提交 `ea560521`。Y01–Y04 已落地：
中央 lock/cache/package 声明接入原厂 ggml/base/vulkan targets；四文件窄补丁只作用于 build-local
副本；私有 move-only metadata arena 由 HYLE allocator 保活。ggml context 的 OOM、溢出和
外部 arena 对齐错误可返回失败。没有修改上游算子算法，旧 CPU ggml BLOCKED 不变。

[结构化记录与原始日志清单](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y1-build-admission.json)
绑定产品 tree、源与补丁身份、实际命令和限制；
[原始日志包](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y1-build-admission-raw.zip)
保留了路径失败及纠正后的完整构建输出。

| W1 实际验证 | 结果与适用边界 |
|---|---|
| 私有 factor static/shared | 各4例通过：3例 metadata/失败与寿命、1例原版 Vulkan RMSNorm→scale→SiLU→residual；宽2048，1/17/128行，每形状3次变化输入，独立 double oracle。使用生成数据，不是真实 checkpoint 权重 |
| 补丁合同 | 5例通过：源/结果身份、未变文件时间戳、篡改/额外文件拒绝；原始 upstream checkout 仍 clean |
| 路线 OFF | 故意传入无效 shader/source 路径并关闭 Vulkan package 查找，configure 通过且未使用这些变量；这里只签配置隔离 |
| static/shared 安装与 relocation | 两套 compute+graph 产品均构建安装成功；复制 SDK 后四个普通 Compute/Graph consumer 构建运行通过，PATH unset、DLL 邻置 |
| 源码嵌入 | 外部工程 add_subdirectory canonical 源码；认证的 ggml/Vulkan-Headers source override 和 FULLY_DISCONNECTED 下构建、两个普通 consumer 通过；不是独立 HYLE source archive 验收 |

安装/源码 consumer 运行的是既有原生 Vulkan 普通路径，不能替代新 ggml managed provider 的
Direct/Graph 验收。内部 static/shared correctness 先于最终 `EP_PREFIX` 目录修正执行，C++ 与
四文件补丁内容相同；最后完整 source build 使用修正后的 CMake。没有因说明/状态更新重跑 GPU/P1。
Khronos validation 由环境请求，日志未见 VUID；未单独证明 layer 激活，不签完整 validation 门。

Windows 源码嵌入曾在 shader host 子项目 compiler probe 失败：object 目录本身243字符，
CMake 安全预算250，MSVC 报生成文件 `C1083/Invalid argument`，本机 `LongPathsEnabled=1`。
通过 owner 内标准 `EP_PREFIX` 把 host build 目录从172字符缩至113字符，最终 source build 和
两个 consumer 通过；无需改系统、搬源码或调大编译器路径预算。工程处理见
[Windows 长路径说明](../../../developer/setup/windows.md#长路径与第三方子构建)。

P0：arena 一次分配、context 一次小对象分配，move 不分配，先释放 context 再退役 arena；
不由 arena 分配 tensor payload。源码认证/补丁只在构建期发生。原厂 graph 的每次编码、临时
资源与同步仍有成本，后续普通入口完整比较必须计入；本批无 P1 结论。

统一 LOC 脚本对上述不可变产品提交沿原22路径加新 provider 路径统计为 **44,559行 / 180个实现文件**，
较前次增加86行；self-check通过，包含 GLSL，不含测试、bench、vendor、CMake/Python。
[原始统计](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y1-implementation-loc.json)。

状态保持 `AUDITED`。后续先固定 Y2 的 native device/queue、loader、配额与部分初始化失败 FIX，
再实现 Y3 受管运行/完成、Y4 普通入口与规则。当前 static SDK 仍有临时 Vulkan 开发 package
依赖；其他平台、完整失败/并发/结果寿命、真实权重、source archive、广组件 selector 门及 P1
尚未完成，聚合验收 `INCONCLUSIVE`，64行/256格不增加 PASS。

## Y2 原生资源接入有限 FIX（AUDITED，2026-09-18）

owner `/root`，baseline `e7e88c62`。有界源码审计已结束：固定 pin 的 dispatcher/instance 与
registry 是进程全局；device 在完整初始化前发布；buffer 在 map/bind 后才记录 owner，失败可漏
native handle；buffer-type context 强持有自身 device；`ggml.cpp` 静态初始化改写宿主 terminate
handler。这些是嵌入基础库的具体接入缺陷，不授权重写算子或增加新算子。

| FIX | 冻结实现责任 | 验证 |
|---|---|---|
| Y05 | 私有 NativeDevice 持有 loader 与独立 upstream instance/device/registry；保留上游 feature/专用 pipeline 决策，按原 physical-device index、软件设备 opt-in 和 validation 选项创建。每次 adapter 边界用栈上 scope 选择对应 Vulkan-Hpp dispatcher，退出恢复线程局部指针；固定 pin 未创建后台 host worker，不让 scope 跨异步 host 回调。设备完成初始化后才发布 | 两个独立 device/loader owner、无效设备/loader 错误、跨线程使用与构造失败回滚；普通用户无新增 Provider 排序/API。后续 route 持有该 owner 并使用同一 native device |
| Y06 | 原版 buffer 的 create/allocate/map/bind 逐步纳入 RAII，实际 VkDeviceMemory 大小通过可选 HYLE QuotaAllocator reservation 计费，失败返还；buffer-type 反向引用改 weak 以消除环。窄 borrow/export 保留 native owner；外部 queue 使用与上游相同的 device-submit→queue 锁顺序 | 配额拒绝/返还、部分初始化故障、同设备 native producer 与原版 region、原 buffer 比 wrapper 活得更久；不以这些用例代签部分 GPU 工作接纳后的 Y3 终态 |
| Y07 | 私有 C++ backend/buffer owner 对 upstream 异常在 adapter 边界转换成 Status；同步内部 region 用于验证资源桥，不承诺未来普通执行同步。取消 vendor 的静态 terminate handler 接管；CMake owner PRIVATE 增加 platform，采用已加载 PFN，去掉 vendor Vulkan loader 静态链接及静态 SDK 的临时 find_dependency(Vulkan) | 迁移既有独立公式 region 到真实私有桥接，确认宿主 handler 不变；W1 对应 factor correctness、DLL 邻置、source/static/shared 的更新声明检查。完整三交付及 Air/Mini 随 Y4 稳定候选完成 |

`KEEP` 所有原版算子/融合/设备能力分支；`FIX` 仅上述运行宿主与失败边界；`DEFER` Y3 的 pending
owner、部分接纳失败 drain、cancel/close/旧结果及 Y4 普通入口规则，仍是本卡必做。新的私有声明
只在本 factor，ggml C 类型不进入安装 SDK，route/Graph 依赖方向不变；本批不改其公开行为。
loader/instance/device 冷期创建；scope 不分配、不加全局执行锁，whole-region 只进入一次 adapter；
quota opt-in 才增加 reservation token，仍需在完整 P0/P1 计入。新增过程性故障测试可包装真实
Vulkan PFN 返回指定错误，正常正确性必须运行真实 GPU，不能用 mock provider 代替。
实现最多两轮纠正，先 W1 适用 correctness，不启动 P1 或重复已完成的长路径调查。

Y2 实施中的具名接入纠正（仍限 Y01/Y05/Y07）：移除 loader 链接后显式声明 build-only
SPIRV-Headers；补丁哈希改变会改变 build-local source 路径，host shader 子构建 prefix 必须含
短补丁身份，避免复用旧 CMake source cache。registry 的 ctx/device 在入 vector 前也用 RAII，
与 device/buffer 部分初始化回滚同属 Y05。依据为真实编译失败日志和对应 `new`→`push_back`
异常路径，不增加算子、公共构建框架或验证仪器范围。

## Y2 实现与验证记录（2026-09-18）

产品提交 `da0322199baca85c71ae8cfcb4d5c39e7fa719db`，审计为 `3caa1103` / `5c7db84e`。
Y05–Y07 的私有 native owner 已落地：独立 loader/instance/device/registry、栈上 dispatcher scope、
完整初始化后发布、buffer RAII/实际内存配额、弱反向引用、原生 borrow/export 与共享 queue 锁。
原版算子/融合不变，私有边界转换创建异常，宿主保留 terminate handler。

MSVC vendor C++ 目标局部使用 `/EHsc-`：原厂 C linkage 的 C++ 函数可能传播分配异常，不能使用
`/EHc` 的不抛异常假设省掉回收；不改变 HYLE 用户编译选项。移除 loader 链接后显式保留 build-only
SPIRV-Headers，静态 SDK 导出不再查找 `Vulkan` 开发 package。短 host prefix 加入补丁身份，
避免升级 build-local source 后命中旧 CMake source cache。

[结构化记录](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y2-native-resources.json)与
[原始日志包](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y2-native-resources-raw.zip)
绑定产品 tree、补丁哈希、命令和限制，保留首次编译失败与设备 identity 反例。

| 实际执行 | 结果与边界 |
|---|---|
| W1 static/shared 私有 factor | 各9例通过、无跳过：3例 metadata、1例真实原版 GPU region、5例原生资源/错误/隔离；PATH unset、所需 DLL 邻置 |
| native 正确性/寿命 | 宽2048、行1/17/128、各3次输入与独立 double oracle；每形状一次由同 VkDevice 的原生 producer 写入；borrow 退役后原 buffer 仍可使用；配额拒绝/返还、双实例跨线程与嵌套 scope |
| 真 PFN 故障 | device 首个 fence、backend 第二个 fence、已分配 buffer 的 bind 失败均回滚；相应 native device/fence/buffer/memory 计数归零。不代签 GPU 部分提交后的 drain |
| 设备 identity 回归 | 新断言在原索引比较下复现8通过/1失败：独立 owner 的本地编号都为0；改为实际 owner 比较后 static/shared 均拒绝外来 buffer type |
| validation | 显式 validation 实例测试实际运行，要求并启用已安装 layer；全套环境也请求 Khronos layer，最终日志无 VUID。只覆盖本批已执行路径 |
| 构建补丁/导出 | 5例源码合同通过；static SDK configure/export 检查通过，find dependency 只有 Threads，未泄漏 Vulkan loader / SPIRV-Headers target。没有据此声称更新后的 SDK 已安装/消费 |

P0：loader/device 与 metadata 在冷期创建；scope 只保存/恢复 TLS 指针，不分配。quota opt-in 才有
每个 native allocation 的 reservation token；原版 graph 每次遍历/融合/编码与转换、同步仍须在
完整比较中计费。本批未运行 P1。构建最多2并行、shader 子进程串行，未改系统或启动安装程序。

统一 LOC 脚本对该不可变提交沿同23路径统计 **44,995行 / 184个实现文件**，比 Y1 增加436行；
self-check通过，包含 GLSL，不含 tests/bench/vendor/CMake/Python。
[原始统计](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y2-implementation-loc.json)。

当前仍为 `AUDITED`，聚合验收 `INCONCLUSIVE`，64行/256格不增加 PASS。Y3 先固定冷构图/分配与
每槽 owner、pending、部分接纳失败 drain、关闭/取消/保留结果。已发现 `ggml-alloc.c` 的多 buffer
context helper 有未检查的 realloc 及 C unwind 清理限制，Y3 必须由受管 region owner 定向解决；
当前仅用于内部 admission 的 `allocate_context` 不能直接继承为最终 OOM 合同。Y4 的普通入口/规则、
真实权重、Air/Mini、完整三交付和 broad selector 门仍待完成；没有新的外部阻塞。

## Y3 完成边界首批有限 FIX（AUDITED，2026-09-18）

owner `/root`，baseline `efe2c9c4`。固定源码的 `ggml_vk_compute_forward` 会在一个 region 内
提交多个 batch；`submit_pending` 不等同于整个 region 成功，后续编码/分配/submit 仍可失败。
`ggml_backend_vk_synchronize` 会提交尚未提交的 transfer/compute context，不能用于失败后盲目清理。
已有 route 的单次 submit 失败处理不覆盖此形状。此次先实现独立可验证的 native 完成边界：

| FIX | 冻结责任 | 验证 |
|---|---|---|
| Y08 | 每个私有 NativeWork 持有 backend、完成 fence 和 NativeDevice。正常 async region 的最后一个上游 compute batch 直接携带该 fence，保留原版融合/分批，不额外提交每 kernel callback；原版强制串行/诊断分支保留，必要时只在 region 尾部追加完成 marker。submit 结果分别报告 failure 与仍可能 pending，不把错误当成资源可释放 | 真实多 batch GPU region、重复执行、双独立 work；真实 PFN 在已接纳首批后使后续 submit 失败，保留原始错误且不继续提交残余编码 |
| Y09 | 私有边界观察 fence；只有完成或 device loss 才可退役。异常时按实际 upstream queue 锁排空 compute/transfer；观察/排空 OOM 仍 pending，保留全部 owner。清理先丢弃未提交 context，再在已证明 quiescent 后回收原版 graph 资源，失败 slot 禁止重用。NativeWork submit 接受强资源 lease，普通 route 后续用已有 pending Run 持有它；此处不再建第二个全局 pending 队列 | 延后故障恢复前 owner 不释放、恢复后恰好释放；正常 poll 不强制 queue idle；失败 slot 拒绝复用；析构的最后防线可阻塞 drain，普通异步丢弃由 Y4 既有 pending owner 保活 |

仅修改本 factor 的私有 C++ seam、锁定 vendor 完成边界与对应测试；不改算子算法、公共 Completion、
route API 或公共性能工具。NativeWork 单对象 host 操作 thread-compatible；同 NativeDevice 的
准备/编码由 route 既有 admission 锁串行，多个 slot 的 GPU 工作可同时 pending。正常路径保留
上游资源与编码成本，新增一次 region 完成 fence/lease；不启动 P1。

Y3 冷期图/存储 owner 仍必须完成：源码检查还确认 gallocr 的多个 metadata 分配有 abort/unchecked
路径。后批须保留成熟 liveness/in-place 规划并定向使其失败可回收，不直接沿用不安全 helper、
不把每个中间张量永久单独分配作为最终方案。本批不代签该分配合同、公开 cancel/close/保留结果，
后者随 Y3 slot 与 Y4 原有 Run/Results 接线验收。W1 static/shared 适用 correctness 先行，纠正预算
两轮，完整路线/三交付/Air/Mini/P1 仍沿本卡执行。

## Y3 私有完成边界实现记录（2026-09-18）

产品提交 `0ab77bcbeefc7af43f821e941c39fa9e38e36c81`，审计 `e0b4b144`。Y08–Y09 已实现：
每槽 NativeWork 保留原版 backend、完成 fence、device/loader 与图/存储强 lease；正常异步路径
直接在原版最后一个 batch 上携带 fence。错误与 pending 分开报告，观察或 drain OOM 不释放
仍可能在执行的资源；已证明完成后丢弃未提交 context 并清理，保留首次错误、禁止失败槽复用。
普通 route 的 pending owner、Direct/Graph 与冷分配接口尚未接线，不能据此签完整 Y3/Y4。

[结构化记录](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y3-native-completion.json)与
[原始日志](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y3-native-completion-raw.zip)
绑定产品 tree、补丁身份、实际命令和未运行项。

| 本批实际验证 | 结果及边界 |
|---|---|
| W1 static/shared 私有 factor | 各13例通过、无跳过，DLL 邻置、PATH unset；显式 validation 实例运行，两份最终日志均无 VUID |
| 独立槽/lease | 2048宽 norm→scale→SiLU→residual，两个槽各三次变化输入、独立 double oracle；先上传全部输入再接纳两个执行，避免第二次同步上传掩盖并发；重复 submit 拒绝，move/析构在 lease 释放前等待退役 |
| 部分接纳失败 | 256次 add region 的第二次 submit 被真实 PFN 包装返回 OOM，首批已实际提交；drain OOM 期间 native buffer/memory 保持，恢复后不提交残余编码，首次错误保留、失败槽拒绝复用，最终原生计数归零 |
| 观察失败 | 实际 SiLU region 后注入 fence 观察 OOM 和 drain OOM；poll 不排空队列，资源保持，恢复后退役且不重复提交 |
| 正常完成结构 | 两份结果均记录 `region_empty_submits=0`，正常 submit 未调用 queue idle；这是 W1 结构验证，不是 P1，也不代签原厂强制串行/诊断分支 |
| 构建/维护 | 5例补丁合同、docs、closure 通过；首次 MSVC 三处 `vk::Fence = {}` 歧义改为显式 null，保留失败日志。上游 checkout 仍 clean |

P0：每槽冷期增加一个完成 fence；每次 region 保留强 lease、栈上 TLS scope、pending/error 状态与
device-loss 原子观察。保留原版遍历/融合/编码及分配成本；正常 poll 不强制 queue drain，异常
wait 与析构最后防线可以阻塞。未运行 P1；没有改写算子算法或新增公共执行器。

统一 LOC 对本次不可变提交沿同23路径统计 **45,279行 / 185个实现文件**，较 Y2 增加284行，
self-check通过，不含测试、bench、vendor、CMake/Python；
[原始统计](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y3-implementation-loc.json)。

本卡保持 `AUDITED`，聚合 `INCONCLUSIVE`，64行/256格不增加 PASS。下一有限动作是 Y3 冷期
graph/storage owner 与 gallocr 的可恢复失败，必须保留成熟 liveness/in-place 规划。公开取消/
关闭/旧结果、device-loss 传播、Y4 默认入口/规则、真实权重、Air/Mini、完整三交付和 broad
selector 门仍未完成；没有用户决策阻塞，不因本次记录修改重建或重跑 P1。

## Y3 冷图与存储有限 FIX（AUDITED，2026-09-18）

owner `/root`，baseline `fbd07798`。固定 `ggml-alloc.c` 的具体缺陷为：new/chunk/hash/node/leaf
metadata 的空指针/abort 路径、free-block 容量耗尽的 abort、部分初始化回收不完整；C 分配循环
不能靠外层 C++ catch 回收尚未登记的局部 buffer。上述只影响本私有 Vulkan 源码副本。

| FIX | 冻结责任 | 验证 |
|---|---|---|
| Y10 | 保留 gallocr 的拓扑/liveness/in-place 算法，metadata 分配采用检查与局部回滚；容量/溢出失败沿 reserve 的 bool 返回，不用 abort/longjmp。失败 owner 只销毁不复用。每个冷存储 owner 使用同设备 buffer-type 的私有副本，仅把 allocation callback 包成不抛异常的失败边界，使 C 的已有逐 buffer 回收生效；不改变算子或全局 allocator | 同补丁 C 源的测试专用副本定点注入各 host 分配失败，统计回收；真实 native 配额拒绝、已分配部分 GPU chunk 后失败；源合同与 static/shared correctness。产品不添加测试 failpoint |
| Y11 | 私有 NativeGraph 接收已按容量构建的 metadata arena、ggml graph 和可选外部 buffer 强 owner，冷期核对原版 op capability、正尺寸与同设备存储，再 reserve/allocate；整体拥有 gallocr、arena、imports 和 NativeDevice。每槽独立，执行由已有 NativeWork 保活该 owner；暴露私有存储字节事实用于验证复用。当前上下游公开 C++ API 不变 | 真实 norm/gate/residual 与长链的存储复用、两图独立、重复执行、保留旧图、OOM/拒绝与析构返还；输入/输出按上游 flags 标记；借入资源保活和跨设备拒绝 |

冷分配只发生在 slot 创建，重复执行不重新调用 gallocr；不以永久逐张量分配替代原规划。此 seam
仍是私有 graph handoff，Y4 的语义 lowering 负责从有限 Plan 计算 metadata 容量、数学/shape
合同与输入/输出 flags，再接自然入口，不把原始 ggml 类型交给普通用户。已有仅 admission 使用的
`allocate_context` 按迁移实际消费清理，未迁移用法不获得最终 OOM 资格。零尺寸/不支持域由上层
固定规则选择其他合格实现或明确拒绝；本批不据此缩减 C1 MUST。实现纠正最多两轮，不启动 P1，
公开取消/关闭、完整三交付和其他机器仍按 Y4 验收。

## Y3 冷图与存储实现记录（2026-09-18）

产品 `f61ae0f9f1d5054fba0c4cc9ace1cc2fafa7e843`，审计 `a9b01593`。Y10–Y11 已实现：
NativeGraph 整体保留 arena、gallocr、外部资源与 device，每槽冷期规划/分配一次，由 NativeWork
保活至原生完成。保留原版 liveness/in-place/块选择，定向补齐空指针、部分初始化、容量与对齐
溢出的失败返回；私有 buffer-type 副本只包装 allocation 的不抛异常边界，让 C 层正常回收。
所有已有私有 region 测试已迁移，旧 `allocate_context` bridge/方法删除，不保留双分配入口。

[结构化记录](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y3-cold-storage.json)与
[原始日志](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y3-cold-storage-raw.zip)
绑定产品 tree、七文件补丁、命令、失败修正和未运行项。

| 实际验证 | 结果与边界 |
|---|---|
| W1 static/shared 私有 factor | 各19例通过、无跳过；同源 norm/gate/residual、双槽重复执行、原生 producer、完成/失败用例全部使用新存储；DLL 邻置、PATH unset，最终无 VUID |
| 内存复用与寿命 | 2048宽256次 add 的计划存储小于逐节点分配的1/4，实际结果64.5；pending、保留图和 readback 期间资源存活，释放后 quota 归零 |
| Host OOM | 同一补丁 C 源的私有测试副本逐个拒绝12个实际分配点，每次返回失败并清空记录中的 host allocation；最后正常分配仍通过。产品没有测试 failpoint |
| 规划容量 | 300个 repeat 输出与交错保留小叶子使原厂256项 free-block 表耗尽；返回失败且回收，没有 abort，不改写原规划算法 |
| GPU 部分分配 | 进程内暂设原厂 chunk-size 为8192并恢复，首块实际分配，后续分配及 memory-type 重试持续返回 OOM；buffer/memory 最终归零，保留原生 Vulkan 错误 |
| 借入资源 | 缺少强 owner、跨设备 buffer 明确拒绝；同设备借入持有至图销毁，完整执行仍由真实 Vulkan 完成 |
| 构建/维护 | 5例补丁合同、docs、closure 通过，上游 checkout clean。保留 weak reference 比较的编译修正，以及首轮17通过/1断言错误；后者将通用 OOM 预期纠正为实际保留的原生 OOM |

P0：每槽冷期拥有一组原版 metadata/chunks，执行不再调用 gallocr；没有新增逐 kernel 分发。
原版 graph 编码和内部执行分配仍计费，不宣称零分配 replay。私有 capability 检查不代替 Y4
语义 lowering 的数学/shape、arena 容量和 input/output flags 合同；原厂有限规划容量仍有限。

统一 LOC 沿同23路径对不可变产品统计 **45,440行 / 188个实现文件**，净增加161行，self-check
通过，不含 tests/bench/vendor/CMake/Python；
[原始统计](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y3-storage-implementation-loc.json)。

当前仍 `AUDITED` / `INCONCLUSIVE`，64行/256格不增加 PASS。后续进入 Y4 普通 Direct/Graph、
默认规则、同设备片段组合与公开取消/关闭/结果保留接线；实际权重、Air/Mini、更新后三交付、
broad selector 门及等责 P1 仍未完成。没有新的外部阻塞，不重复本批验证来代替普通入口接线。

## Y4 同设备普通入口前置 FIX（AUDITED，2026-09-18）

owner `/root`，baseline `94b40163`。本批同时绑定 `compute_provider_ggml_vulkan` 与
`compute_route_vulkan`；Graph 是既有普通入口的直接消费者，不修改其语义。检查实际源码确认：
现有 Context 自建第二套 instance/device，不能直接消费 Y2/Y3 的资源；原 route 队列锁也不覆盖
原版 compute/transfer 的 queue handle。另一具体反例是 borrow 无条件查询 BDA，而公开
`borrow_ready` 只要求 storage/transfer buffer，没有承诺地址 usage/allocation flags。

| FIX | 产品改动与成本 | 适用验证 |
|---|---|---|
| Y12 | 普通 Context 持有既有 NativeDevice，取其准确 instance/device/queue、已启用 descriptor feature 和 loader 身份；删除 route 重复创建/销毁 loader/instance/device 的实现。HYLE PFN 一次绑定，原 Buffer/Run/Completion/配额 owner 保留。原生 submit、异常 drain 与 expert queue 回调进入同一 upstream device-submit→queue 锁；热 submit 使用无分配的私有函数指针/状态回调，不新增逐 kernel vtable。borrow 仅在调用方明确知道 buffer 具有 BDA 资格时取地址，否则保留 descriptor 路径 | W1 static/shared 的 Provider、全部现有 Vulkan Compute/Graph 及两个公开头；同一实际 VkDevice 资源往返、普通 route producer→原版 norm→普通下载；独立 Context、配额、设备失败/部分分配、既有并发/取消/关闭/保留结果。Khronos validation、DLL 邻置、无 PATH 依赖 |

同设备导入测试只证明资源互通，不能代签普通 Program 的 managed region。Y4 后续仍必须接入
冷语义 lowering/固定规则、输入重绑、原生/managed 片段组合，以及部分 region 接纳后的既有
pending Run/Completion。shared 编码与清理的互斥、正常 wait 不锁住独立 producer 也须在该批
固定后实施；不能继承单次 QueueSubmit 失败即未接纳的假设。本批不宣称新覆盖、不运行 P1。
跨平台与完整三交付随稳定 Y4 候选，当前局部纠正最多两轮；不修改公共治理或性能工具。

## Y12 实现与验证记录（2026-09-18）

实现 `01b545f0`，最终验证候选 `c1a8b6ff`（随后两个提交仅修测试夹具），审计 `72aac688`。
普通 Context 已持有 Provider 的唯一 instance/device/queue，读取实际启用的 descriptor feature；
route 原来的 loader/设备创建与销毁代码删除。HYLE 提交、异常 drain 和 expert 回调进入原版
队列锁；正常提交采用栈上 callback/state，无额外 std::function 构造或逐 kernel 虚调用。
借入 Buffer 只有在调用方明确证明地址资格时才查询 BDA，普通 storage/transfer Buffer 可用。

[结构化记录](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y4-device.json)与
[原始日志](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y4-device-raw.zip)
包含不可变代码/夹具身份、命令、artifact hashes、初始结果和一次严格编译失败。

| 最终 W1 验证（WERROR=ON、instrumentation=OFF） | 结果与边界 |
|---|---|
| static | 19 Provider + 34 Direct + 14 Graph 通过；两个公开头与5例源补丁合同通过 |
| shared | 19 Provider + 34 Direct + 13 Graph 通过；两个公开头与5例源补丁合同通过。少的一项是既有仅静态编译的 executable-local host allocation 注入；没有跳过已编译用例 |
| 同设备实际计算 | 普通 Neg Program → 私有原版 RMSNorm → 普通下载，17×2048、独立 double oracle，误差门 `6e-6*(1+abs(expected))`；instance/device/queue 逐个相同，旧图释放与 Context.close 后保留结果仍可读取，最终 quota 归零 |
| 生命周期与失败 | 原有域/视图/重绑/双槽/并发 producer/取消/关闭/原生 OOM/观察与排空用例全部在新设备 owner 上运行；expert callback 抛异常后诊断保留且队列锁释放 |
| 环境与维护 | 两模式 DLL 邻置、运行时 PATH unset、validation 开启，无 VUID/validation error；docs、closure（previous-ref `94b40163`）通过，上游 checkout clean |

初始缓存为 WERROR=OFF，初始通过不代签严格编译。测试中的 Windows getenv 改成持有 CRT
返回值；严格编译首次发现测试专用 C 包含单元的 UTF-8 与原版 FP16 隐式窄化告警。仅该 C
文件增加 `/utf-8`，仅原版源码 include 范围抑制具名4244；产品及 HYLE 自写部分仍 WERROR，
原始第三方源码与算法不变。后续严格构建/测试均通过，没有改变数值或失败门限。

P0：Context 冷期现在承担完整原版 native runtime 的准备成本，不宣称启动更快；原生提交多进入
原版队列锁，但没有新增热期分配、搜索或 timing。普通 Program 仍使用已录制的 HYLE/shader
路径；中间的 ggml region 测试证明资源互通，**不代签 managed Direct/Graph**。

统一脚本沿同23路径统计 **45,277行 / 188个实现文件**，比上一实现减少163行，self-check通过；
[统计](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y4-device-implementation-loc.json)
绑定 `01b545f0`，其到 `c1a8b6ff` 的实现路径 diff 为空。测试/bench/vendor 不计入。

当前保持 `AUDITED` / `INCONCLUSIVE`，64行/256格不增加 PASS。下一步是 ordinary Program 的
managed region：有限数学/shape 规则、冷构图与重绑、原生片段组合、部分提交和完成接入原有
pending Run。真实权重、Air/Mini、完整三交付、原 broad package 门和等责 P1 仍未完成；没有
新的外部 blocker，不重复本批验证。此次文档/状态更新不触发重建或 P1。

## Y4 普通 managed region 有限 FIX（AUDITED，2026-09-18）

owner `/root`，baseline `ae39a9e1`；本批绑定 Provider、Compute route、Graph route。源码依据是
锁定 ggml 的 op_f32 descriptor 偏移处理、RMS/二元/GLU capability 与融合、gallocr 的 input/output
flags，以及现有 HYLE Plan/Recorder/Run。已排除把 HYLE 图整体交给 ggml scheduler、预分配每个
内部 tensor、执行期选 Provider、为每 kernel 建虚调用等方案；规则和 owner 如下固定。

| FIX | 具体实现、边界与成本 | 适用验证 |
|---|---|---|
| Y13 | Provider 私有 typed region 接收 HYLE Operation/TensorDesc，PRIVATE 依赖 compute_ops，仍不依赖 route/Graph。准入正尺寸 dense F32、rank1–4：F32 RMSNorm（可选同类型 scale/bias）、SiLU、Add/Multiply（至少一侧为完整输出形状）、分离式 SiLU gate。原版参数、广播、op capability 和设备工作组限制冷期匹配；不引入新算子算法。每槽一次有容量证明的 arena/graph/gallocr，内部 tensor 由原版规划；端口用可重绑的私有 native buffer wrapper，保持真实 owner、offset/capacity，不虚构 VkBuffer。未绑定槽不可提交，重绑不创建 graph/设备资源 | 原版数值 oracle、width2048和1/17/128行、参数广播、非有限/极端输入、odd View offset；配额/冷期失败、变输入与独立槽、内部存储复用。原版 reserve_n_size 仍含 assert，本批不把它当成可恢复预估 API |
| Y14 | Compute 冷期按最大连续兼容片段分组，仅带 RMSNorm/分离 gate 的组默认选 managed，其周围兼容 SiLU/Add/Mul 一起交付；其余保留已有已准入实现。Graph 仍拥有合法 DAG/死节点/跨片段边界，published 或其他片段使用的值标为 region 输出，避免成熟融合吞掉可见结果。HYLE 只规划边界与自有片段 scratch；Program/Recorder 使用同一 recipe，native 子片段仅保存实际引用值的紧凑映射，避免每片段复制全图状态。CompileReport 分开 native dispatch/managed region，说明 provider 内部容量和执行分配的未知范围 | 普通 Direct norm/gate、普通 Graph norm→scale→SiLU→residual，HYLE native 前后缀及 Recorder 同设备组合；中间 published 输出、分叉、常量与旧结果；原有未选 managed 的空/半精度/stride/LayerNorm 等域保持正确，不从本次有限 region 域删减 C1 MUST |
| Y15 | 原生 command 片段与 NativeWork 按已固定次序提交同一 queue；一个父 Run 继续使用原有 pending 列表。正常 completion 等待最后实际 batch 的 fence，不增加逐 kernel fence/marker，正常 wait 释放 route mutex；provider 编码和 retire 仍在同一 admission mutex 下。部分接纳后失败返回持有失败 Completion 的 pending owner，poll 不阻塞；wait/close 异常 drain native 及 provider 已接纳前缀，只有真实完成或 device loss 才退役。失败 managed 槽不重用。旧单次原生 submit 未接纳失败仍可重试；取消只抑制结果，丢弃 Completion 不立即等 GPU | 双槽、wait期间独立 producer、poll/close并发、取消/丢弃、保留 Tensor/ResultLease、结束与旧结果读取；真实 native prefix→managed→native suffix 的注入提交失败与 drain，继承 Provider 已证实的内部多批部分接纳/观察失败机制，并核对 route 的错误分支；DLL邻置、validation、严格W1 static/shared及两个公开头 |

公开说明同步修正：HYLE 不在 submit 重新规划或搜索，但成熟 provider 可执行自己的图编码与
内部资源管理，不能继续承诺所有路径“无 command recording/无 allocation/无任何 DAG walk”。
优选规则是开发期固定的初始限定域，最终性能资格仍随不可变 C1 候选做等责 P1；普通用户没有
provider 排序、timing profile 或运行期赢家裁决入口。本批不宣称已胜出、不删除未比较算法。
后续实际权重、Air/Mini、完整三交付/relocation/OFF与原 broad 门仍是 Y4 的适用验收，不移到 C7。
有限实现纠正最多两轮；不扩建公共治理/性能仪器，文档状态提交不重建或跑 P1。

## 临时产物回收约定（2026-09-18）

用户要求持续控制磁盘占用。Windows 仍为产品真相；本轮复用一套固定 static/shared 构建目录，
第三方认证源码只放共享 cache，工具与 SDK 独立保存。每个验证批次先归档身份、命令、结果和
必要失败日志，再清理过期构建、重复源码包与可重建中间产物；不按每个提交无限新增副本。
不清理未提交源码、本地模型或唯一验收记录。环境被用户清理后只恢复下一项适用验证必需部分，
不以恢复缓存为由重跑已匹配的历史正确性或 P1。清理本身不计入功能完成度。

### Y13 实现前补充审计：端口起点与共同 schema

owner `/root`，baseline `4848e222`。审查 `TensorDesc::is_dense`、View slice 与原版 F32 descriptor
寻址发现具体反例：stride1 的合法 F32 输入经 `slice(start=1, step=4)` 可形成 dense 描述而
实际起点是 byte1。描述本身不能证明动态绑定端口的四字节对齐。Y13 因此在冷期为不能证明
对齐的 region 输入固定加入既有 HYLE Copy 适配；已知对齐常量和原生计算结果直接导入，
不在 bind/submit 改选 Provider。适配 buffer 参与既有 liveness/quota，dispatch 和复制均计费。
补 byte1 起点、改变起点重绑和原输入保留的回归；不得用 bind 拒绝代替支持原有合法 View 域。
这个首批保守转换的成本需随 C1 等责 P1 验收，不能宣称优选已经胜出。

共同 schema 的 RMSNorm 当前只接受可选 scale；表中 bias 是相邻 Add 的 region 能力，
不是新增 RMSNorm 参数。实施沿已有 schema，删除不可达的第三参数路径，不改共同操作合同。
上述为同一 Y13–Y15 的有限接入纠正，不增加算子算法、公共配置或治理工具；预算仍为两轮。

### Y13–Y15 首轮候选与定向纠正

产品 `5a96b46c` 首次严格编译因显式 Operation 构造被拒绝，`7ececfc6` 修正构造后编译通过。
W1 static、无 PATH、Khronos validation：Provider 19/19；Direct 36/38；Graph 14/15；
两个公开头和五个源码补丁合同通过。新加的部分接纳后失败/close重试、正常wait期间独立producer、
取消与重用、byte1 dense View/Recorder、2048×1/17/128行的多输出分叉图均通过，无 VUID。
本批仍未验收，保留原失败日志；下一次只纠正以下具名问题，不放宽数值门。

1. 产品数值反例：原版 `swiglu.comp` 与 `unary.comp::op_silu` 直接求负无穷除以正无穷，
   得到 NaN，违反已准入的 SiLU(-Inf)=+0。固定增加两处精确 source patch：仅对负无穷选择
   activation=+0，门控继续乘原第二输入，保留 0×Inf/NaN 的传播；有限域公式、fusion、
   pipeline/资源组织不改。这是第1轮产品语义纠正，更新 patch/NOTICE 身份并补普通 Graph
   SiLU 非有限回归；不因此删除原厂执行器或另写完整算子。
2. 配额断言错误：原版 backend 初始化把 `prealloc_size_add_rms_partials` 固定为1024B，
   `ggml_vk_preallocate_buffers` 在每槽首次执行分配，后续重放复用。配额已经经 native
   reservation 覆盖原厂 VkDeviceMemory，
   并非泄漏。测试在两槽各首次执行后记录容量，验证后续重绑/重放不增长，仍检查完整释放；
   文档区分运行时配额与编译报告中的未知 provider 容量，不宣称 submit 零分配。
3. 旧 report 用例把 managed gate 仍计为一个原生 dispatch；改为1 native +1 managed，
   继续验证报告不保活常量、自定义 payload 或设备。其数值/寿命约束不变。

### Y13–Y15 普通入口的 W1 局部验证

最终产品 `a7ebf205047c86a8b0724fbbebac3fa27f4a57b0`，tree
`bbdb71536846ec3adb92147f981321bfb4b64ec8`。原版 managed region 已进入普通 Direct、Graph
和 Recorder，包含固定的对齐适配、published/跨片段边界、独立槽重绑，以及已有 pending owner
下的完成与失败处理。[结构化记录](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y4-managed-entry.json)
和[原始日志](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y4-managed-entry-raw.zip)
保留首轮失败、一次语义纠正、最终命令及身份；未放宽数值门。

| 实际验证 | W1 static | W1 shared |
|---|---|---|
| 原版 Provider 与私有失败/存储边界 | 19/19 | 19/19 |
| 普通 Direct、Recorder、生命周期 | 38/38 | 38/38 |
| 普通 Graph、多输出/分叉与生命周期 | 16/16 | 15/15 |
| 三个公开头，分属两个 header suite | PASS | PASS |
| 五个源码补丁合同 | PASS | PASS |

shared 少一个已有的静态 global-new 失败注入用例，不是运行时跳过。所有最终测试无 skip、
无 VUID/validation/sync-hazard 诊断；DLL 邻置并 unset PATH，严格 Release/WERROR、shipping
layout，构建 parallel2、重型验证串行。没有运行安装器或修改电源、注册表和防火墙。

新入口已验证2048宽、1/17/128行、多个公开中间输出、原生前后缀、两独立槽与变输入；byte1
dense View、Recorder、取消/旧结果/关闭亦通过。真实已接纳 native→managed 前缀之后拒绝
native suffix，保留首次错误与全部 owner；poll 不强制等待，drain 失败后保活，恢复后安全退役。
正常 wait 期间另一个 producer 可以提交。Provider 的原有多批失败和冷分配用例同时通过。

P0：冷期创建 typed recipe、每槽原版 graph/gallocr 和固定边界 copy；内部保留成熟融合/存储
规划。submit 不做 HYLE 规划或赢家搜索，但原版编码和每槽首次1024B辅助缓存仍存在，配额已
计入；两槽预热后的重放不增长，完整释放后配额归零。适配、首次准备和融合损失都须计入后续
等责比较；本批没有 P1 或性能胜出结论，也不删除尚未比较的自有实现。

[统一 LOC](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y4-managed-entry-loc.json)
为 **46,079行 / 191个实现文件**，沿同23路径、含GLSL、不含测试/bench/vendor，self-check PASS。
统计绑定 `8ca72e6b`；脚本另外核对该23路径到最终候选的所有 blob 完全相同，复用计数。

三个 owner 继续 `AUDITED`，整包 `INCONCLUSIVE`；64行/256格不增加已验收 PASS。Air/Mini、
真实本地权重片段、完整源码/静态/动态 consumer 与 relocation/OFF、broad selector 门及 C1 P1
仍未完成。没有外部用户决策阻塞。用户进一步明确：接入应包含现成能力的普通默认路由，
不能把执行通道或一个示例 region 当作完整 Provider 接入。下一有限动作先按已固定 ggml pin
核对 C1 操作分派与当前映射，区分已绑定、待绑定、需适配和确实不支持，固定下一组有限 FIX；
随后对该组完成 Air/Mini 等适用验证，复用本次未改变的事实，不重复研究已有执行边界。
本次证据/状态修改不触发产品重建或 P1。

## Y16：Windows MASM 与原厂 ASM 组合（重新定界并 AUDITED）

owner `/root`，baseline `6957f964`。C1-AA 的原 selector 全因子配置复现
`CMAKE_ASM_COMPILE_OBJECT` 缺失：Sched 已启用 `ASM_MASM`，之后固定 ggml 的
`project("ggml" C CXX ASM)` 注册通用 ASM，把父项目的 `.asm` 文件重新归入 ASM。
通用 ASM 的编译规则只存在于 vendor 子目录，故父目标生成失败。这是成熟 Provider 的 CMake
组合污染，不是 Sched 算法或汇编内容缺陷；修复归当前 AUDITED 的 Provider owner。

[具名最小复现](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-masm-audit.json)
和[原始脚本/日志](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-masm-audit-raw.zip)
已在同一 W1 工具链验证：原形状报告父 fcontext 文件 LANGUAGE=ASM 并生成失败；仅在 vendor
block 限定通用 ASM 后缀为 `.s/.S`，则保留 LANGUAGE=ASM_MASM，实际 fcontext 汇编及静态库构建通过。
这是实现前的机制证据，不代签真实 Provider 组合或任一计算/性能验收。

有限 FIX **Y16**：仅在本 factor 的现有 vendor block、MSVC 分支设置
`CMAKE_ASM_SOURCE_FILE_EXTENSIONS` 为 `s;S`，然后仍使用原厂 `add_subdirectory`。
增加真实父工程 MASM target 与 HYLE Vulkan source embedding 同时存在的配置/编译回归；
原 full-factor selector 配置必须恢复，并完成已选门。保留当前 ggml pin、九处源码补丁和全部
运行语义；不修改 Sched、公共构建/selector 或性能仪器，不重写 vendor 源文件列表。

这属于 Y1 构建隔离/组合边界的第二次有界产品纠正，预算不增加；只裁决这一个可复现反例。
适用验证为 Windows 实际组合回归、selected factor/直接消费者/header/交付及原 broad 门。
MSVC 以外分支不改变，已有未变的运行时/跨平台证据按身份保留；该配置修复不单独启动 P1。
实现前本节只冻结修复；后续实施和验证事实如下，C1-Y 的其他未完成项保持不变。

### Y16 实施与隔离验证推进

产品修复 `472f1fe0` 保留父工程 MASM 语言归属；真实组合回归在 `904b1fce` 完成。
该候选已通过 full-factor configure、Provider 三项 CTest（真实 GPU、源码合同、父 MASM
组合）及本次 selector 的96项基础回归/逐头门。回归夹具先后漏传 portable Vulkan SDK 的
include/library 与 SPIR-V package 路径，补齐后实际通过；保留初次失败，不把它们归为计算缺陷。
这些结果不代签剩余组件合同、三交付或性能；原始结果暂存本轮 `.scratch/validation/onednn-aa`，
后续按同批记录归档。

候选 `27a67023` 的 component-contract 随后复现另一项**测试注册遗漏**：Vulkan OFF 时私有
optional-absent factor 仍存在，但整个测试 fragment 被 ENABLED 条件跳过，没有测试聚合目标，
因此 CPU-only + tests ON 配置被现有检查器拒绝。有限修正仅将既有、无需 SDK 的 Python 源码
合同测试移到条件外，并使用现有 module test aggregate helper；GPU 执行测试仍只在路线启用
时注册。不改产品、依赖选择、公共治理或性能合同，不扩产品纠正预算。修正后只运行 selector
选中的受影响检查和未完成的原组件/交付门，已通过的无关基础门不重跑。

`98d3331e` 修正后4项 selected 检查及原 component-contract 实际通过；Vulkan OFF 的源码合同
测试无需 SDK，CPU-only + tests ON 不再因缺少模块测试目标失败。原 broad selector 的112项
检查已具备通过记录，详见[同批验证与复用依据](../../../../evidence/correctness/compute-llm-moe-c1/onednn-normalization-selected-validation.json)。
两次夹具 SDK 路径遗漏、OFF 注册失败和隔离 cache 缺归档均保留原日志；后者只复用本机已有
锁定包补齐，没有修改公共合同检查器。此处完成 Y16 的组合/隔离修复及原 broad 门，不代签
Y4 尚未完成的 GPU-on 三交付、Air/Mini、真实权重片段或性能资格，owner 继续 AUDITED。

## Y17–Y19：成熟逐元素族批量绑定（重新定界并 AUDITED）

2026-09-19，baseline `dff6a509`，owner `/root`；产品仅属于
`compute_provider_ggml_vulkan`、`compute_route_vulkan`，Graph 是普通入口和直接消费者。
这是既定 F03/F04、缩放/残差/门控及通用数值链的成熟 Provider 接入，沿用 Y1–Y16 的设备、
资源、失败、完成和交付边界。当前 registry 的 AUDITED 重新限定为下表，不沿用旧自写算子队列。

固定 clean ggml `456172ec` 的 `ggml-vulkan.cpp` supports_op、`unary.comp` 和 `ggml.c`
已核对：Abs/Neg/Exp/Tanh/Sigmoid、Sub/Div/Repeat 均有实际 Vulkan 实现；Sub/Div 只允许
右侧向左侧广播，不能沿用 Add/Mul 的换序。当前 HYLE `region.cc` 仅绑定 SiLU/Add/Mul，
`vulkan_program.cc` 又要求 RMSNorm/门控 anchor，故只扩 enum 映射无法接通普通 Direct。
这里不宣称 vendor 所有 dtype/算子合格，先沿当前 F32 dense rank≤4 域完成一个有限族。

| FIX | 固定实施范围 | 验收与成本 |
|---|---|---|
| Y17 | 私有 typed recipe 批量映射 Abs/Neg/Exp/Tanh/Sigmoid 到原版 `ggml_unary`；保留已有 SiLU，不增加 HYLE kernel。scalar 映为 ggml 单元素 tensor，HYLE 描述符仍为 rank0 | 独立 double oracle；dense rank0/1/2/4、尾宽、±0/NaN/Inf、变化输入；F16/BF16、任意 stride、rank>4/空域继续已有 native 路径，不宣称 vendor 已覆盖 |
| Y18 | Sub/Div 保持输入顺序；至少一侧为完整输出 shape。仅左侧需要广播时调用原版 Repeat，再 Sub/Div；Add/Mul 继续合法交换 | scalar/vector 在左和右、非交换结果、普通 Direct 和 Graph；Repeat 的一个真实中间 tensor/dispatch 与 gallocr 存储全部保留并计费，不能隐去或写成零成本。两侧都需扩展的域仍走已有 native |
| Y19 | 普通 Vulkan 冷编译按已资格化的语义域接纳连续 managed span，不再需要 norm/gate anchor；保留 published/fork/不支持节点边界、origin adapter、每槽资源与自然完成 | 强校验真实 managed region，独立槽/rebind/旧结果、固定 Recorder、原生前后缀和失败/drain；以仍属 native 的 ReLU 迁移原边界故障夹具，不能删除此风险或让夹具失效 |

本批只固定接入/语义选择，不预称性能最优；同一 Provider 的算法/融合、资源及完成封装复用。
P0 为原有 managed 编码、边界 copy，加上具名的可选 Repeat；无运行期搜索/计时、新执行器、
热 metadata、公共 API、依赖或构建变更。所有未支持语义由当前 native 实现承载，比较前不删它们。
Log/Sqrt/Rsqrt、ReLU/GELU/Softplus、LayerNorm 和半精度 vendor 域各有数值/布局资格需求，
不借这一族顺手接入或重写原厂算法。

预声明：先 Windows strict static/shared 的三个 selected factors（含实际 GPU、现有 OOM/关闭/
失败和源码合同），DLL邻置、PATH unset、validation layer，构建 parallel2且重型验证串行；
随后 Air/Mini 同候选、真实本地权重片段、source/static/shared consumer 和整包适用 P1 沿主卡
完成。未变公开头和 build/package 输入按身份复用，不额外全库重建。算法/默认规则稳定后才
冻结性能候选，完整边界成本比较决定最终偏好与自有实现取舍。纠正最多两轮；具体原厂数值
不合格时记录该域并重新定界，不放宽 oracle，不把接入测试算作 C1 整包或性能 PASS。

### Y17–Y19：W1 实现与局部验证

`4c309c6e` 实现上述七个原版算子映射、scalar 及有序广播，普通 Direct/Graph 无需 norm/gate
anchor 即可绑定。Provider/执行/资源封装复用，未新增 shader 或原厂算法补丁。
[W1结果](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-pointwise-w1/README.md)
保存同一干净候选的 strict static/shared：各19 Provider、40 Direct，17/16 Graph，以及各两个
源码/真实 MASM 组合 CTest，首轮全部通过、零 skip、无 VUID/同步诊断。shared 少一个既有
static-only 全局分配失败夹具；原混合片段接纳后失败/drain 用仍属 native 的 ReLU 保留风险覆盖。

新增独立公式覆盖 rank0/1/2/4、特殊值、两侧 scalar/vector/column 广播、不可交换结果、Recorder，
以及无 norm anchor 的完整 Graph、公开中间值、独立槽/变输入/旧结果。统一 LOC 为46,989行/
193实现文件，不含测试/bench/vendor，自检PASS，本批产品净增34行。

三个 owner 仍按当前有限卡 AUDITED；本次不签 C1-Y/C1 完成、默认性能优选或自有实现淘汰。
Air/Mini 正准备同一候选；已有 Mac 环境缺少 ggml 原厂构建所需 glslc，工具准备只发生在验证机，
不修改产品构建/依赖锁或降低交付要求。真实模型片段、完整三交付和 P1 等剩余项保持未验收。

### Y17–Y19：Mac 严格编译的夹具修正

同产品候选在 Air 的首次配置通过，原版 shader 与 ggml Provider 产品源码编译通过；Provider 测试
`region_test.cc` 的五处 Vulkan 结构体部分初始化被 Clang 的
`-Werror=missing-field-initializers` 拒绝，尚未开始 GPU 测试。修正只将夹具改为完整零初始化
后设置 `sType`，所有字段值与断言保持不变，不关闭警告、不改产品或性能合同。
原失败日志保留；按 selector 验证受影响 Provider 测试，再以完整源码同步至两台 Mac 继续未完成门。
`4c309c6e` 产品及其已有 W1 事实不变，不因此重跑 Windows Direct/Graph 或启动 P1。
该夹具修正已通过 W1 strict static 的19个 Provider 用例及源码/MASM 两项检查，零 skip、
无 validation 诊断；selected configure、docs、closure 同时通过，116项无关门跳过。

夹具修正 `132e7497` 继续 Air 编译后，进一步发现 route 冷期 origin-copy 的 `PlanStep`
聚合初始化遗漏显式 trailing fields，同样触发严格 Clang 警告；首次构建尚未编译到此 TU，
不将早期 Provider 编译通过误写成全部路线通过。

**实现前有限纠正 Y19-P（AUDITED）**：owner `compute_route_vulkan` / `/root`，baseline
`132e7497`。仅补齐这一个 `PlanStep` 的原有默认值：空 custom、groups={1,1,1}、空
push_constants；不改公开类型、算法、原生指令、资源和热路径。这是本批第一次产品编译可移植性
纠正，预算仍至多两轮。按 selector 验证 route/Graph，随后同不可变候选继续 Air/Mini；无新 P1。

Y19-P 的 W1 strict static selected configure、40个 Direct 与17个 Graph 用例通过，零 skip、
无 validation 诊断；未改 Provider 产品，复用其已通过事实。此处仅补原有默认初始化，不新增
算子或性能优选结论。两次 Mac 初次编译失败均保留，不将未运行的运行时验证记为通过。

`69b3414b` 已通过 Air 全部产品 TU；用 `ninja -k 0` 完成其余测试 TU 的一次编译核对，避免
逐个遇错停止。余下仅有两处夹具类型问题：生命周期测试的三个 `PlanStep` 也需补同样默认值；
原版 allocator-only C 夹具引入的私有头含未使用 helper，沿既有 MSVC 的 vendor include 诊断边界，
仅在该 include 内抑制 GCC/Clang 的 unused-function，不改原厂文件、自有代码警告、算法或断言。
这些为测试专用可移植性修正；不增加产品纠正轮数，不跑 P1。

## Y20：MoltenVK 必需设备扩展（重新定界并 AUDITED）

owner `compute_provider_ggml_vulkan` / `/root`，baseline `ceace753`。Air 严格编译、19 Provider、
40 Direct、17 Graph 和源码合同均执行通过，零 runtime skip；但日志在87次实际设备创建中
重复出现唯一一类 `VUID-VkDeviceCreateInfo-pProperties-04451`，因此 validation gate 为 FAIL，
不能把数值用例通过写成路线验收通过。初次驱动在末尾检查中退出1，原始 receipt 的 RUNNING
不能覆盖这个实际失败事实；保留日志并另记最终裁决，不重跑未改变候选求 PASS。

固定原版 `ggml-vulkan.cpp` 已枚举 `ext_props`，但设备扩展列表遗漏 `VK_KHR_portability_subset`。
[Vulkan 设备创建合同](https://docs.vulkan.org/refpages/latest/refpages/source/VkDeviceCreateInfo.html)
要求设备宣告该扩展时启用它；instance 的 portability_enumeration 已正确处理，不能代替设备扩展。

有限 FIX **Y20**：仅在既有 build-local 精确补丁内，从已取得的 device extension 列表匹配该名称，
存在时加入创建列表，不存在时不请求；不依赖 Apple 宏，不改变 feature、shader、region、队列或
执行期规则。原始 ggml cache/pin 保持不变；更新补丁结果/树/NOTICE 哈希，仍由同一 Provider
CMake owner 和 package 描述投影。不存在设备扩展枚举、重复 dispatcher 或新 runtime。

这是本批第二次产品纠正，继续沿至多两轮预算。预声明为 W1 strict static/shared 与 Air/Mini
同候选的实际 Provider/Direct/Graph、源合同；Air validation 必须零 VUID，W1 验证没有此扩展
的设备仍可执行；Mini 沿已声明的 correctness/portability，不代签 validation 或 Formal 性能。
原完整三交付和其余 C1 MUST 保持未验收；selector 选出的构建/package 门按实际影响执行，
未变无关门按身份复用。只增加冷期枚举结果匹配，热路径不变，不单独再开 P1。

初次 Mac 失败及此前编译修正已[完整归档](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-macos-portability/README.md)，
统一 Compute+Graph LOC 为46,990行/193文件，自检PASS。Y20 已实现条件扩展启用与完整补丁身份；
W1 strict static 的19 Provider、40 Direct、17 Graph 与源码/MASM 两项检查通过，无 VUID。
shared、修正后的 Air/Mini 与本次 lock/package 改动选出的其余适用门仍待完成；不从首次通过
或旧 broad 验收自动推导当前 package PASS。

### Y20 同候选跨平台局部结果

`3c39428e` 已完成本批 W1 static/shared、Air 与 Mini 的实际 Provider/Direct/Graph：分别为
19/40/17、19/40/16、19/40/17、18/40/17用例，全部通过。W1 各两个源合同/MASM CTest，
两台 Mac 各一个源合同 CTest 通过；W1/Air 无 VUID，Air 原87次设备创建错误归零。
Mini 无 validation layer，其唯一专用实例检查由 W1/Air 覆盖；命令预先排除该项，其他实际
用例零 runtime skip。shared Graph 继续少一个已有 static-only 分配故障夹具。

[同候选证据与明确限制](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-y20-cross-platform/README.md)
保存所有原始命令、日志、二进制哈希、工具身份和 selector。两个 Mac 验证副本未回写产品；
重型验证串行、parallel2，Windows DLL 邻置/PATH unset。没有再次运行 P1。

Y17–Y20 的这批 correctness 可停止重复运行；三个 owner 仍 AUDITED，完整 C1-Y/C1 为
INCONCLUSIVE。新 lock/package 的其余 selected 门、真实权重、完整三交付、默认优选规则与
性能/淘汰仍未完成，CPU oneDNN 的既有性能资格 BLOCKED 不变。统一实现LOC仍46,990行/
193文件，已核对计数路径字节未变；清理四份过期源码包回收约125MiB，不把清理或证据算新能力。

### Y4 交付消费者迁移与剩余验证

baseline `20517db8`，owner `/root`。普通 Graph FFN 样例仍断言五个节点全部为 native dispatch；
当前两个上投影、下投影和应用 affine 共四个 native dispatch，独立 SiLU 门控已由一个 ggml
managed region 承载。有限迁移仅将报告断言改为4 native / 1 managed；保留五节点、存储报告、
独立 double oracle、变化输入、重复运行和原 Graph 销毁后的资源寿命检查，不改产品/API/算法。

预声明沿 Y4 执行同一不可变源码快照的 Windows/Air source archive、installed static/shared
独立 consumer，迁移 SDK 并隐藏 producer，关闭其他计算路线，离线依赖及许可证/exports 检查；
Windows 继续 DLL 邻置、PATH unset、非交互错误模式。新 lock 选出的广组件门随后完成，
不重复已结束的 Y17–Y20 focused correctness，不为样例维护启动 P1。未运行交付仍为未验收，
该 FFN 的生成权重不代替主卡的本地2B真实权重片段。

### Y4/Y20 交付与所选回归结果

`351ec13d` source snapshot 与 `3c39428e` 产品相同，已完成 Windows/Air 同一独立源码包的
source、relocated static/shared 普通 Direct/Graph，合计12个 consumer；保留原生 MatMul、
ggml managed gate 和应用 SPIR-V 的真实混合 FFN、独立 oracle、变化输入及资源寿命断言。
Y20 新 lock 选出的112个 gate 全部通过：96基础/逐头等、1组件合同、15组件三交付；8个未选
项保留 selector 理由。[完整交付证据](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-managed-delivery/README.md)
保留首次离线 cache 缺锁定 Highway archive 的失败，补入已有认证归档后只重跑失败门。

重型验证串行、parallel2；Windows DLL 邻置/PATH unset，Air 零 VUID，没有 P1。先核对归档
再清理 Windows/Air producer 及45个重复组件 build，累计释放约10.63GiB逻辑字节；保留 SDK、
独立 consumer 和共享缓存。当前统一 Compute+Graph LOC 为46,990行/193文件，自检通过。

随后使用上述已验证 SDK 完成[真实权重归一化片段](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-real-weight-norm/README.md)：
Windows/Air static/shared 共4个独立 consumer、72次 Direct/Graph 提交通过，实际本地2B
BF16 embedding 和第一层 RMS 参数、1/17/128行、独立 double oracle、双槽/变输入/旧结果、
输入不变和配额退役。Graph 严格绑定一个 ggml managed norm；不把 BF16 Direct 的 native
路径标为 ggml。该新 consumer 的 source delivery、真实 Gather/gate/residual/state 仍未完成。

原模型13文件重新校验通过；C0登记的 Windows reference/量化派生二进制路径已不存在，
按原 recipe 恢复或找到保留副本后才供 C3/C6 消费；不因此撤销历史收据或重新下载原模型。
三个 owner 保持 AUDITED，完整 C1-Y/C1 仍 INCONCLUSIVE；剩余能力映射、四路线本包资格、
默认性能优选与替代实现淘汰待完成，CPU oneDNN 性能资格 BLOCKED 不变。

## Y21–Y24：TopK 双输出成熟实现接入（重新定界并 AUDITED）

2026-09-19，baseline `d27c14a387f17d62c43c2eb09c3148c5fa5a6374`，owner `/root`。
`compute_route_vulkan` 拥有 Y21 的冷计划/SSA/原生适配，`graph_route_vulkan` 拥有多输出
lowering/发布接线，`compute_provider_ggml_vulkan` 拥有 Y22–Y23 的成熟算法语义与资源组合；
三者共同完成 Y24 普通消费者。公开 Program/Results/Recorder 和自有 SIMD 不扩改。
既有交付队列已全部结束且归档，产品源码与已完成的有限调查身份相符。
[已登记源码审计](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-topk-source-audit.json)
只作实施依据，不作产品正确性或性能证据。本次不重开调查，不续写其他自有算子队列。

### 具名需求与来源

- 冻结矩阵 F04.6：router F32 score → values + I32 indices，k=1/配置值/E，E=256/288/384/512/896；
  同分稳定、NaN 最后、largest/smallest、±0 同分；两个输出的资源和完成合同与普通入口一致。
- 沿既有 schema 的任意合法轴、rank1–8、byte stride/broadcast 与空非选择域，不把成熟库
  rank4/连续最后轴接口差异直接写成 HYLE MUST 缺席。F04.6 明列 F32/I32，消费方为浮点
  router score 和整数 indices；本批固定 F32 values + I32 indices。半精度 TopK 不是本行
  已冻结消费档位，当前明确拒绝且不宣称四路半精度 TopK 完成；已存在的其他路线能力保留。
- ggml pin `456172ec733a135778adcd32d00e576a58232e45`。`ggml_top_k` 明确不保证顺序；
  `argsort.comp`/`argsort_large.comp` 已有完整小/长行 bitonic 排序拓扑，但仅比较浮点值并
  在 DESC 反转输出，不能直接满足稳定同分和两方向 NaN-last。
- `ggml_get_rows` 已有 F32/I32 等 storage 路径；其 I32 shader 使用 uint load/store，可用于
  原始 F32 位载荷收集。reshape/view/必要 cont 在 Provider 内保留，发布 view 必须核对
  ggml 外部 buffer 分配约束；不能把有 view_src 的 tensor 当独立存储导入。
- HYLE 公共 Program/Results/Recorder 已表达多输出；缺口在冷 PlanStep、RegionNode、SSA、
  published/liveness 和普通 prepare/Graph lowering，毋须新的公开执行器或热期分派。

### 固定有限 FIX

1. **Y21 冷期多输出接线**：PlanStep/RegionNode 使用输出 ID 集合，实际 arity 推导、逐输出类型
   和 SSA 校验、完整 input 拓扑、重复/越界输出拒绝，逐输出 last-use/published/资源规划。
   Custom callback 保持已有单输出合同，明确拒绝其他 arity。已有单输出消费者完整迁移。
2. **Y22 成熟排序语义适配**：保持固定 ggml 两个 argsort shader 的排序网络、dispatch、barrier
   与 scratch 管理，仅把比较关系改成原始 F32 ordered key + 原始索引；合并 ±0、NaN-last、
   有效元素在 padding 前。按 order 构造键并统一顺序写回，不能在 DESC 再反转稳定 ties。
   更新既有精确 build-local patch 身份/NOTICE；原 pin/cache 不变。该范围明确重新定界原
   C1-Y“不更改 ggml 算子算法”约束，只允许所列比较语义修正，不能借机自写完整 TopK。
3. **Y23 Provider 双输出组合**：原版 argsort → 前 k index view/必要 materialization → 原版
   GET_ROWS 取原始位 values，保留 Provider graph/gallocr/Work。按实际节点数预估 metadata
   capacity，完整溢出/OOM 回滚。正常 dense last-axis region 保留上下游组合；其他布局复用
   已有 View/Copy 整理，原始位保留、还原输出逻辑轴，所有真实 copy/scratch 如实计费。
   长行由原版设备特征选择；必须验证两台适用 Vulkan 设备的实际准入，不静默 CPU fallback。
4. **Y24 普通消费者与资源验证**：独立 stable scalar oracle + values 位比较；实际 E/k、尾宽、
   k=1/k=E、两方向、flags、全相等/全 NaN/NaN payload/±0/±Inf/subnormal、多轴、strides、
   空域和拒绝域。Direct/Graph/Recorder 双输出，分别只发布一个输出、下游复用、两个槽、
   三轮变化输入、旧结果保留、原 Graph 销毁、关闭/失败/OOM 和现有 mixed-region 回归。

### 成本与验收边界

- 原版 bitonic 完整排序 O(rows * padded_width * log²(width))，不是 O(N) selection；输出
  indices 全排序存储、长行 ivec2 scratch、必要 view materialization 和输出收集全部计费。
  k=1 的复杂度明确保留，不在未比较时宣称 SOTA 或默认赢家。
- 新逻辑主要发生在准备期；热期仍为同一 native region 执行、绑定和原有完成通道。
- Windows strict static/shared、Air/Mini 实际 GPU、独立 oracle/生命周期和源合同；Windows/Air
  三交付的普通 TopK/图消费必须同批落实，不拖到 C7。新 lock 的 selected gates 遵守现有
  selector，未变且身份匹配的门复用；不修改公共仪器，不为文档运行 P1。
- 调查仅锁定 ggml Argsort/GetRows 两条现成路径，最多60分钟；之后进入有限实现，至多两轮
  纠正。遇真实数值/布局/资源反例精确记录；不不断切换 peer/扩大研究。

### Y23-P1：存储重解释必须保留原图依赖

首轮 W1 strict static 编译通过，19 Provider 与两个源码/组合 CTest 通过；Direct 41/50
通过，9项 TopK/相关失败寿命测试发生主机异常，Graph 在首个新 TopK 用例中止。原失败及
一次单例构图诊断已保留，未把未结束的 Graph 或任何 TopK 签 PASS。

诊断显示值输出只展开1个节点；锁定源码 `ggml_view_tensor` 只建存储别名，没有设置 op/src
依赖。因此 raw-score 与 gathered-value 两处重解释把父运算从图中遗漏，后续 allocator 查找
未登记的 view root 失败。这是 HYLE 适配使用接口错误，不是成熟排序算法或测试 oracle 问题。

本批第一次有限纠正（owner `compute_provider_ggml_vulkan`）：只把这两处同形状别名改为原版
`ggml_reshape_2d/4d`，保留 op/src 与四字节存储重解释；metadata 节点数不变。删除临时诊断
输出，先验证失败的 TopK/资源与 Graph 用例，再继续同候选的剩余适用门。不修改 shader、
数值门、公共治理或性能仪器，剩余纠正预算一轮。

### Y21–Y24 实现与 W1 static 定向结果

冷 Plan/SSA、Provider region 和 Graph 已支持 TopK 两个输出；原版 Argsort/GetRows 组合保留
成熟执行与存储机制，布局适配沿既有 View/Copy。一次产品纠正后的定向结果为 **10 TopK +
1 mixed-prefix 失败寿命 + 2 Graph = 13 个用例通过**，不是完整 C1 或跨平台签收。

其中 byte-origin 用例初次误将非对齐 offset 传给 `borrow_ready`，随后又对非 dense slice 使用
零拷贝 reshape；两者都违反既有接口准入，失败如实保留。最终用合法 Tensor::view 保留非 dense
描述，并另测满足设备对齐的非零 native offset 捕获输入。只修测试构造，不弱化产品检查，
不修改数值参考或门限。前10项与最终剩余3项按相同产品源码复用，不重复整个未变测试集合。

独立原始位/稳定排序参考覆盖实际专家宽度与尾宽、1025/4097 长行、任意轴/rank8、NaN payload、
±0/Inf/subnormal、动态/捕获输入、两个输出、Recorder、OOM、部分接纳后的排空和旧结果保活。
Windows DLL 放在 EXE 旁、运行时 PATH unset、validation layer 无 VUID/同步错误；未运行 P1。
原始失败、单例诊断及纠正后的日志绑定于
[定向记录](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-topk-w1-focused.json)。

当前统一脚本为 **47,189 行 / 193 实现文件**，不含测试/bench/第三方，自检通过。三个 owner
继续 AUDITED；shared、Air/Mini、TopK 三交付、新候选所选门及整包性能/淘汰仍未完成。
CPU oneDNN 性能资格 BLOCKED 不变，不把先前 managed 片段的三交付代签给新 TopK。

Y24 的交付样例现已加入实际 TopK：Direct 重绑定三组输入并对两方向 values/indices 作稳定
排序位级参考；Graph 执行 Abs → TopK，values 经 Neg、indices 经 Copy 后独立发布。复用
既有 compute/graph C++ package 消费者，不新增公共工具或执行接口；样例的三交付待实际运行。
同时纠正 Graph Vulkan 公开头“拒绝全部多输出”的过时注释；删除整行注释后代码逐行相同，
不为该注释重跑构建或性能。产品语义候选仍为 `5c1e27a8`。

### Y23-P2：内部别名必须计入常量根的字节起点（AUDITED）

2026-09-19，baseline `5b22060111e3c4f0ed1a0c20ca80f64e6f984aa6`，owner `/root`，
产品 owner 仅 `compute_route_vulkan`。已安装 shared SDK 的独立最小 consumer 复现：合法
rank8、dense、byte-origin=1 的捕获输入在 TopK 准备/建槽时返回
`ggml Vulkan region native buffer range`。原始输入由 `Tensor::view` 正步长 slice 得到，
没有非法 borrow 或非 dense reshape。失败与重现源码见
[原始记录](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-topk-origin-repro.json)。

源码因果已收敛：`lower_topk` 的私有降秩别名没有独立 `constant`，`lower_region` 只检查
alias.byte_offset，没有合入 alias.root 常量的 native_offset；`allocate_values` 实际绑定却
包含两者。原版 ggml 的四字节准入拒绝正确，错误属于 HYLE 冷适配，不能弱化该检查。

本次使用 Y21–Y24 剩余的一轮产品纠正，固定为：

- 对齐证明按真正的常量根 native_offset 与累计 alias.byte_offset 合并；先分别取余再合并，
  不引入地址求和溢出。动态输入起点仍未知，沿已有 byte-addressed Copy 适配。
- 已对齐的根/别名继续直接进入 region；不增加公开 API、对象字段、热期检查或新算法，
  未对齐域只支付原本就应有的一次布局 Copy。第三方 pin、shader 和 provider 实现不改。
- 定向回归覆盖 rank8 捕获 dense View 的0/1/2/3/4/7起点、值与索引原始位、输入保护区；
  Graph 常量经过 slice 后的累计偏移，包含原始起点与别名偏移相加后重新对齐，校验不多复制。
  两个槽、旧结果与普通执行沿既有 TopK/managed 生命周期回归。

先运行 W1 修复反例及 Graph/managed 定向集，再完成 selector 对本次私有实现差异选出的门、
W1 shared 与 Air/Mini 对应 route/Graph 验证，Windows/Air 的 source/static/shared 普通消费。
不重跑未变 shader 的独立算法资格，不启动 P1。已通过的旧候选局部结果保留其原身份；
新反例使本域尚不能签收。旧候选96个 native 门和1个组件合同门已完成，15个通用交付门在
首门途中明确中止，不能写112项通过；修复候选按实际影响接续未完成门，不丢弃它们。

### Y25：排序输入的设备 descriptor 对齐（重新定界并 AUDITED）

Y23-P2 的 `a50b2ef4` 修正了根起点遗漏，W1 strict 编译通过，但新增 Direct 回归执行失败：
原版 `ggml_vk_tensor_subbuffer` 触发 `allow_misalign || misalign_bytes == 0`。Graph 回归未运行，
不能写通过；[失败记录](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-topk-descriptor-failure.json)
保留候选、二进制和日志。Y21–Y24 原定两轮纠正预算已用尽，本节按具体新反例重新定界，
不继续沿旧预算试错，不扩大到其他算子或 Provider 算法。

baseline `a50b2ef405662e12af8c701555c8bfad0601fedd`，owner `/root`，产品 owner 仍仅
`compute_route_vulkan`。固定 ggml 的 `ggml_vk_argsort` 14322–14323行调用不允许偏移的
`ggml_vk_tensor_subbuffer`；后者8300–8304行要求 `minStorageBufferOffsetAlignment`。
已接逐元素/norm 的偏移感知 shader 可以接受四字节起点，因此不能把所有 region 输入统一
升级对齐、给无关已合格域增加 Copy。Provider 公共准入不是本次新增 API。

有限 FIX：在冷 region 的已有逐值分析中标记直接被 TopK 消费的输入，仅这些输入要求
max(4, minStorageBufferOffsetAlignment)，其他输入继续四字节；根起点和累计别名偏移分别
取余再合并。中间结果由 ggml allocator 自行对齐。新增标记只在准备栈帧中的临时向量存在，
不增加常态 Program/Invocation 字段、热期搜索或第三方补丁。回归增加设备对齐起点与
root=1、alias=alignment−1 的重新对齐组合，同时保留4字节但未满足设备对齐的反例。

本次源码调查仅上述两个具名函数、最多30分钟，产品修正最多一轮；如仍失败即记录具体
blocker/裁决，不再在本节追加尝试。验收范围沿 Y23-P2 的两 route factors、适用平台和
三交付；已有 managed/原版排序验证只按未变身份保留，不签 C1 整包或性能。

### Y25 实现与适用平台/实际交付结果

产品 `7f4a3fb6`、测试/交付快照 `5f4df55b` 已完成根与别名起点相加、逐输入原厂对齐需求的
冷适配。只向直接被 TopK 消费的输入施加设备 descriptor 对齐，其他原有偏移感知核不增加
Copy；Graph 的root=1、alias=alignment−1恢复零复制。没有新的算子算法、公开API或热期决策。

[完整定向证据](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-topk-origin-correction/README.md)：
W1 static 51 Direct+20 Graph、shared 51+19，Air/Mini static各51+20，共283项通过；
Windows/Air同一source archive的源码、relocated static/shared各6个普通消费者通过，新增
rank8捕获起点1/4/7、值/索引双输出保持原始位，并保留既有混合FFN及应用SPIR-V消费。
两个新增回归另有先行定向运行，不重复计入283。首次Graph夹具缺少私有声明及include路径的
编译失败保留；补齐测试include后产品不再变化。

Y25最终selector选两个route factors与compute/graph交付；Windows按两factor重新配置/构建
后的两个EXE hash与已通过测试完全一致，因此复用对应结果，未重复运行。原新lock选出的
96个native门及1个组件合同仍保留`5b220601`身份；15个通用交付门正在固定Windows验证副本
`C:\hyle-scratch\active-vk\v`的`5f4df55b`快照上串行执行，不签112门聚合PASS，也不反向
覆盖canonical产品。实际Vulkan三交付不替代这些通用门。

W1 DLL邻置/PATH unset、W1/Air无VUID/同步错误，没有P1。归档逐文件hash核对后清理两平台
producer和展开源码，释放1,933,074,359逻辑字节；SDK、consumer及共享依赖保留。统一LOC
为47,202行/193实现文件（不含测试、bench、第三方），自检通过。

三个owner保持C1 AUDITED；本组修复和上述适用验证停止重复运行。通用交付、checked索引/
Scatter/Scan、完整C1真实权重片段、默认性能优选和淘汰仍待完成；CPU oneDNN资格BLOCKED
不变，64行/256格共同矩阵尚未签收。

## Y26：受检 Gather 的原版 GetRows 接入（重新定界并 AUDITED）

2026-09-19，owner `/root`，baseline `04643d187d6255822bd7c46fdbf8afffb1ad498d`。
当前三个 Vulkan owner 仅增加本节有限 FIX；旧 CPU BLOCKED、Y25 已完成的验证与正在运行的
固定快照通用交付保持各自身份。本节满足 C1/F02.1 的 embedding/索引读取，Scatter、Scan
另行审计，不以本节代签它们或整个 C1。

### 需求、承载与冻结边界

- 主消费：本地 2B BF16 embedding 表按 I32/I64 token 索引读取，再接已有 norm/逐元素 region；
  正交消费是有 stride、非末轴、标量/多维索引与原始位载荷的表查询。共同 Gather 合同要求
  checked 在写该算子 output 前检查全部索引，越界返回 OutOfRange，不能留下部分输出。
- 锁定 ggml `456172ec` 的 `get_rows.comp` 支持 I32 的原始字读取；F16/BF16 原厂路径经过浮点
  转换，不能证明任意 NaN payload 原位保持。该 shader 直接用索引寻址，没有范围检查。
  `region.cc` 的 TopK 已证明 I32 GetRows 与真实 reshape DAG 的资源组合。
- 裁决：BORROW 原版 GetRows，保留其 allocator/runtime；HYLE 仅补索引检查、原始字布局与
  有条件输出 Copy、完成/失败适配。输入不下载到 CPU，不另写 Gather payload kernel，不改 vendor。
- 普通入口不变：`prepare(context, Operation(Gather{...}), arguments)`、Graph 普通 Gather 节点、
  Recorder 显式输出。准备时固定布局/Provider；提交时不探测候选、改规则或分配检查容器。
- 域为已准入 plain F32/F16/BF16/I32/I64/Bool、I32/I64 indices、任意合法轴/rank≤8/正或零
  byte stride 与 byte origin；空索引无 payload，非空索引即使结果为空仍遵守检查合同。
  非空物化与原厂域仍受设备 buffer/workgroup 和 I32 元素数量限制，超界冷拒绝，不截断。

### 有限 FIX 与成本

| ID | 允许改动 | 必需证明 |
|---|---|---|
| Y26-A | 私有 Program 标记检查状态；每个需要检查的运行槽冷建4字节 host-visible storage buffer，Recorder 合并时保留各状态 owner。只有带检查的 Run 建可选状态集合；提交前归零/适用 flush，GPU 检查只在失败时置位，完成后适用 invalidate/读取 | 不与 scratch 复用；host 读写只在无 pending/真实 GPU 终态；非 coherent 规则、quota/OOM、独立槽、取消/丢弃/close/部分提交；算法失败退休该槽，不毒化 Context |
| Y26-B | GPU 检查/转换原始 I32/I64 indices 为连续安全 I32；原版 GetRows 只接已检查安全索引。复用 View/Copy/Fill/Concat 将任意轴布局变成原始 I32 字行；窄字奇数行补齐，结果以原有 Copy 的受检变体恢复原描述 | 全部原始位保留；负数/边界/I64高位、非对齐起点、广播/标量/空/尾块/rank8；任何坏索引使本 Gather 输出保持原样，原厂不得越界 |
| Y26-C | provider 私有 region 只增加明确的 unchecked I32 二维行读取域；普通 checked 入口由路线适配后调用。Graph 复用原有纯 SSA lowering，Recorder 按原有重映射传递检查标记 | Direct/Graph/Recorder 实际走 GetRows；两个槽与旧结果保留，失败后新槽可用；不增加公开 API、通用执行器或其他 owner 改动 |
| Y26-D | 位级独立 oracle、故障/生命周期/真实 BF16 表片段与普通交付消费者、用户说明 | W1 static/shared、Air/Mini 对应 factors；W1/Air source/static/shared；selector 适用门，未运行不签 PASS |

静态账单：检查 O(indices)，取数 O(output)，必要 layout/行补齐及最终条件 Copy 分别计费；
原厂 managed 编码成本保留。合法连续整字行复用 View，准备不能从 dense 推导未知绑定起点。
检查状态常驻4个逻辑字节及实际 Vulkan allocation/quota，host 只观察状态，不观察 tensor payload；
非检查程序无状态 buffer/集合，Run 仅增加一个私有可选 owner。每个受检原生 batch 带 shader→host
可见性屏障，最终父完成点覆盖全序列；采用
[Vulkan host/device 可见性合同](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)
和[非 coherent mapped memory 合同](https://docs.vulkan.org/spec/latest/chapters/memory.html)。
不宣称性能优选，最终同合同比较必须包含这些必要成本。

调查限于已登记 ggml GetRows 与 Vulkan 同步/内存合同，累计最多60分钟；产品纠正最多两轮。
先完成语义/资源门，API/算法固定后沿 C1 的原 P1 bundle 预算验收，不因本节开启微基准或
重跑 CPU 性能。失败保留具体反例并按协议裁决；全部 Y26 门完成前 owner 保持 AUDITED。

### Y26 实现与 Windows static 首轮证据

已实现Y26-A–C：复用原版 I32 GetRows 载荷读取，HYLE 补GPU索引检查、必要原始字布局/补齐和
条件发布；每个受检程序运行槽拥有私有4字节逻辑状态，真实GPU退休后观察错误。普通无检查
程序不分配状态buffer/集合。失败槽按原协议退役，Context及新槽仍可用；未改vendor或公开API。

[首轮固定工作树证据](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-gather/w1-static/README.md)
绑定baseline `858b035b`及完整patch/hash：W1 strict static的19 provider + 60 Direct + 21 Graph
共100个用例全部通过；新增10个定向用例不重复计数。公开头、原版source合同及实际MASM parent
组合均通过，Vulkan validation无VUID/同步错误。Windows实际EXE使用DLL邻置/PATH unset和
非交互错误模式；编译类检查保留编译器环境。首轮没有产品纠正或P1。

此前Y25的15个通用交付门已经在`5f4df55b`固定验证副本完成，共45条source/static/shared路径。
证据继续保留独立身份，不将`5b220601`的96 native + 1组件合同改签到其他候选。
归档核对后清理45个重复producer目录，共9,492,056,710逻辑字节；SDK、consumer和共享cache保留。

Y26 shared、Air/Mini、真实本地BF16 embedding片段和source/static/shared消费仍未运行；
后者的输入/独立oracle及外部consumer已准备，不能算运行PASS。三个owner保持AUDITED，完整C1
Scatter/Scan、固定优选/淘汰和矩阵签收未完成，CPU oneDNN资格BLOCKED不变。
当前统一实现LOC为47,554行/193文件，不含测试、bench、第三方，自检通过。

### Y26-F1：原版 GetRows 的冷调度上限（第一轮有限产品纠正）

baseline `f6c37c33`；原产品仍为`3a59c5d8`。原版`ggml_vk_op_f32`为GetRows限制Y/Z并由
 shader循环覆盖，X必须按512个实际shader lane覆盖，`ggml_vk_dispatch_pipeline`在超设备上限时
触发断言。当前checked布局适配未拒绝过宽行，unchecked I32成熟直通还绕过512-lane准入。
已用真实Context的冷准备测试降低缓存能力界限，**不分配大payload、不提交非法dispatch**，
复现3次错误接纳；原始失败保留在Y26归档。该注入只证明拒绝合同，不代替实际GPU正向验证。

固定纠正：仅在`vulkan_region.cc`的managed匹配及Gather布局适配中检查512-lane能力与
`row_words <= maxComputeWorkGroupCount[0]*512`；超界返回Unimplemented。
增加一个checked/unchecked与workgroup/grid边界回归。原GetRows算法、shader、热期、ABI及
状态语义不变。按selector验证受影响route factors，未变provider按相同身份复用；平台与实际消费者保持原范围。消耗两轮产品纠正预算中的
第一轮，不扩展到其他算子或公共治理。

### Y26-F2：真实 BF16 宽行暴露的上游 GetRows 工作组配置（第二轮有限纠正）

2026-09-19，owner `/root`，baseline `807bc22bb1b50cc6eaf9bc03d7f8a72be417a597`。
本节先固定 SCOPED/AUDITED 边界；三个当前 Vulkan owner 仍为 AUDITED。

- F1 的受影响 Direct/Graph 在 Windows static/shared、Air/Mini 分别通过82/81/82/82例。
  原 provider 用例按未变身份复用；这些用例没有证明任意宽行的真实模型片段。
- 本地2B的真实BF16 embedding子表（34行、宽2048）在source consumer首次实际运行失败：
  Direct第一个输出的元素1024为`0000`，独立原始输入为`bc11`；此前1024个元素一致。
  这是位保持合同的明确反例，不放宽容差，不继续推进其余交付来寻找PASS。
  [失败归档](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-gather/wide-row-counterexample/README.md)
  保存首次运行、最小诊断、宽行回归及先前独立的consumer编译失败。
- 锁定ggml `456172ec` 的 `get_rows.comp` 固定 `local_size_x=512`，I32专用编译使用uint
  原始读取/写入；但 `ggml-vulkan.cpp` 创建 `pipeline_get_rows[GGML_TYPE_I32]` 时将
  `wg_denoms[0]` 写成1024。host只录制`ceil(row_words/1024)`组，宽于512字的行会漏写。
  此前小尺寸边界没有跨过原生工作组，F1的512-lane上限仍是修正后应遵守的准入界限。

固定FIX仅为：在既有带输入/输出摘要的vendor patch manifest中，将上述I32 pipeline的单个
分母1024纠正为512，并同步文件结果摘要、NOTICE及既有dependency lock的对应内容/说明摘要；
保留原pin、shader、算法、原生资源和ABI。
不修改上游checkout或共享source cache，不新增payload kernel，也不扩公共构建/性能工具。
增加跨511/512/513/1024/1025字边界的普通Gather位级回归，包含窄字适配与原生I32直通。

P0：恢复覆盖整个输出所需的工作组数；复杂度O(output)，没有新增metadata、拷贝、分配或同步。
验证限于selector选中的provider及两个直接route factor、原source/补丁合同、W1 static/shared、
Air/Mini，随后继续既定W1/Air真实权重source/static/shared旅程；原失败和每次候选身份保留。
本次不运行P1，正确性通过不签性能胜出。此FIX消耗Y26两轮产品纠正预算中的最后一轮；若仍有
产品失败，按具体反例裁决而不无界追加修复。完整C1仍缺Scatter/Scan、优选/淘汰和矩阵签收。

F2单值补丁现已实施。W1新增宽行回归先在原实现复现FAIL，纠正后60个dtype/边界/checked
组合逐位通过；这只签该回归，不替代完整平台与真实片段交付。结果及源码身份见
[W1定向纠正记录](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-gather/wide-row-counterexample/w1-correction.json)。
完整selector的其余适用门尚待执行，未签全量PASS；本轮未运行P1。统一实现LOC仍为
47,564行/193文件（不含测试、bench、第三方）；F2只改vendor配置及测试，HYLE实现集合未变。

### Y26-F2 平台回归与实际三交付结果

固定候选`d54ddc5a`的[平台/交付证据](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-gather/platform-delivery/README.md)
已归档：Windows static/shared分别102/101例，Air/Mini static分别102/101例，共406项通过；
Windows两种构建的source及MASM组合检查各2项通过。Mini的validation专属用例因该层缺失明确
不运行、不计PASS，其他三组validation无VUID或同步错误。

Windows/Air以同一source archive完成source、relocated static/shared共六条实际消费。真实本地
BF16 embedding子表34×2048，1/17/128行的Direct与Graph各5次变化输入重放，独立位级Gather/
FP64 norm oracle、旧结果、受检失败/新槽恢复及owner/quota均通过。Direct用冷建3个Gather/2个Norm
槽保留快照并交替释放下游持有的工作输入；Graph用两个完整片段槽。最初consumer覆盖仍被Norm
持有的Gather结果，被产品正确拒绝；仅纠正冷槽安排，原失败保留，没有弱化寿命检查。

Windows/Air的具名producer与展开源码在归档核验后合计清理1,876,666,772逻辑字节，保留SDK/
consumer/cache。此组已通过的验证停止重复运行；全局selector适用门在C1固定候选收口，旧候选
证据不改签。三个owner继续AUDITED；Scatter/Scan、完整C1片段、优选/淘汰和矩阵仍未签收，
没有运行P1，CPU oneDNN资格BLOCKED不变。

## Y27：F32 Scan 原版前缀和接入（SCOPED / AUDITED）

2026-09-19，baseline `6900c5db`，owner `/root`；产品owner仅
`compute_provider_ggml_vulkan`和`compute_route_vulkan`，Graph为真实直接消费者。
这是C1/F02.4中浮点前缀消费及公开Scan完整性的有限子集；I32受检前缀、Scatter另行裁决。
先固定本节并提交，随后改产品代码；不借此重开CPU oneDNN资格或改公共治理。

源码审计限定锁定ggml `456172ec`的`ggml_vk_cumsum`、三份`cumsum*.comp`、
`ggml_vk_op_f32`与supports_op，以及当前HYLE Scan语义/region适配。调查到此停止。
原版支持F32 inclusive、subgroup arithmetic与full-subgroup能力，保留按行数/宽度选择的
单遍或两遍原生程序。HYLE共同合同允许F32重结合；不能把该实现当作I32溢出检查。

具名静态反例：单遍shader把XYZ展开为row，host对513行分配512×2组，shader没有总行数检查。
不能把513行直接交给原版；同时两遍路径的输入要求设备descriptor对齐。固定适配是将行数
展平为2D，按原厂512行/262144行的两级XYZ tile边界分为完整前缀与余数，分别cumsum后原版
concat；有偏移的段先原版cont以保有独立对齐。最多三个Scan，不拆每行/每kernel，不改vendor
shader、pin或运行时。精确覆盖的行数保持一个原版cumsum。首次候选前复核完整dispatch分支确认
大于262144行还需处理第二级余数，已在首次产品候选之前固定；不是运行失败后追加算法。

| FIX | 有限实现 | 成本与必需验证 |
|---|---|---|
| Y27-A | 私有region准入F32、最后轴inclusive Scan；lower到原版cumsum及上述最多三段DAG | 原有allocator/资源/完成不变；原版单遍O(N)，两遍的块前缀重读成本如实计入，不能预称最优；ragged行额外cont尾段/concat输出，冷tensor节点上界16 |
| Y27-B | 路线将任意合法轴/rank≤8/stride/origin变成dense二维原生输入，Scan后恢复轴；exclusive以零列与inclusive前缀的View/Concat组合 | 复用已接View/Copy/Fill/Concat；空输入无payload，宽1 exclusive直接Fill；必要dense物化/对齐/结果重排分别计费。冷检查I32元素/设备buffer与原厂两类workgroup上限，无热期路由/计时、新状态或payload kernel |
| Y27-C | 普通Direct/Graph独立FP64 prefix与生命周期/文档 | 0/1/17/127/128/129/255/256/257/1023/1024/1025/4096/4097/8193宽，1/17/128/513/1025及262144边界行；中/负轴、rank8、奇数/零stride、byte origin、非二进制/NaN/±Inf；exclusive不以inclusive减当前值。两槽/5次变化输入/旧结果、Graph上下游组合、metadata拒绝、既有OOM/错误排空回归 |

适用验证：selector的provider、Direct/Graph factors，W1 static/shared、Air/Mini；W1/Air
source/static/shared普通片段沿C1固定bundle。首次W1先验证具名边界，再运行该候选适用回归；
测试不得下载模型或把诊断输入写成真实权重。API/算法固定后的等责P1仍按C1原预算，本节不采样。
原厂设备能力不具备时在冷建槽沿既有NativeGraph准入返回Unimplemented，不运行替代算法。
产品纠正最多两轮；具体新反例按协议裁决，不无限改换实现。未运行平台/交付/整包门保持未证明。

### Y27 实现与四构建局部结果

候选`198cccba`已实现Y27-A/B/C的F32接入与普通测试。原版cumsum承载全部前缀数值计算，
没有新增shader、vendor补丁或公开API；冷layout适配及原版DAG分段保持原资源/完成机制。
[局部证据](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-scan/README.md)已归档：
W1 static/shared分别108/107例，Air/Mini static分别108/107例，共430项通过；W1先行6例
不重复计数，source/MASM检查各2项通过。没有产品纠正或P1，Mini缺少validation层的专属用例
明确未运行。所选provider/Direct/Graph回归到此停止重复执行。

数值验证沿已冻结Scan合同使用独立FP64前缀及前缀绝对值和误差界限，覆盖native两类程序、
两级row尾块、任意轴/stride/起点、exclusive、非有限值；两个槽、5次变化、Graph上下游与旧
结果保活通过。三交付、真实C1组合、整包优选与全局package gates仍未签收，不能由此次回归代签。
统一实现LOC为47,759行/193文件，不含测试/bench/第三方，自检通过。Mini四份旧验证源码在
当前源码和原始证据核验后清理344,914,592逻辑字节，保留当前build/cache。

### 后续受检整数 Scan 的有界来源核对（仅审计输入）

当前ggml原版CUMSUM只准入F32。第二个具名来源
[GPUPrefixSums](https://github.com/b0nes164/GPUPrefixSums/tree/98d93a4e9ed2f3c8353119515bf9be90a2e137ad)
已锁到`98d93a4e9ed2f3c8353119515bf9be90a2e137ad`，只读取ReduceThenScan的HLSL/WGSL与许可证，
[输入摘要](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-scan/checked-integer-source-inputs.json)
可复取。源码实现的是U32模加，未提供I32全前缀溢出检查或失败不写输出；其README也将WGPU端
列为测试用途/未完整可移植。不能将它直接登记为已合格Vulkan Provider。

可借鉴其reduce→spine scan→propagate分层算法，checked域仍需有符号宽累加、检查全部真实前缀
（含exclusive最终total）与全局条件发布。局部block前缀越界不等于最终带carry的前缀越界；
不能用总和检查或U32模加代替。当前仅固定缺口与来源，调查到此停止；I32的产品FIX、临时量/
dispatch账单与具体实现须先独立定界，Y27不授权自写整数或Scatter核。

后续[I32独立Y28](llm-moe-c1-tensor-state.md)现已定界、实现并完成候选`f0115f96`的391项
Direct/Graph四构建回归；该域由route的宽整数shader补缺，不属于ggml Provider能力。
ggml产品未因Y28改变，F32原版路径同候选回归通过。完整C1三交付/优选和Scatter仍未签收。

### Y34 所选工程门：已完成

候选`ec240ac9`的[112项所选工程门](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-reduction/engineering/README.md)
全部资格化：96项普通工程检查、1项component-contract、15项默认三交付（45条消费）。
既有Y34三个C++ suite及两个公开头按源码、产物、163条未变编译命令与机器身份复用；
Provider的源码/MASM组合另行执行，未把旧版本门直接代签为本候选。8项未选理由保留。
本次没有产品、测试、公共工具改动或P1；W1/Air此前432例/1296次实际GPU三交付不重复。

归档核验后清理45个重复producer，共9,492,226,425逻辑字节。实现仍48,737行/197文件。
本批验收到此停止重复；C1其余Provider覆盖、完整成本比较/条件规则/存量淘汰与CPU Q2
BLOCKED仍待处理，C1及共同矩阵没有签收。C1全部适用验收完成后通知用户，在C2前等待提问。
