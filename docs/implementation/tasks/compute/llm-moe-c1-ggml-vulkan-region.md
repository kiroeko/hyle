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
