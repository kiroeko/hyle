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
