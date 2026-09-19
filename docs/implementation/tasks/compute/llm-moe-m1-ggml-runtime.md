# M1-G：ggml 可恢复资源运行时与共同源码接入

## Closure card

- component / owner：`compute_ggml_runtime`、`compute_provider_ggml_vulkan` / `/root`。
- baseline：`9401082b116334ca6dc9cfbd4e426bf11c497e0d`；新身份已在 `9b8da003` 以 QUEUED 登记。
- trigger：[M0施工清单](llm-moe-provider-integration-ledger.md)准入四个ggml后端；原CPU owner的
  [BLOCKED条件](compute-provider-ggml-v1.md)要求独立source-patch factor。当前Vulkan已经拥有
  非abort context补丁与build-local source机制，但其独占vendor target不能重复用于第二个后端。
- 主纵切：同一份ggml原版norm→MatMul→状态输出region，冷准备metadata、pool与workspace，
  更换activation后重复执行；下一层模型片段接入复用该seam。
- 正交纵切：两份独立invocation的普通F32工程计算图并发运行，共享只读常量；prepare OOM、
  部分线程启动失败、取消与销毁。native资源测试不冒称普通Direct/Graph或完整模型已经接通。
- scope：共同vendor源码/补丁/三交付；共享metadata owner；CPU fallible pool、持久plan与
  caller-owned workspace；现有Vulkan只做依赖/资源所有权提取与相应回归。
- non-goals：任何自研算子研究/扩张/优化/比较；修改上游数学kernel；模型加载；新CPU Provider
  全量lowering；CUDA/Metal接口施工；调整oneDNN阻塞；公共selector/性能仪器重写。
- dependencies：core/error/mem与锁定ggml；Vulkan原生依赖仍仅由选中Vulkan的闭包取得。
  vendor C类型只在私有实现头，普通用户入口不变化；不产生第二个公开C产品。
- tier：P0。检查资源初始化、allocation/thread/lock、同步与复用；本卡不进行性能计时或自研竞速。
- 预算：资源seam新增源码/设计2小时；每个实现根因最多两次纠正。改变owner/语义/交付形状
  必须重新定界；不能以本卡名义重构Sched/Mem或扩张数学算法。
- 机器：Windows与WSL的CPU正确性/失败与三交付；Air/Mini的CPU可移植性与适用三交付；
  现有Vulkan受影响资源旅程在RTX4070/Air验证。重型任务串行、parallel2，无UAC/电源策略改动。
- stop：本卡有限FIX及适用门完成后关闭独立运行时范围；原CPU Provider仍需自己的新有限卡，
  不能把source-patch完成当作Provider/64域已覆盖。

## 已核对的上游资源事实

锁定研究/拟实施源均为独立ggml `456172ec733a135778adcd32d00e576a58232e45`：

1. `src/ggml.c::ggml_init`通过GGML_MALLOC创建context，失败abort。现有Vulkan补丁已把
   context/mem_buffer失败改成NULL且检查padding/alignment；提取复用，不另写分配器。
2. `src/ggml-cpu/ggml-cpu.c::ggml_threadpool_new_impl`对pool/workers分配直接解引用，
   pthread/Win32线程创建失败assert；POSIX mutex/cond初始化结果未检查。
   需要精确回滚已经创建的线程、同步对象与完整大小的worker内存，保留原生失败码。
3. `ggml_graph_plan/ggml_graph_compute`公开支持持久threadpool与调用方work_data；同一个pool
   保存当前graph/plan，因此每个并发execution slot有独立pool/plan/workspace。单slot线程兼容，
   prepare/run/close不并发；不同slot可以并发。完成后再释放arena和workspace。
4. `ggml_cpu_init`每次入口先分配/释放一个临时context，且compute每次都调用它。
   把time初始化放入原有one-time分支，消除此项热路径资源创建；数学查表内容保持字节不变。
5. ggml arena内构造函数以容量不变量保护metadata，不是可泛化的fallible建图API。
   私有builder必须在调用前按tensor slots/graph overhead做checked容量预算；本卡不宣称任意
   外来ggml graph安全，不全局把assert改成NULL后让算子继续空指针解引用。
6. 现有Vulkan metadata owner、source patch verifier和package登记可提取到共同owner；
   保持原Vulkan算子/融合/shader字节。第三方缓存原件只读，补丁仍仅作用build-local副本。

## 能力与形状审计

| 能力 | 层级 / 承载 | 当前事实 | 有限处理 |
|---|---|---|---|
| fallible metadata context | 资源机制 / BORROW+接入 | Vulkan已有，CPU未接 | 共享现有owner/补丁，checked容量与allocator lifetime |
| persistent CPU execution slot | 资源复合 / BORROW | ggml公开plan/pool可复用，创建失败不完整 | 私有RAII pool/plan，独立slot，冷分配workspace；调用原版graph_compute |
| partial start failure | 资源机制 / 接入补丁 | 分配、mutex/cond、thread failure未回滚 | 返回错误，停止唤醒并join已创建线程；不改数学与调度算法 |
| source/build/package identity | 交付机制 / BORROW | 当前Vulkan独占ggml目标、CPU旧lock | 一个vendor源码owner；backend按选中路线启用，保持原生执行器独立 |
| model/Direct/Graph lowering | 上层复合 | M0只有映射，未接完 | 留在M1-R/M2具名卡，本卡不签覆盖 |

### 已冻结有限 FIX

| ID | 允许改动 | 适用验收 |
|---|---|---|
| G1 | 新runtime factor接管共同ggml源、context补丁与私有prepare_source；Vulkan保留后端专用补丁；依赖锁与许可证同步 | pin/hash不符拒绝、原cache不变、同build只定义一套ggml目标、CPU/Vulkan关闭隔离 |
| G2 | CPU pool创建的NULL/参数/native-init检查与部分创建回滚；持久pool失格时返回错误；不加入热路径test hook | 两次allocation及mutex/cond/thread失败覆盖；真实线程半启动回滚、可再创建与close；无abort/泄漏 |
| G3 | 私有metadata与CPU execution RAII：checked arena容量、独立pool/plan、Mem workspace、明确borrow lifetime、取消/错误翻译 | norm→MatMul→状态重复运行、两个slot并发、只读常量共享、old output、allocator失败、abort/close |
| G4 | CPU one-time time初始化替代每次临时context；不改查表或算子实现 | 原版计算差分；源码/测试分配账单证明steady不建context/pool/workspace |
| G5 | 迁移现有Vulkan metadata依赖与source-contract测试；三交付/导出/源码包闭包 | 现有Vulkan资源回归；CPU及CPU+Vulkan source/static/shared、relocation与route-OFF |

G1–G5已完成以下资源/交付审计并冻结；本卡两个owner进入AUDITED后实施。任何数学文件修改均不在授权内。

### API / ownership / 成本冻结方向

- 本factor提供Provider内部的`MetadataArena`与`CpuExecution`，不安装ggml头，不增加普通用户
  的Provider选择。CPU后续普通工厂/Graph映射在其具名owner中接入。
- metadata容量从已检查的tensor slots/graph capacity求得，payload由caller/后端持有；workspace
  由Mem分配，graph/常量借用必须活到execution完成。每个并发slot独立，禁止共享可变cplan。
- prepare拥有一次pool和workspace创建；execute只调用既定原版graph_compute。cancel在上游
  abort callback边界观察，属于已完成返回；close只能在本slot无在途调用时执行并join线程。
- 无process级allocator覆盖。故障测试仿照已有Vulkan的“原版patched源+测试局部替换”，
  测试链接身份独立；产品没有注入开关，不用资源耗尽整台机器模拟故障。
- CPU可分发初始profile使用安全ISA floor，禁GGML_NATIVE/OpenMP及环境线程策略写入；更高ISA
  由后续成熟CPU Provider的固定资格规则选取，不把本机指令集直接变成SDK floor。
- 只复用既有CMake lock/cache/package注册机制；关闭的GPU后端不探测SDK、不下载、不链接。
  私有构建脚本不是新的公共下载/构建入口；Python仅是已有source patch构建工具。

## 验证与状态

- G1–G5实施前：source/pin/hash、JSON、docs、closure（显式previous-ref）与diff check。
- 产品：本factor native单测及原版数学差分；allocation/thread部分失败；独立slot并发；
  WSL ASan/UBSan与适用TSan；逐头/ABI不泄露；真实CPU三交付与受影响Vulkan资源旅程。
- change-impact selector预声明：新增component及共同vendor/package seam会选择较宽门；
  `queue-plan.json`中的native/package门尚未执行，不能写PASS。登记/审计提交只完成docs/closure，
  所选产品门并入本卡稳定候选统一执行，不在每次文档/状态提交重复重建。
- source/static/shared消费者的DLL邻置、干净PATH、SetErrorMode和缓存合同沿用现有工具。
  Windows产品真相，WSL/Mac只作验证副本；用短的任务专属build目录并在收据归档后清理。
- P1不适用本资源补丁阶段，不创建性能card；M2真实Provider完整消费性能仍必须完成。

原`compute_provider_ggml`保持BLOCKED。独立factor完成维护/升级、OOM、并发与三交付合同审计后，
才可依原恢复条件另建CPU Provider卡回到SCOPED；本卡不代替该状态迁移或其适用产品验证。

## AUDITED 决定与七问

1. **来源/升级**：共用`456172ec`，CPU旧`ggml`lock迁移到此内容；保留Vulkan后端补丁身份，
   context条目移至新的共同runtime manifest。private helper仍验证before/after完整SHA和唯一
   replacement，升级必须逐hunk重新核对；数学/查表/调度循环不是补丁范围。原版源码与MIT许可
   通过现有lock/package分发，不改共享cache。补丁维护面限`ggml.c`资源初始化、`ggml-cpu.c`
   pool资源/CPU初始化以及必要私有声明；Vulkan已冻结数学改动的字节不变。
2. **消费与语义**：只签资源seam，主纵切和两个独立slot覆盖真实graph_compute。私有arena
   预算使用ggml公开overhead，先检查size_t/ptrdiff与容量；算子内部容量assert仍是不变量。
   后续lowering须声明节点/metadata上限，不能把原始ggml任意指针作为普通入口。
3. **复用**：直接使用ggml原生pool/barrier/graph_plan；不把Sched伪装成其线程池。
   MetadataArena从现有Vulkan实现提取；payload/workspace仍使用现有Mem，错误使用Status/native
   translators。core/error/mem不重开。线程优先级normal、空affinity、poll0，不动系统策略。
4. **成本**：每slot冷建一个pool和workspace；context创建、CPU time initialization、线程创建
   均不在steady执行。原版融合识别/数学执行留在Provider内部，不参加HYLE Provider搜索或竞速。
   最多一个明确native-error输出参数用于pool创建，产品不增加测试分支/全局注入状态。
5. **接口**：`MetadataArena`和`CpuExecution`是不安装的Provider内部C++ RAII类型；必要的
   `ggml_hyle_threadpool_new(params, native_error)`只在固定vendor与adapter间使用，不是公开C产品。
   native_error为0的NULL表示分配失败；POSIX/Win32线程与同步初始化错误保留实际码。
6. **三交付/构建**：一个runtime owner创建ggml-base和按选中路线启用的backend targets，
   现有Vulkan owner只保留原生适配器。CPU backend在CPU闭包或显式runtime开发验证中启用；
   CPU-OFF与Vulkan-OFF均受门控，不通过本factor引入CUDA/Metal。target与license注册沿原机制，
   不把私有头安装给普通SDK用户。共同seam的实际三交付仍待G5验证，不以本审计代签。
7. **易用/验证**：普通用户入口不变；后续M1-R/M2接自然工厂。故障fixture链接精确patched C
   translation unit，用测试局部malloc/aligned-allocation/thread-create替换真实失败点；仍创建
   真实线程并join已成功部分，正常算术fixture运行原版计算。生产二进制不含这些注入入口。

原CPU Provider恢复条件的“独立patch factor审计”已有具名承接；本轮仍保留其BLOCKED状态，
先完成资源实现与验证，避免把审计文档误算为已接入。公共component/package所选门仍待候选统一运行。

## 实施检查点（2026-09-20，未关闭）

- G1–G4产品代码已落地：共同源码/分离补丁、CPU fallible pool、部分线程启动回滚、私有
  MetadataArena/CpuExecution、持久plan/workspace与一次性初始化。Vulkan只迁移资源依赖；
  原有数学补丁条目保持完整字节身份，未开始冻结自研的研究或对比。
- Windows MSVC 19.50 shared开发形状通过4个CTest：6项源码合同、7项C++资源/计算用例、
  原版CPU C TU局部故障fixture与context fixture。涵盖真实F16 MatMul组合、两slot并发、
  cancellation后复用、workspace OOM、两处分配失败、每个thread-start失败位置与精确join。
  成功线程是真实OS线程，产品无fault hook。该结果不代表三交付或其它环境已通过。
- 新vendored owner仅在既有validation-policy数据表增加映射；selector算法未改。其54项
  工具回归、docs/closure与diff检查通过。共同component/package seam选择的较宽产品门尚待运行。
- G5仍在进行：Vulkan回归、CPU与组合三交付、route-OFF、WSL sanitizers和两台Mac的适用门
  尚未完成，不能写PASS；本卡和两个owner保持AUDITED。没有新增64域或整模型验收通过。
