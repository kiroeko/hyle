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

### 拟定有限 FIX

| ID | 允许改动 | 适用验收 |
|---|---|---|
| G1 | 新runtime factor接管共同ggml源、context补丁与私有prepare_source；Vulkan保留后端专用补丁；依赖锁与许可证同步 | pin/hash不符拒绝、原cache不变、同build只定义一套ggml目标、CPU/Vulkan关闭隔离 |
| G2 | CPU pool创建的NULL/参数/native-init检查与部分创建回滚；持久pool失格时返回错误；不加入热路径test hook | 两次allocation及mutex/cond/thread失败覆盖；真实线程半启动回滚、可再创建与close；无abort/泄漏 |
| G3 | 私有metadata与CPU execution RAII：checked arena容量、独立pool/plan、Mem workspace、明确borrow lifetime、取消/错误翻译 | norm→MatMul→状态重复运行、两个slot并发、只读常量共享、old output、allocator失败、abort/close |
| G4 | CPU one-time time初始化替代每次临时context；不改查表或算子实现 | 原版计算差分；源码/测试分配账单证明steady不建context/pool/workspace |
| G5 | 迁移现有Vulkan metadata依赖与source-contract测试；三交付/导出/源码包闭包 | 现有Vulkan资源回归；CPU及CPU+Vulkan source/static/shared、relocation与route-OFF |

以上仍是SCOPED拟定集；审计冻结并进入AUDITED后才允许产品改动。

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
