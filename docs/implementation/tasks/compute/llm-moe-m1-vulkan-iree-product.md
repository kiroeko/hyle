# M1-V：IREE 普通入口与成熟低精度片段

## Closure card

- owner：`/root`；当前阶段 AUDITED；baseline `31be591067ea9a1a6773cb640bc138a00cf35c39`；
  SCOPED 检查点 `c0e65357`。实现必须在本次 AUDITED commit 之后。
- 本卡 owner：`compute_provider_iree_vulkan`（已在 baseline 登记 QUEUED）、`compute_ops`。
  后者因具名压缩格式尚无自然 descriptor factory 的新消费者需求从 CLOSED 重开；不重审旧数学。
- 直接消费方：既有 AUDITED 的 `compute_route_vulkan`、`graph_route_vulkan`；同设备 native owner
  `compute_provider_ggml_vulkan` 继续由[父卡](llm-moe-m1-vulkan-admission.md)绑定。父卡新的 IREE
  资源/调用改动必须等本卡完成 AUDITED 的有限合同；已实现 recurrent 纵切保持。
- trigger：M0 F02.4 / F18 的 checked I32 Scan 与压缩 FP8 / MXFP4，及父卡保存的 Windows/Air
  数值、外来资源和单队列自动续执行资格。IREE 单独 packed-FP4 已失格，继续由原版 ggml 承载。
- 主纵切：驻留的 E4M3 + F32 block128 / UE8 group32 输入，经普通 MatMul Direct/Graph 接原版
  IREE 程序；MXFP4 经原版 ggml。正交纵切：无模型标签的 checked integer prefix，重复执行、
  两 invocation、输入不变、旧结果与失败不发布输出。
- scope：自然格式描述、锁定 C runtime/离线制品、私有资源适配、准备绑定、普通 Direct/Graph、
  共享队列的自动推进与失败、三种 C++ 交付。依赖只向共同语义、Mem/Error/原生资源；Graph
  单向消费 Compute。禁止把 IREE C API/VM/MLIR 变成 HYLE 用户必须使用的接口。
- non-goals：全部冻结自研数学的扩张、调研、优化、比较、删除；修改上游数学/编译器；模型
  loader/量化器/推理引擎；全部64×4接线；公共构建/验证/性能系统重写；新通用 Queue 或线程循环。
- tier：P0；明确元数据、原版线程、GPU copy、F32 region workspace、编码/分配与通知成本。
  不在 M1 签 P1/SOTA，不提前比较自研；默认仅按数值/设备/软件适用域冷绑定成熟程序。
- 预算：沿父卡总调查截止 **2026-09-20 12:02:33 UTC**，不新增四小时。基础 Q-VK 资格已经停止；
  剩余调查只允许这张卡的格式/原生资源/交付接缝细化。每根因最多两次纠正；实施和预声明验证另记。
- 机器：Windows RTX4070、Air/MoltenVK 真实执行与三交付；WSL/Mini 只按公共语义/OFF/host 风险
  适用性选择。parallel2、重型串行、Windows DLL邻置和无 UAC；Windows 是唯一产品真相。
- stop：有限功能与适用验收齐全才 CLOSED；必要接缝未满足则精确记录缺口，不能以资格探针代签。
  整包 M1-V 还必须完成父卡剩余门；新增完整覆盖仍 0/256。**M5 前停下并等待用户明确继续。**

## SCOPED 审计边界

下表用于固定有限职责，还不是产品修改授权。

| ID | 正常消费与已知事实 | 审计出口 |
|---|---|---|
| I1 | 用户已有压缩数据，需要具名布局描述；现有 MatMul 已支持 floating logical dtype 与 blocked/opaque storage | 共同语义只加 typed descriptor factories 与物理格式合同，复用普通 MatMul；不改旧 QuantizedMatMul 的整数 Q4 含义 |
| I2 | 原版 runtime + 参数化 VMFB 已可由 C++ source/static/shared 消费 | 独立 provider 拥有锁、资源 overlay、recipe/制品、许可、module CMake；运行端无 Python/编译器，OFF 不获取 |
| I3 | 同一个 ggml VkDevice/queue、IREE 延后提交、Air 只有一个 compute queue | 复用 NativeDevice::with_queue；原版 proactor 续执行，显式 native feature 与 ready/completion；不在持有 queue lock 时进入 IREE |
| I4 | Context memory_quota 明确包括实际 VkDeviceMemory 与 staging | 审计 IREE 的 allocator/slab/输出资源；逐实际 native allocation 预留并在释放后退账，禁止无声绕过 quota |
| I5 | checked Scan 包括 exclusive 最终总和，错误时不能写公开输出 | 使用已验证原版宽 scan/标准比较归约；私有输出/错误标志保留至完成，异步读回标志后才发布结果或继续下游 |
| I6 | 普通 Direct/Graph、多个成熟 region、重复运行与两请求 | 保留各 provider 程序；HYLE 只管理片段边界、取消/关闭/accepted-prefix drain、旧结果；执行不搜索、不竞速 |
| I7 | 完整 C++ 消费与实际依赖可迁移性 | source/static/shared、prefix 搬移、隐藏 producer、header/符号/许可证、独立 oracle、资源故障与适用 host/OFF 门 |

### 必须固定的自然形状

- 具名格式描述 E4M3 的 scale 粒度/类型及数据/scale 字节布局，MXFP4 原版32值/17字节块；普通
  用户不制造 StorageFormatId、不知道 Provider 排名。FP8 的逻辑输出为 F32，MXFP4 初始合格域
  为 F16 activation/output、F32 accumulation；固定输入物理方向与 MatMul 的逻辑 `[M,K]*[K,N]`。
- IREE context 懒建，只在实际准备本 provider 的程序时创建；所有 IREE 对象保留 native owner，
  避免 Device 与 provider 的 shared_ptr 环。一个 HAL device group/module，独立 invocation VM。
- 普通 submit 不等待 GPU 计算、不依赖用户反复 poll 才推进后续片段。原版每队列 completion
  thread 与 proactor runner、跨片段通知计入 P0；不为 native-only 程序引入同样常态成本。
- 旧自研实现保留冻结；接线到成熟实现属于本卡，不修改其数学来迁就或击败成熟实现。

## 预声明验证

1. 状态/文档阶段仅 selector、closure（明确 previous ref）、docs、diff；产品门待不可变候选。
2. 格式 schema：合法/非法 dtype/shape/block、required bytes 与算术溢出、物理方向、公共头；
   适用 Windows/WSL/Air/Mini 共同语义门，不以某一平台的 enum 声明计为路线覆盖。
3. Windows/Air 普通 Direct/Graph：checked I32 中间/最终溢出、FP8 两种 scale、MXFP4 原版
   逻辑 oracle；变化形状、同设备成熟片段组合、两 invocation、repeat、旧结果、输入不变。
4. 生命周期/失败：未启用 feature、错误设备/容量/布局/对齐、quota 与部分构造回滚；accepted
   dependency/submit/观察失败、失败不写输出、后段不运行、取消/放弃 Completion、close/drain。
   插桩只签适用 host adapter；不能代签 GPU driver。
5. Windows/Air 三交付真实普通 consumer、prefix 搬移、隐藏 producer、offline 与许可/资源/私有
   符号；Windows/WSL/Mac 适用 OFF 配置、公共语义及 selector 所选真实消费者。
6. P0 静态成本、候选复核与统一 Compute+Graph LOC；不跑 P1。归档后清理本卡具名 scratch，
   保留共享依赖缓存及本地2B模型。未运行门保持未决，不预写 PASS。

本卡启动实现统计仍为 **51,953 行 / 222 文件**，排除测试和第三方。当前产品门未执行。

## AUDITED：有限 FIX 与普通接口冻结

资格证据来自父卡 `checked-queue-qualification.json` 与 `native-progress.json`。源码接缝为同 pin
IREE `hal/drivers/vulkan/{logical_device,queue,allocator,buffer,sparse_buffer}.c`、`async/operations/semaphore.h`
和原版 ggml native queue/feature 创建；只核对资源/生命周期，不研究冻结数学。

| FIX | 固定裁决 | 普通形状、失败与成本 |
|---|---|---|
| I1 | FIX 共同 `low_precision.h` 格式描述；KEEP MatMul 逻辑推导 | `Float8ScaleMode`、`float8_activation_description(m,k,mode)`、`float8_weight_description(k,n,mode)`、`mxfp4_weight_description(k,n)`；只描述已有压缩内容，不量化、不加载模型 |
| I2 | BORROW 原版 IREE runtime/VMFB，FIX module-owned 资源 overlay 与打包 | 参数化原版程序离线编译，运行端不调用编译器/Python。源码和两类SDK使用同一锁与程序；原版 compiler 不进入消费者依赖闭包 |
| I3 | FIX ggml 实际启用 feature 与共享 queue 接缝；BORROW 原版异步续执行 | query/enable synchronization2；发布真正启用的 timeline/scalar/F16/storage16/Int64。IREE QueueSubmit2/BindSparse 经 NativeDevice::with_queue；完成用原版 semaphore-wait proactor，GPU payload不回落CPU |
| I4 | FIX 可选实际显存计账 | private resource hooks 覆盖所有 IREE VkAllocateMemory/FreeMemory；调用前预留 QuotaAllocator，分配失败回滚，释放实际memory后退账。只有显式quota才维护token记录；不改原版pool/数学 |
| I5 | BORROW 已资格化 checked Scan/FP8，FIX 私有输出发布 | I32暂存结果与标志由原版程序生成；异步完成及必要4字节读回后才复制公开输出/继续依赖片段；失败保留原输出。FP8保留region级F32 workspace，MXFP4调用原版ggml，不假称native FP8 GEMM |
| I6 | FIX 本路线的片段接线和自动推进 | native-only原路径保持；含IREE时按实际完成继续下段，不提前向同队列提交等待尚未入队的工作。两请求VM/资源独立；最终Completion覆盖已接受前缀和晚到callback，取消/close先退休再释放 |
| I7 | FIX 独立消费者、交付和验证 | 按本卡及父卡预声明门，不把原生probe代签产品；固定候选后执行适用selector/资源/三交付，未运行仍未决 |

### 格式与选择

FP8 两种模式复用既有 opaque storage 和 F32 logical dtype。activation 的物理 E4M3 字节为
row-major `[M,K]`；weight 物理为 `[N,K]`，逻辑为 `[K,N]`。payload 后接 scale table：
F32 block128 的 table 分别为 `[M,K/128]` 与 `[N/128,K/128]`，按4字节对齐；UE8 group32
分别为 `[M,K/32]`、`[N,K/32]`。首个合格域要求完整块，factory 明确拒绝余块与溢出。
公共具名格式ID描述数值编码，不表示 Provider；消费者只用普通 MatMul 与上传/绑定。

MXFP4 采用原版32值/17字节块，物理 `[N,K]`、逻辑 `[K,N]`，block axis=0、stride分别为
17 和 `K/32*17`；逻辑dtype F16，普通 F16 activation × weight，F32 accumulation后输出F16。
其原版 `mul_mat` / cast / cont 成本明示。没有把 FP4 改叫整数 Q4，也没有修改 QuantizedMatMul。

IREE Scan 首个普通域为 dense I32、末轴、正形状；其余已存在功能保持冻结，广覆盖与布局成批接线
仍归 M2。prepare 校验实际feature、布局、容量、workspace和设备上限，只选择资格范围；热执行
不探测、不搜索、不计时，不要求模型标签或用户排序。模型/硬件/软件 identity 继续沿 M1-R。

### 原生 owner 与异步边界

- `compute_provider_iree_vulkan` 私有独立 Device/Program/Invocation 持有原版 HAL/VM/池；
  `compute_route_vulkan` 保有公共 Context/Program/Invocation/Completion，Graph 只 lowering。
- IREE device 懒建并保留 ggml NativeDevice 的独立副本，不保留其上层 Device，消除引用环。
  一个共享 HAL group/module，独立 VM contexts；来源VkDevice/queue不能由IREE销毁。
- 初版采用已验证有界GPU复制：输入、输出与必要flag readback明确计费；不以未经资格的私有
  borrowed-buffer 优化阻塞接入。保留原压缩输入，禁止展开整模型为常驻F32。
- 每个IREE invocation预留自己的native copy命令、状态与续执行记录；原版程序仍可能分配/编码。
  IREE proactor回调只推进片段，不能阻塞等待GPU；同一个Runtime的修改仍由本路线mutex保护。
  completion/error路径必须覆盖回调尚未返回的保活，close不能在锁内等待需要该锁的回调。
- IREE resource overlay只包围资源创建/释放与queue host访问。实际memory计账不包含driver/pipeline
  内部分配，与现有ContextOptions合同一致。显式quota路径允许按live allocation数扫描token表；
  默认无quota无该表/锁；模型数学无变化。
- 依赖锁、patch/原版程序制品、许可、离线资源映射、policy中的具名owner/platform事实归本factor
  的接入改动；不改变公共selector算法或引入新的治理机制。资源hash/升级必须显式失败，不能静默换pin。

自然调用保持 `vulkan::prepare(context, MatMul{}, activation_desc, weight_desc)` / `Scan{}`，以及
普通 `graph::vulkan::compile`。本卡只增加格式说明与私有适配，禁止要求用户直接管理 VM/MLIR。
七问裁决：成熟实现承载、明确有限域、复用Mem/Error/native owner、P0完整计费、私有C边界、
三交付同源、普通接口可达；实现及适用验收尚未签字。调查未新增预算，按上述有限FIX开始实施。

### I3/I6 实施细化：VM 挂起与失败前缀

原版 `vm/invocation.c` 允许 FP8 在一次调用内挂起；同步 invoke 后才释放输入 fence 会形成等待环。
使用原版 asynchronous invoke，私有适配仅把锁定程序的 HAL semaphore wait 交给原 proactor；
不建立 HYLE 线程、轮询循环或 kernel 调度器。回调返回之前仍保留 invocation，普通执行不用用户 poll 推进。

`vulkan/queue.c` 的 deferred submission 在依赖就绪时才取得 epoch，故失败后一个无依赖 barrier
可能越过此前 deferred 前缀，不能作为排空证明。有限 FIX 改为暴露原 completion thread 已有
stop/cancel-deferred/drain-pending 路径的完成 semaphore。仅 VM/执行基础设施错误使该 IREE
owner 终止并取消其余请求，重建 owner 后恢复；数值 overflow 由私有 flag 返回，不触发此路径。
它不销毁借用的 VkDevice/queue，也不停止同设备 ggml owner。此故障域须由普通入口说明和失败门
验收；当前仍是实现中，不因 native 数值测试通过而签 I6/I7。调查预算不延长，不新增数学范围。

### I3 适用 host 验证发现的有限资源修复

候选 `778aa484` 在 Air 的原生 Scan 重复执行触发堆损坏。一次 ASan 构建同时插桩 HYLE
adapter 与原版 IREE runtime，确认 `async/platform/posix/proactor.c` 的
`iree_async_proactor_posix_drain_pending_semaphore_waits` 在 final callback 之后写
`wait_op->base.next`，而原版 `async/operation.h` 明确允许 final callback 释放该操作。
当时 Windows 原生门通过，但尚未证明 IOCP 的对应完成路径安全；其既有 PASS 不代签 POSIX，
也不能排除普通混合链复用该操作内存时的缺陷。

本项属于已审 I3/I6 的 callback 存储归还接缝，owner 与数学/普通接口范围不变。固定唯一 FIX：
在调用 final callback **之前**清理操作上的 tracker 引用，callback 返回后不再访问操作；
使用现有精确 hash 资源 overlay，保留原版队列、等待、worker 和数值程序。最多两次同根因
纠正；重用已经复现的 Scan/repeat/自动推进/尾回调测试和一次 ASan 纠正验证，随后执行普通
Release 回归。登记本 FIX 时尚未修、未签 PASS；后续纠正结果见原生检查点。
不以该具体缺陷重启 Q-VK 调查或冻结算子研究。

### I3/I6 普通混合链发现的 IOCP 对应修复

普通入口候选 `9deedc43` 在 Windows 的 IREE → ggml 混合 Graph 中出现读零地址；一次局部异常
诊断把调用栈定位到 `ggml_vk_graph_cleanup` 释放 context 的 shared_ptr。原版
`async/platform/iocp/proactor.c:iree_async_proactor_iocp_drain_pending_semaphore_waits`
也在 final callback 后写 `wait_op->base.next = NULL`，违反同一原版操作归还合同。HYLE 的
callback 已释放 operation，并可立即创建 ggml region 的执行资源；回调之后的写因此可能破坏
刚复用该地址的对象。此前对 Windows 路径安全性的判断撤回。

有限 FIX 与 POSIX 相同：只在精确 hash overlay 中把 tracker 清理移至 final callback 前；
不改上游数学、调度器或 ggml 算子。仍属已审 I3/I6，不重开调查，最多两次同根因纠正。
Windows 首次 HYLE ASan 配置没有插桩原版 IREE C runtime，未检出不能当作反证。I7 同时固定
本 provider 在 HYLE ASan 配置下启用原版 IREE ASan，确认真实编译参数；不改公共插桩系统。
登记本项时尚未修复/验收，原失败及源码事实必须保留，普通产品仍未关闭。

## 实施检查点：共同格式与 MXFP4 普通入口

AUDITED 检查点为 `8b3c5c30`，随后产品候选 `d5cd5568c179d0f9bbff1cd528b306f3a9bde558`
实现 I1，以及 I5/I6 中独立的原版 ggml MXFP4 纵切。
[产品收据](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/mxfp4-product.json)
及同目录 raw archive 保留完整 selector、命令、原始测试、Windows/Air源码身份和统一LOC。

- 新 `low_precision.h` 提供四种FP8布局和MXFP4的具名描述；按公共普通MatMul推导F32/F16输出，
  不增加数学实现或量化器。非法块、物理字节溢出、逻辑/物理方向已验证。FP8的格式可构造，
  其Vulkan普通执行仍未接完，不能因此签覆盖。
- 普通Vulkan MatMul与Graph已绑定原版ggml MXFP4 `mul_mat` + F16 Cast，F32累加。压缩输入
  直接以具名原版布局驻留；不把逻辑F16元素数当作压缩字节Copy，也不解压整模型。动态F16
  activation仍可能有既有对齐Copy，原厂F32结果、最多八份split-K暂存和编码均是实际成本。
- 支持正尺寸2D、K完整32值块；在既有buffer范围之外，冷期保守限制M/N与split-K grid/暂存。
  压缩borrowed端口须native descriptor对齐，错误格式与非对应普通算子明确拒绝。
- Windows shared的9个CTest门、Air/MoltenVK static的8个门全部通过；包括三个变化形状、
  独立逻辑oracle、repeat3、两个invocation槽、旧结果、压缩输入不变、quota归零，以及Graph
  两个原版MatMul串接并发布中间结果。Air只接收经前后文件hash核对的Windows提交前向delta。
- 复用两端已有本包build，无新增native构建树。公共头/manifest使selector选出更宽的完整计划，
  已原样保存；本检查点只运行预声明四factor及头门，其余selected/交付/host/OFF仍未决，未写PASS。

统一 Compute+Graph 当前 **52,110 行 / 224 文件**，不含测试/vendor；相对本卡启动增加157行
格式与接线实现。IREE native owner、queue/quota/异步桥、普通checked Scan/FP8、真实混合链和
整包三交付仍待实施/验收；本卡保持AUDITED，聚合INCONCLUSIVE，新增完整覆盖 **0/256**。
不因这次检查点关闭M1-V，也不提前研究/比较/改动冻结自研算子。M5前停止点保持。

## 实施检查点：IREE 原生资源与异步程序

原生实现 `aa26de0f`、最终可移植候选 `dee6ad642fc0952d0c47a46daa36330db8291e82` 的
[产品收据](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/iree-native-checkpoint.json)
与 raw archive 保存首次失败、有限纠正、完整 selector、前向源码身份和实际测试日志。

- I2/I3/I4 已实现独立原版 C runtime、离线参数化制品、同 ggml VkDevice/queue 的 owner、
  实际启用 feature、原生内存配额和原版 proactor/VM 异步推进；没有运行时编译器或新 HYLE worker。
- checked Scan 与两种 FP8 scale 的私有 native program 已能重复执行、保留独立请求输入，
  检查中间/最终 I32 overflow flag。普通入口的私有结果发布与下游截断仍待接线。
- VM/执行基础设施错误经原版 stop/drain 路径等待计算与传输两个逻辑 lane 退场；同一物理队列
  不能只等待一个 lane。失败使该 IREE owner 停止接受新程序，同设备 ggml/native owner仍可使用。
  数值 overflow 不停止 owner。回调尾部返回之前 invocation 仍 pending，不能重用或释放。
- Windows shared、Air/MoltenVK static Release 各通过 **7项native + 6项source** 测试。Air首次
  Release失败后，一次原版runtime与HYLE adapter同时插桩的ASan明确复现POSIX释放后写；按上节
  唯一FIX纠正后，ASan与Release同一组测试通过。ASan不签GPU/driver内部，也未启用leak检测。
- 源码准备验证原版和overlay身份，拒绝非预期修改；更新只影响自己已登记的文件。包含绝对路径、
  drive、parent traversal、symlink/junction边界保护。失效Windows构建副本已归档并回收约199MiB。

P0包括原版运行线程、每请求VM/资源、原版分配/编码、显式quota表，以及接线所需GPU copy和FP8
region F32 workspace。它们不是零开销承诺，也没有运行P1或比较冻结数学。

完整selector有120个selected、13个skipped，本检查点只签上述native门；普通Direct/Graph的
checked Scan/FP8、混合链、cancel/close/失败不发布输出、产品三交付、剩余host/OFF与全selected
仍未决。本卡/父包保持AUDITED、聚合INCONCLUSIVE，新增完整覆盖仍 **0/256**。
统一Compute+Graph **53,003行 / 230文件**，不含测试/vendor；M5前用户停止点保持。
