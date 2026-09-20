# M1-V：IREE 普通入口与成熟低精度片段

## Closure card

- owner：`/root`；当前阶段 SCOPED；baseline `31be591067ea9a1a6773cb640bc138a00cf35c39`。
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
