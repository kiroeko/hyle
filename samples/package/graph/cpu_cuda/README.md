# CPU+CUDA 组合

选择 `HYLE_COMPUTE_ROUTES="cpu;cuda"` 即取得组合、完整 CPU/CUDA Provider 和现有 Sched。
source、shared SDK、static SDK 使用相同 C++ 入口；安装消费仍链接 `hyle::graph`。
单路线 SDK 不包含组合，示例在 configure 时明确报告缺少 `cpu;cuda`。

```powershell
cmake -S samples/package/graph/cpu_cuda -B C:/hyle-demo -G Ninja `
  -DHYLE_SAMPLE_SOURCE=C:/public/project-hyle/hyle
cmake --build C:/hyle-demo --parallel 1
ctest --test-dir C:/hyle-demo --output-on-failure -j 1

# 已安装 SDK：省略 HYLE_SAMPLE_SOURCE，传 CMAKE_PREFIX_PATH。
```

`cpu_cuda::compile` 接受显式 CPU/CUDA Context、普通 host resource 和同 CUDA Context 的
pinned staging/device 资源。CPU 数学直接使用 host backing；跨端张量沿既有 Copy 规划经 staging，
`staging_workspace_budget_bytes` 限制每个 Invocation 的暂存，不能据此推断总显存限额。
CPU-required 节点使用 `kCpuTarget`，普通数学使用 `kCudaTarget`；仍使用 Graph 原有 placement、
格式、模型/制品条件和预先固定的 Provider 规则。compile/prepare 绑定程序，submit 不重新选择。

`Invocation::create` 为一个可变槽准备固定状态，借用调用者 Scheduler 和原生 Queue。
创建独立槽与独立 Queue 可维持多个在途任务。bind 后 prepare_replay，submit 返回 OwnedResult；
wait/result 成功后取得 Results lease。任何旧结果或在途任务都会阻止该槽改绑和覆盖。
普通外部输入输出仍由应用保活；lease 保活 Graph workspace，不让外部可变存储自动不可变。

引擎可创建一个 `compute::ResourceBudget`，按实际 host/device/staging resource 登记额度，
在 Graph `CompileOptions::resource_budget` 和 `WeightSlotOptions::resource_budget` 中传相同 handle。
它复用 Mem 的严格配额：编译/运行 metadata、materialized constants、workspace及两个迁移槽分别
计费，所有使用同一 scope 的实例共享限额；底层实际释放后才归还额度，旧结果保活不会提前归还。
loader 自己拥有的权重/KV可用 `reserve` 显式计账，其 Reservation 必须晚于实际 backing 释放。

`resource_report()`按 storage ID 返回该 scope 当前请求字节、峰值、上限和 metadata/constant/
workspace/migration/external分类；边界 tensor views 按 canonical allocation/range 合并，观察值
可能已经被 scope 计费，不能再次直接相加。额外登记资源与不具备域归属的 Provider bounds单独报告。
私有成本未知或不能归属时，严格 scope 拒绝；显式 `allow_unattributed` 为部分预算，不能证明总显存
满足限额。allocator rounding、native resource/Context/Queue/Scheduler/driver开销和其他进程占用
不在已管理请求字节中；引擎必须保留余量，scope不预留系统物理内存。

`replay_report()`和 `dependency_route()`查询成功准备后的实际边界：native ordering、host wait与
两跳 staging/bytes，另报告 weight device wait/host barrier。bind/reset使准备失效；host-wait拒绝
策略也覆盖组合新增的 weight 路径。报告不能仅凭 queued/pending 就证明硬件实际重叠。

`compute::cuda::WeightSlots::create` 冷分配两组固定容量 pinned/device backing 和实际 TensorDesc/ID 目录。
迁移机制属于 CUDA Compute 路线；Direct 用户可包含 `hyle/compute/cuda_weights.h` 并链接 `hyle::compute`，
不需 CPU、Graph 或 Sched。Graph 组合的 typed bind 只增加其所有权与执行依赖连接。
upload 选择 host-accessible tensor 片段，先检查 ID、格式、重复和总容量，再暂存原始编码字节并
执行真实 DMA。source TensorView 可以来自 Buffer subview，越界由 Mem/TensorView admission 拒绝。
一种矩阵可以是 FP16，另一种可以有不同 dtype/physical encoding；每片须匹配其登记描述。
该机制不量化或转换，也不负责模型加载、专家预测、缓存或淘汰。

用 `invocation.bind(value, batch, id)` 保活对应批次并建立依赖。共享 CUDA lane 记录原生 Queue wait；
delegated episode 无此原生端点时使用显式 host barrier。Direct 消费者可调用 batch.order(queue)，
并保留 batch 至其最终计算完成；若在 CPU 读数据，必须先 wait。
同槽的上传、批次或 Graph 绑定尚存时 upload 会立即拒绝复用；reset/rebind 也受结果 lease admission 保护。

取消只阻止尚未接纳的后继；已接纳原生前缀仍须完成退役。公有句柄析构不隐藏等待。
Invocation.close 只排空并关闭该槽；WeightSlots.close 只排空该池上传，不关闭调用者 Context/Queue。
Context、resource、Queue、Scheduler、allocator 和外部 tensor backing 的借用寿命见公开头。

示例包含16KiB普通主存CPU权重的真实MatMul、最多512字节的图暂存与CUDA矩阵/CPU结果消费，
另有CPU normalization/casts→CUDA gated FFN→CPU；没有把所有CPU权重转存到pinned。
实际实现和适用运行验收状态见[当前 H1 卡](../../../../docs/implementation/tasks/compute/compute-h1-cpu-cuda-composition.md)。
