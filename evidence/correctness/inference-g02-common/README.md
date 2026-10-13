# G02 共同多 Queue 接缝候选阶段证据

本记录只签已实际运行的阶段证据，**不是 CLOSED 或整个 G02 完成**。活动卡见
[多 Queue batch](../../../docs/implementation/tasks/graph/inference-g02-multi-queue-batch.md)。
最终 API 为每 lane 的 Context、Queue、host workspace；删除原单 Queue 构造签名，迁移
现有消费者，保原 region/slot、F2 upload/plan、page permission 与 actual retirement。

Windows canonical 是产品输入，WSL只单向验证副本。内容比较后只复制变化文件，保依赖与
未改产品的时间戳；第一次复制18文件，per-lane staging产品轮再复制6文件。现有Linux
source-build/成熟依赖复用，后续消费者纠正只重编一个TU和链接，12 jobs。

实际证据见[receipt.json](receipt.json)与[原始日志](raw.zip)：

- Windows Graph机制/Executable（含新cold lane/精确Context Queue测试）、graph_batch与其
  public header通过；全Graph15头用本机CUDA SDK include通过。CPU factor树最初全头扫描
  缺CUDA include的实际失败保留，没有修改公共checker或将缺依赖写成语义NA。
- WSL RTX4070、锁定FI原数学：6组不同dtype/D/paged-ragged的两CUDA Context/两Queue，
  每region独立staging/device residency/metadata与两独立invocations，正常数学/旧结果/replan
  通过；同Context共享metadata额外回归通过。**两CUDA Context不是CPU＋GPU真数学证明**。
- 6个故障阶段通过：两Context的upload/plan部分prefix、first accepted region错误、unknown
  跨Queue前缀阻止后继，drain真实等待且保guards，失败不发布/不重用。原精确页append/F4
  old/new/old及真实第二request ragged重排也沿迁移接口通过。
- 原生前两次因global staging仍属第一Context而失败；per-lane staging后修正。一次replan
  断言错误按独立metadata已先更新一region的真实topology纠正；失败日志保留，不签为PASS。

首轮Bash/WSL命令包装未可靠传播native exit code；依据日志识别FAIL，随后用Python
subprocess直接记录实际returncode/elapsed并退出。最终完整consumer实际exit0，不由工具
显示的首轮exit0推断。脚本仅scratch编排，不增加产品/公共工具。

上述为原native接缝阶段；当前不可变candidate是
`26d7c2cb7212f5abdd3866baaf886b8fdbca02e5`。新增证据已并入receipt/raw.zip，旧阶段和
全部失败保留，**Graph与graph_batch仍CANDIDATE，未签CLOSED**：

- 公开CPU+CUDA消费者通过F16/BF16、GQA8/2 D256、两request跨页/尾长：CPU RMS/Cast→
  native paged→CPU Cast/Neg，两Queue、准确page frame、重排、旧output和非均匀独立oracle
  通过。普通source与relocated static/shared SDK均实际编译运行，只用public入口。
  内部故障夹具不代签public SDK资格。静态SDK原缺cudart配置的FAIL保留。
- 宿主ASan地址、UBSan、TSan数据竞争facet各跑native/public CUDA/public hybrid三消费者
  并通过，Ninja compdb核对变更TU的实际instrumentation。ASan用
  `halt_on_error=1:protect_shadow_gap=0:detect_leaks=0`，TSan用进程局部setarch-R及
  `halt_on_error=1:report_thread_leaks=0`；不抑制地址/UB/data-race报错或改退出码。
  [外库LSan](../foundation-f5-native-cuda-host-asan/README.md)与
  [NVIDIA诊断线程](../foundation-f5-native-cuda-host-tsan/README.md)控制及INCONCLUSIVE
  限制继续保留，不能称整套GPU/driver sanitizer通过。
- 公开hybrid冷创建每dtype18个caller allocation点均注入失败并回滚，live计数归零，随后
  正常四episodes通过。共36失败点，不代签native opaque私有分配上界。
- Windows CPU、CPU+CUDA、CPU+Vulkan、ggml CUDA和oneDNN直接消费者影响门通过。
- MacBook Air实查Mac16,12、macOS27.0.1、AppleClang21：原CPU+Apple公共消费者、取消/
  生命周期合同、finite/control与public batch通过；九个common/batch/CPU头通过。原全
  目录误扫CUDA头与遗漏Metal shim配置FAIL保留，实际修正扫描域/文档既有17F109 shim。
  2012项必要输入核对后只复制23个变化文件；这是影响回归，不签新Apple联合分页。

Windows单CUDA大测试因用户反馈电脑卡顿被显式终止；当时GPU约77%、显存约11.5GiB。
原log/已过阶段保留，该correctness门及后续Vulkan门仍UNPROVEN，不能签PASS或CLOSED。
后续本机编译4 jobs，暂不恢复重型GPU run；原build/tree与累计投入不重置。

结构P0/独立源码复检未发现共同接缝产品缺陷：冷O(lane capacity+active bindings+regions)，
输入span只在create借用，完整QueueBindings复制一次；warm仍原prepare_replay/submit，没有
第二runtime/queue/plan。保原FI waits与private UNKNOWN，没有新吞吐宣称。统一LOC报告
取candidate：全仓库2,835,108、自实现native174,381，净增43 code行；graph目录14,875，
graph_batch src345。后两者是工具的目录边界，前者含其他route头、后者不含公开batch.h，
不能当精确owner并集。整个G02仍需Apple/Vulkan准确联合分页与其错误/退休旅程；G01及
G03–G10由当前goal继续实施。
