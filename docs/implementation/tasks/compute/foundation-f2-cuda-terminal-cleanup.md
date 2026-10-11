# F2 CUDA：真实晚失败后的资源释放反例

状态：AUDITED；owner仅compute_backend_cuda；父轮仍是[F2七FIX](foundation-f2-batched-transfers.md)。
baseline/改前registry ref：`7e91e9ad5b40485e555f579053101e31dfc2e16b`。
原产品candidate为03162add0；compute/Apple/Vulkan保持其CANDIDATE，F1历史不重开。

## 具名反例与源码裁决

独立CudaCopyLateFailure在Windows/RTX4070/driver617.14/CUDA13.3实际正常组ready及读回后，
释放原producer的memset＋原illegal-address kernel。日志到actual fault后在cuda_memory.cc:313
的`released == CUDA_SUCCESS` fatal退出。这是F2-06的晚失败/teardown未完整实现，不是fixture PASS。
56dbabad5的first-use加载TIMEOUT及7e91e9ad5的产品fatal均保留，不重置原调查/纠正预算。

本地CUDA13.3 cuda.h的CUDA_ERROR_ILLEGAL_ADDRESS及同类终态定义明确要求进程退出/重启；
[NVIDIA类型合同](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html)也规定后续CUDA工作持续返回终态错误。
cuMemFree/cuMemFreeHost及context操作可以回报此前异步错误。不能把这一错误当成功释放，也不能
在HYLE析构中自动reset共享primary context；其他native消费者也可能持有它。

有限源码核对仅cuda_memory.cc两种owned allocation释放、cuda_backend.cc原终态分类与context
push/pop、cuda_internal.h，以及当前锁定SDK上述合同。其他native allocator/runtime/API不在本delta。

## 固定单项FIX（F2-06既定范围）

- 成熟native算法继续BORROW；只修已有owned resource的终态释放边界。正常路径仍原native free；
  malformed handle/未知释放错误保留原诊断，不把所有错误忽略。
- 用与原Completion终态集合一致的九个Driver错误作私有classification；terminal free/context-scope
  失败允许HYLE逻辑owner/metadata正常退出，不制造host进程二次fatal。借入payload仍不释放。
- terminal GPU原生allocation是否实际释放不能证明，明确为UNKNOWN，由CUDA进程重启处理；
  requested/live logical owner计数与native residency分开，不伪造physical bytes回收。
- 不增加公开API、常态字段、queue、reaper、driver retry或kernel，不自动恢复/重置GPU。
- 同类external-owned resource和已有ScopedPrimaryContext的pop保持同一终态合同；成功pop仍核对
  expected context，非终态故障不放宽。只在当前AUDITED CUDA owner内修改。

## 验证与停止

沿原独立late target，Windows/WSL各build1/test1、4GiB、300/1800/120门：正常host组先ready，
writer producer真实晚失败后失败组不得发布，原错误保留，旧组仍可读，guard/HYLE metadata实际归零。
15秒fixture watchdog介入即FAIL；ready/退休各5秒。两原fault target独立进程，不在毒化Context中串跑。
精准复跑8个CudaCopyBatch/Program、34 common及原owned/borrowed/adopted resource释放直接消费者；
shipping ordinary CPU/CUDA数学及hook absence沿原SDK矩阵。未改变的其他owner证据按scope复用。
源码正常free路径/成功context恢复成本保持原等级；对非法地址后的native物理回收不签PASS。
一个固定修复集、原失败预算与停止规则继续；仍有material矛盾则裁决，不无限扩建native工具。
