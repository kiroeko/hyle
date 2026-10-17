# G09 直接修复与实测

本目录保存实际工程结果，不作为开发、测试或继续修复的准入条件。

## Metal 线程组资源

修复 `d77f63204` 在 dispatch 前读取实际编译 pipeline 的静态线程组内存需求与设备上限。
Hyle 分页/不等长注意力的默认 full tile 超限时使用较小 tile，并再次检查资源需求。
32,768 bytes 是本机每线程组共享临时空间上限，不是模型、KV、GPU buffer 或整机内存上限。

[Air 实测摘要](air-metal-summary.json)与[脚本、命令和原始日志](air-metal-results.zip)
对应 `08e02b27c`，实机为 MacBook Air Mac16,12 / Apple M4 / macOS 27.0.1 / AppleClang 21。
两个 Metal 验证层实际开启；十个测试进程全部退出 0：

- 原 D256 / GQA 8:2 / Q8-KV / mixed Q1:17 pending 消费：F16 与 BF16 各 64 步，
  跨页及两 Graph 输出数值通过；原 58,368 > 32,768 断言未再出现。
- 原注意力数值回归 7 项，包括 D256 长 prefill、plain/Q8、独立 value 深度、window、
  noncausal、small softmax weights / large values 和 subnormal query。
- 原 public contracts、pipeline、encoded（4 项）、paged joint、sampling、public effect、
  symmetry，以及原生 completion notification 生命周期（4 项）。

归档 SHA256：`e40fa3e124b6735a8c00ef99f041599a386a612cce48fa844c4028fe280bf608`。
这些是正确性和设备验证结果，不是性能样本。

## 完成通知

`1dbfce9a0` 将 Apple/Vulkan 组合 Invocation 的终态通知改为同一 monitor 下的广播。
原实现先 signal 自动复位 Event，后清 running；观察者可能消费信号后重新睡眠，直到下次
1ms 超时。新实现保留 callback 尾部与旧 episode 的生命周期，在终态锁内广播并发布
running=false，复用同样经过该锁。Completion wait 保留现有的首错观察语义；close 直接等待终态通知。

该实现的原 contracts、pipeline、encoded 4 项与 native notification 4 项在 Air 上增量重编后
全部通过，Metal API/GPU 验证实际开启，详见 [Air 最终摘要](air-direct-summary.json)。

新增[真实 Graph 同步回归](../../../tests/graph/cpu_completion_publication.cc)在 Windows Vulkan
与 Air Apple 均一次通过，覆盖 callback 尾部、多独立 owner 观察者、单 worker 进度、close
等待与立即复用。通过私有 completion dependency 观察同一 owner，避免并发 settle 同一
move-only wrapper；没有产品测试钩子或新公共 API。该测试使用真实 CPU 数学，不提交 GPU
数学命令。它验证新同步协议的正确性，旧 timer 实现也可能通过这些生命周期断言。

原始命令与日志：[Windows](windows-publication.zip)、[Air](air-publication.zip)；
机器可读结果：[Windows](windows-publication.json)、[Air](air-publication.json)。

## 完整 replay 性能

实际候选产品为 `1dbfce9a0`，对照为原 `9e2f25c469634a6f723e76eaf1aa2db5973bca9a` SDK。
相同 4M F32 输入与 CPU/GPU 数学、真实 copy、固定 slot 数和输出，依次覆盖独立 CPU-first、
独立 GPU-first、两个 microbatch，以及必要串行链。两个 microbatch 均先 submit 后 wait。

每路线三轮交替进程顺序，六个独立进程；每进程每形状 8 次 warmup、31 个 warm 样本，
即每侧每形状 93 个样本。每个 slot 的完整 4M 输出在计时前后检查，每次 replay 仍检查原
scalar oracle。计时从 Python ctypes 调用开始，包含 prepare、submit、checked owning
result、结果释放与 Scheduler drain，包含两侧相同 FFI 开销。cold create、首 replay、
close 及进程总耗时另存；性能进程关闭 GPU 验证层。所有样本与形状保留，全部进程退出 0。

下表为各侧全部 93 个 warm 样本的中位数（ms，原基线 → 当前产品）：

| 路线 | 独立 CPU-first | 独立 GPU-first | 两个 microbatch | 串行链 |
|---|---:|---:|---:|---:|
| Windows Vulkan | 8.610 → 5.025 | 8.512 → 4.983 | 14.912 → 12.145 | 11.561 → 8.488 |
| Air Apple | 2.666 → 2.091 | 2.914 → 1.884 | 5.610 → 4.753 | 5.452 → 3.025 |
| Air Vulkan | 9.836 → 3.286 | 10.806 → 6.936 | 13.709 → 13.124 | 13.980 → 4.777 |

这是这些实际机器和消费者的直接测量，进程间扰动仍可见。完整逐进程中位数、p90/p99、
冷成本及全部样本见 [Windows 摘要](windows-direct-summary.json)与
[Air 摘要](air-direct-summary.json)。没有从本次样本单独分离出 terminal monitor 改动的稳定
额外收益；表格反映相对原基线的完整 G09 实现。CUDA 本轮未改、未重跑，不外推本表。

复现脚本、构建命令、源码/对象/SDK/实际映像身份与完整 raw：
[Windows 归档](windows-direct-latency.zip)、[Air 归档](air-direct-latency.zip)。
Windows 首次计时发生短暂重复启动且输出路径复用，已标记 `CONTAMINATED_NOT_A_RESULT`
并保留，表格仅使用随后唯一目录的六进程最终结果；没有筛选有利样本。

- Windows 归档 SHA256：`4fb490193d9e0db8f23ad2938b6c7ce9a6ebcbc000787156b528ba6b26959cd9`。
- Air 归档 SHA256：`6b19c2e595e7ff69afc268b5a35b08f23caa4211e41058a20ba9f6962a27d5dd`。
