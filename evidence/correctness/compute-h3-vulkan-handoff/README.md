# H3-07 原生队列交接资格

产品代码 `e9c5e2c3b`，普通 CMake 消费者 `97f9332ac`；后续文档提交不改变实现。
Windows MSVC 19.50.35728/RTX 4070/617.14 与 Air Mac16,12/macOS 27.0.1/AppleClang 21
各自七项 source 通过：原两入口、迁移 MatMul、组合 Q8 FFN/正交管线、资源寿命 contracts、
原生接纳后失败，以及固定 GPU 时间线。Windows 编译 owned peak 0.504 GiB，Air peak RSS 727.3 MiB。

Failure 注入只拒绝 transfer copy 已成功提交后的 compute handoff；copy 仍为真实 VkQueueSubmit，
不是 mock Provider。原 copy fence 退休后返回原 ResourceExhausted，不发布 ready 权重、不允许
失败槽复用，close 保留诊断，最后 owner 释放后 device/staging 预算归零。旧 EXCLUSIVE buffer
的同队列上传还保持正确。完整数字与程序输出在两端 LastTest 日志中。

P0 只装饰原录制 PFN，执行既有 sample affine shader 的 512 个真实 dispatch 和 16,776,960 字节
真实 WeightSlots payload；每个 compute 不动点和全部 payload 均独立检查。query reset 在测量前，
两个 family 的 timestampValidBits 均核实，使用同设备计数与 period：

| 机器 | compute / transfer family | 有交集的 dispatch bracket | bracket 交集总量 |
|---|---|---:|---:|
| RTX 4070 | 2 / 1 | 17 | 694,784 ns |
| Apple M4 | 0 / 1 | 2 | 63,207 ns |

这是本机原生独立传输机制的有界证据，不是任何 provider 的吞吐 WIN、跨设备时钟比较或
任意 driver/模型的重叠保证。旧单计算队列 no-overlap 探针仍保留，不能改写为 PASS。
准备的 capability/上传报告与执行路径一致；公开 ready 仍包含 compute ordered prefix。
Semaphore 可见性与下载 host barrier 依据 [Vulkan 同步合同](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)。

统一脚本不可变统计：Compute+Graph 68,204 行/301 个文件；全库非测试实现 163,431 行/652 个文件。
此处只签 source/P0，installed 三交付和 WSL 缺实 GPU 的限制各自留在 H3 卡与交付证据。
