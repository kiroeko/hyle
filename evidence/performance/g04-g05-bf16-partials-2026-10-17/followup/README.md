# C2 无窗口专用化、C3 K 预取与剩余差距

本轮产品保留 C1 的可选 BF16 partial 和 C2 的无窗口专用化；C3 预取和 v12 vendor
候选均未采用，canonical vendor 仍为 v11。C1 的实际显存和速度收益见[前轮报告](../README.md)。

## 已确认的残差位置

C1 的同源 plain BF16 KV 对照，使用相同数据冷精确展开并吸收 scale。现有 ordinary
完整调用中，FP8/BF16-partial 路径仍慢约 5%。新独立小 trace 共 512 个 timed episodes、
1024 个关联 kernel。normal/replay、两分配顺序、两调用顺序的八个矩阵均显示：

- attention 两边都为 grid40×1×2、block32×4、shared49152 bytes；FP8 attention 多
  5.19–5.42µs；merge 两边 grid460、block32×4、shared8704 bytes，FP8 反而少
  1.32–1.36µs，GPU 合计多 3.85–4.08µs。
- normal 两边每次均为八项资源查询、两次 kernel launch；replay 均只剩 GraphLaunch、
  EventRecord、EventSynchronize，查询消失后 attention 差距仍在。checked_wait 包含 GPU
  和调度等待，不能称为纯 CPU 成本。
- 原生 BF16 与 FP8 的实际 SASS、加载、转换和参数/scale 路径不同。旧 F32-partial
  四角中的载体差额不能泛化到当前 BF16-partial 列。静态依赖链不能单独证明动态瓶颈。

## C2：仅去掉无窗口场景的冗余检查

专用 BF16 variant 已由 admission predicate 限定 window_left=-1，却继承了开启窗口的
DefaultAttention。C2 仅将继承和 Base 两处模板参数改为关闭窗口，独立 causal mask 保留。
实际八条 ISETP.LT.U32.OR 消失；目标仍246寄存器，无 stack/local spill。所选默认 D256
F32 prefill、F32 merge、BF16 merge 的完整128bit指令和控制字与 C1 相同。

C2 通过公共 Direct/Graph/replay、17项原生测试、同输入全输出精度比较、memcheck、
synccheck 及 feature memcheck。它是语义等价专用化，不宣称有稳定的额外产品提速。
后续 C2/C3 完整 native_cache TU 保持原 flags，仅编译实际验证设备的 SM89；产品构建
架构配置没有改变，C1 双架构产物保留。其他归档 member 逐字节保持。

## C3：不采用

只在专用 BF16 marker 域，将每线程首个16B K packed load 提前到 softmax 前，decode、
shared store、第二个 K load 的源位置和全部同步点不变。实际 SASS 确认首 load 已跨过
softmax，编译器也把第二 load 提前到首 decode 前；不能称第二 load 的机器码调度不变。
250寄存器、stack/local0，shared/CTA不变；默认 D256 F32 prefill 和两种 merge 与 C2
完整 codewords 相同。数学/精度/设备检查全部通过，但实际延迟方向不稳定，故未采用。

## 固定样本结果

每组都是32768 pairs：两种分配顺序×两种调用顺序×normal/replay，每矩阵32 blocks×128
pairs。前两组 C1/C2、C2/C3 的两臂同11MiB cap、同 FP8 resident KV、同 BF16 partial。
第三组保留实际 plain BF16 peer 的 BF16 KV/22MiB cap，对比 C3 的 FP8 KV/11MiB cap，
冷展开不计入 warm 指标；它不是只改变一个载体的反事实。C1/C2 与 C2/C3 从实际
provider/native_batch/native_cache object 构造符号隔离闭包；所有 weak/COMDAT 与引用映射
核验，fatbin逐字节不变，link map确认没有误取原始 FI 三个 member。实际不同 hoststub
及资源查询绑定被测路径。没有重新编译基线 kernel、删样本、减负对照或追加样本。

| 对照 → 候选 | normal完整调用µs | replay完整调用µs |
|---|---:|---:|
| C1 → C2 | 81.687592 → 81.295579（−0.48%） | 77.559899 → 77.379053（−0.23%） |
| C2 → C3 | 80.316810 → 79.935948（−0.47%） | 77.098391 → 76.842867（−0.33%） |
| plain BF16 KV → C3 | 74.827959 → 78.366180（+4.73%） | 76.805963 → 80.252612（+4.49%） |

C1/C2、C2/C3 的描述性 block 区间都跨零，C3 各矩阵有快有慢。小 pooled 改善不是稳定
收益，C3 更没有补齐 peer 差距。不同进程的均值不能串乘成总收益。完整矩阵和限制见
[汇总](results-summary.json)、[矩阵](all-matrices.csv)；C1原始主判不变。

## 硬件计数器访问

Nsight Compute2026.2.1 实际 --query-metrics 返回 ERR_NVGPUCTRPERM；进程 exit0
不代表读取成功。没有 counter 数据、没有启动 ncu profile、没有修改系统权限。
Windows 当前进程不是 elevated administrator。WSL 的 profiling counter 访问需要在
Windows 宿主启用，见 [NVIDIA WSL requirements](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html)
及 [ERR_NVGPUCTRPERM 官方说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)。
本机已核安装 NVIDIA App11.0.9.251；官方对应入口为 System → Advanced → Developer
→ Manage GPU Performance Counters，修改需管理员确认。这里只读查元数据，未打开应用。

因此当前只能确认残差主要位于 attention，不能精确分配到解码指令吞吐、访存等待或
occupancy。K 预取没有稳定提速也不支持把首个 K load 等同完整瓶颈。下一项已准备的
有限采样使用现有 C2 binary，每臂至多两个 warm normal attention launch，不需新变体。

构建、源、隔离证明、codegen、全部测试、普通采样和计数器拒绝原件见 [raw.zip](raw.zip)，
文件校验清单为 [archive.json](archive.json)。完整 vendor shadow 不重复归档；锁定来源、
补丁、2576文件逐项清单足以重建。
实际产品选择、源/归档身份和12项执行记录见 [decision.json](decision.json)。
