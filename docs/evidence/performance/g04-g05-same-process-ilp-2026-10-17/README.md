# G04/G05 双 K 加载：同进程对照

2026-10-17。用户同意先做同进程原版/改版对照，完整流程收益成立后采纳。
本轮结论是**保留原产品实现，暂不采纳双 K 提前加载**。G04/G05 此前剩余的性能差距没有
在本轮被解决。治理系统保持停用，没有为本实验恢复状态卡、资格或预算门禁。

## 实际对照

实机 WSL2 / RTX 4070 / SM89 / driver 617.14 / CUDA 13.3。复用已编译的完整
`native_cache.cu` 原版与改版对象；改版只在既有 paged FP8 direct-stage D256 K 路径中，
将两次 packed load 放到各自转换之前。FP8 驻留、F32 partial/merge、原解码、NoFill/tail、
tile、V 和其它域均不变。

原对象不是 PIC，本次实际采用同一 executable 的对称静态对象克隆：两侧各自的
provider、native_batch、native_cache 定义符号及 COMDAT signature 改名，内部引用同步
重映射，共用其余未改 Hyle/CUDA 库。link map 验证原 archive 三成员没有被额外抽入；
fatbin 字节保持原样。实际属性查询及 timeline 均确认原版 prefill 使用 248 个寄存器，
改版使用 250 个，local 为 0、block 128、dynamic shared 49,152 bytes。

两边在同一进程交替执行原三形状：D256 window、D256 prefill、D64/Dv128 noncausal
holdout。每侧有两个轮换 invocation slots。两种分配顺序 × 两种执行顺序 × 三形状 ×
八组 × 每组 64 对，合计 6,144 对、12,288 个完整 episode；每侧每个矩阵 64 次 warmup。
1023/767 KV 尾页另作数值检查。主指标事先确定为完整 episode 的配对 block 算术均值，
包含 update_batch、run、checked wait、输出取得与释放；不计 oracle 或日志。所有尾样本保留。

128 次独立 double oracle 检查通过，最大绝对误差 0.00099498，小于原 0.008；原版与
改版输出逐位相同，原 FP8 KV 字节未改变。关闭、drain 与资源释放均完成。

## 结果

| 分配顺序 | 执行顺序 | 完整 prefill 耗时变化 | 改版更快的 block |
|---|---|---:|---:|
| 原版先 | 原版先 | −0.26% | 4/8 |
| 原版先 | 改版先 | +3.06% | 4/8 |
| 改版先 | 原版先 | −0.13% | 3/8 |
| 改版先 | 改版先 | +1.90% | 3/8 |

全部 prefill 样本平均耗时为 **95.115 → 96.144µs（+1.08%）**。p50 为
78.657 → 77.360µs，p99 为 430.824 → 636.092µs；不能用 p50 改善替换主指标。
未改 window/holdout 也有明显完整流程波动，因此本次不证明改版存在稳定性能回归，
但它同样未提供稳定收益的依据。

另一次同程序 Nsight 仅用于解释。12,288 个 episode 与 16,384 个 timed kernel 按真实
correlation/NVTX 唯一匹配。四组 attention GPU 均值变化为 −10.22%、+0.78%、−3.34%、
−6.22%，merge 约在 ±1.8% 内，也未取得全顺序稳定收益。profiler 的 host 时间不用作
普通性能结论。

GPU 最后一个 kernel 结束到同步返回约 1.9–3.7µs，首个 launch 返回到 GPU 开始约
26–32µs。未发现 G09 式 1ms 轮询；这些观察不能直接归因于某个 driver/OS 机制。

## 设备资源查询假说

继续只读核源码与原 BF16 peer：两边每个 prefill 都是两次 kernel launch，相同的
八项资源调用（GetDevice ×2、DeviceGetAttribute ×3、Occupancy ×1、FuncSetAttribute ×2）。
原有 trace 没有启用全部 CUDA API 捕获，查询缺行不能当成零成本；submit 中未捕获的
10–13µs 也不能全归这些查询。

另用真实 D256 prefill/merge kernel 与原参数，做未 profile 的资源查询组合计时，没有
提交 attention 数学。暖态八调用组合平均 **0.464µs**、16 个 block 的中位数 0.441µs，
约为原完整 episode 的 0.49%。这是没有 pending 数学工作的普通查询成本，不是实际
enqueue 后关键路径成本的严格上界。本次结果不支持为了这组查询引入新的缓存状态。

冷创建日志仅描述 CUDA 属性查询初始化之后的 Context/arena/program/两 slots 构造，
不是进程首次 CUDA 初始化；关闭与进程总耗时另列。所有顺序、尾样本、失败及计时口径
保存在原件中，没有修改主指标、裁尾或补采有利样本。

## 原件

- [机器可读结果](result.json)。
- [同进程源码、构建命令、实际身份、全部 raw 和独立重算](same-process-results.zip)。
  SHA256：`a64ad441d02ac9e08b5ad8b7e27c495d1133c7203c9b2aa174eebd587ccca273`。
- [资源查询结果](query-summary.json)与[查询源码、参数、全部 raw 和独立复核](query-followup.zip)。
  查询使用的 merge 精确为 `<8,32,4,2,float,bfloat16,int,true>`，原控制对象符号的地址、
  大小和节不变，仅增加命名别名；不是以其它同寄存器数 kernel 代替。
  归档同时保存两次 host 编译尝试、既有 trace 的 API 区间和未捕获残差，不另采 GPU 数学。
  SHA256：`6b32d7af917de94c35eab6b57b234801af64eab9d375e760f342dc8b26937c49`。
