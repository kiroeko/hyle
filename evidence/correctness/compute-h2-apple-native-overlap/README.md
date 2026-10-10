# H2 Apple原生copy/compute时间线

最终candidate37744b0ea，MacBook Air M4/16GiB/macOS27.0.1，原锁定MLX与真实Metal程序。
本项P0只证明原生机制，未进行速度比赛/用户timing profile/统计WIN。

初始4轮16次MatMul的整个command-buffer区间相交，因粒度较宽仅保留diagnostic。随后固定两轮
单次4096×4096原MLX MatMul；test-process原生objc selector记录实际dispatch次数，确认每个
compute command只有一次dispatch，与原WeightSlots的16MiB Metal blit分别使用独立原生queue。
completed handler在commit前注册，从真实GPUStartTime/GPUEndTime读取记录；有计算提交事实
才上传，不用两个pending或CPU墙钟推断重叠。原算术和GPU执行代码没有替换成mock/HYLE自研。

两轮single-dispatch compute与blit的GPU执行区间相交约56.386ms、52.764ms，所有native command
status为Completed；原MLX矩阵数值也检查通过。原生区间证明工作可重叠，不给出独立DMA引擎、
每周期并行或吞吐收益承诺；带宽/占用/调度可显著改变copy成本，16MiB非单纯空闲copy性能结果。

[原诊断](overlap-run.log)、[单dispatch原始记录](overlap-single-run.log)、
[固定检查](overlap-single-verdict.json)与[实际运行](overlap-single-result.json)保持不同身份。
[fixture](single-dispatch-probe.mm)只进入验证exe，不改变产品、公共仪器或MLX全局编译模式。
CLT无xctrace，使用此有界原生记录，不无预算扩建通用instrumentation。
