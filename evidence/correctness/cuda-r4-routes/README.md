# R4 C4：CUDA 普通入口与真实 FFN/capture 检查点

[当前施工卡](../../../docs/implementation/tasks/compute/cuda-v1-four-route-foundation.md)仍为 AUDITED。
已实现路线 factory、默认绑定、Q4 packing owner 的路线名称，以及单 GPU Graph compile/prepare_replay
冷 helper。公共头没有 CUDA SDK 类型；资源与 stream 专家互操作复用已有 backend。

## 产品验证

Windows MSVC 19.50、CUDA 13.3、RTX 4070 SM89，Release shared / Werror、CPU OFF、低并行 2。
两个公共头与 Direct 两个 GPU 测试通过；Graph 三个 GPU 测试最终通过。Direct 和 Graph 分别覆盖
M=1/17、K=H=128 的 FP32 dense、FP16 dense、Q4→dense 和 Q4→Q4 FFN；两个独立槽与 stream、
每槽三组不同 activation、固定常量权重；中间为实际 HYLE SwiGLU。混合链中三个 Provider 都由
普通 factory 固定规则选择，caller 没有排序、profile 或 native callback。

Graph 检查真实 kCaptured（默认 PreferCapture 与显式 RequireCapture）、三段选定 Provider、无
隐式 tensor copy、常量借用、并发槽和无 host-wait dependency；另走 M=7 的显式 command 路线。
重绑定使 preparation 失效，重新冷准备后再运行；真实 CUDA pending prefix 下释放忙碌 Invocation
和 Executable、关闭 Context，Completion 仍保活 captured program/每槽状态，最终数据正确。

数值 oracle 在 CPU 用 double GEMM 累加，并保持 FP16 scale/dequant、阶段输出与 SwiGLU 的数值
舍入边界。Q4 常驻字节只包含 packed weights 与 FP16 scales。正常三次重放的 HYLE metadata
分配计数不增加，全部 owner 析构后平衡；不能由此声称 vendor/driver 私有分配严格为零。

## 一次测试构造纠正

初次 CTest 3/4 通过，Graph 寿命测试在 120 秒被终止。它先用 host callback 阻塞设备，再销毁
另一个空闲槽；该槽的普通 cuMemFree 析构可能等待设备进度，形成夹具自己制造的等待。将空闲槽
清理移到阻塞前，并为额外 gate Completion 冷预留容量后，只重建/重跑 Graph：3 个用例全部通过。
产品代码未修改；两个头与 Direct 已通过且未变化，不重复。原始超时与纠正日志均保留。

## 证据 / 限制

`windows-shared.zip` 保存 configure/build/CTest/LastTest、DLL hashes、配置、脚本、selector 和 LOC。
运行前真实 CUDA/cuBLASLt/CRT DLL 全部邻置 exe；非交互、串行、有限 timeout。selector policy 只为
两个实际路线增加 CUDA ON/CPU OFF 配置，现有 45 个工具测试通过。

本检查点使用维护 factor 验证真正实现；`hyle::compute/graph` 的公开产品 roots、source/static/shared
与用户样例属于接下来的 C5，尚未签 PASS。中央 manifest 选出的其它 package/native 门保留未运行，
不把本 checkpoint 计为完整 R4 完成。没有 P1；最终候选固定后才按原预算启动。

统一脚本 Compute + Graph 全实现 **32,662 行**，不含测试/第三方，self-check PASS；增量 100 行为
路线入口/冷绑定 glue，真实计算复用 C1–C3 原生 Provider，规模不是进度百分比。
