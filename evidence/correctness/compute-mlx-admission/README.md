# MLX delegated-runtime 接入研究

范围和判据见[研究卡](../../../docs/implementation/tasks/compute/foundation-architecture-admission.md)。
本目录的独立 C++ fixture 只验证公开 MLX API，不是 HYLE Metal backend 或 Provider；不签发性能资格。

版本：MLX v0.32.2 (`1f8e74e3f12f31365464a6867c6579f0e9b29d85`)。
使用官方 PyPI `mlx_metal-0.32.2-py3-none-macosx_26_0_arm64.whl` 中的同版 headers、dylib 和 metallib；
SHA-256 `e6abeac9ac5265830c9c1541b6f96e9be37a85c2446763a46ad466c63a3837ab`。
这条 C++ API 调查不依赖 Python interpreter/ABI，不代替 HYLE source/static/shared 交付资格。

预声明：F32 gate/up → SwiGLU → down，M=4/K=1024/H=256，两线程各三次变化输入；double oracle，
逐元素绝对误差不超过 `1e-4 * max(1, abs(expected))`。输入按主机 page 对齐并观察是否真实借入；
保留 caller array 至 `wait()` 完成；线程分别 drain 后检查释放。相同 compiled function 在线程迁移前后
记录 trace 次数，不用 cache 名称推断无热编译。`synchronize` 是实验观察点，不是推荐产品热路径。

运行结果：待实验。未测试设备故障、OOM、并发提交、取消、HYLE Completion bridge、原生 Metal interop、
allocator 硬配额、完整模型精度或 P1；不得从该实验推出这些资格。
