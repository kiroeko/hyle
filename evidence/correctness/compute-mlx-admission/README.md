# MLX delegated-runtime 接入研究

范围和判据见[研究卡](../../../docs/implementation/tasks/compute/foundation-architecture-admission.md)。
本目录的独立 C++ fixture 只验证公开 MLX API，不是 HYLE Metal backend 或 Provider；不签发性能资格。

版本：MLX v0.32.2 (`1f8e74e3f12f31365464a6867c6579f0e9b29d85`)。
使用官方 PyPI `mlx_metal-0.32.2-py3-none-macosx_26_0_arm64.whl` 中的同版 headers、dylib 和 metallib；
SHA-256 `e6abeac9ac5265830c9c1541b6f96e9be37a85c2446763a46ad466c63a3837ab`。
这条 C++ API 调查不依赖 Python interpreter/ABI，不代替 HYLE source/static/shared 交付资格。

预声明：F32 gate/up → SwiGLU → down，M=4/K=1024/H=256，两线程各三次变化输入；double oracle，
逐元素绝对误差不超过 `1e-4 * max(1, abs(expected))`。输入按主机 page 对齐并观察是否真实借入；
保留 caller array 至 `wait()` 和显式 stream `synchronize()` 完成；线程分别 drain 后检查释放。相同 compiled function 在线程迁移前后
记录 trace 次数，不用 cache 名称推断无热编译。`synchronize` 是实验观察点，不是推荐产品热路径。

运行结果：MacBook Air `Mac16,12`、macOS 26.6.2、Apple clang 21.0.0、SDK 27.0。
初版 `66dea696` 的[运行日志](run-macbook-air.log)六次通过；随后按源码发现纠正完成观察点，
纠正版 `eb94ae05` 的[运行日志](run-checked-macbook-air.log)再次六次通过，最大绝对误差
`7.30185e-09`，六次输入真实借入并在最终释放，跨线程 trace 从 1 变成 2。未测量延迟。

初版[链接警告](build-macbook-air.log)为 target=26.0 而 dylib 最低=26.2；纠正版使用
`-mmacosx-version-min=26.2`，[build log](build-checked-macbook-air.log)为空、命令 exit=0。
两次都由 Windows 对应完整 Git archive 导出到一次性 Mac 目录；没有远程产品回流或全局 SDK 安装。

独立复现：解压上述 wheel 后，以其 `mlx/include`、`mlx/lib` 路径替换命令内占位路径：

```sh
clang++ -std=c++20 -O2 -pthread -mmacosx-version-min=26.2 \
  -I/path/to/runtime/mlx/include probe/main.cc \
  -L/path/to/runtime/mlx/lib -lmlx -Wl,-rpath,/path/to/runtime/mlx/lib -o mlx-probe
./mlx-probe
```

未测试设备故障、OOM、并发提交、取消、HYLE Completion bridge、原生 Metal interop、
allocator 硬配额、完整模型精度或 P1；不得从该实验推出这些资格。
