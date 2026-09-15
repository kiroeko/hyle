# 四路线有限架构实验

范围与预声明判据见[研究卡](../../../docs/implementation/tasks/compute/four-route-architecture-decision.md)。
这里是独立研究程序，不是已交付的 HYLE Provider。源码由 Windows canonical checkout 管理，macOS
使用完整 Git archive。没有性能计时或 SOTA 签发，也不验证完整 OOM/device-loss/三交付。

- `mlx_custom.cc`：MLX 0.32.2 C++ API，GPU MatMul → 自定义 Metal SwiGLU → GPU MatMul。
  同线程六次变化输入；double oracle 与编译 trace；仅最终输出处显式同步。
- `vulkan_replay.cc`、`two_pass.comp`：一个原生 pipeline、两个 compute pass、显式内存 barrier；
  录制一次、三次变化输入、最终 fence 后 exact oracle。host-visible buffer 是同步反偏置探针，
  不是独显权重存储或性能推荐。只检查正常生命周期，不把失败时进程退出当生产 RAII。

MLX 原生扩展由公开 `fast::metal_kernel` 接入；源码生成/编译是冷启动成本，缓存命中也仍有框架 metadata/
allocator 成本。实验中的 kernel 仅证明可组合，不是 HYLE 已胜出的 SIMD/kernel 资产。

结果在执行后记录；未运行项不标通过。
