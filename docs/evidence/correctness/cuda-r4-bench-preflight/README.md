# CUDA 完整 FFN 比较夹具：编译与数值预检

产品代码沿 `f1cf11121b5d4ad9529e3079b82b7e19717e2420`；新增的 benchmark source 与四个
实际 exe 逐文件 SHA256 在 [summary](summary.json)，[原始配置/编译/检查日志](raw.zip)及
[归档哈希](sha256.json)一并保留。本记录不签 A/A、P1、原生包装成本或默认 winner。

- W1-WIN MSVC 19.50.35728、CUDA/nvcc 13.3.73、RTX 4070/SM89；driver API/runtime 13030，
  NVIDIA 驱动发行版本另由主卡 inventory 记录为 610.62。
- Ninja 并行 2、static shipping、Werror、host caller P 核 2、实际 CUDA/CRT DLL 与 exe 邻置。
  两种 placement 的 Direct/Graph 四个 exe 编译通过；本次数值检查执行普通 placement 两个 exe。
- 两侧均为 Q4 gate/up → precise SwiGLU → FP16 dense down，M=1/16、K=4096、hidden11008、
  output4096；两个不同 resident 输入、独立 output/run-state，固定 packed Q4 常量 46,497,792 bytes。
- Direct 和 Graph 均通过两种形状、两个输入的全部输出检查；Graph 要求实际 `kCaptured`。
  两侧输出在本次数据中逐值相同；相对独立 double-accumulation/FP16 stage oracle 的最大绝对误差
  为 0.00048828125。误差门在执行前已写入夹具，未按结果调整。

夹具初始配置缺少本目录 `enable_language(CUDA)`；最初共享常量与各 invocation resource 不属于
同一声明资源域，Direct/Graph 均按合同拒绝。修正仅在 benchmark：显式准备实际 tensor views，
两侧借同一个 device resource，仍各自持有独立 buffers、queues 和两槽；没有改变产品资源检查。
初始失败日志保留。测试后的 clang-format 仅改变空白，不触发重建或重跑数值检查。

成熟参考保留 SGLang pin 的 kernel 算术与 indexing，使用精确配置和同 HYLE Provider 生命周期。
它只随 benchmark 构建；未增加产品 Provider、公开 API、运行时寻优或普通用户配置。
