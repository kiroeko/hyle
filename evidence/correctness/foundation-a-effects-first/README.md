# Foundation A effects 与 CPU+CUDA：第二批 source 事实

活动范围为 [A-04](../../../docs/implementation/tasks/graph/foundation-a-route-symmetry.md) 与
[A-CUDA-01](../../../docs/implementation/tasks/graph/foundation-a-cpu-cuda-variants.md)。
所有结果为 AUDITED 实施检查点；没有代签 Apple、本轮完整 sanitizer/installed/红队或 CLOSED。

Windows MSVC19.50，shipping HYLE_TESTING=OFF；库按实际≥12GiB j4，test1。
复用独立 CPU+Vulkan shared 和 CPU+CUDA static source 树，仅真实 target/deps，不运行全目录。

- 共同40/40 PASS：原39项加 unused resource 声明不虚构 effect carrier；共同 effect helper
  提取保持原 SmallVector0 和原 footprint 判定。两批增量8.218+7.985秒。
- 内部 CPU+Vulkan consumer 与原七个 effects/alias/OOM 用例8/8 PASS；资源 write→GPU math→read，
  跨片段 alias、缺 footprint/GPU carrier 拒绝、旧结果 guard。原 first build FAIL86.328秒因
  私有 fixture target 未链接其直接 ggml/native Provider；修正真实依赖后 build89.531秒 PASS。
- [公开 C++ consumer](../../../tests/graph/public_effect_consumer.cc) 只包含安装公开头，无内部
  Provider/Graph runtime：原 make_custom_provider/make_custom_kernel 声明真实 CPU sidecar
  读写端口，原 GPU 数学。正常/接受后失败保留写入前缀但停止后继、cross-fragment alias、
  零 tensor output effect、close 后 old result 和用户 Context 再用均 PASS。
  首批 build80.250秒，完整 shape 补证增量7.297秒；两次journey各0.250秒。
- [CPU+CUDA consumer](../../../tests/graph/route_symmetry_cuda.cc) 实际 RTX4070/SM89，两个精确
  shape、表外拒绝、CPU→CUDA→host 数学、旧结果 guard 与有限 family quota 不按点拆分 PASS；
  build80.360秒/journey0.875秒。只新增冷 typed factory，原compiler/执行器/Provider 保持原机制。

上述 gate 各目录只是不可覆盖日志名，原 FAIL 与预算持续累计：effects 内部 build175.859秒，
public effect build87.547秒，CUDA finite build80.360秒；不是重新给同 gate 1800秒。
共同 static gate 加上第一批证据累计311.952秒；未运行项不记 PASS。
源码自身旧混合 EOL 在一次私有编辑中被展开，已逐行恢复未改原 bytes；保留有限真实 diff，
编译使用的是同语义修改，换行不构成新的产品 candidate 或性能资格。
