# CUDA U2/U3：Linux CPU+CUDA 动态交付

source与搬移后的offline独立SDK均通过两条路线加载/执行。源码Direct1项/Graph7项/CPU3项，
SDK GPU Direct166项/Graph151项、CPU独立3项通过，无失败或skip；61个固定步骤一次完成。
CPU测试包含F32数学、原Q8 codec与Graph变化输入重放；不做模型级CPU+GPU联合推理。

命令、日志/XML和哈希见[receipt](receipt.json)、[归档](delivery.zip)、[用例并集](sdk-test-union.json)。
SDK HYLE头/库只来自搬移prefix，原Torch机器cache与ggml参考资产是声明依赖。
未设置LD_LIBRARY_PATH，原ELF RPATH/原CUDA12与13组合的loader实际通过。

34个packed合同按原完整case隔离进程，保持组预算、shape、精度和门，并集逐名核对。
完整shape数学复用同原源[两平台native 651/649](../cuda-final-u3-f32-native/README.md)，
不把shipping检查当新shape全集或整模型推理。原CPU数学/已闭性能轮未改变；本批无P1。
Linux mixed static及U4/U5/U6仍待，本页不签全部CUDA完成。
