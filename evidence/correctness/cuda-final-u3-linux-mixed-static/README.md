# CUDA U2/U3：Linux CPU+CUDA 静态交付

source与搬移offline独立SDK的两路线实际加载/运行通过。源码Direct1项/Graph7项/CPU3项，
SDK GPU Direct166项/Graph151项、CPU独立3项通过，无失败或skip；61个固定步骤一次完成。
原依赖对象复用，模式变化仅编译必要HYLE/消费者对象；CPU普通F32/Q8/Graph重放保持。

命令、日志/XML和哈希见[receipt](receipt.json)、[归档](delivery.zip)、[用例并集](sdk-test-union.json)。
SDK取搬移prefix的HYLE头/库和原CPU动态ISA模块；Torch cache/原ggml参考资产为声明依赖。
未设置LD_LIBRARY_PATH，原ELF loader实际通过。不是模型级CPU+GPU联合推理。

原34个packed合同按完整case分进程，保持原组预算、shape/精度/门，并集核对相等。
完整shape数学复用同原源[两平台native全集](../cuda-final-u3-f32-native/README.md)，不把
shipping检查当新的shape全集。CPU算法/已闭性能轮未改；本批无P1或新性能赢家。
当前U2/U3承诺的Windows/Linux、CUDA-only/mixed三交付已齐；U4/U5/U6仍未完成。
