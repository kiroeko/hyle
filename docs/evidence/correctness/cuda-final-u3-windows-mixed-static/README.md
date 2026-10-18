# CUDA U2/U3：Windows CPU+CUDA 静态交付

普通CUDA配置只覆盖CPU=ON，source/static同时交付两条独立路线。源码构建、Direct1项、
Graph7项（含本地Qwen实际权重）及CPU3项通过；install/prefix搬移、离线独立SDK配置/链接通过。
SDK固定GPU交付检查Direct166项/Graph151项，CPU独立3项全部通过，无失败/skip。
原已通过configure复用，未重新配置；其余59步一次完成，没有新增数学或全shape重跑。

记录及原日志/XML见[receipt](receipt.json)、[归档](delivery.zip)、[用例并集](sdk-test-union.json)。
Windows运行PATH只有System32/Windows，必要DLL邻exe；SDK只取搬移prefix的HYLE头/库。
单独CPU检查覆盖F32 Direct MatMul、原Q8 codec和变化输入的Graph replay，采用Provider要求的workspace。
这不是模型级CPU+GPU联合推理，不重开CPU已关闭的性能/算法轮。

完整数学shape、格式/精度/资源资格复用同原源的[两平台native全集](../cuda-final-u3-f32-native/README.md)，
不把317个shipping检查写成新的559 shape全集。原source-id仍f99368c68335266b，原CPU数学保持。
本批无P1或新WIN。Windows CUDA-only/mixed三交付子格均已完成；Linux shared/mixed及U4/U5/U6仍待。
