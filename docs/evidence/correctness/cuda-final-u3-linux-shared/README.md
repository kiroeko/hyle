# CUDA U2/U3：Linux CUDA-only 动态交付

源码构建、普通Direct1项/Graph7项、安装搬移、离线独立SDK配置/链接和实际ELF加载通过。
SDK Direct166项、Graph151项通过，无失败或skip；完整数学shape复用同原源的
[native 651/649](../cuda-final-u3-f32-native/README.md)。50组检查覆盖60个GGUF角色/type、34个
non-GGUF、必要完整current/family/state与本地Qwen原权重，不把shipping数目当新的shape全集。

原命令/记录/日志/XML与哈希见[receipt](receipt.json)、[归档](delivery.zip)、[用例并集](sdk-test-union.json)。
HYLE头/库仅从搬移SDK取得，Torch原cache与独立ggml reference为声明依赖；LD_LIBRARY_PATH未设置。
原RPATH和原CUDA12/13共存组合实际运行通过，不据此签其它版本/机器。

第一次configure的WSL shell分号转发错误保留，`--exec`同ID纠正通过。Direct packed合并进程
达到11979/12282MiB并在300秒超时，原FAIL保留；同ID第二次改为原34个完整case各自进程，
66.953秒内通过，合并并集逐名相等；Graph同原case分组亦通过。没有改shape、误差门或数学，
没有把timeout作性能输赢，没有提高原组预算。已通过source/install/link等全部复用。

Linux mixed shared/static及U4/U5/U6仍待；本批无P1，不签整个CUDA收口。
