# F4 WSL CPU+CUDA native-risk journey

固定candidate29d3c7a3e，实际GPU/Ubuntu/GCC/NVCC身份及输入hash在job；原source树增量，
只新F4普通Compute/Graph直接消费者，库按实时可用内存j4/j2/j1、test1。
原CUDA host callback挡住原Queue已接受copy；pending时cancel、放弃全部F4 controls、
concurrent wait与close三场景禁止writer发布，source实际native use保留到退休。最终原source
数值保持，Store metadata余额归零，无新增copy/math kernel或GPU抢占承诺。
首次fixture在native use退休回调之后、原CUDA reaper清理末个descriptor/pool metadata
之前销毁借用allocator，触发QuotaAllocator断言。调整allocator先于endpoint构造、在全部
F4 controls释放后显式有界观察原reaper cleanup，三个场景通过；产品candidate不改，
原exit -6/时间/log保留，不把native use计数为零当全部owner已经销毁。

原失败/时间/原日志SHA完整封存；本结果不代签其他平台、installed SDK或F4/F5终局。
