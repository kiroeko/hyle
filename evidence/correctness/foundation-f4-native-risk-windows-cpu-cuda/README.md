# F4 windows cpu-cuda native fault journey

固定产品candidate 29d3c7a3e，新增test-only fixture。只构建F4普通Compute/Graph直接消费者，
复用同route/toolchain/configuration树；库按可用内存j4/j2/j1，test1。

原CUDA host callback挡住实际copy Queue；已接受F4 dirty clone仍pending时，cancel/
释放所有F4 controls/concurrent ticket.wait与Store.close均不发布writer、不提前释放源native use；
实际退休后旧源值保持、metadata allocator归零。原native cancel无抢占合同未改变。

全部原FAIL/elapsed和原日志SHA保存在raw-logs.zip；可读.log只清理行尾空白。
本证据不代签SDK、WSL、其余风险、P0/终局红队或F5 attention。
