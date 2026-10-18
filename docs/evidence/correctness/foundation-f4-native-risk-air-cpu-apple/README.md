# F4 air cpu-apple native fault journey

固定产品candidate 29d3c7a3e，新增test-only fixture。只构建F4普通Compute/Graph直接消费者，
复用同route/toolchain/configuration树；库按可用内存j4/j2/j1，test1。

原MLX GPU checkpoint挡住原native owner；后续实际F4 Metal copy pending时，cancel/
释放全部F4 controls/concurrent wait与close均不提前发writer或回收源native use。原MLX
late prefix Internal故障经原copy依赖传播，candidate不能finalize；退休后旧源值保持、
metadata allocator归零。checkpoint只在测试中复制既有native buffer，不是产品数学kernel。
首次abandon fixture在LeaseUse回调已执行但原native owner仍可持有retired Job时
立刻要求allocator归零，原FAIL保留；最终在显式Context.close join后检查余额，不给F4新增隐式wait。

全部原FAIL/elapsed和原日志SHA保存在raw-logs.zip；可读.log只清理行尾空白。
本证据不代签SDK、WSL、其余风险、P0/终局红队或F5 attention。
