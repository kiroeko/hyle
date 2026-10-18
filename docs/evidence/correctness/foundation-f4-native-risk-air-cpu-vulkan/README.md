# F4 air cpu-vulkan native fault journey

固定产品candidate 29d3c7a3e，新增test-only fixture。只构建F4普通Compute/Graph直接消费者，
复用同route/toolchain/configuration树；库按可用内存j4/j2/j1，test1。

原Vulkan实际GPU copy已接受；沿现有private设备fn table给fence poll/wait/drain注入
OUT_OF_HOST_MEMORY。unknown不发writer、不完成candidate、不回收live clone；逻辑取消后
未证明退休时cold rearm拒绝。cancel及abandon恢复原设备观察后实际退休、旧源值保持。

Windows首轮fixture把blocking wait放在解除注入之前造成自阻塞，随后误把未取消slot
的no-op rearm当新admission；这些fixture FAIL/已用预算保留。没有产品修改或换目录重置。

全部原FAIL/elapsed和原日志SHA保存在raw-logs.zip；可读.log只清理行尾空白。
本证据不代签SDK、WSL、其余风险、P0/终局红队或F5 attention。
