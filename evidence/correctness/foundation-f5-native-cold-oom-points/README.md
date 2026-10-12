# F5 Apple/Vulkan controller逐位置冷OOM

baseline4dd42b4cf，AUDITED NATIVE-OOM-POINT-02，test-only修改，无产品/API/公头变化。
沿已有F16/BF16 paged reference recipe分别创建实际typed Direct与原native Graph factory
controller。成功cold create测经caller allocator的allocation次数（硬上限128）；每次只在
一个位置失败，必须ResourceExhausted、释放所有已分配metadata、维持真实resource owner。

| 实际profile | Direct F16/BF16各位置 | Graph F16/BF16各位置 | 合计 |
|---|---|---|---|
| Win RTX4070 Vulkan | 59 | 59 | 236 |
| Air M4 Apple/MLX | 53 | 53 | 212 |
| Air M4 Vulkan/MoltenVK | 59 | 59 | 236 |

原生684位置全部PASS。CPU既有FaultAllocator仅抽到统一test helper，原Win/WSL CPU各
Direct14/Graph45位置复核通过，另118位置，总802。每个profile原正常数学、精确页/F4
old-new-old、metadata版本、old result、byte caps、关闭/escaped result/quota0与ragged
多请求append消费继续通过，不把冷allocation拒绝写成GPU执行失败或accepted partial。

Air运行前实际读取Mac16,12、16GiB、macOS27.0.1/26A434、Apple Clang21.0.0与CMake4.3.4，
每native runner再次核对完整identity和当前可用内存，Mini未使用。实际GPU/driver信息在
结果日志，软件Vulkan没有代签。库按可用内存j4/j2/j1、consumer1，重型跨机器串行。
Windows canonical→WSL/Air disposable副本只复制hash不同输入；没有反向复制产品。

预算仍原profile日志root/config/tree与configure300/build1800/consumer180累计：本次追加
日志而非新预算。每profileconsumer累计11.282/18.888/23.843秒，全部<180，完整命令/
inputs/历史FAIL/hash/identity见[receipt.json](receipt.json)及各raw.zip。没有使用CUDA Direct
一次追加预算。普通公头/产品形状未变，复用匹配独立头/common证据；未回放全库、原算子或P1。
变化选择与skipped reasons见[validation-plan.json](validation-plan.json)，docs/closure/diff另验。

覆盖只限经公开options caller allocator的controller/native invocation/F1/F2 metadata。
MLX/ggml/driver私有heap或device allocation、accepted H2D/native错误/unknown/abandon等
完整组合不由这些cold OOM位置代签；18 installed SDK、san/P0/guide/终审仍继续原F5六FIX。
goal保持active，产品owners保持AUDITED，没有用本轮PASS重定义全goal完成。
