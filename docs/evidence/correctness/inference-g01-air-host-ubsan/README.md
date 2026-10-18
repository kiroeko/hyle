# G01 MacBook Air Vulkan 宿主 UBSan

candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`。实查Mac16,12、macOS27.0.1、
10 cores/16GiB，只用Air，不借Mac mini。[receipt](receipt.json)/[raw.zip](raw.zip)保
实际commands、logs与Ninja compdb，region/public consumer/native反例三TU插桩已核对。

同公开F32/F16 D64/128/256 resident math与旧结果/复用旅程通过；三个native反例通过：
precision cache两模式/双顺序、大指数F32值、非法D冷拒绝和零quota回滚。只签实际
MoltenVK/Air与宿主UBSan，shader/driver/private bytes不由此覆盖，不外推全部GPU设备。

2013项必要输入校验、只复制差分，canonical仍Windows；测试维护输入单独同步，原树/
cache/elapsed复用。8路远端编译，与本机重型build串行。没有新shader数学或CPUfallback。
TSan、CPU风险/交付、P1和终审仍待完成；本记录不是G01/G02或整个goal结束。
