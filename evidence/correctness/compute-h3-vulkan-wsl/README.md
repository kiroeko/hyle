# H3 WSL 三交付：编译/安装资格，GPU运行未资格

Canonical Windows产品97f9332ac，同库补充fixture d2d1e86e3。archive SHA与全source同步、相同文件
保留mtime、新fixture SHA均在此处收据。GCC14.2/C++23，原Source/SDK缓存复用，不从WSL反向改产品。

实际source、installed shared/static的编译/安装、原staging真rename、三新消费者链接和三个新增
安装公开头通过（result.json的十二阶段全exit0）。genuine Vulkan-only缺CPU与CPU+CUDA缺Vulkan
在find_package阶段给出实际拒绝。四Vulkan公开头禁用native macro而无native SDK include时通过；
shared/static六消费者ldd没有not found，不设置LD_LIBRARY_PATH补救。

systemd hyle-h3-vulkan-build.service实读MemoryMax=12884901888、MemorySwapMax=0，exit0后unit
inactive/MainPID0；完整运行15min53.798s，peak 5.4GiB，swap peak0。build1，重型验证串行。

当前WSL只有llvmpipe软件设备，用户明确接受C3及后续H1–H3的这个验证边界。
这里**没有运行GPU数学、资源寿命或overlap**，这些仍为UNQUALIFIED，不是PASS或NOT_APPLICABLE。
Windows和Air真GPU三交付/合同/P0由其独立证据签，不用WSL链接或设备枚举冒充运行。
