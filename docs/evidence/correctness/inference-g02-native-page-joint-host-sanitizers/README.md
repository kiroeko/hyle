# G02 联合 CPU/native page/CPU 六组宿主风险

不可变产品candidate `1cd0893979e1672a0f6fac98b783a6a3bfcabbca`；actual Mac16,12/M4/
macOS27.0.1/AppleClang21，复用六个native-host tree，cpu+actual native routes、四jobs
串行。每route ASan/UBSan/TSan 的 public joint consumer 与全部新联合风险最终exit0。
[receipt](receipt.json)、[raw](raw.zip)保实际插桩编译命令、candidate28输入SHA、各次日志/
验证脚本/parent和原FAIL；累计455.074s，不重置原失败或树。

ASan detect_leaks0，无LSan；全部仅宿主插桩，不签shader/driver或native private pool。
Apple TSan沿原G01显式同一MTL command buffer commit/completed callback边界模型；原
Metal cache与产品头不改，不用suppression。

另保真实原Apple joint TSan FAIL：final atomic decrement与ResultStorage销毁free被报告
为竞争。相同机器/编译器的独立纯C++程序，没有HYLE、Metal或Sched，采用release-RMW＋
最终acquire fence：两refs及三refs/两个独立producer均复现原FAIL，acq_rel对照与只补
最终counter acquire模型各256lifetimes通过。独立源/原始日志及copied头均在raw内。
[LLVM21.1.3原实现](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.3/compiler-rt/lib/tsan/rtl/tsan_interface_atomic.cpp#L486)
中OpFence模型未实现；[下载身份](upstream-source.json)保精确URL/SHA。此源码事实与
实际编译器独立复现共同支持工具边界判断，不把一般san成功当C++内存模型证明。

只在验证util头副本、真正atomic counter上、最后一个drop的原acquire fence后加
`__tsan_acquire`。TSan已记录release RMW并合并release clocks；不新增intermediate
drop同步、额外payload同步或跨对象边，无suppression。原util头/canonical/依赖cache
SHA不变，普通release-only成本不为工具改成acq_rel，util没有伪reopen。两个joint
targets和HYLE宿主owner以同一copied头编译，parent和before/after SHA在raw内可核。

Vulkan首次清单用了不存在的vulkan_context.cc，在成功build后脚本assert失败且测试未
启动；改清单为实际vulkan_device.cc后同原树续跑，原build/validator85.740s保留。
只最终六组资格通过；原未建模TSan仍FAIL，不写成原始无模型的PASS。

本证据不代签新joint relocated static/shared/逐头、Windowscompile/link、原G02其它
保留门或整个goal，也不代签token/s、全平台capture或private内存硬上界。
