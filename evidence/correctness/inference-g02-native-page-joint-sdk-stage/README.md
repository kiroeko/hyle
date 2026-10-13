# G02 联合分页四份公开 Graph SDK

不可变candidate `1cd0893979e1672a0f6fac98b783a6a3bfcabbca`；actual Mac16,12/M4/
macOS27.0.1/AppleClang21。CPU+Apple、CPU+Vulkan各static/shared，原cached producers
定向四jobs构建并install到新staging，再搬为relocated SDK。四份普通消费者只链接
`hyle::graph`、只用relocated includes；公开编译命令不引用canonical/source/cache。
[receipt](receipt.json)、[raw](raw.zip)保命令/身份/consumer前后SHA、编译命令、原始日志
和四consumer-source；累计302.419s，全部阶段exit0。

每份执行同一实际CPU ABS/Cast→原native page→CPU Cast/Neg模型片段，Apple12组144/
Vulkan6组72episodes；static和shared合计432episodes，独立double oracle、三D/dtype/
causal、双slot/topology/旧结果/精确frame与上传费用通过。公开domain consumer另验two
page nodes/shared input、所有错误node/reader/layout/schema/resource/frame/shape拒绝、
拒绝保旧binding、全空输出及冷compile/create/prepare/bind逐OOM和backing/host/shared
预算。source-private native fault/FIFO/driver fixtures没有混进SDK。

各份独立编译graph/batch.h、cpu_apple.h或cpu_vulkan.h、graph/cpu.h，合计十二TU，
严格warnings。普通数学consumer保canonical独立bit encoder不变；公开domain helper
原测试用SIMD half converter只替换为test-only IEEE scalar carrier（所用值均精确可表），
不把optional SIMD helper当SDK必备API。原canonical与实际consumer SHA逐个保记录。

这些SDK实际资格不签Windows GPU/math、全模型token/s、capture/delta planning或native
private内存硬上界；Windows新joint compile/link仍由本轮单独记录，原G02其它未决门保。
