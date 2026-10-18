# Native page 三交付中的搬迁 SDK 阶段

产品candidate `520027a57`。actual MacBook Air Mac16,12/M4/arm64/macOS27.0.1/
AppleClang21；Apple/Vulkan各static/shared，共四份新搬迁SDK全部PASS。生产复用
原selected native树，fourjobs，只build实际route、batch/state及直接Graph交付依赖。
新的stage安装后搬迁到relocated，未覆盖原CLOSED证据对应SDK。

consumer仅find_package/链接公开hyle::compute；编译命令逐项核对只来自relocated/include，
不借canonical include或内部target/native proxy headers。每份均实跑普通page GPU数学
（Apple F16/BF16三D/两causal，Vulkan F16三D/两causal）、双slot/64代际/旧output、全空
output/旧空result、foreign resource、fixed backing budget与cold OOM（Apple52、Vulkan55）
拒绝保持旧准备可运行。每份另独立编译batch_paged_native.h、typed route batch头和
共同controller头，共十二个standalone header TU通过严格warnings。

SDK的compute closure不承诺可选SIMD转换helpers；test使用独立compiler IEEE half与
scalar BF16 bit carriers，保原独立数学decoder。这个test-only adapter只改变数据准备，
没有复制产品头进SDK或改变产品/API/数学。Apple static初次consumer-build1.047s FAIL
因消费者未用descriptions局部变量；删除dead local后通过，原日志与累计保留。

receipt.json/raw.zip保四份每阶段身份/命令/输入/严格编译/退出码/elapsed/累计及所有失败，
public consumer源码和oracle carrier也归档。结合原实际source阶段签三交付有限公开消费，
不代签shader/driver/LSan/native-private bytes、Windows资格或不可变P1。新native host
ASan/UBSan/TSan仍在活动卡执行，整个G01/G02/G03-G10仍未完成。
