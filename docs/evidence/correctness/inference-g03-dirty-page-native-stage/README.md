# G03 GP01/GP02 实际 native 增量消费与退休反证

2026-10-14，八产品输入SHA与actual命令/硬件在[receipt](receipt.json)，原始输入1–8、
全部尝试、runner、CPU新阶段与crash report在[raw](raw.zip)。实际Mac16,12/M4/arm64/
macOS27.0.1/AppleClang21，四jobs、Windows/Air heavy串行。Windows canonical单向覆盖
可丢弃source副本；不改immutable P1 source/cache/原数学。

当前GP02先原nonblocking observe再原take_mutable；首次只有take时，实际Apple abandon
成功native工作仍不能返回writer。这个真实公开生命周期缺口已修正，保原FAIL/input。
当前CPU public64与arena/F2/F4九test重新PASS，不能拿旧4f67 bytes代签新代码。

[Apple F16/BF16](apple-public-math-2.log)各64、[Vulkan F16](vulkan-public-math-1.log)64，
共192 final episodes：GQA8/2 D256、page16、Q1/17混合prefill/decode、两request长度
17→80/17→48、跨页、active output变化与原generation保持。每请求新K/V两行，每组
64步共196608 dirty字节，无整页/历史搬运；独立double causal加权全输出通过，未写
history参与oracle。referenced页不可覆写，unreferenced page9独立可写；旧result拒绝
prepare，下一显式cold prepare确实释放旧native imports，取得下一dirty writer。

[Apple五risk](apple-joint-3.log)是原MLX实际GPU checkpoint之后接纳的F2 queued phase，
成功、cancel、真实prefix error、abandon与closed；观测前source uses=1、无ready output/
writer，actual retirement后source uses=0。只有成功/成功abandon能返原writer并生成exact
frame；取消、失败或closed不经unbind逃出。
[Vulkan四risk](vulkan-joint-4.log)使用真实QueueSubmit/WaitForFences：成功accepted1/wait1，
拒绝accepted0、unknown accepted1/实际wait1并保原error、closed accepted1/wait1。
无假成功fence，unknown/failed不发布数据或writer。依赖phase借原native ordering。

Vulkan最初fixture把原UMA host-ready短copy当成必须有native admission；后续明确empty
logical phase→dependent真实copy，原collector路径和产品选择不改。还曾错误地期待GPU
拒绝时完全无submission，实际empty logical prefix已接纳，必须用其token观察tail error。
早期fixture异常退出又未先drain borrowed resources，造成SIGSEGV；原crash report显示
context teardown晚于resource owner。已补test-only scope drain并保FAIL，不宣称产品破坏
borrow合同。LLDB因noninteractive debug权限拒绝，原ips提供真实stack，未扩大权限。

[原F3直接消费者](f3-run-1.log)十四tests全部PASS，重链接当前common库；source SHA与
canonical匹配。API没有新增state/controller/allocator/拷贝；P0为原ready page O(1) unique
与原F2 O(accepted phases) nonblocking观察。显式metadata prepare/host observation费用
和native private UNKNOWN仍保留，不签低等待pending producer→attention或G06/G09。

这是source/math/风险阶段；host san、relocated static/shared/逐头仍待实际签发。没有
Windows GPU run、全模型token/s、Vulkan BF16、整G03/goal完成声明。
