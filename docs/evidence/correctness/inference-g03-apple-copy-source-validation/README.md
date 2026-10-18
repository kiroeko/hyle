# G03 Apple readonly source 候选宿主与三交付验证

固定 candidate dab23ba92256cad21f2e83af6141eb8ad51264c5；2026-10-14。
[五组 actual 记录](receipt.json)、[raw/编译命令/输入/脚本/显式模型](raw.zip)与
[变更影响计划](validation-plan.json)。全部实际 terminal0，身份 MacBook Air Mac16,12
arm64/macOS27.0.1；没有 Windows Apple binary journey。原 source 交付证据见
[对应阶段](../inference-g03-apple-copy-source-stage/README.md)。

ASan、UBSan、TSan 各实际运行来源64（固定program/两slots、精确bits、旧owner、partial
subview、容量/foreign/spoof/mutable拒绝、注入allocator绑定/准备零新增），F16/BF16
ready producer→dirty页→原attention各64步，以及五个原actual native copy lifecycle风险。
编译命令明确查 changed apple_memory.mm / apple_copy_program.cc / apple_copy_recording.mm
与真实消费者的对应 -fsanitize。ASan detect_leaks=0，未宣称 LSan/shader/driver/private
MLX pool 资格。TSan 复用已独立反证的 final-counter acquire 与同 Metal command-buffer
completion 模型；原模型字节保于 raw，非 suppression，产品/util/cache 不变。

relocated static/shared 使用同一 canonical 公共consumer字节：source64/native128仅
hyle::compute，原 CPU→Apple页→CPU joint768仅 hyle::graph；每包四独立 strict headers，
真实输出与权限全部PASS。产品SDK build只选对应公共闭包/四jobs，安装后重定位；独立
consumer compile commands 不含源码树 include，原 apple_memory 安装头逐字节匹配。
原 source/direct graph/copy8/copy风险已经通过，不重复闭合数学 P1。

selector selected docs/closure/configure/Apple header/直接 graph_route_cpu_apple/compute与
graph delivery，由上述真实 source、独立头、direct joint 和两种SDK覆盖；保141个跳过
理由。公共头只增加合同注释，原 symbol/layout/依赖图不变，但仍实际验证全部相关交付。
docs/closure 的当前 verdict 由 terminal 的机械检查给出，不由本 raw 的测试结果代签。

红队重读：只有原 ImportedArrayStorage 可进入家族资格；伪造 domain/host与普通异资源
不进入。actual resource owner/range仍保，原 copy_span/recording 原 device/alias核验未
删；写端严格 identity未变；bind failure不改现有binding；不增公开字段/Completion/queue。
这关闭 ready来源固定slot缺口，整步 import native费用/ready host waits保留；完整pending
page到attention、整个G03/goal仍未完成。
