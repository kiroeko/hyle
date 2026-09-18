# C1-Y17–Y19 macOS 可移植性首轮

**结论：FAIL，未签 C1-Y 或路线完成。** 候选 `ceace753` 在 Air 严格 Release/WERROR 编译，
19 Provider、40 Direct、17 Graph 用例及源码合同均运行通过，零 runtime skip；三个套件却分别
出现19、44、24次相同的设备创建 validation 错误，合计87次。

唯一 VUID 为 `VkDeviceCreateInfo-pProperties-04451`：固定 ggml 设备创建遗漏设备已宣告的
`VK_KHR_portability_subset`。[Vulkan 合同](https://docs.vulkan.org/refpages/latest/refpages/source/VkDeviceCreateInfo.html)
要求启用此扩展；数值用例通过不能代替合法的设备使用。有限纠正另在
[当前卡 Y20](../../../../docs/implementation/tasks/compute/llm-moe-c1-ggml-vulkan-region.md#y20moltenvk-必需设备扩展重新定界并-audited)
审计，不能回写此候选为 PASS。

[结构化结果](result.json)与[原始日志](raw-bundle.zip)包含三次首次编译失败、实际修正、Windows
对应 selected 回归、最终 Air 数值/生命周期输出、VUID、完整命令、工具和二进制身份。驱动在
日志后置检查退出1，原始 receipt 留下的 RUNNING 字段不是通过结论，未修改原始失败记录。

产品修正仅 `69b3414b` 的一个冷期 PlanStep 显式默认初始化；`132e7497`、`ceace753` 为测试夹具
可移植性维护。全部其他图、算法、特例、oracle 与生命周期断言保留。Air 使用 shaderc2026.4；
Mini 的兼容 shaderc2026.3 与 SPIR-V headers 已准备核验，尚未运行此候选，不能写 Mini PASS。

[统一实现 LOC](loc.json)：Compute+Graph **46,990行 / 193文件**，23条完整实现路径，含GLSL，
不含测试/bench/vendor，自检PASS。完整三交付、真实权重、剩余绑定和合格性能仍未验收。
