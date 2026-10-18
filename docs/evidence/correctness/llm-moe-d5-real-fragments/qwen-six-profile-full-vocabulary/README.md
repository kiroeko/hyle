# D5 六个 Qwen 画像完整词表批量复核

Qwen3.5-2B 与 Qwen3.8-27B 的官方、primary GGUF、fidelity GGUF 六个固定画像，
共 48 个 route/entry 格，原先因 R01 只测七行而退回待验。本批在普通 Direct/Graph
入口保留 H=2048/5120 和完整 V=248,320；GGUF 的 embedding/head 使用各自真实
角色格式。输出头无损分片并核对全部 logits，Gather 检查跨片索引、全局越界及失败不覆盖
已发布结果。Qwen3.5 官方输入来自用户本地 checkpoint 的有界夹具
`4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155`；
其它巨型画像使用固定 header 和格式合法的非零合成 payload，不声称持有完整权重。

旧收据中的 R02–R05/R13–R15 或 R14–R15 成功片段继续作为历史证据；本批同一
owner suite 重新运行这些片段和完整 R01，按精确画像、路线及入口重新组合格 verdict。
Windows 不可变 `3b326cd1` 上六个选中 factor、configure 均 PASS；CPU Direct
在测试判定修正提交 `b2ef3eb0` 上增量重编并完整运行 PASS，CPU Graph 在仅删除
未使用测试变量的 `57cd5a67` 上同样 PASS。后两次为明确选择的 owner gate 直接
构建/CTest，未声称各自重新取得完整 selector verdict。Windows 测试夹具环境实际设置，
CPU/CUDA/Vulkan 的 Direct/Graph 没有 D5 SKIP。

MacBook Air `Mac16,12` 上，`b2ef3eb0` 的 Apple Direct 与 Graph factor
各自取得 selector `PASS`，普通入口分别 104/104、77/77 且没有 SKIP。
Mac 上 CPU F16 的初次逐元素测试失败是验证口径错误：固定 ggml ARM NEON 上游
用 FP16 向量累加，而旧 oracle 假设 FP32 累加。六次真实宽度投影的独立 FP32
oracle 对照最大绝对差 0.001953、整体 RMS 相对误差 0.00037–0.00043；
修正只对 ARM `ggml-cpu` 使用最大绝对差与 RMS 双门，其他 Provider 仍用原逐元素门。
Mac Vulkan 首次因系统 Python 清除 `DYLD_LIBRARY_PATH` 找不到 loader；在
Python 进程内设定已登记的 Homebrew loader 路径后，Direct factor PASS，Graph
owner 的单独完整 CTest 也 PASS。环境失败不是产品 verdict。

Mac CPU Graph 的额外跨平台检查仍未通过：一个 F16 测试沿用同一旧 oracle，另一个
把官方 BF16 权重转成 F32 后强制要求 oneDNN 融合，在 ARM 上选择了 `hyle-native`。
它不是本卡所定 Windows CPU 产品机的六画像签署依据；不得把 Mac 整体 selector
写成 PASS。该测试形状和 ARM 默认选择需单独定界修正。D5-A5 的外部三交付与
D5-A6 的适用 P1 尚未运行，本批不代签。

机器、候选和逐入口判定见 [receipt.json](receipt.json)。
