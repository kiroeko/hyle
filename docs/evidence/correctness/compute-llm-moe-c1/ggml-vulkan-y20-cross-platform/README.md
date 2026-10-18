# C1-Y17–Y20：同候选跨平台局部验证

产品候选 `3c39428e3fdf1911b284f7632d891f663ad82f31`，tree
`43ae3b6ff6cd064fb928d5978f36edbca1278c99`。**本批 focused correctness PASS；C1-Y/C1
收口仍为 INCONCLUSIVE。** [结果](result.json)与[原始日志](raw-bundle.zip)保留完整命令、工具、
二进制哈希、source 身份、未完成项和 selector，不代签完整三种 C++ 交付或性能。

| 同候选实际运行 | W1 static | W1 shared | MacBook Air | Mac mini |
|---|---:|---:|---:|---:|
| Provider 用例 | 19 | 19 | 19 | 18 |
| 普通 Direct / Recorder / 生命周期 | 40 | 40 | 40 | 40 |
| 普通 Graph / 生命周期 | 17 | 16 | 17 | 17 |
| 源码合同 / Windows MASM 的 CTest 条目 | 2 | 2 | 1 | 1 |
| Validation 无 VUID / 同步诊断 | PASS | PASS | PASS | 未安装 layer |

全部实际用例通过，零 runtime skip。shared Graph 少一个既有 static-only global-new 故障夹具。
Mini 在命令中排除唯一需要 validation layer 的专用实例用例；同一用例已在 W1/Air 实际执行，
Mini 只签 correctness/portability，不签 validation 或 Formal 性能。

Air 首轮的87次 `VUID-VkDeviceCreateInfo-pProperties-04451` 已归零；其
[失败记录](../ggml-vulkan-macos-portability/README.md)保持不变。修正只在冷期从已有枚举结果
启用设备宣告的 portability_subset；原厂算法、shader、融合、热路径与 oracle 不变。

四环境构建/运行串行，外层并行2；Windows DLL邻置、PATH unset、错误弹窗模式固定，不运行
安装器或更改电源/UAC。两台 Mac 使用完整 canonical Git bundle/archive，不反向覆盖产品。

lock/补丁身份变化的 selector 仍选择112项全局门；本目录只证明上表。剩余全局 package 门、
真实权重、完整 source/static/shared consumer、合格 P1/规则与存量淘汰均未签 PASS。

统一[LOC](../ggml-vulkan-macos-portability/loc.json)为 **46,990行 / 193文件**，不含测试/bench/vendor。
该23路径从计数候选 `ceace753` 到本候选的全部 blob 相同，自检PASS。按 SHA 核对后删除四份
可由 Git 重建的旧源码压缩包，回收131,352,040 bytes；当前候选、模型和全部日志保留。
