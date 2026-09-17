# Air 性能验证启动环境纠正

产品保持 `8e45b104`；两个未进入计时的启动失败发生在测量快照 `3894b72e`。

1. 任务启动器通过 `/usr/bin/python3` 启动 selector，实际子进程丢失 `DYLD_LIBRARY_PATH`，
   native tests 在 Context 创建时找不到已安装的 loader。只读子进程对照确认直接使用解析后的
   `/Library/Developer/CommandLineTools/usr/bin/python3` 可保留该变量；纠正启动器后两个 native
   factor 的 17 项测试全部通过。
2. 进入 A/A 后，统一性能仪器按既有环境合同主动清理 `DYLD_LIBRARY_PATH`，第一次 native
   Context 再次拒绝启动。正式产品进程的 A/A guard 随即拒绝计时，未产生有效样本。

纠正限于 Vulkan benchmark 的 Apple 冷准备：CMake 定位已安装的 `libvulkan.1.dylib`，候选和
参考都通过已有公开 `ContextOptions.loader_path` 使用同一个 loader；不修改公共仪器的环境
清理、统计门槛、产品 API、shader 或热执行。新的 Apple 测量快照是 `a41d43d5`；Windows
`3894b72e` 的合格 P1 保持原身份，不重跑。

[raw.zip](raw.zip) 保留 36 个原始文件及逐项 hash，包括两次启动器、selector 结果/receipts、
实际失败输出和 CMake cache；从 Air 取回后全部重新校验。[identity.json](identity.json) 记录
归档 SHA256 和两份快照。此记录仅说明启动修正，不能代替 Air A/A 或正式 P1 的实际结果。
