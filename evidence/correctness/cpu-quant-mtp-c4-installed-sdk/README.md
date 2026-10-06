# C4 安装 SDK 的普通 CPU 消费

Windows 固定消费者候选 `216242d2fabdfae8e69d00872786aac0148442c1`，
安装 SDK 候选 `0bea4748333fbb94ff7e132119e4d12ee299dd4a`。
两者 `include/src/cmake/CMakeLists.txt` 无差异，复用已搬移的 C3 安装产物。

- Windows static/shared 各用 `find_package(Hyle)` 构建普通 Direct、Graph 消费者，
  每个执行 5 项，共 20 项通过。原始命令、日志、JUnit 及消费者 CMake 在
  [Windows 收据](windows/summary.json)。没有编译或链接源码构建中的 Hyle 产品库。
- 静态 SDK 的数值参考也链接 SDK 导出的固定 ggml 库；动态 SDK 的参考使用匹配的
  源码构建 ggml 静态库，Hyle 产品调用使用安装 DLL。这些原库仅计算独立参考数值，
  不把 ggml 对象传入产品。GTest 是消费者测试依赖。
- 各 exe 使用独立输出目录，Windows DLL 邻置；离线依赖 cache、SDK 搬移和资产资格
  复用 [C3 交付](../cpu-quant-mtp-c3-mixed-batch/README.md) 的相同产品身份。
- 本批覆盖 target projection/confidence、全词表 Markov、Qwen 混合格式完整 MTP、
  两份 Flash 混合格式三阶段 sidecar，以及原版 FP4 projection 接缝。
  完整官方主干和三阶段资格仍来自已保存的源码消费者，不能由这五项替代。

Linux static/shared 的同批 20 项已通过，见 [Linux 收据](linux/summary.json)。
该 SDK 候选 `11462ba08825d86fbf63e179fa2f4a591b2d37d9` 与本批产品源码也无差异。
两系统合计 40 项安装消费检查；源码完整程序、安装入口与未变依赖资产证据共同使用。

资源边界见 [资源资格](../cpu-quant-mtp-c4-resources/README.md)，成本见
[P0 审计](cost-audit.md)。C5 全部准入制品矩阵尚未完成，不能把这批 SDK 检查代签 104 档位。
统一脚本重新统计为 **61,475 实现行 / 250 文件**，不含测试/vendor/bench，见 [LOC](loc.json)。
已清理本批独立消费者目录：[Windows](cleanup-windows.json) 1,804,127,513 字节，
[Linux](cleanup-linux.json) 113,143,016 字节；保留 producer、SDK、源缓存与原始收据。
