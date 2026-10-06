# C5 更新后的安装 SDK

Windows/Linux 的 installed-static、installed-shared 普通 Direct/Graph 各8项：共64项实际通过。
产品与数值消费者候选为 `46a55325a465a94da3d79d2e5e92b0cb6b2cfb62`，CPU-only ggml0.26，
原 core `ffa4e8b80930029a35991f94e7c8a93cd67730ab`，资源 patch/source 身份未再改变。

- 两系统均重新构建、安装本次已修复的大编码表准入 shim；不是旧 SDK 代签。
- 每组覆盖真实2.16GB Q8_0 embedding 的末端/重复/越界、退役与新 slot 恢复，I32 hash，
  以及 nested NF4、AWQ、CT、MLX affine4/MX4/MX8 六个实际宽度代表。
- 产品只经独立工程 `find_package(Hyle)` 链接。static reference 使用安装导出的原 ggml；
  shared reference 使用同身份原 ggml 静态 archive。reference 不替代产品实现。
- Windows DLL 放各自 exe 同目录；两个入口有独立运行目录，未并发复制到同一位置。
- 沿用 C3 已搬移 prefix 的未改变资产与 offline cache 合同；完整 C4 branch 安装消费者已另有证据。
  本目录没有宣称重新跑64个完整模型或零分配。

[Windows 收据](windows/receipt.json)、[Linux 收据](linux/receipt.json)保存命令、候选、日志和耗时，
四份 XML/平台均8项、无失败和 skip。各平台 `summary.json.remaining` 是归档当时的待办快照；
另一平台现已完成，当前剩余范围以 C5 施工卡为准。

此目录证明更新 SDK 的必要接缝；完整参数/required-region对账及最终收口仍须完成。

归档后核对CMake owner、绝对目录边界及活动进程，删除本批Windows/WSL临时SDK消费者，
释放1,327,652,425 bytes（约1.24GiB）；保留producer、安装SDK、原源码和依赖cache。
清理明细见 `cleanup-windows.json` 与 `cleanup-linux.json`。
