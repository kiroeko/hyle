# R5 Apple：首个 Direct / Graph 产品检查点

此记录只证明已运行的局部正确性和源码构建，不签 R5 完成、三交付或性能。

- Windows canonical baseline `47751365`，本次 staged product tree 与逐文件 SHA256 见
  [source-manifest.json](source-manifest.json)。同一 manifest 在 Air 验证副本逐文件核对。
- Air `Mac16,12` / Apple M4 / macOS 27.0，AppleClang 21，SDK 27.0，deployment 14.4。
  MLX v0.32.2 Metal-only，CPU/CUDA/Python/文件格式关闭，源码/依赖锁与缓存采用本轮准入。
- Release static producer，`HYLE_DEV_FACTORS=compute_route_apple;graph_route_apple`，
  `HYLE_COMPUTE_APPLE=ON`，`HYLE_WERROR=ON`，Ninja parallel 2。
  [configure](configure.log)、[build](build.log) 通过；MLX 外部头标为 SYSTEM，HYLE 警告门保持。
- [实际 CTest/GTest 原始输出](native-last-test.log)：4/4 CTest，内含 9 个 native GTest。
  两个通用逐头测试检查 OFF 分支；native 测试 TU 首先 include 路线头，实际编译其启用分支。
- 已测：自然 typed MatMul、Q4/group64、SiLU；Q4 独立逐 nibble 解码标量 oracle；
  decode/短 prefill 的完整 Q4 FFN；变化输入、两槽、旧结果、Graph/constant 句柄释放；
  MatMul + 同 runtime custom Metal、只 trace 一次；非法 dtype/shape/group、pending 输入/常量、
  未提供/错误 custom lowering；关闭排空、并发等待、丢弃 completion；原生环境事实。
- Windows Apple OFF 的两个路线头 2/2，配置不查 Metal/MLX；尚不代表全 selected gate 通过。
- 产品 `d900a8ce` 的 [统一实现 LOC](loc-d900a8ce.json) 为 **33,815 行 / 116 文件**，
  含旧实现及 Objective-C++、排除测试/bench/第三方，self-check PASS；不作为完成百分比。

## Metal 编译环境

Air 原先只有 CLT，`xcrun metal` 缺失；完整 MLX 源码构建据此失败。JIT 仍构建少量基础
`mlx.metallib`，因此不能借 JIT 或现成 wheel 绕过源码交付。验证机独立工具目录采用 Apple
官方 CDN 的 Metal Toolchain 17F109：

- [Apple 资产](https://updates.cdn-apple.com/2026MobileAssets/mobileassets/022-22264/722AB304-567A-4611-A7CA-1A94FA951B0B/com_apple_MobileAsset_MetalToolchain/78ED9FD0-8B39-4EEF-B0EE-67013D77BE16.aar)
  SHA256 `52b695754914d150d630fe149cacc76a6d72052d6a62dab1f2b92f3e68267a56`，687865856 bytes。
- 解压的 `metal`、`metallib` 均经 `codesign --verify --strict`，Apple Root CA 签名链与实际工具 hash
  见 [receipt](metal-toolchain-receipt.json)。compiler 为 Apple metal 32023.883。
- [验证环境的窄 xcrun wrapper](xcrun-metal-validation.sh) 仅将 Metal CLI 映射到该原厂工具，
  SDK 仍取系统 CLT。没有改变产品 shader、仓库 vendor 源码或全机 xcode-select 默认值。
  资产中的 toolchain 没有 Info.plist，原 `xcrun --toolchain` 发现失败，故使用此显式环境入口。
- 产品 ON 现在在获取第三方源码前检查 Metal CLI；AUTO 缺工具时不可用，OFF 不做检查。

## 未完成

普通 `hyle::compute/graph` roots、metallib 的源码/安装后邻置与 relocation、shared/native runtime
唯一性、mini、完整故障/OOM/取消/并发风险、host 插桩、文档旅程和不可变候选 P1 仍待后续 A1–A6。
本目录不是路线最终验收或性能胜出证据。
