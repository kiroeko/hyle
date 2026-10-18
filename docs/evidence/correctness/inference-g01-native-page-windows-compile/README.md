# Native page Windows 定向公共链接

冻结产品 `520027a57`，复用原 C:/h7/inference-g01-native-vulkan source/shared 树，
四 jobs，215.016秒。真实 Vulkan route/provider/runtime 与 compute_batch 新页入口
编译/链接成功；普通 consumer仅链接公开 hyle::compute，静态断言 move-only slot，并
链接 recipe/version/table、create/prepare/bind/submit/wait/report/close/drain 公共调用。
batch_paged_native.h、vulkan_batch.h、共同 controller 头另独立三个 TU 通过 /W4 /WX。

这里只签 compile/link/逐头，未运行 executable、GPU/math 或宿主风险；实际 Air 数学与
SDK/宿主证据另有阶段，不能冒称 Windows/NVIDIA 数值与性能资格。raw.zip 保存原配置/
构建日志、consumer、原 parent、实际四 TU 的编译命令和执行脚本；receipt绑定九个
canonical 与 candidate 的精确 SHA、命令/退出码/elapsed。产品候选没有修改。

原依赖配置中 IREE source authentication 及锁定 ggml shader generation/库内重编成本
也计入215.016秒，没有因等待或新日志重置。只启动具名 public target，不构建全库、
不跑本机重型 GPU，也未借此改公共构建工具。Vulkan不可变P1和完整G01/联合消费仍待。
