# R4 C5 公开产品交付证据

产品 candidate：`f735271b725d2f3b1d9016dd35a2a9d7b0d43b5e`，tree
`6c9a9836b3659d3f8fccd29dc89ac26ec85b1e54`。完整 Git archive SHA256：
`28d3d95779b5c2199871303a1602deba42edb736f7b8e7fb9cd271a1b014f609`。
Windows 是唯一产品来源；WSL/Air/Mini 均使用该完整 archive。

| 已执行范围 | 实际结果 | 原始证据 |
|---|---|---|
| Windows public source / relocated shared / relocated static | 每种 Direct + Graph 2/2 PASS；默认三个 Provider 的 Q4 gate/up → SwiGLU → dense down，三次变化输入，Graph 要求实际 captured | [Windows 日志/命令](windows-delivery.zip) |
| WSL 同一三交付 | 每种 2/2 PASS；真实 NVIDIA SM89 | [WSL 日志/命令](wsl-delivery.tar.gz) |
| CUDA-only SDK | Win/WSL 两种安装均含实际路线头、六个 CUDA factor、CUDA 13.3 依赖和字节一致的 Marlin LICENSE/NOTICE；无 CPU/oneDNN factor/头 | 上述两个归档中的 sdk-projection.json |
| Windows CPU-only / no-backend | 各 2/2 PASS，并执行 static install；配置禁止查找 CUDA，发布投影不含 CUDA 入口 | Windows isolation 日志 |
| Windows core / backend-only | core 不带 CUDA 发现/发布；显式 backend factor 不带三个默认 Provider、两个 route 或 CPU factor | Windows isolation 日志 |
| Windows component-contract | PASS，既有配置合同完整执行一次 | Windows component-contract.log |
| Air / Mini | 各 CPU-only、no-backend、core 三种 CUDA OFF 配置/发布投影 PASS；不构建或签 CUDA runtime | [Air](air-projection.tar.gz)、[Mini](mini-projection.tar.gz) |
| 八个旧 owner 的 P0 | 相对 d2e631b4 的公共头及全部旧 factor 源目录 Git tree 相等；没有新 CPU 算法、layout 或热路径成本 | [源码投影](p0-unchanged.json) |

普通消费者只链接 `hyle::compute/graph`，不使用维护 factor 或 Provider 排序。Windows 启动前实际
HYLE、CUDA、CRT DLL 邻置，非交互且 timeout 有界；重型串行、构建最多两 jobs，SDK consumer
构建串行。Linux static producer 显式启用 PIC。实际 toolchain/driver/device 在各 cache/日志中。

## 尚未完成与检查夹具纠正

本记录是交付检查点，**不是完整 R4 terminal**。主卡的 CUDA P1、WSL 更完整的 factor 边界回归/
适用 host 插桩，以及中央 seam 自动扩展的 selected gates 仍未完成，不写 PASS。八个旧 owner
仍归 AUDITED 的 [C5 卡](../../../docs/implementation/tasks/build/cuda-r4-product-delivery.md)。

初次 backend-only 检查误把顶层维护选项传给嵌入 parent，按构建合同被拒绝；改为真实顶层 factor
配置后通过，原失败日志保留。Air 的系统 Python 不支持 Path.write_text 的 newline 参数，任务本地
检查器改用 open 后完成；产品/公共工具未因这两项改变，没有重复已通过的 GPU 交付。

另有独立的 [selector 配置冲突](../../../docs/implementation/tasks/build/validation-cuda-cpu-coexistence-v1.md)：
自动合并合法 CPU/CUDA 测试时得到 oneDNN ON + CPU OFF。该工具反例单独修复/验收，不能把自动门
未运行误写成三交付失败，也不能以三交付通过代替其它适用门。

## 身份、规模与继承

[原候选/CPU 性能完整快照](inherited-baseline.json)保留原始测量身份和用户批准的一项例外。
[统一 LOC](loc-f735271b.json) self-check PASS：Compute + Graph **32,662 行**，全库自实现不含测试
127,847 行；不含测试/第三方，不把源码规模或本文计作完成度。原始文件 SHA256 在
[manifest](sha256.json)。
