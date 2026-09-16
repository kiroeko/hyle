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
| 中央 seam 的 selected gates | 106 项全部 PASS，按精确身份复用前序证据，保留 skipped reasons | [最终汇总](selected-summary.json)、[原始命令/receipts/日志](selected-raw.zip) |

普通消费者只链接 `hyle::compute/graph`，不使用维护 factor 或 Provider 排序。Windows 启动前实际
HYLE、CUDA、CRT DLL 邻置，非交互且 timeout 有界；重型串行、构建最多两 jobs，SDK consumer
构建串行。Linux static producer 显式启用 PIC。实际 toolchain/driver/device 在各 cache/日志中。

## 尚未完成与检查夹具纠正

本记录签署 **C5 交付范围 CLOSED**，不是完整 R4 terminal。主卡的 CUDA P1、WSL 更完整的 factor
边界回归/适用 host 插桩及 cuBLAS OOM 尾项仍未完成，不写 PASS。八个旧 owner 已按
[C5 卡](../../../docs/implementation/tasks/build/cuda-r4-product-delivery.md) 完成交付收口；六个 CUDA
owner 继续保持主卡 AUDITED。

初次 backend-only 检查误把顶层维护选项传给嵌入 parent，按构建合同被拒绝；改为真实顶层 factor
配置后通过，原失败日志保留。Air 的系统 Python 不支持 Path.write_text 的 newline 参数，任务本地
检查器改用 open 后完成；产品/公共工具未因这两项改变，没有重复已通过的 GPU 交付。

另有独立的 [selector 配置冲突](../../../docs/implementation/tasks/build/validation-cuda-cpu-coexistence-v1.md)：
自动合并合法 CPU/CUDA 测试时得到 oneDNN ON + CPU OFF。该工具反例单独修复/验收，不能把自动门
未运行误写成三交付失败，也不能以三交付通过代替其它适用门。

上述工具修复及实际 broad configure 已通过。首次 component-contract 的 MSBuild tracking file
遇到长路径失败，原日志保留；短路径接续只执行未通过项。上游 /MP 未被 /m:2 完全约束，发现后
限定该验证进程族的亲和性，后续未完成交付在同提交 detached Windows 副本中使用 Ninja -j2。
迁移时主动终止刚开始的 Graph 检查，最终实际完成它；没有重跑前序 95 个 PASS gate，没有 P1。
完整修正过程与最终 106 个 PASS receipts 都包含在 selected-raw.zip。

## 身份、规模与继承

[原候选/CPU 性能完整快照](inherited-baseline.json)保留原始测量身份和用户批准的一项例外。
[统一 LOC](loc-f735271b.json) self-check PASS：Compute + Graph **32,662 行**，全库自实现不含测试
127,847 行；不含测试/第三方，不把源码规模或本文计作完成度。原始文件 SHA256 在
[manifest](sha256.json)。
最终 selected 归档的哈希单列于 [selected manifest](selected-sha256.json)。
