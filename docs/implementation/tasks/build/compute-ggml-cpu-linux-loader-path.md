# C3 Linux ggml CPU 原生程序的装载路径修正

<!-- MODULE REVIEW: AUDITED -->

- owner：`compute_ggml_runtime`；baseline `788f9e80`，原候选 `c96c7c6e`。
- trigger：C3 WSL 完整 CPU source 编译成功，Direct PASS；普通 Graph 在创建 invocation
  时返回 NotFound `ggml CPU AVX2 program`，原 consumer exit12。
- [失败与诊断](../../../../evidence/correctness/compute-unified-c3/wsl-cpu-loader-failure/)
  保留原批次、CTest 和只在 scratch 编译的诊断；诊断不代签普通入口。

## 范围和预算

正常旅程是 source/shared/static CPU Graph 与模型无关数值管线，在具备已锁定完整依赖时
直接运行；用户不应自行为 HYLE 私有 CPU 程序设置 loader 环境。直接消费者是 CPU
Direct/Graph，包括后续 CPU+CUDA、CPU+Vulkan SDK；Windows 和非 AVX2 Apple 域保留原资格。

只定向检查 Linux 原生 CPU 程序取得/装载与运行资产；不改数学、CPU 优选、ggml 版本/ABI、
通用 Platform loader、全局环境、公共配置或治理仪器，不扩大成 Mem/其它模块重审。
P0：冷装载路径修正，submit 成本不变。预算为一次具体 loader 审计、一个有限路径修正、
原 source 两消费者以及原 shared/static 剩余交付；现有 objects/失败预算保留。
实际适用 Linux CPU/mixed 域必须验证，未运行不写 PASS；达到预算仍失败按具体原因裁决。

当前源码事实：CPU 原生 MODULE 与共享 runtime 在同一 build 输出目录，模块绝对路径
`dlopen` 成功。runtime 的实际 ELF RUNPATH 只含 Mem/Platform/Error；仅在诊断进程添加
原 MODULE 目录后，原 Graph consumer exit0。安装 shared 库已有 sibling `$ORIGIN` 合同。
这些证据尚未代签正常默认路径；下一步冻结 target-owned 路径的有限 FIX。

## AUDITED：有限 FIX 与验收

| 能力族 | 当前证据 | 动作与 owner |
|---|---|---|
| 原生 CPU 程序冷装载 | Linux source 缺自身 MODULE 搜索目录；绝对打开和单进程目录覆盖均成功 | FIX，仅 `compute_ggml_runtime` target 的 Linux build RPATH |
| 已安装 CPU 程序取得 | package 已给 shared library sibling `$ORIGIN`，static app 经统一 helper 取得 runtime 文件目录 | KEEP，随本轮 Linux shared/static 原消费者验收 |
| 数学/优选/ABI/完成保活 | 源码未变；原程序可实际运行，故障发生于 invocation 创建、接纳之前 | KEEP，不变更算子、规则、后台机制或 hot path |

固定 FIX：仅在 AVX2 MODULE 已构建且 `CMAKE_SYSTEM_NAME=Linux` 时，向
`hyle_compute_ggml_runtime` 的 `BUILD_RPATH` 追加该真实 target 的输出目录。
不固化源码绝对路径到安装 SDK；不改 INSTALL_RPATH、父工程全局属性或进程环境；
不绕过 AVX2 程序缺失/ABI错误，不退回另一数学实现。

API/usability 与成本冻结：公开 C++、对象布局、资源借用、终态和并发均不变；普通入口无需
额外 loader 配置。只有冷动态装载搜索路径改变，无常态 allocation/copy/atomic 成本。
适用检查：原 Linux source 两消费者（无 LD 覆盖）、MODULE 缺失诊断与恢复、原 Linux
shared/static 两普通消费者和搬迁/资产；CPP/头/算法未改，Windows/Air 与 P1 不重跑。
前序[原 F32 修正](../compute/vulkan-f32-matmul-admission-repair.md)及其实际证据仍保留；
本卡不推翻、不重开那一数学范围，也不以本次 source 修复代签剩余 C3/H1–H3。
