# R4 CUDA native-helper 采集修正资格

候选 `ea8a7101`，仅 benchmark/validator 接线；没有改动产品算法。六个 Direct/Graph
primary、placement、native-cost targets 在 W1-WIN 用 MSVC / CUDA 13.3、Release/static、
Ninja 并行度 2 构建成功，实际 CUDA/CRT DLL 放在 exe 同目录。

## 已运行

- 从实时机器 probe 派生 `native-runtime-pair`：caller CPU2、process mask `0x14`，
  两颗不同物理 P 核、AboveNormal；SDK helper 保持原生管理。
- 32 次独立 CUDA prefix 全部正常退出；与正式采集相同的进程预算、caller singleton 和 priority。
  prefix 复用此前独立 CUDA 程序，没有链接 HYLE。
- 7 项冷拒绝检查全部通过：Direct/Graph 各自缺失与 FAIL receipt、单 CPU 进程预算、
  缺失 parent mask、错误 priority，均在 GPU 计时前退出 86。
- Direct 与 Graph 各 16 个完整 FFN A/A 进程通过原门，覆盖两种同实现路线、两个 placement、
  正反执行顺序与正反分配顺序；每进程同时测 M1/M16。没有增加样本、调整 5% 因子中位门或
  10% 单进程失稳门。

[原始回归/A/A、构建日志和执行脚本](raw-and-recipes.zip)、[汇总](summary.json)、
[二进制与 DLL 身份](artifacts.json)、[统一实现 LOC](loc.json) 和 [hash](sha256.json)。

## 裁决与边界

Q1–Q3 修正及进度资格 PASS，满足主卡解除采集 BLOCKED 的条件。这不是 P1、默认 winner 或
整个 CUDA 路线的完成证明。有限启动不保证所有 driver/Windows 组合均无故障；此次只签当前 W1。
旧单 CPU 采样继续排除。正式 bundle 必须核对准确 artifact/compiler/DLL/validator/policy 身份；
新鲜构建若身份不同，按原合同取得自己的 A/A receipt，不能借用本 build 的 PASS。

统一脚本统计 Compute + Graph 全部实现（不含测试、第三方）仍为 **32,665 行 / 107 文件**，
self-check PASS。旧实现也在此规模中，LOC 不是本轮完成度。
