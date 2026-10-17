# 开发说明

按实际问题实现，运行与改动相关的构建、测试或性能实验，说明结果与未验证项。旧治理流程及其
状态、审批、调查预算和收口规则已删除；历史记录不决定开发权限或下一步工作。

- [环境准备](setup/README.md)：Windows、Linux/WSL、macOS、依赖与离线构建。
- [编码与接口约定](standards/coding-conventions.md)：模块边界、错误、寿命、并发和交付。
- [性能比较](performance/README.md)：公平比较、统计、仪器误差与实际硬件限制。
- [维护操作](maintenance/README.md)：实现速查、SIMD 路由与生成物。
- [调试案例](debugging/README.md)与[故障排查](troubleshooting/README.md)：可复用的诊断方法。
- [维护脚本](../../scripts/README.md)：构建和诊断工具。
- [当前工作主线](current-work.md)：用户确认的工作范围与优先事项。

用户用法放在 [user](../user/README.md)，设计解释放在 [architecture](../architecture/README.md)。
实验报告链接原始数据和复现命令即可，不要求填模板、登记状态或维护重复账本。

[CI](../../.github/workflows/build.yml) 直接运行三系统基础组件构建、CTest、统计解析及 SIMD
配置生成检查；不代表 Compute/Graph 或 GPU 已全部验证。设备测试与性能实验按实际改动运行。
