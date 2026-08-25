# HYLE 维护者文档

本目录帮助维护者下一次正确、快速地完成工作。一次任务做了什么属于
[implementation](../implementation/README.md)；这里保存从任务中提炼出的可重复规则和工具。

- [开发标准](standards/README.md)：工程章程、编码与公开面规则。
- [模块 review](review/README.md)：怎样从需求、功能、设计、性能、易用性和文档重新审一个模块。
- [性能工程](performance/README.md)：benchmark 仪器资格、统计、证据和专项方法。
- [环境与依赖](setup/README.md)：Windows、Linux/WSL、macOS、离线依赖。
- [维护操作](maintenance/README.md)：日常速查、SIMD dispatch 等专门维护说明。
- [调试案例](debugging/README.md)：棘手问题的症状、证据链和可复用诊断方法。
- [故障排查](troubleshooting/README.md)：已知问题、快速定位和恢复路径。

新增文档前先问：它是否能指导下一位维护者重复完成某项工作？若只描述一次任务的状态或结果，
应放 implementation；若在解释产品为什么这样设计，应放 architecture。
