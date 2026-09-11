# 正确性资格证据

本目录保存跨组件正确性资格审计的事实性摘要。它不创建产品生命周期状态，也不把覆盖率、测试数量
或一次全库运行转换成模块 `CLOSED` 结论。当前状态仍只看
[`module-status.json`](../../module-status.json)。

证据按依赖波次记录：

- [Wave A：Core / Error / Math / Container-intrusive / Util-core / Time](wave-a-2026-09.md)。
- [Wave B：Timer-queue / Platform / Mem](wave-b-2026-09.md)。
- [Wave C：Obs / SIMD / Container / Util-runtime](wave-c-2026-09.md)。
- [Wave D：Sched](wave-d-2026-09.md)。

原始 coverage JSON 和构建日志位于产生它们的隔离 scratch build；仓库只保留可重复命令、输入
commit/tree、摘要和逐风险解释。覆盖率只用于发现没有触达的路径，逐能力 verdict 必须由语义测试、
静态/编译期反证、平台证据或诚实的 `UNPROVEN` / `NOT_APPLICABLE` 给出。
