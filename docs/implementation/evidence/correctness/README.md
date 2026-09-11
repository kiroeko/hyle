# 正确性资格证据

本目录保存跨组件正确性资格审计的事实性摘要。它不创建产品生命周期状态，也不把覆盖率、测试数量
或一次全库运行转换成模块 `CLOSED` 结论。当前状态仍只看
[`module-status.json`](../../module-status.json)。

Wave A–E 与 public-journey 页面是各波结束时的证据快照；其中“留给 Qxx”或
`UNPROVEN(Qxx)` 描述的是当时的停止点，不是另一份永续状态。后续语义组合由下一波和
[public journeys](public-journeys-2026-09.md) 闭合，终局平台/交付解析统一看
[Q09 平台、Sanitizer 与交付资格](platform-delivery-2026-09.md)；不带后续批次标记的 `UNPROVEN`，
以及 Q09 页仍列为 `UNPROVEN` 的路线，才是本次资格结束后尚未执行的边界。

证据按依赖波次记录：

- [Wave A：Core / Error / Math / Container-intrusive / Util-core / Time](wave-a-2026-09.md)。
- [Wave B：Timer-queue / Platform / Mem](wave-b-2026-09.md)。
- [Wave C：Obs / SIMD / Container / Util-runtime](wave-c-2026-09.md)。
- [Wave D：Sched](wave-d-2026-09.md)。
- [Wave E：IO / Process](wave-e-2026-09.md)。
- [跨模块公开 API 旅程](public-journeys-2026-09.md)：数据准备、异步运行时、模型文件、服务连接与工作进程。
- [平台、Sanitizer 与交付资格](platform-delivery-2026-09.md)：Windows/macOS/Linux backend、
  ASan/UBSan/TSan、source/static/shared 与 hosted CI 边界。
- [终局裁决](final-adjudication-2026-09.md)：范围、新鲜度、伪证明与剩余 `UNPROVEN` 的红队结论。

原始 coverage JSON 和构建日志位于产生它们的隔离 scratch build；仓库只保留可重复命令、输入
commit/tree、摘要和逐风险解释。覆盖率只用于发现没有触达的路径，逐能力 verdict 必须由语义测试、
静态/编译期反证、平台证据或诚实的 `UNPROVEN` / `NOT_APPLICABLE` 给出。
