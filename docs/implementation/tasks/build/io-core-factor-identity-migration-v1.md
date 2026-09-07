# IO Core factor 身份迁移

<!-- PROJECT UNIT REVIEW: COMPLETE -->

## Scope

- owner 于 2026-09-08 明确要求内部基础传输因子不得与完整用户产品同名：内部 factor 使用
  `io_core`，用户产品 façade 继续使用 `io`；项目尚未发布，不保留旧 component/target alias；
- baseline：`4ccd6836094ab8974fc737fa891ccdbf40a144d7`；迁移前 registry 快照 SHA-256 为
  `a4eb0dcb280c75893932ca5433a7faca20e542bf8d0cf023eb10638ca25b2acc`；
- public product、namespace 与 include 形状保持 `hyle::io`、`hyle::io` 和 `<hyle/io/...>`；只纠正
  维护者 factor 身份；
- non-goals：改变 Proactor/IoContext/file/network/TLS/HTTP 行为，提前实现其他 IO factor，改变公开
  C++ 语义，新增 Native FFI/C API，运行性能采样或重写 closure/build 工具。

## Decision and finite checks

中央 component ID、真实 target、源码/test/bench owner 与 package detail target 分别成为
`io_core`、`hyle_io_core`、`src/io_core` / `tests/io_core` / `bench/io_core` 与
`hyle::_factor_io_core`。IO 产品 façade 继续叫 `io`，当前只聚合 `io_core`；公开头通过中央
`SPLIT_PUBLIC_HEADERS` 暂由新 factor 拥有。TLS/HTTP/HTTP2 头和实现会在各自 `AUDITED` 轮冻结 owner 后
从该临时 ownership 中原子迁出，本身份任务不替那些能力签署结论。

registry 永久记录一对一 `io -> io_core` migration，并在原 ledger 槽位延续现有 `SCOPED` 状态，
不删除或重置 v1 closure card。验证只包括显式 previous transition、closure/schema/tooling checks、
`io_core` 精确 factor build/test/header，以及 `io` product 的 source/static/shared consumer。产品能力与
性能整改属于当前 v1 模块轮的 `AUDITED` 之后，不混入本身份迁移。

## Result

迁移已完成：

- registry 在原 ledger 槽位保存 `io_core` 并永久登记 `io -> io_core`；使用迁移前 `HEAD` 的显式
  transition check 通过，旧 `HYLE_DEV_FACTORS=io` 配置以 unknown factor 的明确诊断失败；
- 中央 manifest、真实 target、目录、测试/benchmark owner、package dependency owner、validation
  selector、LOC owner、当前架构/维护文档和 export discriminator 已原子迁移；当前合同中不存在旧
  `hyle_io`、`HYLE_BUILD_IO`、`HYLE_IO_BUILD`、`src/io`、`tests/io` 或 `bench/io`；
- W1-WIN 的 `io_core` 精确 factor 配置与 11 个 factor 测试全部通过；用户产品仍由
  `HYLE_COMPONENTS=io` / `hyle::io` 选择，并在独立 scratch 中通过 source embedding、relocated shared
  SDK 与 relocated static SDK 三条真实 consumer；共享/静态内部 artifact 是
  `hyle_io_core`，没有兼容 alias；
- closure、文档、LOC/selector/tooling 与 whitespace 合同通过。身份迁移未改变公共 C++ API、算法或
  性能，也不替 `io_core` 或协议 factors 签署能力结论。

本 build unit 为 **COMPLETE**；`io_core` 产品轮继续保持 `SCOPED`，下一门仍是有界能力审计。
