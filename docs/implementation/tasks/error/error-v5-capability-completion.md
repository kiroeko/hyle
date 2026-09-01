# Error v5 消费者能力补全

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮不是无新事实复查。Error v4 关闭后，全局能力展开与真实 IOCP 消费路径共同暴露了一个已存在的
> 原生错误类别缺口；成熟 peer 与普通系统集成 journey 还暴露了三项需要有限裁决的 API 缺口。本轮只
> 重开这些受影响能力，不重做已经签署的 Status 表示、传播宏或全部性能研究。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`error` / Error / `error-v5-capability-completion` / `REOPEN`；
- baseline：`afdb6b81ab4e8e3ad9d790be87a80dc9242c1cec`；前轮：`error-v4`；
- trigger：`ERROR-CAPABILITY-2026-09-01`；
- `ERR5-R1`：Windows IOCP 实现反复读取 `WSAGetLastError()`，但 Error 只有 POSIX errno、Win32、
  HRESULT 与若干计算后端 domain。现有 IO 代码只能把 WinSock 数值压成无类别负整数；这满足“真实
  consumer 暴露 missing-capability defect”的重开条件；
- `ERR5-R2`：普通 registry/cache/parser journey 需要“成功时非空借用、失败时传播 Error”的形状。
  当前 guide 只能要求用户写 `Result<std::reference_wrapper<T>>`，公开类型没有直接表达 borrow 的
  ownership 与 `value()/operator*()/operator->()` 语义；
- `ERR5-R3`：边界适配器需要把运行时诊断参数附到 Status，同时维持 HYLE 默认不抛异常。当前只有
  literal/string 路径，没有统一的 no-throw 动态格式化入口；
- `ERR5-R4`：标准库与第三方 adapter 普遍交付 `std::error_code`。HYLE 文档要求每个调用者自己比较
  category 并映射，形成重复、容易丢失类别的冷边界代码；
- capability provenance：[`capability-factorization-v1`](../project/capability-factorization-v1.md)
  的 `ERROR-A-04/06/07/08`；真实 WinSock 消费位于 `src/io/proactor.cc`，其下游迁移仍由 IO owner
  在自己的轮次完成；
- 初步受影响面：Error 的 borrowed-result、native-domain/translator、冷路径动态诊断与标准错误适配，
  以及 architecture/user guide、correctness、三种 C++ 交付和定向 P1 证据；
- 初步依赖结论：Error 保持只依赖 Core，不引入 platform、obs、fmt 或其他第三方产品依赖；OS 文本
  查询与 C++ 标准库 adapter 是 Error 自己的边界机制；
- 暂定 tier：`P1`。现有 Status/Result 热路径仍复放 Error v4 的正常 consumer 合同；新的 borrowed
  result 属可能高频 primitive，WinSock、`std::error_code` 与动态诊断属于冷边界并以复杂度、allocation
  和异常遏制为主；
- 暂定 non-goals：异常框架、任意 payload/cause/source-location、MultiError、运行时 domain registry、
  日志、Native FFI/C API、稳定 C++ ABI、IO 模块迁移、相邻模块收口、公共性能仪器改造和 Formal 认证。

当前仅登记 `REOPENED` 新事实；closure card、API freeze 与有限 `FIX` ledger 尚未冻结，不授权产品修改。
