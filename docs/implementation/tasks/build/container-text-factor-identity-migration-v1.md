# Container Text factor 身份迁移

<!-- PROJECT UNIT REVIEW: ACTIVE -->

## Scope and audit

- demand：owner 于 2026-09-07 明确要求把过于宽泛的内部 factor ID `text` 改为
  `container_text`，并同步 roadmap；项目尚未发布，不需要保留旧 component/target alias；
- baseline：`df09e441b94d5e8bfa886806ef47fcb6d8a55e93`；
- 事实：该 factor 的全部公开头位于 `hyle/container/*`，公开类型属于 `hyle::container`，用户只从
  `hyle::container` façade 取得它；内部真实消费者是 `util` 与 `io`。它不是独立 `hyle::text` 产品；
- 问题：现有中央图、target、状态 ID 与 package detail 名为 `text/hyle_text`，和同产品的
  `container_concurrent/container_queue` 不一致，会让维护者误判 owner 与产品边界；
- non-goals：改变公开 C++ header/namespace/API、Unicode/path/string 算法、依赖方向、性能 workload、
  用户 façade、Native FFI/C API、稳定 ABI 或其他 component identity；
- budget：实现与测试 45 分钟；一次迁移，无 compatibility alias，无通用 rename framework 扩张。

## Decision

中央身份改为 `container_text`，真实 target 改为 `hyle_container_text`，源码/test owner 目录改为
`src/container_text` 与 `tests/container_text`。公开头仍是 `hyle/container/{path,string_*}.h`，公开
namespace 仍是 `hyle::container`。`util`、`io` 与 Container façade 的直接边改为
`container_text`；package detail target 随之成为 `_factor_container_text`。

因为 `module-status.json` 是状态真相，本次不能删除旧记录再伪造一个后期新组件。registry 永久保存一条
`component_identity_migrations`：`text -> container_text` 及本任务证据。transition checker 只允许满足
以下全部条件的迁移：

1. 一对一、源/目标合法且不相同，当前 manifest/registry 只有目标、没有源；
2. 目标占据源原来的有序 ledger 位置，不借迁移重排其他 component；
3. 状态按源记录到目标记录继续走合法状态边，不重置历史或跳级；
4. evidence 是 repository 内现存文件；未登记的删除、重命名和重排继续失败。

这保留“身份 ledger 不被静默重写”的原意，同时允许预发布 owner 对错误命名做一次显式、可追溯修正。
历史 task 中描述当时 `text/hyle_text` 的事实不批量改写；当前 architecture、developer、user、status、
CMake/CI/package 与活动 resume task 必须使用新名，并在历史 terminal task 增加迁移说明。

## Finite FIX and verification

| ID | action | gate |
|---|---|---|
| `CTID-01` | closure checker/schema 支持上述显式一对一迁移，补合法/非法 transition tests | tooling test、canonical registry、previous snapshot |
| `CTID-02` | 原子迁移 component、target、目录、visibility macro、provider owner 与所有真实直接边 | all-product/component contract、selected factor/product build |
| `CTID-03` | 更新 roadmap、build architecture、维护说明、当前状态与历史迁移注记 | `check-docs.py`、全仓当前名/旧名分类扫描 |

停止条件：三项完成并由真实 source/static/shared Container consumer 证明新 target/package closure；若发现
公开 API 或产品 namespace 必须改变，停止并回到 Container `AUDITED`，不能在本 build task 顺手修改。

