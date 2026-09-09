# 治理执行性与文档压缩 v4

## Task card

- baseline：`31c9bbac25dd878c2fc791ba70641488e44a57d6`；
- trigger：v3 已统一 terminal 状态，但只读审计证明 CI 尚不能拒绝未经审计直接修改 `CLOSED`
  产品；build/package 历史索引已经出现 `ACTIVE/CLOSED` 冲突；当前人读文档 473 份、70,555 行，
  其中 243 份 task 与 17 份性能说明共 32,012 行不在当前权威或 receipt 证据链中；
- objective：让已有状态机真正约束产品修改，删除没有信息量的 registry 字段和瞬时状态；清除不再
  服务当前合同的历史文档与 legacy 工具，同时保留 26 个 `CLOSED`、2 个 `QUEUED` 组件及其当前
  terminal record、性能证据和 Git 可恢复性；
- scope：closure checker/selector 及 focused tests，registry/status/治理文档，implementation 历史、
  错位 architecture/developer 文档、legacy performance adapter 与说明；
- non-goals：修改 `include/hyle/**`、`src/**`、组件图、CMake 产品合同、benchmark、raw evidence、
  模块 API/算法/状态结论，或运行全库 C++ build、sanitizer、三交付和 P1；删除 factor 测试登记时只
  运行 selector 选出的该 factor correctness 门；
- stop：当前 record/evidence 断链、组件状态或 candidate 漂移、selector 不能机械拒绝未授权产品修改，
  或删除项仍被当前权威/receipt 链需要时停止；否则维护测试和链接检查通过即结束。

## 有限工作集

1. Registry 只保留有信息量的状态、record、candidate commit、性能证据与摘要；删除重复 gate、tree、
   verdict、round/date 和瞬时 `REOPENED`。
2. Selector 对 source/public/factor seam、中央图归属变化和已登记 vendored 产品输入检查 owning factor
   状态；产品字节进入仓库前后 owning factor 都必须已经处于 `AUDITED`，同一提交不得一边完成审计
   一边写实现，也不得在 `CANDIDATE/CLOSED` 后继续偷改产品。
3. 删除 task marker 这一未被采用的第二状态；build/package 作为横向合同随受影响改动验证，独立重构
   使用 project task，不伪装成稳定产品 component。
4. 删除不在当前 authority/receipt 传递链中的历史 task/evidence；Git 历史承担精确恢复。
5. 清理仍在当前入口中的旧状态、过时计划、一次性 debug 日志和 legacy 仪器说明；不压缩真正的用户
   选型指南或当前 terminal record。

## 预声明验证

- `python tests/tooling/check_module_closure_test.py`；
- `python tests/tooling/validation_selector_test.py`；
- 受影响 legacy tool 的 focused tests 或删除引用检查；
- change-impact plan 及其选择的 `container_intrusive` factor correctness；
- `python scripts/check-module-closure.py --previous-ref 31c9bbac`；
- `python scripts/check-docs.py`；
- `git diff --check`；
- diff 反证不含产品源码、组件图、benchmark 与 raw evidence。

## 结果

- Registry 已迁移到 schema 4：组件集合、状态、当前 record、candidate commit、性能证据、margin 与摘要
  相对 baseline 零语义漂移，仍为 26 个 `CLOSED`、2 个 `QUEUED`。删除可从 Git 导出的 tree、重复
  terminal verdict/gates、round/date 和瞬时状态。
- Closure checker 会逐提交验证一组 push 中的状态迁移；selector 会逐产品提交验证 owner 在提交前后
  都是 `AUDITED`。focused fixtures 已覆盖合法完整生命周期、同提交审计加实现、`CLOSED` 偷改、测试/
  文档维护、vendored 输入，以及“未登记工具只允许纯删除”的非对称规则。
- `docs/` 从 455 份、68,550 行压缩到 175 份、30,543 行，净减少 280 份、38,007 行；实际删除
  281 份被替代页面并新增本记录。当前 terminal record、其性能解释、raw receipt 回链、用户指南、长期
  架构和可复用排障记录均保留；精确旧历史仍可由 Git 恢复。
- 删除 8 个已经脱离公共入口的 legacy benchmark/stat adapter，以及相应的旧 parser test/CMake 登记；
  当前通用性能仪器、产品 benchmark、raw evidence 和保留脚本不变。
- 验证：closure focused test、selector 39 项测试、schema 4 对 baseline 的逐提交 transition 检查、
  change-impact `READY`（无 unknown/authorization issue）、`container_intrusive` 4/4、193 份受检
  Markdown/750 个相对链接、Python compile 和 `git diff --check` 全部通过。
- 产品边界反证：`include/`、`src/`、中央组件图、根 CMake、`bench/`、raw `evidence/`、`third_party/`
  相对 baseline 均无改动；因此除测试登记触发的精确 factor 门外，全库 build、sanitizer、三交付和 P1
  按 scope 不适用。本轮横向任务结束。
