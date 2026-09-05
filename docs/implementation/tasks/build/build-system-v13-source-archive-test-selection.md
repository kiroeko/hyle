# Build/package v13：源码归档测试选择

<!-- PROJECT UNIT REVIEW: ACTIVE -->

> Build/package 是横向开发单元，不进入 `module-status.json`。本轮由 Platform v7 的真实
> Git-less 源码归档验证触发，不改变 Platform 的产品范围或替它签发状态。

## Closure card

- unit / owner / round / kind：`build-package` / build-package maintainers /
  `build-v13-source-archive-test-selection` / `REOPEN`；
- baseline：`4be4b7e0d95f053d57a55e12d50d4c7f4064a76e`；
- trigger：`BUILD-SOURCE-ARCHIVE-GIT-ONLY-TEST-2026-09-05`。受支持的完整源码交付不携带
  `.git`，但 `HYLE_BUILD_TESTS=ON` 仍无条件注册维护者 LOC 测试；该测试调用 Git 解析
  `HEAD`，使产品/模块测试全部成功的合法归档得到伪失败；
- real consumers：拿到完整授权源码归档并在无 Git metadata 环境中运行模块测试的客户、
  以及用归档做三平台候选验证的 HYLE 维护者；
- scope：仅在 Git 可执行程序和 checkout metadata 同时存在时注册
  `hyle_loc_report_test`；Git-less 源码仍注册并运行其余适用产品、模块、文档与 closure
  合同；补充源码交付文档和真实归档回归；
- non-goals：修改 LOC 统计定义、伪造 Git identity、关闭其他测试、改变产品 target/API、
  修改模块状态或重写通用 closure/performance 仪器；
- tier / cost：`P0` configure-time selection；新增一次常量条件判断，没有产品二进制或
  运行时成本；
- finite FIX：`BUILD-V13-SELECT-01` 正确限定 Git-only test；
  `BUILD-V13-DOC-02` 写清源码归档的测试边界；`BUILD-V13-ARCHIVE-03` 在真实无 `.git`
  归档中证明 LOC test 未注册且其余所选测试通过；
- required gates：Git checkout 中 LOC test 仍注册并通过；Git-less archive 中不注册它；
  Platform 精确 factor 的其余 tests；source/shared/static Platform consumer；docs/closure/diff；
- budget / stop：一个测试选择条件、一段文档和一轮归档验证。若必须修改 LOC 统计器或
  屏蔽任何非 Git-only 测试才能通过，则 `BLOCKED`。

## 候选

尚未生成。

## Verification 与裁决

尚未裁决。
