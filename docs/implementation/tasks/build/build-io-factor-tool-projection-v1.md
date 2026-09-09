# Build/package：IO 因子工具投影校准

<!-- PROJECT UNIT REVIEW: CLOSED -->

> Build/package 是横向开发单元，不进入产品模块 registry。本任务只校准拆分后的维护工具，
> 不改变 `io_core` 候选产品、API、算法、二进制或性能证据。

## 任务边界

- baseline：`3c4989534c28b08109673be18fde0bee31217dd4`；
- trigger：`io_core` 最终 LOC 收据把已经独立的 HTTP、HTTP/2、TLS 头错误计入核心传输层，
  同时漏计 Buffer、Endpoint、Local 与 Operation；P1 输入策略仍把
  picohttpparser 归给 `io_core`，而其唯一产品 owner 已是 `io_http`；
- scope：只修正 LOC 的五个 IO 因子边界、picohttpparser 的证据 owner，以及对应工具合同；
- non-goals：修改 `include/hyle/**`、`src/**`、CMake 产品图、模块状态、性能合同或重跑 P1；
- cost：纯 Python/JSON 维护工具，无产品运行时成本；
- stop：若校准需要改变产品边界或统计核心算法则 `BLOCKED`，否则合同通过即 `CLOSED`。

## 裁决

- `io_core` 只统计核心 transport/runtime 的公开头和 `src/io_core`；
- `io_tls`、`io_http`、`io_http2`、`io_transfer` 分别拥有自己的公开头与源目录；
- picohttpparser 仅进入依赖 `io_http` 的 P1 输入闭包，不污染核心 transport 的证据身份；
- 同步更新 IO 因子化后已经过时的 selector 反向消费者断言；这只是中央图当前语义的合同投影，
  没有复制第二份依赖图。

## 验证

| Gate | Verdict | Evidence |
|---|---|---|
| LOC boundary | `PASS` | `python tests/tooling/loc_report_test.py` |
| impact/evidence selector | `PASS` | `python tests/tooling/validation_selector_test.py`，32 项通过；新增 parser 正、负归属断言 |
| product isolation | `PASS` | diff 不含 `include/hyle/**`、`src/**`、CMake 产品图或 registry；P1 运行 0 次 |
| docs / closure / diff | `PASS` | 本任务提交前运行公共 checker 与 `git diff --check` |

终态为 `CLOSED`。只有新增、移动 IO 因子，LOC 再次跨因子计数，或 vendored 输入进入错误 P1
闭包时，才新开有限 build/package 工具任务；不得借此重开已冻结产品。
