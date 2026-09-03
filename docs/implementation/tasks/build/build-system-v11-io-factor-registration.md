# Build/package v11：IO 因子身份登记

<!-- PROJECT UNIT REVIEW: CLOSED -->

> Build/package 是横向开发单元，不进入 `module-status.json`。本卡只建立 IO 分阶段施工所需的机器身份，
> 不授权移动、修改或验证 IO 产品实现。当前状态仍只由
> [`module-status.json`](../../module-status.json) 决定。

## Closure card

- unit / owner / round / kind：`build-package` / build-package maintainers /
  `build-v11-io-factor-registration` / `REOPEN`；
- baseline：`a93a2b9293b7f2a4b93add0880d4e1a722d9f191`；
- trigger：IO 当前把 completion/file/net/IPC、TLS、HTTP/1、HTTP/2 与完整传输 workflow 聚在一个
  约 7,926 行实现单元中。已接受的 Stage C 能力审计证明这些族具有不同 lifecycle、可选依赖、测试和
  并行施工边界；模块方法又要求新 factor 在任何产品代码修改前先登记 ID 并进入 `QUEUED`；
- real consumer：以后以 `hyle::io` 使用完整产品的源码/static/shared C++ 消费者，以及分别维护 IO
  transport、TLS、HTTP 和模型传输的 HYLE 开发者；
- scope：保留 `io_tls/io_http/io_http2/io_transfer` 四个内部 factor ID；让中央 manifest 明确区分
  “已登记”与“已有 target”；保持普通 product `all` 和 maintainer all-factor 可构建；拒绝显式选择
  尚无 target 的保留项；同步 registry、状态投影、架构和 IO 施工计划；校准整套 component contract
  中仍按旧 Container 依赖图要求完整 Platform/Mem façade与 `time/util_core` factor 的过时断言；
- non-goals：拆分 `src/io`、修改公开 API/算法/性能、声明新依赖、创建空 target/测试、把新因子加入
  `hyle::io` façade、改变 TLS/H2 feature 语义、重开已关闭模块、修改 Compute/Graph；
- tier / runtime performance：P0 structural。登记与选择仍是 configure-time `O(V+E)`；没有产品
  source、target、dispatch、metadata 或运行时路径变化；
- budget / stop：一个 manifest schema slice、一个文档/registry slice、配置正负合同各一次。若现有
  产品配置或 all-factor 维护路径必须选择保留项，停止并回退，不为登记制造假 target；
- required gates：previous-registry transition check、status projection、product IO/all configure、
  maintainer all-factor configure、reserved-factor negative configure、docs、diff whitespace；
- state：`CLOSED`。身份登记、选择隔离和现有组件合同均已验证，本轮不进入 IO 产品实现。

## 决策与冻结 FIX

现有机器 ID `io` 将承载语义上的 IO core（completion/file/net/IPC）；不为了概念命名迁移已经存在的
状态身份。新增四项先只是 identity reservation：

| ID | eventual owner | 本轮机器合同 |
|---|---|---|
| `io_tls` | TLS transport/config/session lifecycle | `QUEUED`; no target/deps/root |
| `io_http` | HTTP/1 framing/client-server semantics | `QUEUED`; no target/deps/root |
| `io_http2` | HTTP/2 connection/stream semantics | `QUEUED`; no target/deps/root |
| `io_transfer` | bounded artifact/model-transfer workflow | `QUEUED`; no target/deps/root |

| ID | frozen action | expected evidence |
|---|---|---|
| `BUILD-V11-ID-01` | manifest 与 registry 以相同拓扑顺序登记四项 `QUEUED` ID | closure checker + status table |
| `BUILD-V11-SELECT-02` | `HYLE_KNOWN_COMPONENTS` 表示登记全集；`HYLE_IMPLEMENTED_COMPONENTS` 表示有 target 的集合 | maintainer all-factor configure 不含 reservation |
| `BUILD-V11-FAIL-03` | reservation 禁止 target metadata、依赖和 product root；显式选择 fail closed | negative configure message |
| `BUILD-V11-PRODUCT-04` | `hyle::io` 暂时仍只投影现有 `io` 根，普通 all/product 行为不变 | IO/all configure diagnostics |
| `BUILD-V11-DOC-05` | 架构明确区分 25 个登记身份、21 个实现 factor 与 14 个产品 | docs checker |
| `BUILD-V11-FIXTURE-06` | source-embedded Container fixture 从中央现树要求 `core/error/container` façade与实际 7-factor 闭包；不再要求不完整的 Platform/Mem façade | full component contract |

本轮不会把 Stage C 候选依赖直接抄入当前图。每个新 factor 的 `AUDITED` 卡必须根据届时真实源码、公开
类型和已经存在的底层 factor 独立冻结直接边；如证明底层存在具体能力缺口，才形成具名 reopen trigger。

## Reference snapshot

本卡只用 reference checkout 校准施工边界，不把它们变成产品依赖：llama.cpp
`0df974d777c904dda1da3b00faa7769c6310ae74`（随仓 ggml
`59fbdbe5b1d989f9226ca9189256ed4ebcfa01dc`）、standalone ggml
`d4716378882593333721eb33f153144b6885caf2`、Asio
`8806a6803cde7054c3049d3666d3ec36786568c5`、Seastar
`9b95b250cc64af11b84fa5ea3d677f76b37fdef0`，均为 2026-09-03 的干净本地快照。

## Verification 与裁决

| gate | current result | evidence |
|---|---|---|
| registry transition / projection | `PASS` | `check-module-closure.py --previous C:\hyle-scratch\io-factor-registration-pre-a93a2b92\docs\implementation\module-status.json`：25 components；状态投影一致 |
| IO and all-product configure | `PASS` | WSL GCC 14.2 fresh Ninja trees；IO-only 与 14-product feature-OFF 均配置成功，resolved factor 中没有 reservation |
| maintainer all-factor selection | `PASS` | `HYLE_DEV_FACTORS=all` 配置成功并只解析 21 个 implemented factor |
| reserved-factor negative contract | `PASS` | `HYLE_DEV_FACTORS=io_tls` 按预期失败并报告 registered/QUEUED/no target；同规则写入 full component contract |
| full component contract | `PASS` | WSL GCC 14.2：`HyleCiComponentContract.cmake` 全部通过；同时校准 stale Container fixture |
| docs / whitespace | `PASS` | `check-docs.py` 检查 390 Markdown / 1,071 links；`git diff --check` 通过 |

首个 WSL preflight 使用系统默认 GCC 13.3 时由既有 GCC 14+ 版本门在生成前正确拒绝；上述证据全部改用
显式 `CC=gcc-14 CXX=g++-14` 的合格环境。没有启动 Windows 原生测试可执行文件。

当前工作树 LOC 口径：全库代码 `436,895`；自实现含测试 `111,204`；自实现不含测试 `70,416`；
IO 实现 `7,926`。本轮未修改后三项所覆盖的产品实现；全库增量来自 CMake 合同与任务文档。

终态裁决：`CLOSED`。本轮只完成可逆的身份登记和 fail-closed 选择机制，没有创建空模块或让计划能力
进入用户产品。下一合法动作是为现有 `io`（语义 IO core）建立自己的 closure card 并由 `QUEUED`
进入 `SCOPED`；四个新 ID 各自在轮到时独立进入 `SCOPED/AUDITED`。只有 reservation 泄漏进产品、
all-factor 再次选择无 target ID，或状态/manifest 集合漂移，才重开本 build/package task。
