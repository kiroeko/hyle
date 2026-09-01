# Core v4 编译合同投影重签

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由 Core v3 关闭后的真实 delivery 审计定向重开。Core headers、API、算法与布局仍冻结；本页只
> 重签 `hyle::core` 向 C++ consumer 传播的 build-owned compile contract。

## Closure card（`CANDIDATE`）

- component / owner / round / kind：`core` / Core + build-package / `core-v4-build-contract-projection` /
  `REOPEN`；
- baseline：`fe944155e6ada4e68997aaa3fa7d52f27b0769e0`；前轮：`core-v3-foundation`；
- trigger：`CORE-BUILD-CONTRACT-01`。Core-only target 仍传播由 Container 需求产生的 SSSE3 许可和
  MSVC `/EHsc`；这击中 v3 的“delivery shape 或 compiler semantics 改变”重开条件；
- evidence：[`Build/package v8`](../build/build-system-v8-core-contract-isolation.md)；
- scope：保持同一 Core 产品实现，只把公开 build usage requirements 缩回 Core 的真实最小集合；重放
  source/static/shared、编译环境事实、无二进制/FFI 和直接 consumer delivery；
- non-goals：新增或删除 Core capability/API；改变 fatal/cache/compiler/environment 实现；重审
  Container 算法/性能；全库 no-exception 支持；Native FFI；纳秒 benchmark；
- tier：`P0 STRUCTURAL`，timing 不适用；正常路径不得因 CMake target 增加 runtime dispatch、调用、
  metadata 或 ISA requirement；
- machines：`W1-WIN`、`W1-WSL`、`M1-MAC`；instrument：`closure-correctness-v1`；
- required gates：Core-only target graph/compile-options、normal 与 exception-disabled compile、三平台
  Core tests、source/static/shared relocated consumer、无 Core binary/symbol、Container effective
  SSSE3 ownership、文档/易用性、独立 red team；
- stop：上述门全部通过则建立不可变 candidate 并关闭；任一受支持平台不能保持 Core 最小合同或
  Container 等价交付则写出单一 blocker，不扩大为全库构建重写。

## SCOPED decisions

1. Core 的公开 usage requirement 只允许语言级别、头解析/编码和 Core 当前 TU 语义所必需的选项；
   不允许因另一个产品的 inline fast path 传播 ISA flag，也不替父工程选择 exception mode。
2. Container 公开 `probe.h` 的同一 SSSE3 route 保持不变，但编译许可必须由
   `hyle_container` 直接 `PUBLIC` 传播；这是所有权搬迁，不授权改 probe 算法或最低 ISA。
3. exception-disabled 只签 Core-only TU：`HYLE_HAS_EXCEPTIONS == 0`、正常 Core API 可编译运行。
   本轮不宣称其他 HYLE 产品或第三方闭包支持 `-fno-exceptions`。
4. Core 仍是无 binary、无 Native FFI、无第三方依赖的 `INTERFACE` 产品；static/shared 只是统一 SDK
   选择，不能制造 Core artifact。

冻结预算：实现 60 分钟、验证 60 分钟；最多一次实现修正。性能仅作 P0 structural/code-option
裁决，不运行纳秒 benchmark。

## 能力、易用性与反 benchmark 审计

审计反查了当前 HYLE 生产消费者，并以 Highway `63f4b16c`、Abseil `b7adf83d`、Folly
`eeac00030`、ggml `d4716378` 与 llama.cpp `0eadefeb` 的同类开发者任务作形状对账。结论不是按 peer
符号数量求并集，而是确认每个公开入口能回答一个正常 consumer 问题：

| 公开 family | 正常开发者任务 / 真实依据 | AUDITED 决策 |
|---|---|---|
| fatal、debug 与 typed context | 推理引擎和 HYLE 内部 invariant；昂贵开发期检查；失败时给宿主少量结构化上下文 | `KEEP`；成功路径只有冷分支，debug-disabled 不求值，context/observer 只在 failure path 付费 |
| failure observer | 把不可恢复事件同步转给宿主已有 crash/diagnostic 管线 | `KEEP`；显式 install/restore 比暗示跨 DSO 或进程级 RAII ownership 更诚实 |
| compiler contracts | 已证明的 alias、branch、loop、inline、allocation 与 unreachable 前提 | `KEEP`；名称按语义而非 benchmark case，unsupported compiler 安全退化；不增加无 consumer 的 `HOT`、blanket no-sanitize 或自造 nodiscard |
| cache layout / three prefetch intents | 隔离独立写字段；区分复用读、一次读和即将写入的 hint | `KEEP`；三种 intent 与 Abseil/成熟 compiler primitive 同形，具体收益仍由拥有访问分布的 consumer 签署 |
| current-TU environment | 多平台、多 compiler、sanitizer、exceptions/RTTI 条件编译 | `KEEP`；全部是数值预处理 facts，没有 runtime registry/dispatch |
| compile-time version / visibility | header identity 与 build-owned export decoration | `KEEP`；没有 linked getter、Core binary 或伪 Native FFI |

公开 headers 中未发现 `benchmark`、`fast/slow`、`for_test` 或 test-only ABI；测试只验证上述语义、
failure bound 与 optimized code shape，没有反向塑造公共 layout/metadata。当前 HYLE 内部尚未直接使用的
debug context、observer、non-temporal/write prefetch 和少数 annotation，也分别对应成熟库与正常宿主
任务，并且不向常态对象增加字段，因此不能仅按“内部调用次数为零”删除。

易用性结论是 `PASS`：按任务 include `assert.h`、`compiler.h`、`cache.h`、`environment.h` 或
`version.h`，没有巨型 umbrella；最短 invariant 是一个 macro，需要信息时才 opt in message/typed
fields；危险的 assume/restrict/prefetch 前提已在 user guide 就近说明。比较式 CHECK 大全、任意格式化、
backtrace/minidump、runtime CPU/NUMA、assume-aligned wrapper、observer RAII 与 C/Native FFI 分别属于
Error/Obs/Platform/SIMD、标准 C++ 或未来具名宿主需求；没有当前 consumer 证据，不准入本轮。

## 冻结 FIX ledger

| ID | Finding | 有限动作 | 验证 |
|---|---|---|---|
| `CORE4-01` | 全局 build contract 把 Container-only SSSE3 许可泄漏给 Core | 从 `_build_contract` 移除，并由 `hyle_container` 直接 `PUBLIC` 拥有同一 compiler-conditioned option | 中央 target property 正/负合同；Container 三交付 compile |
| `CORE4-02` | 全局 `/EHsc` 替父工程选择异常模式，并覆盖 Core-only no-exception TU | 移除显式 `/EHsc`；增加 `HYLE_HAS_EXCEPTIONS == 0` 的 Core package journey | source/static/shared Core normal + no-exception compile/run |
| `CORE4-03` | 构建所有权和用户例外边界尚未写成当前合同 | 更新 build architecture、Core user guide 与机器合同 | docs/closure checker、package inventory |
| `CORE4-04` | 公共 component contract 在 macOS 生成需要 Platform 的子工程时未传项目规定的 14.4 deployment target，导致中央 gate 在到达 Core 断言前失败 | 只让该脚本的全部 child configure 继承中央 macOS floor；不改产品 target、API 或平台 floor | 三平台 component contract |

Core headers、API、算法、layout 与 runtime 行为的产品修改集为空。任何新 API、compiler spelling、cache
常量、fatal 语义或 Container 算法变化都必须退回 `SCOPED/AUDITED`，不能借本轮顺手实施。当前为
`AUDITED`，只授权以上三项。

## 实施结果（候选冻结前）

- `CORE4-01`：`hyle_build_contract` 已移除 SSSE3；原 compiler-conditioned option 原样迁到
  `hyle_container PUBLIC`。中央合同同时检查全局负空间与 x86 Container 的直接 owner。
- `CORE4-02`：全局 `/EHsc` 已移除；Core package/source journey 新增 no-exception executable，并在
  编译期要求 `HYLE_HAS_EXCEPTIONS == 0`。
- `CORE4-03`：build architecture 与 Core guide 已说明 target ownership、no-exception 边界和
  Container ISA 隔离。
- 冻结前预检：W1-WIN 与 W1-WSL 的 Core tests 各 `6/6`，Core source/static/shared 各 `2/2`，
  Container source/static/shared 各 `1/1`，中央 component contract 均通过。它们是候选建立前的实现
  反馈，不替代对不可变 candidate 的三平台验证。

## Candidate identity

- commit：`350fc39a25adf953c0f13c7404635b4624d5c6f1`；
- tree：`dafe004eb0651ac0fccc500c658b7cd577e52f4f`；
- `AUDITED -> CANDIDATE` 前态 registry：
  `C:\hyle-scratch\core-v4\module-status-audited.json`；
- 候选后允许的改动只有验证 evidence、task/registry/receipt 与不改变候选解释的文档事实。任何 CMake、
  sample、公开/内部 header、Container target 或验证工具语义变化都必须退回 `AUDITED` 并建立新候选。

### Rejected candidate 1

`350fc39a25adf953c0f13c7404635b4624d5c6f1` / `dafe004eb0651ac0fccc500c658b7cd577e52f4f`
的 W1-WIN、W1-WSL Core/Container 三交付、Core tests 与 component contract 通过；M1-MAC 的 Core tests
`6/6`、Core 三交付各 `2/2`、Container 三交付各 `1/1` 也通过。但 Mac component contract 在其
SIMD exact-target child configure 因未传 `CMAKE_OSX_DEPLOYMENT_TARGET=14.4` 失败，尚未执行后续中央
断言。该失败是验证工具的跨平台投影缺陷，不是 Core/Container 产品失败，也不能写成中央 gate PASS。

按 `CANDIDATE -> AUDITED` 退回后，只允许实施 `CORE4-04`，然后建立新 commit/tree 并从头重放；首个
候选保留为 rejected fact，不复用其 terminal verdict。`CANDIDATE -> AUDITED` 前态 registry 位于
`C:\hyle-scratch\core-v4\module-status-candidate.json`。当前为 `AUDITED`。

`CORE4-04` 已实施：component contract 从中央 `HylePlatformRequirements.cmake` 读取现有 14.4 floor，
并把它传给在 Mac host 上生成的每个 child configure。冻结前 W1-WIN/W1-WSL component contract
通过；这仍只是新候选建立前的工具反馈。

### Candidate 2 identity

- commit：`7ae3de400b5d7b598972ebf4913d336b60517483`；
- tree：`b6c04d0e97c93724a9136c34e4ea55b6e4449092`；
- `AUDITED -> CANDIDATE` 前态 registry：
  `C:\hyle-scratch\core-v4\module-status-audited-2.json`；
- candidate 1 的任何结果不自动继承。第二候选从头运行固定 gates；候选后产品、CMake/tool 语义或
  sample 变化必须再次退回 `AUDITED`。

当前为 `CANDIDATE`。
