# util-runtime-v3 重审与重构记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前机械状态只看 [`module-status.json`](../../module-status.json)。本轮由已关闭候选之后发生的
> Container 公开字符串/索引合同替换触发；在进入 `AUDITED` 并冻结有限 `FIX` 集前不得修改产品实现。

## Reopen intake

- component / owner / round：`util` / runtime utility owner / `util-runtime-v3`；task kind：`REOPEN`；
- trigger ID：`UTIL-V3-CONTAINER-PUBLIC-CONTRACT-01`；
- baseline：`8eb8926a7b15b879d8c96628b1f2c35c804d27b9`；
- trigger evidence：v2 candidate `f3b341cb2ae81154f81bc45052a651e773c5e2b3` 之后，Container 删除
  `string_ref`/旧 Swiss surface，Registry 的全部 key/help、hash/index、tests 与公开 include 已迁移到
  `StringView`/`FlatHashMap`。这满足 v2 的“公开合同、算法、布局或直接依赖变化”重开条件；
- affected scope：完整 Registry capability/public surface、Container/Error/Mem/Core 依赖合同、复杂度与
  source/static/shared 交付；demangle 作为同一 runtime component 的未变能力做双向回放；`util_core`
  保持独立、已关闭且不在本轮；
- provisional functional finding：当前指南要求按名解析后缓存 typed factory/handle，但 Registry 没有
  expected-O(1) 的 lookup-only 入口，只有 lookup+invoke 的 `create()` 或 O(n) 枚举。该项由当前
  llama.cpp/ggml 的 backend-reg-by-name 真实旅程支持，必须在 `AUDITED` 前裁决，尚不预先决定 API；
- north-star identities（2026-08-31 remote check）：llama.cpp
  `9723942adc518b43c4b95dc4dce6906903eb5e09`；standalone ggml
  `36da57138425487184aa1da2eee2cde155909c6f`；llama.cpp 随仓 ggml 与前者的 registry shape 对照；
- provisional tier / profiles / instrument：`P0`；`W1-WIN`、`W1-WSL`、`M1-MAC`；
  `closure-correctness-v1`；Formal `NONE`。Registry 是启动/服务发现，demangle 是诊断冷路；没有 profile
  证明二者进入 token/kernel 热路径；
- next：冻结 closure card 进入 `SCOPED`，再形成 capability-family map、完整 public trace、六问与有限
  gap ledger。动态库 discover/load/unload/quiescence、backend/device 领域选择和 proc-address 先作为
  owner-boundary 输入，不自动下沉到通用 Registry。

`CLOSED -> REOPENED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\util-runtime-v3\module-status-before-reopened.json`，SHA-256
`10B756BFFF43C9D8A9BEC6519738A24381AD93B0E19CED66BC2FBACB13C8AC1F`。

## Closure card（进入 `SCOPED` 前冻结）

待下一相邻状态填写；本节为空不授权产品施工。

## Capability-family coverage map

待 `SCOPED` 后审计。

## Capability / gap ledger

待 `SCOPED` 后审计并冻结。

## Owner 六问直答

待审计。

## Candidate

尚无。

## Verification manifest

尚未进入候选验证。

## 独立红队

尚未执行。

## Closure decision

尚未裁决。
