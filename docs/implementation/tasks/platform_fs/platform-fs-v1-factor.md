# Platform fs v1：同步 namespace 与 FileHandle 因子

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

## Closure card（`SCOPED`）

- component / owner / round / kind：`platform_fs` / Platform / `platform-fs-v1-factor` / `NEW`；
  baseline `f0662cdee82dd72d5813d17a4765f0fea0f16d87`；进入原因是已接受的
  `B-F09 SPLIT_PRIME`；
- 真实消费者：model/cache loader、IO async adapter、配置和 backend discovery；北极星是最新
  llama.cpp/ggml 的同步 model read、cache/temp/atomic replace 与 directory discovery；
- 范围：UTF-8 synchronous namespace query/mutation、directory visitor、安全 caller-owned tree removal、
  one-cursor FileHandle short/exact/all/seek/sync、cwd/temp/home、secure temp create 与 atomic replacement
  所需机制；
- 边界：lexical Path/Text 归后续 Container/Text，mapping/residency/shared storage 归 Mem VM，positional/
  vectored/direct/async/watch/transfer 归 IO；不把 OVERLAPPED/proactor 或 mapping state 塞入一字 handle；
- 预期依赖：`PUBLIC core,error,time`，secure temp 允许 `PRIVATE platform` entropy mechanism；普通用户仍
  只选择 `hyle::platform`；
- tier / workload：`P1`。primary 是 warmed regular model file 的 open + complete sequential read + close，
  对 direct OS/Folly File 同语义路线，W1-WIN 5% non-inferiority；holdout 是 temp write/sync/rename
  lifecycle，只作 consumer budget，避免把 durability 与非 durability 混比；
- 常态成本：FileHandle 一字、无 heap/refcount/lock；read/write 至多一次 native call，exact/all 为
  O(bytes/native-limit)；visitor 常态逐项零分配，递归删除辅助空间 O(depth + max_path)；buffer-fill
  查询不分配，owning convenience 显式支付；
- peers：Folly `eeab4246827eae9a11c0fa22529bff1a258aa83a` File、Boost
  `cb94f33fcb13e90922905c2d8105508e71fa6db4` Filesystem；OS filesystem ABI 是 correctness authority；
- correctness：normal、empty/zero、EOF/short transfer、large offset/overflow、missing/permission/OOM/system
  failure、UTF-8/symlink/race threat model、move/release/close/destruction、recursive depth/width、property/
  stress、ASan/UBSan；单 owner cursor 的并发 success 为不适用；
- delivery：exact factor、Platform product source/static/shared + relocation、三平台、Mem/IO direct
  consumers、header isolation、export/import 与旧 FFI 反证；
- 总预算与仪器共享 Platform v3 六小时/九十分钟上限；最多一次性能机制诊断和两次产品修正；
- 非目标与 registry 一致；完整 capability map/public trace/有限 `FIX` 和 API freeze 前不得迁移源码。

当前状态仅为 `SCOPED`。

## Audit verdict（`AUDITED`）

正向证据覆盖 llama.cpp model/config/cache 的同步 open/read/stat/temp/replace，peer 为 Folly File 与
Boost Filesystem；反向追踪覆盖 `fs.h/fs.cc`、混合 convenience、MappedFile 调用方、package sample
与 IO/Compute/Mem 消费。现有 FileHandle、namespace、visitor、安全递归删除、temp 与 rename 组合已
形成完整 portable subset；本轮没有缺失文件操作需要新增。冻结票只有：

| FIX | frozen shape | cost / boundary | verification |
|---|---|---|---|
| `PLAT3-F01` | `platform_fs` 独立拥有同步 fs；environment 与 CPU owning helper 从混合 convenience 拆开；Windows temp 使用 host secure entropy | FileHandle 一字；普通 read/write 至多一次 syscall；不携带 mapping/async/path owner | exact factor/header isolation/direct IO-Mem edge |
| `PLAT3-F02` | warmed regular file 的 open + complete sequential read + close 同合同 P1；temp write/sync/rename 为 holdout budget | 计入 setup/bytes/teardown，不与 cached scalar 或无 durability route 混比 | W1-WIN 5% gate + A/A qualification |
| `PLAT3-D01` | guide/sample/architecture/performance 与 source/static/shared relocation | 无 product cost | docs/scans/red team |

API freeze 为 `FROZEN`（2026-09-03）：公开头自含、FileHandle/namespace/temp/replace normal journeys 与
Windows/WSL 精确 factor 已通过；P1 不能反向改变一字 handle、把 setup/teardown 移出 normal workload，
或增加 benchmark-only 入口。MappedFile/SharedMemory 已由依法审计的 `mem_vm` 唯一拥有。file lock/
preallocate/watch/direct/positional/vectored/async 与 lexical Path 保持明确 `DEFER/ROUTED`。产品无第三方
实现依赖、无 Native FFI；公开 namespace 仍是 `hyle::platform`，factor 名只服务内部独立构建。

## 候选前实现检查点

- F01--F02 与 D01 的 source/header/convenience 分离、精确 target、测试、sample 和文档已经落地；
- W1-WIN `51/51`、GCC 14 `-Werror` `58/58`，公开头自含通过；GCC 隔离构建还发现并修复了私有测试
  seam 错用 host visibility macro 的真实 factor 缺陷；
- 在该检查点 M1-MAC、sanitizer、Platform product 三交付、Mem/IO direct consumers、P1 与红队尚未
  裁决；后续进度以紧接的 Candidate 节为准。

## Candidate（`CANDIDATE`）

product commit/tree 是 `e9b9a43d4782c60d84badc02b1a41fbe21207a00` /
`c6429316279aaac1e0f5d7a7a71872a60cf967b6`。F01--F02/D01、公开 API/一字 handle 与 warmed
open-read-close normal workload 已冻结；W1-WIN、GCC 14 `-Werror`、ASan/UBSan、公开头自含、Windows
exact factor 及 Platform 三交付已通过。M1-MAC、Mem/IO direct consumers、正式 P1 与最终红队仍待签署。

首个共同 candidate 因 private Platform host entropy 的 Apple `getentropy` 声明缺失退回 `AUDITED`；
本 factor 没有 API/算法改动，修复 host 后重放验证。

## 第二候选（`CANDIDATE`）

共同 product commit/tree 是 `b1a526d6fdb05b6bb42c4cb041788959e74511de` /
`f808df6fcbd0b5549777e0c6939240a3feab9512`。M1-MAC 的公开头与实现测试已随五组件共同重放通过；
F01--F02/D01、API、一字 handle 与 warmed open-read-close workload 未改变。sanitizer、Mem/IO direct
consumers、正式 P1 与最终红队仍待裁决。

## 最终验证与六问（`VERIFIED`）

本 factor 与 Platform v3 共用 immutable product candidate
`b1a526d6fdb05b6bb42c4cb041788959e74511de` / tree
`f808df6fcbd0b5549777e0c6939240a3feab9512`，完整矩阵、红队与前态 snapshot 见
[Platform v3 主记录](../platform/platform-v3-capability-factorization.md)。W1-WIN 实现/公开头、W1-WSL
Release/ASan/UBSan/TSan、M1-MAC Release/ASan、Platform 三交付、Mem 与 IO 直接消费者、动态符号/旧
FFI 扫描均 `PASS`；mapping/shared-storage 公开 owner 已从本 factor 清除。

[Platform P1 报告](../../evidence/performance/benchmarks-platform.md)中的 warmed 4 MiB
open/read/checksum/close 行 8/8 strata 为 `PARITY` 且 5% NI `PASS`，median ratio
0.996192--1.010228，最低 confidence lower 0.966165。合同完整计费 setup/bytes/teardown；不与 cached
scalar、mapping 或无 durability 的不同语义混比。结构门确认 FileHandle 一字、short I/O 至多一次 native
operation、exact/all 为 O(bytes/native limit)、visitor 常态不收集 owning list。

Owner 六问：产品只用 C++23/compiler/OS ABI，Folly/Boost 只是需求与 peer；当前 portable synchronous
namespace/FileHandle/temp/replace 能力完整，lexical path 与 async/direct/vectored/watch 有明确 owner；没有
公开 C/Native FFI 或稳定 C++ ABI；factor 精确构建及 façade 三交付全部通过；普通用户只选择
`hyle::platform`，guide 明确 UTF-8、short transfer、单 cursor、symlink/race 与 durability 边界。F01--F02/
D01 全部 `COMPLETE`，七项 aggregate gate 全为 `PASS`，无 blocker；当前进入 `VERIFIED`。

## Closure decision 与 terminal receipt

本 factor **`CLOSED`**：F01--F02/D01、七项 aggregate gate 与所有适用验证均通过。已知限制是同步
one-cursor owner、调用者负责并发和 symlink/race threat model，且 P1 只覆盖 W1-WIN warm regular-file
完整读取；lexical Path、watch、positional/vectored/direct/async I/O 与 mapping storage 有各自 owner，
没有 Native FFI 或稳定 C++ ABI 承诺。

只有真实 consumer 的 fs correctness/security/ownership/易用性缺陷、支持 OS 击穿 UTF-8/offset/short
transfer/durability 合同、公开 API/layout/dependency/delivery 改变、material 正常负载回归，或正式准入
一个应由同步 fs 拥有的共同 mechanism 才能重开。

immutable candidate 的四项 LOC 为 `434,612 / 110,971 / 70,261 / 2,061`，由
`loc-report.py --module platform_fs --commit b1a526d6fdb05b6bb42c4cb041788959e74511de --self-check`
重算通过。`2,061` 是 factor-owned `src/platform_fs`；共享公开头计入 Platform conventional boundary。
前态 snapshot/hash 与 Platform 主 receipt 相同；后续 registry 绑定本 terminal receipt 的真实提交与首个
`CLOSED` anchor。
