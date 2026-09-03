# Platform fs v1：同步 namespace 与 FileHandle 因子

<!-- MODULE CLOSURE RECORD: ACTIVE -->

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
