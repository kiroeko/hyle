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
