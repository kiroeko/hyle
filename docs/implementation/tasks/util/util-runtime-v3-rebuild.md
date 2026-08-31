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

## Closure card（`SCOPED`）

- component / owner / round ID：`util` / runtime utility owner / `util-runtime-v3`；task kind：`REOPEN`；
- baseline commit：`8eb8926a7b15b879d8c96628b1f2c35c804d27b9`；该提交已包含 Container v2
  迁移后的 Registry 当前实现，后续 reopen/scoped 提交只有任务与状态；
- demand / defect：
  - `UTIL-V3-DEP-01`：Registry 的全部字符串/索引公开合同已从已签 v2 identity 漂移，必须按当前
    `StringView` 语义重查长度、ownership、hash、OOM、复杂度和 delivery；
  - `UTIL-V3-FIND-02`：启动期登记后，普通 backend/plugin consumer 需要按名解析并缓存 factory，
    当前只有 lookup+invoke 或 O(n) 枚举，指南与公开能力不闭环；
  - `UTIL-V3-INFER-03`：必须把当前 llama.cpp/ggml 的 backend registration、reg/device enumeration、
    name/type selection、dynamic load/unload 和 proc-address 逐项映射到 util、platform、compute/module
    owner，不能把“参考过 ggml”替代真实路线；
  - `UTIL-V3-SURFACE-04`：Registry 与 demangle 的全部公开入口、C++ 易用性、Native FFI 清洁度和
    source/static/shared 交付按六问重签；
- normal consumer workload：startup owner 显式登记约 4--32 个内建或已加载 backend factory，检查
  collision/OOM，按名字取得 factory 或直接构造，把解析出的 typed descriptor/object 缓存到服务对象；
  初始化结束后可冻结并枚举帮助信息。日志/backtrace 在正常线程把 C++ symbol 写入 owning string 或
  caller buffer；二者都不进入 token/kernel 内循环；
- scope：`include/hyle/util/registry.h`、`include/hyle/util/demangle.h`、`src/util/demangle.cc`，当前
  `hyle::util` component 的 public/internal 边界，直接 dependency/consumer，C++ API、ownership/
  lifetime/failure/concurrency/复杂度，tests/package journey，source/static/shared delivery，以及
  architecture/user/implementation/performance 文档；
- non-goals：重开 `util_core` 或依赖模块；把 backend/device 类型、capabilities、按类型/最佳设备策略、
  proc-address schema、动态库搜索/评分/卸载/quiescence 下沉到通用 Registry；async-signal-safe parser；
  captured arbitrary callable；模型/token/KV/serving policy；Native FFI/C API；稳定 C++ ABI；P1 timing、
  all-operation SOTA、Formal 证书或下游 compute/graph 收口；
- public/internal/higher-layer initial boundary：Registry 只拥有通用显式名字→factory 元数据、启动期
  mutation 与 immutable publication；platform `DynLib` 只拥有 loader handle；backend descriptor、设备
  集合、选择与插件资源排空由未来 compute/module owner 组合。demangle 的 sink 仍是 shared C++ SDK
  internal seam，不是公开 callback 或 FFI；
- direct dependencies / consumers：`PUBLIC core,error,mem,container`；没有直接第三方产品依赖。
  仓内真实产品 consumer 为 0，selected package sample 是交付 fixture；未来推理引擎是准入 demand，
  llama.cpp/ggml 是真实 domain evidence，不冒充 HYLE 已完成 engine；
- performance tier：`P0`。Registry 登记为 O(key+help)+expected-amortized O(1)，find/create 为
  O(key)+expected O(1)，freeze 为 O(1)/零 allocation，enumeration O(n)；demangle 为
  O(input+output)，允许平台 ABI 的临时 allocation/serialization。不得新增 lookup timing 专用状态，
  不运行纳秒 benchmark，不继承旧 N=256 数字，不声明端到端或跨平台 SOTA；
- resource/layout budget：lookup-only 路线不得 allocation/lock（frozen）或增加每项常态 metadata；
  mutable lookup 可持现有 mutex 但 callback/factory 不在锁内；新增公开入口优先返回现有
  `EntryView` value snapshot，不引入 `std::function`、owning string 或第二张 frozen table；
- total budget：审计 60 分钟、一个产品批次 45 分钟、三平台/三交付/红队 75 分钟；最多一次产品
  批次、一次机制纠正和一次受影响重跑；不得顺手修改 closure checker、构建系统或公共性能仪器；
- machine / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC` / `closure-correctness-v1`；Formal `NONE`；
- pinned strong peers / domain inputs：
  - inference lifecycle：llama.cpp `9723942adc518b43c4b95dc4dce6906903eb5e09` 与 standalone ggml
    `36da57138425487184aa1da2eee2cde155909c6f` 的 backend registry；
  - index lower bound：Abseil `2c004366e983c5be8334ac1ea3d4420e8fbcbea7` flat hash；
  - demangle：Folly `23ebefdf96c116700d3e134f04d95a2b60a509de` 与 Boost
    `78505eab9b203e3e26e83893d0685f160133fa0f`；
- correctness gates：normal、empty/zero、key/help 长度上限、OOM/invalid input、duplicate/priority、
  lookup/create 等价、move-only args、view/owner lifetime、mutable/freeze race、creator/visitor reentry、
  stress、ASan/UBSan/TSan、三编译器/平台 ABI 与 public-header self-containment；同步 utility 的
  cancellation 为语义不适用，不能写虚假 PASS；
- delivery gates：isolated selected `util` build；source embedding、relocated installed static/shared
  C++ package consumer；中央 component graph、真实 target links、headers、Windows import/export 与 Unix
  loader identity；不制造 C/Native FFI journey；
- expected output：capability-family map、完整 public trace、有限 gap ledger/FIX、一个 candidate、三平台
  correctness/sanitizer 与三交付证据、P0 cost structure、六问、文档/红队、限制/reopen triggers、receipt
  与四项 LOC；
- stop condition：全部 `MUST` 为 `COMPLETE/ROUTED` 且所有适用门有真实 verdict则 `CLOSED`；否则在
  预算内以具名 blocker 停为 `BLOCKED`。动态 module lifecycle 只有 owner+完整路线才可写 `ROUTED`，
  不能用“未来再说”掩盖缺口。

`REOPENED -> SCOPED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\util-runtime-v3\module-status-before-scoped.json`，SHA-256
`5396D080BA423700C4E421F8E42E76B01878C95E4A1DE8041E4DF5B09619B483`。

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
