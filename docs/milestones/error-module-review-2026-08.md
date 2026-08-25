# error 模块严格复审（2026-08）

> Jira 式施工与证据记录。稳定设计见
> [`../architecture/error-design.md`](../architecture/error-design.md)，用户写法见
> [`../guide/error.md`](../guide/error.md)，FFI 见 [`../guide/c-error.md`](../guide/c-error.md)，
> 点值只见 [`../performance/benchmarks-error.md`](../performance/benchmarks-error.md)。验收方法以
> [`module-closure-methodology`](../architecture/module-closure-methodology.md) 为准。

## 0. 身份与状态

| 项 | 值 |
|---|---|
| 模块 / owner | error / canonical Windows workspace |
| review 类型 | 完整复审；core 保持锁定，只回验编译期宏接缝 |
| 起点 | `2ce66a2e6b347e9ec0ca136bd1a217c5aad23646` |
| 旧证据 | [`error-module-reclosure.md`](./error-module-reclosure.md)，仅作历史输入 |
| 当前状态 | `CANDIDATE` 前施工；本页不得先写绿色 |
| candidate | 尚未冻结；任何产品/公开头/CMake/claim benchmark 改动后重置 |
| CI | GitHub Actions 额度受限；最终若不触发，登记 waiver + 三平台本地真机证据，不能声称持续回归已通过 |

## 1. 审查单元与依赖

| 边界 | 文件/target | 身份与裁决 |
|---|---|---|
| C++ surface | `include/hyle/error/{status,result,format}.h` | 产品；Status/Result/context/native/format |
| implementation | `src/error/{status,c_error}.cc` | product hot/cold implementation + FFI bridge |
| internal bridge | `src/error/status_access.h` | 只供 module FFI shim 最后一层转移 ownership |
| native-C helper | `include/hyle/c/error_helpers.h` | fixture-only source inline；非产品、非动态 FFI |
| direct FFI | `include/hyle/c/error.h` + 6-symbol manifest | managed receiver 产品候选；opaque handle |
| build | `hyle_error` | 零 binary upstream shared/static target |
| compile-time seam | `hyle/macros.h` | core-owned header contract；不引入 `hyle_core` runtime edge |
| direct downstream | obs/platform/time/mem/simd/container/util/sched/io/compute/graph | C++ Status/Result 与各自 FFI bridge |

语法扫描覆盖 include、Status/Result 调用、`status_access`、C header、exports、tests/samples 与 CMake
PUBLIC/PRIVATE edges。语义扫描覆盖 TLS last-error、bool/NULL+诊断、私有 status enum、OS translator、
异常、静默 discard 和正常协议状态。中央 graph 中 error 无 runtime upstream；所有公开返回
`hyle_status` 的下游 targets 已 PUBLIC 依赖 error。

已发现 sched/io 未审 C schema 仍维护 TLS last-error。它们不成为 error 的第二产品面；迁移是各自
closure 的 P1，但在本轮 adoption ledger 中保持显式，不能被“error 已实现”掩盖。

## 2. 真实场景、能力与负空间

| 任务 | ownership / failure | 当前裁决 |
|---|---|---|
| 只返回运营失败 | small copyable value | `Status`；OK/code-only 零分配 |
| 返回值或失败 | value/error sum | standard expected-shaped `Result<T>`，composition error arm 强制非 OK |
| 多层语义传播 | unique rvalue normal path | 扁平文本 + 几何 prefix，累计 O(final bytes) |
| OS/COM 边界 | one canonical + one native detail | errno/Win32/HRESULT 公共 adapter；Win32 W→UTF-8，locale errno 非 UTF-8 时退化为 ASCII |
| backend recovery | one domain/code | fixed `NativeError`；vendor translator 留在所属模块 |
| 清理/聚合/文本 | caller controls policy | `Update`、`IgnoreError`、caller-buffer/opt-in owning format |
| managed 接收下游失败 | NULL 或一个 owner | 6-symbol opaque receiver FFI；SafeHandle/Java owner |

明确不做：任意 payload map、StatusBuilder/stream、自动 source location/backtrace/log、异常层级、
RPC serialization、WinSock domain、managed-side Status construction。重开必须有真实 consumer。

## 3. 成熟实现与算法裁决

| peer / local revision | 审查结果 |
|---|---|
| Abseil `81c993af4fb88b3a51829a796df02251fa27102f` | 借 unique-release load fast path、same-rep assignment/equality、StatusOr 非 OK invariant、errno taxonomy；payload/Builder 不复制 |
| Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` | Expected/value-error shape 与 intrusive ownership 对照；exception_wrapper 不作为 HYLE operational path |
| Boost `df8eeb90846648464e3c8e378e312f0cec772e83` | System/Outcome/LEAF 的 domain、checked failure、typed dynamic detail 对照；当前单 native detail 足够 |
| C++23 `std::expected` | 保留 observers/monadic codegen；以同尺寸 `ResultError` 补 Status 语义不变量 |

实现增量：unique destroy 避免常见 atomic RMW；same-rep assignment/equality shortcut；长 owning
format 使用 `resize_and_overwrite`；Win32 native text 改 W→UTF-8，locale errno 文本增加严格
UTF-8 验证与 ASCII fallback；public WithContext 始终 inline 并调用不同名 slow path；传播宏使用
唯一临时名；moved-from Status 置为 code-only Unknown，`ResultError` 改为同尺寸 composition，保证
移动或普通 API 都不能让 Result error arm 出现“error 但 OK”。

复杂度表以 [`error-design`](../architecture/error-design.md#4-算法与复杂度合同) 为稳定信源。
normal unique rvalue context 无隐藏 O(k²)；反复 lvalue/shared snapshot 的 O(k²) 是显式历史保留成本。

## 4. FFI 破坏性重做

旧 `struct { uintptr_t rep; }` + 30-symbol C++ parity candidate 已删除，不承担兼容。新任务面只有：

- per-artifact bootstrap；
- NULL=OK opaque owning handle；
- code/native/message/to-string borrowing query；
- destroy once；
- fixed-width POD、caller-buffer、显式 `__cdecl`；
- owner lifetime 固定时允许 immutable concurrent query；query 与 destroy/close 竞争非法；
- UTF-8 精确 byte length 与 embedded NUL 保留；截断是可能切开 code point 的 byte prefix，完整解码前必须重试。

code-only error 使用 process-lifetime sentinel，跨 FFI 仍零分配。Java/.NET 正负旅程使用 test-only
producer shared artifact；error 自己不为测试增加生产 factory。incompatible fixture 必须在普通
symbol lookup 前被 bootstrap 拒绝。

## 5. 正确性与交付矩阵

| 门 | Windows | WSL/GCC | macOS/AppleClang |
|---|---|---|---|
| error Release tests/C schema/export/header | 初轮 4/4；candidate 后重跑 | pending | pending |
| all-component affected-edge build | pending | pending | pending |
| ASan/UBSan/TSan | pending | pending | 适用项 pending |
| Java/.NET exact artifact | pending | N/A（最小 native shared call 仍 required） | N/A（最小 native shared call 仍 required） |
| shared/static/source package | pending | pending | pending |

当前初轮 MSVC 只证明实现可编译；managed fixture、文档和 benchmark 尚在变化，不能形成 candidate
证书。

## 6. 性能证据任务

旧 2026-07 点值没有 raw paired JSON、预声明 margin、reverse order、placement shape 和当前 peer
pin，统一降为 historical/`INCONCLUSIVE`。新证书只跑 error：

- code-only runtime/opaque construction，防 constant folding；
- 真实四层 Result success/code-only error；
- message size 覆盖 SSO 两侧与长文本；
- unique/lvalue context 按 contract 分表；
- prvalue normal Update 与 stored-lvalue mechanism 分表；
- caller-buffer/owning format；
- native/payload 不同能力标 `NOT_COMPARABLE`。

采样前预声明 hot non-inferiority margin `m=3%`、`U=1.03`；Windows/WSL/macOS 每格至少 15 独立
process，保存 candidate-first、peer-first 与适用 code-shape raw JSON。任何同合同 SOTA
`FAIL/UNRESOLVED` 或无法解释的 exact LOSE 保持黄色。

## 7. 文档与最终待办

- [x] architecture 改为稳定设计，不再固化旧 raw C parity；
- [x] C++ guide 回答什么情况选什么、成本和坑；
- [x] FFI guide 只从六个真实 export、managed ownership/bootstrap 出发；
- [ ] current performance 页与 raw evidence；
- [ ] Java/.NET samples/runner；
- [ ] 三平台、sanitizer、shared/static/source；
- [ ] frozen-neighbor adoption 与 package affected edges；
- [ ] 十问红队复核、四项 LOC、candidate commit/tree、commit/push。

上述任一项为空，本模块仍是黄色。
