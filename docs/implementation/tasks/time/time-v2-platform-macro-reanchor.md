# time-v2：公开头跨平台组合与当前身份重签

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由
> [`Time current-identity replay`](time-current-identity-replay-2026-08.md) 的
> `TIME-REPLAY-WINMAX-01` 触发。它不重新设计 Time；只把已经落地的 Windows 宏可组合性修复绑定到
> 当前 Time 候选，并复核受影响的跨平台、交付与性能证据边。

## Closure card（`SCOPED`）

- component / owner / round ID：`time` / time maintainers / `time-v2`；
- task kind：`REOPEN`；baseline commit：
  `565b710fd6dad1a50588992d336dddb958326d99`；trigger：
  `TIME-REPLAY-WINMAX-01`；
- 进入原因：Platform v1 发现并修复了一个真实的 Time public-header defect：Windows SDK 的
  `max(a,b)` 宏会改写 `Duration::max()`。旧 time-v1 candidate 不含该修复，旧卡却仍宣称
  candidate 后产品/API 零漂移；
- normal consumer：C++23 用户可以在任意正常 include order 下包含 Windows/第三方 OS 头和全部
  Time 公开头；source embedding、installed static 与 installed shared 三种形状都必须成立；
- scope：`clock.h` / `common_timeout_queue.h` 的 macro-safe current identity；Time 自有组合回归；
  public-header self-containment；当前 Windows、Linux/WSL、macOS Time tests 与 selected-component
  source/static/shared relocated consumer；直接组件边；旧 P1 scheduler replay 的当前深回放；
- non-goals：新增或删除 Time 能力；clock/timer 算法、layout、ownership 或 public C++ 语义变化；
  重构 sched/platform；新 benchmark 姿势或新采样；Native FFI/C API；稳定 C++ ABI；Formal；
- direct dependencies / consumers：保持 `PUBLIC core container_intrusive`；代表消费者为 platform、
  sched、io，只验证被本次 public-header/交付变化触及的编译闭包；
- tier / normal performance task：保持 `P1`。唯一已准入 P1 task 仍是
  `scheduler-timer-lifecycle-64`；5% margin、W1-WIN、`closure-perf-v1` 和 pinned Seastar peer 不变；
  public-header token 修复不改变该 workload 的算法、计费、layout 或 codegen，故本轮只深回放已资格化
  bundle 与精确 provenance renewal，不重采样，也不产生新 SOTA/Formal 声明；
- resource / complexity target：所有运行时复杂度和 metadata 为零变化；宏安全写法必须在无
  `NOMINMAX` 的污染 TU 中通过，且 macro-free 预处理语义/Release codegen 不变；
- total budget：90 分钟；静态审计 20 分钟、三平台 correctness/delivery 60 分钟、红队 10 分钟；
  性能采样预算为 0；产品修改最多 0 个，test/doc 修复最多一个批次；
- machine / instrument：W1-WIN、W1-WSL、M1-MAC；correctness/delivery 使用
  `closure-correctness-v1`，工程性能使用 `closure-perf-v1` retained replay；
- strong peer：不重新打开能力差异；宏组合合同来自 Windows SDK/C++ 静态成员调用规则，旧 P1 peer
  仍为 Seastar `fd71a5b4c71b91abd4e93c9d20f88e1fd7465976`；
- correctness gates：污染 `min/max` 的 include-order TU；全部 Time headers self-contained；正常、
  boundary、overflow、timestamp、timer lifecycle 的既有 suite；当前 MSVC/GCC/AppleClang 真平台；
  本轮没有共享状态或算法变化，新增 sanitizer/TSan 工作负载不适用；
- delivery gates：time selected root 的 source embedding、relocated installed static/shared C++
  consumer；中央 `time PUBLIC core container_intrusive`；Windows DLL staging 与 Unix loader-relative
  路径；
- expected output：一个 current product candidate identity、Time-owned macro-composition test、三平台
  correctness/delivery verdict、工程 evidence replay、六问直答、限制/reopen triggers 与四项 LOC；
- stop condition：任何当前平台 compile/package failure 是具名 blocker；若 macro-safe test、三平台
  suite/交付和工程 evidence replay 全通过，且 product diff 仍限于已审修复，则 `CLOSED`。不因没有
  新采样而继续研究。

## 有限审计与冻结 FIX

| ID | demand / evidence | coverage | action | verification / negative space |
|---|---|---|---|---|
| `T2-ID-01` | `TIME-REPLAY-WINMAX-01`；公开头在 Windows macro-polluted TU 的真实编译缺陷 | 当前产品已修，但 time-v1 identity 未重签 | `FIX` evidence/identity | Time-owned polluted-TU test；candidate diff 与 macro-free codegen 结构核对 |
| `T2-TEST-01` | 修复目前只由 platform test 间接守护，time-only 根不会编译该 TU | `PARTIAL` | `FIX` test ownership | Time test 同时包含全部四个公开头；不要求用户定义 `NOMINMAX` |
| `T2-DELIVERY-01` | public source header 与安装头均受 include order 影响 | `PARTIAL` until replay | `FIX` evidence | 三平台 source/static/shared relocated consumer + public headers |
| `T2-PERF-01` | scheduler provenance 整文件漂移，产品算法/charged phases 不变 | `ROUTED` to independent instrument task | `KEEP` product / replay evidence | `QUALIFIED + REVIEWED_NON_MATERIAL`；Formal 与新性能声明均不续签 |
| `T2-DOC-01` | time-v1 仍写 candidate 后 product/API 无 drift | `PARTIAL` | `FIX` current record only | v1 保持历史原文；本轮 task、registry、performance freshness 给出当前解释 |

冻结 FIX 仅为：新增 Time 自有 macro-composition test、完成当前三平台/交付/证据回放、更新本轮记录与
机械状态。产品实现修改集为空；发现算法/API/能力缺口必须另行回到 `AUDITED`，不能顺手扩面。

## `AUDITED` 结果与 owner 六问

candidate→baseline 的 Time 产品 diff 已逐项枚举：只有 `clock.h` 和 `common_timeout_queue.h` 的十处
`Duration::max()` token 变为 `(Duration::max)()`；`src/time`、全部 public declarations、layout、
ownership、算法、组件边和正常 workload 均为零变化。全部 Time public header 的 `min/max` 调用扫描
确认其他数值边界已经采用 parenthesized static-member 写法。v1 的 capability-family map 与 public-
surface trace 因而仍完整适用；本轮受影响族只有 public-header composition / delivery，覆盖从
`PARTIAL` 进入候选前的 `COMPLETE`。

1. **第三方产品路径：**Time 产品没有直接或间接第三方实现；直接产品依赖只有 HYLE `core` 与
   `container_intrusive`。标准库提供 chrono/value mechanism，OS API/compiler builtin 提供各平台 clock/
   sleep/counter；它们不是第三方模块。GoogleTest 只属于 test，Seastar 只属于 P1 peer，不进入产品
   artifact 或 public type。升级触发点仍是标准库/OS/compiler 改变实际时钟、宏组合或 codegen 合同。
2. **推理引擎功能：**量化推理的 monotonic deadline、OS-thread wait、CPU/profile accounting、Unix
   timestamp 与 caller-owned timer container 均为 `COMPLETE`；fiber/executor/periodic/cross-thread
   routing 完整归 sched/io，calendar/timezone/text protocol 归标准/协议 owner。当前证据下没有未处置
   `MUST`；本轮不按 ggml API 求并集，也不承诺未来不增长。
3. **拟真性能：**P1 task 仍只包含 W1-WIN 的 scheduler timer lifecycle source replay、pinned Seastar
   peer 与 5% NI。算法/layout/charged phases 未变；当前 verdict 要等 retained bundle 深回放后填写。
   不允许外推为全部 clock/sleep、全平台或 Formal SOTA。
4. **旧 C ABI / Native FFI：**产品树没有公开 C header/schema、bridge、未装饰 C export、C/managed
   sample、专用 option/target/package metadata或 FFI-only layout；内部 OS C API/asm 仍是 private
   mechanism。shared library 只有 C++ decorated Time symbols。预验证 verdict：`CLEAN`。
5. **构建与导入：**中央图保持 `time PUBLIC core container_intrusive`，Time-owned headers 与 target
   owner 一致；source/static/shared 的当前跨平台 consumer verdict 在 verification 后填写，不能以
   producer build 代替。
6. **C++ API 与文档：**构造、typed/raw clock、checked timestamp、deadline/sleep、queue lifecycle 和
   teardown 形状均未变化；宏污染 TU 不再要求用户知道 internal seam 或 `NOMINMAX`。architecture/user
   guide 的选型与陷阱仍对应当前 API；time-v1 的历史“candidate 后无漂移”由本轮记录纠正，不回写
   历史 task。最终 verdict 在三平台 journey 通过后填写。

## Candidate

- product commit / tree：
  `565b710fd6dad1a50588992d336dddb958326d99` /
  `4534070226c16b9ba37aa7ecb189b9f447eb6892`；
- baseline 与 candidate 相同：当前产品已包含 Platform v1 落地的 macro-safe 修复，本轮没有再次修改
  `include/hyle/time` 或 `src/time`；candidate 只重新绑定该 current identity；
- public contract / layout / algorithm / resources：零变化；十处 parenthesized static-member call 在
  未定义 `max` 宏时具有相同 C++ 语义和 Release codegen，在污染 TU 中从 compile failure 变为正常；
- test artifact：`4a438c9fa0def99d0baba50faed0b53e5ada4daa` 增加 Time-owned
  all-public-header macro-composition TU，不属于产品 candidate；
- direct consumer / component graph：保持 platform/sched/io 与
  `time PUBLIC core container_intrusive`；
- harness identity：retained P1 bundle 由已提交的 provenance-renewal verifier 深回放，和 product
  commit/tree 分开记录。
