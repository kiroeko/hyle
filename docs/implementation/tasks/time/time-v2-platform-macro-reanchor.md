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
