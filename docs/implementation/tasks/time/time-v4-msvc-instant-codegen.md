# Time v4：MSVC `Instant::now()` 代码形状修复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮不是重跑 benchmark 后追逐约 0.5 ns 的点值。Time v3 关闭后，对其冻结的普通 host-phase
> consumer 做最终对象检查，发现支持的 MSVC Release 工具链违反了“默认路线不增加调用壳”的既有
> 结构合同；这是可复现的 compiler/codegen reopen trigger。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`time` / Time /
  `time-v4-msvc-instant-codegen` / `REOPEN`；
- baseline：`03a14a91744697ce6b5c8803b651886afa82cef3`；前轮：
  `time-v3-host-clock-factor`；
- trigger：`TIME-MSVC-INSTANT-CODEGEN-2026-09-02`；
- `TIME4-R1`：冻结的 `time-host-consumer-v1` 在 W1-WIN MSVC 19.50、Release `/O2 /Ob2`
  中执行同一 64-phase 工作：两次 steady-clock 采样、同一饱和 deadline、非负 elapsed、remaining
  与 checksum。候选与直接 chrono 路线不是不同语义的比较；
- `TIME4-R2`：v3 retained bundle 的四个 strata 仍全部通过 5% non-inferiority，候选中位数约慢
  2.0%--2.2%（约 0.55 ns/phase）。这个点值不单独授权产品改动；
- `TIME4-R3`：对象级机制检查复现：直接 chrono phase 已展开到 `_Query_perf_frequency` /
  `_Query_perf_counter`，而 `Instant::now()` 路线每个 phase 仍有两次外联
  `std::chrono::steady_clock::now()` 调用。`/Ob3` 可消除差异，但它是过宽的全局策略；
- `TIME4-R4`：局部实验表明，在 `Instant::now()` 的 MSVC 作用域使用受特性检测保护的
  `[[msvc::flatten]]` 后，两条 phase body 的规范化指令逐项相同。相同处理若下沉到
  `monotonic_now()`，会让普通单次读取比直接 chrono 更激进地展开并造成不必要的代码膨胀，故拒绝；
- 反向范围检查：单次 `monotonic_ns()`、`realtime_ns()` 与 `Instant::elapsed()` 已和各自直接
  chrono consumer 同形；本轮不修改它们；
- 初步受影响面：`Instant::now()` 的 MSVC 默认系统钟代码形状、一个 maintenance-only Release
  codegen probe、现有 Time focused correctness/P1/source-static-shared 验证与事实文档；
- 暂定 tier：`P1`，material margin 仍为 5%。结构 gate 要求冻结 phase 中 HYLE 不比直接路线多
  HYLE/chrono 调用壳；计时只确认正常 consumer 没有回退，不以亚纳秒点值决定 API；
- 暂定非目标：公共 API/layout/语义或能力变化，timer_queue，global `/Ob3`，私有 CRT/QPC
  重实现，公共 compiler macro，build-system 全局开关，Native FFI/C API、仪器重写与 Formal。

当前仅登记足以击穿旧结论的新事实；closure card、完整受影响面与有限 `FIX` 集尚未冻结，
因此不授权 canonical 产品代码修改。

## Closure card（`SCOPED`）

- component / owner / round / kind：`time` / Time /
  `time-v4-msvc-instant-codegen` / `REOPEN`；baseline 与 trigger 同上；
- 真实 consumer / normal workload：沿用已冻结且资格化的 `time-host-consumer-v1`，一个 runtime
  phase batch 执行 64 次 begin/end monotonic sample、饱和 deadline、nonnegative elapsed、remaining
  和 observable checksum；同合同 peer 是同一 translation unit 中直接拼写的
  `std::chrono::steady_clock` 路线；
- 范围：MSVC `/O2 /Ob2` 下无参 `Instant::now()` 的 default system-clock consumer codegen；
  maintenance-only optimized object probe；既有 Time correctness、三交付和 P1 consumer 的定向回放；
- 非目标：公开 API/layout/语义/能力、其他 clock helper、timer_queue、全局 compiler option/macro、
  私有 Windows CRT/QPC 实现、跨模块重构、公共仪器、Native FFI/C API 与 Formal；
- 依赖/消费者：产品依赖仍为 `time -> core`，没有新第三方或 target edge；公开 header 变化会由
  Platform/Sched/IO 的既有 compile journey 和 source/static/shared Time consumer 覆盖；
- tier / margin：`P1`，W1-WIN `closure-perf-v1` 仍使用 5% material margin。结构机制 gate 先于
  计时，要求冻结 consumer 中 HYLE 与 direct route 都到达相同 native clock calls，且 HYLE 不多出
  wrapper call；
- 成本合同：不增加 allocation、lock、branch、dispatch、state、metadata 或额外 clock read；不得以
  全局 `/Ob3` 换取该局部形状，也不得强制普通 `monotonic_ns()` 产生更大的展开 body；
- 调查/实现预算：诊断已经结束；从 `SCOPED` 起最多一个局部产品修改、一个结构 probe、一次 focused
  correctness/delivery 回放和一次冻结 P1 回放，不再扩展 benchmark 姿势；
- machines：W1-WIN 为缺陷与 P1 主机；W1-WSL、M1-MAC 只验证特性保护后的编译/正确性/交付，
  不制造 Windows codegen verdict；
- correctness gates：公开 header self-containment，Instant now/elapsed 与 clock 单调/边界既有 suite，
  MSVC Release object shape；无 allocation/OOM/cancellation/concurrency 新风险，均不因本轮改变；
- delivery gates：Windows source/static/shared consumer、WSL/macOS focused compile/test；API 与 symbol
  surface 负向 diff；
- 预期输出：一个只在支持该 attribute 的 MSVC 上生效的局部 source hint、一个不进入安装产物的
  codegen regression gate，以及同步的 task/performance/current-status 事实；
- 停止条件：有限 `FIX` 完成即形成 candidate；若局部 hint 不能在默认 Release 工具链消除调用壳，
  或引起普通路径膨胀/跨平台失败，则 `BLOCKED`，不升级为全局优化政策。

`SCOPED` 只冻结上述合同；必须完成受影响 public trace、有限 `FIX` 集与 API freeze 后才可修改产品。
