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
