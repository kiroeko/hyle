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

## Capability / affected-surface audit（`AUDITED`）

| Family / route | natural consumer | baseline code shape | coverage | action |
|---|---|---|---|---|
| `Instant::now()` default system clock | runtime/inference phase begin/end | MSVC phase 比 direct chrono 多两次外联 `steady_clock::now()` 壳 | `PARTIAL` | `FIX` |
| `Instant::now(source)` | deterministic/selected clock | 一次 `source.now()`，typed route inline、ref route 显式间接 | `COMPLETE` | `KEEP` |
| `Instant::elapsed[_ns]()` | ordinary elapsed convenience | 单次 HYLE 与 direct chrono probe 指令同形 | `COMPLETE` | `KEEP` |
| `monotonic_now/ns` | raw deadline/runtime clock | 单次 HYLE 与 direct chrono probe 同形；在 phase 中可由调用者自然展开 | `COMPLETE` | `KEEP` |
| realtime/coarse/profiling/CPU clocks | timestamp/diagnosis | 不经过 `Instant::now()`；单次 realtime 对照同形 | `COMPLETE` | `KEEP` |
| timer queue/provider/value conversion/sleep | 已冻结 Time/timer factors | 无受影响调用边、布局或语义 | `COMPLETE` | `KEEP` |

### Public trace、依赖与交付影响

- 唯一待改公开声明是 header-inline `Instant::now()` 的函数体；签名、exception contract、返回布局、
  overload resolution、clock domain 和 observable value 不变；
- `Instant::elapsed()` 也调用 `monotonic_now()`，但同合同独立 probe 已证明它与 direct chrono 都保留
  同一个标准库调用，因此没有 HYLE 额外壳。对它加 flatten 会改变而不是恢复 peer code shape；
- 将 attribute 放进 `monotonic_now()` 的实验虽能修复 phase，却使单次 raw read 强制展开为较大 body，
  与 direct chrono 单次调用不同；拒绝这个更宽的方案；
- `realtime_now/ns` 在 MSVC 已自然到达 `_Xtime_get_ticks`，不受本缺陷影响；
- 无产品 target、component graph、第三方依赖、installed header inventory 或 symbol surface 变化。
  只在 `tests/time` 注册一个 maintenance-only object target 和 MSVC optimized audit；
- source/static/shared 用户都在自己的 translation unit 实例化这个 inline body，因此必须使用匹配的
  header/toolchain；现有三交付 consumer 足以验证可用性，Windows object probe 裁决具体 codegen。

### 冻结有限 `FIX` 集

1. `TIME4-CODEGEN-01`：仅在 MSVC 且 `HYLE_HAS_CPP_ATTRIBUTE(msvc::flatten)` 为真时，把
   `[[msvc::flatten]]` 作用于 `Instant::now()` 的 return block；其他编译器保持原代码；
2. `TIME4-REGRESSION-02`：增加不安装、不链接进产品的 phase probe；MSVC Release audit 必须证明
   HYLE/direct 两个函数均无 HYLE/`steady_clock::now` wrapper edge，且规范化指令序列相同；
3. `TIME4-DOC-03`：更新本轮 task、Time 性能事实、registry/status；API/guide 因用户形状未变无需改写。

没有其他 `FIX`；新功能、其他 helper 的强制展开、全局 compiler policy 或新的计时姿势均越界。

## API / usability freeze

- 状态：`UNCHANGED/FROZEN`；公开名称、签名、include、layout、语义和错误模型全部不变；
- C++ 消费者仍自然写 `const auto begin = hyle::time::Instant::now();`；不暴露 MSVC attribute、
  fast/light 变体、宏或 backend 选择；
- C ABI / Native FFI：仍为 `N/A/CLEAN`，本轮不创建跨语言 surface；
- source/static/shared：该函数是 header-inline C++ API，三种交付都使用相同 header contract；
- 陷阱仍是 v3 已记录的 clock-domain、ManualClock relaxed-value 和 borrowed ref lifetime；本修复不引入
  新用户可见陷阱。

## Owner 六问（`AUDITED` 直答）

1. **能力：**v3 能力账本不变且没有新缺口；本轮只修已有 `Instant::now()` 的实现质量。
2. **第三方/依赖：**无第三方产品依赖变化；标准库 chrono 是机制，Time 仍只依赖 Core。
3. **性能：**同合同 consumer 已确认，基线仍在 5% 门内但存在明确额外调用机制；修复后以 codegen
   equality 为主证据、冻结 P1 回放为非回退证据，不追求固定亚纳秒胜负。
4. **C ABI/交付：**无伪 C ABI；不改变静态/动态/source 产品图，只新增 test-only object/audit。
5. **C++ 易用性：**调用形状完全不变，且无需用户传宏或选择 fast API。
6. **文档：**architecture/user guide 无产品决策变化；只更新可审计事实和性能 limitation。

审计结论：只有 `TIME4-CODEGEN-01..03` 获准进入实现；接口与 workload 均冻结。

## Candidate

- product commit / tree：`a9b9ff3117ab647581a6b846318a8254a6cd148f` /
  `6b5026fa1bbc0e916a0f81fa707ed6f8bbe75301`；
- `TIME4-CODEGEN-01`：`Instant::now()` 仅在 MSVC 且 attribute 可用时对局部 return block 使用
  `[[msvc::flatten]]`；没有改动 `monotonic_now()`、其他 clock helper、API、layout 或 target options；
- `TIME4-REGRESSION-02`：test-only 64-phase object probe 与共享的成熟 COFF parser 已接入 Time
  focused target；不进入安装或产品链接；
- W1-WIN MSVC 19.50 Release/static、`HYLE_WERROR=ON`：Time header/runtime/codegen tests 3/3；
  HYLE/direct chrono phase 均为 143 条规范化指令，逐项相同，且没有 HYLE 或
  `steady_clock::now()` wrapper edge；
- pending：冻结 P1 consumer 一次 qualification、W1-WSL/M1-MAC compile/correctness、Windows
  source/shared/static delivery、最终 docs/red-team/aggregate gates。未运行项不写 `PASS`。

`AUDITED -> CANDIDATE` 的前态 registry 快照位于
`C:\hyle-scratch\time-v4-msvc-instant-codegen\module-status-before-candidate.json`，SHA-256
`3344E69E501BFDEC2F0FF55DDFF8D0B78BDA7D0EB74F92BC13C878FC90FA91E8`。candidate 后若改变
产品实现、公开合同或冻结 workload，必须退回 `AUDITED`；只补验证与事实记录不改变 candidate identity。

## Verification manifest

全部 verdict 绑定 immutable product candidate
`a9b9ff3117ab647581a6b846318a8254a6cd148f` / tree
`6b5026fa1bbc0e916a0f81fa707ed6f8bbe75301`。

| Gate / risk | Verdict | Evidence / limit |
|---|---|---|
| normal/header correctness | `PASS` | W1-WIN MSVC 19.50 与 19.44 Release/WERROR 各 3/3；W1-WSL GCC 14.2 与 M1-MAC AppleClang 21 各 2/2；非 MSVC 均真实编译 test-only probe |
| MSVC optimized code shape | `PASS` | 19.50 与 19.44 `/O2 /Ob2`：Instant/direct 64-phase 均 143 条规范化指令、逐项相同，无 HYLE/chrono wrapper call edge |
| P1 engineering performance | `PASS` | 原 `time-host-consumer-v1` contract/hash 未变；W1-WIN clean n=15 bundle `QUALIFIED`，4/4 strata 5% NI `PASS`；exact 为 3 `PARITY` + 1 `INCONCLUSIVE`，不签微观点值胜出 |
| source/shared/static delivery | `PASS` | W1-WIN selected Time source consumer、relocated shared SDK consumer、relocated static SDK consumer 各 1/1；tests-off artifact 不含 probe/audit |
| dependency/build impact | `PASS` | 产品 diff 只有 header-inline body；`time -> core` 与产品 factor graph、installed header inventory、target options 均不变；新增 target 仅位于 tests/time |
| API/layout/symbol/FFI | `PASS / N/A` | 签名、overload、layout、语义不变；Windows DLL 仍为原 12 个 decorated C++ exports，无 C/Native FFI surface |
| docs/usability | `PASS` | 用户调用与陷阱不变，故 user/architecture 无伪改写；performance 页记录 v4 结构机制、NI verdict 与 limitation，tool README 说明回归入口 |

P1 retained bundle 位于 `C:\hyle-scratch\time-v4-perf-evidence\manifest.json`；contract、manifest、
summary、controls、observations SHA-256 分别为
`707029C26EB677DCECFC1478CCA207D6A2A5D64B305EB04C43391F14D0CB158E`、
`4207D5E656E34C8017B3944CE7B908CC5D09D44EA06BBE4225F54A34227E125E`、
`901508FC76615C5AD7471C8DB77793D49053DFC4622CECF7189A244EB1A24DC8`、
`7B45ECD630A554A20CA15C184376743DAF34B73ADFAC819F137E47C13F589319`、
`078EE44C7472D61FB25220917309FC079CBFD31A576C904D81A58DF9AA638F24`。deep verifier 返回
`QUALIFIED`、`paths_verified=true`、`provenance=EXACT`、`environment=CURRENT`。

## Independent red team

| Attack | Result / disposition |
|---|---|
| 把约 0.55 ns 点值当产品需求 | 拒绝；授权来自同合同对象中额外 call-shell 机制，P1 只签 5% NI，不创建 API |
| 在 `monotonic_now()` 中统一 flatten | 实验会让单次 raw read 比 direct chrono 更激进地展开并增大 body；拒绝，改动只落在已复现的 `Instant::now()` |
| 顺手处理 `elapsed/realtime/deadline` | 独立 probe 已证明 elapsed、raw monotonic、realtime 与 direct route 同形；无缺陷，不改 |
| 用全局 `/Ob3` 或公共 macro | 拒绝；没有产品 target option 或 Core/build-system 改动，局部 attribute 不泄露给用户 |
| 只在最新 MSVC 偶然成立 | MSVC 19.50 与支持范围内 19.44 的默认 Release gate 都通过；feature test 让其他/未知编译器安全保留原语义 |
| test seam 进入 SDK 或改变组件图 | tests-off source/shared/static smoke 全通过；probe 是未安装 OBJECT target，脚本不是产品依赖 |
| exact instruction gate 诱导未来盲改 | gate 只绑定这一支持编译器的冻结同源 consumer；未来 compiler/stdlib 变化时要求检查两边机制，不能自动授权 API/layout 修改 |

红队未发现公开合同、correctness、lifetime、dependency、delivery 或未处置性能机制缺陷；没有理由扩大
`TIME4-CODEGEN-01..03`。
