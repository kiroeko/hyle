# SIMD v7：Compute 驱动的普通 f32 row primitive

## Closure card

- component / owner：`simd` / SIMD；round=`simd-v7-compute-row-primitives`；
- baseline commit：`df54eef5352019e25ea1694fc5fec17583ae9d87`；上一 terminal candidate 为
  `e72089c196f6b38f897524087bc53201a0fd8df4`；
- trigger 与证据：Compute Native 的 18/19 executable checkpoint 已形成真实推理 consumer：
  [`elementwise.cc`](../../../../src/compute_provider_native/elementwise.cc) 对 dense f32 `Rsqrt`、`Minimum`、
  `Maximum` 和 `Select` 仍逐元素 scalar；[`reduction.cc`](../../../../src/compute_provider_native/reduction.cc)
  不能复用现有 extrema，因为当前 `simd::min_value/max_value` 对 `+0/-0` 不满足 Compute 已冻结的
  NaN propagation + signed-zero 合同。以上是缺失的通用连续 row 机制，不是 tensor/layout 或 benchmark
  名称；
- 正常 consumer 任务：对已验证的连续 f32 rows 执行 reciprocal-sqrt、IEEE `fmin/fmax` 风格逐 lane
  选择、byte-bool 条件选择和确定 extrema reduction；普通调用一次 pointer/count entry，fixed/fat
  Compute kernel 可在 target TU 内直接组合；
- 主产品纵切 / 具名来源：量化 decoder 的 elementwise、normalization/attention 前后处理；直接 consumer
  是 `compute_provider_native`；
- 正交反偏置纵切：批处理数值管线对连续 f32 arrays 执行相同四类操作，证明 API 不携带 Tensor、axis、
  broadcast、模型或 Graph policy；
- 跨模块 public journey 与每段 owner：SIMD 拥有连续 row 数值语义与 ISA route；Compute 拥有 dtype/layout/
  broadcast/alias validation、row partition 和 provider execution；Platform 继续提供 CPU facts；
- scope：只审计并交付 f32 reciprocal-sqrt、f32 pairwise `fmin/fmax`、byte-bool f32 select、以及现有 f32
  extrema reduction 的 signed-zero 合同；迁移 Compute dense row route 并保留 scalar strided/broadcast
  fallback；
- non-goals：重跑 v6 的 422-op 迁移、增加 f64/narrow/tensor broadcast API、模型 fused op、线程池、改变
  runtime decision authority、按微基准建立 HYLE/xsimd 例外，或修改未受影响 SIMD family；
- 直接依赖 / 直接消费者：依赖不变，仍为 Core + Platform 与锁定 Highway/xsimd build-only mechanism；
  直接消费者为 Compute Native，SIMD 不反向依赖 Compute；
- 性能 tier 与原因：`P1`。P0 先证明 O(n)/O(1)、零 allocation/lock、一次 ordinary dispatch 和 target body
  无逐元素分派；P1 只运行一个 Compute-real dense-row group，与同语义 scalar oracle及 direct locked-
  Highway body核对，固定 5% material non-inferiority margin；
- 适用机器 / 仪器 / material margin：W1-WIN 做普通/fixed codegen、correctness 与有界 P1；Linux 与两台
  Apple 做 target correctness/交付；5%；
- 调查与纠正预算：Highway/xsimd 各一次源码合同审计；公开 API/语义一轮；实现至多两轮；P1 一次完整
  qualification，若 preflight 不合格不靠重复采样找结论；
- 预声明验证：zero/null、tails、exact alias、NaN/inf/subnormal、`+0/-0`、非 0 bool bytes、scalar/
  selected differential、Compute dense/fallback differential、逐头、三平台、sanitizers、source/static/shared、
  selected target/codegen、文档、红队与有界 P1；
- stop / blocker 条件：若 Highway/xsimd 不能在不增加第二次 dispatch/临时 buffer 的情况下表达精确语义，
  保留最小 HYLE semantic adapter；若 API 必须吸收 tensor/broadcast policy，停止并把能力留在 Compute；
  material P1 失败最多纠正两轮，之后如实 `BLOCKED`。

## 能力与形状审计

| 能力族 | 层级 | 准入旅程/来源 | 语义与边界 | 当前路线 | 覆盖状态 | 动作 |
|---|---|---|---|---|---|---|
| full-precision f32 reciprocal sqrt row | 原子机制 | Compute dense `Unary(kRsqrt)`；普通数值数组 | `out[i] = 1.0F / sqrt(x[i])`；不得把 ISA approximate-rsqrt 冒充 exact；完整输入/输出可原位 | 现有 SIMD 只有 `sqrt`，Compute scalar loop | `MISSING` | `FIX` |
| f32 pairwise number min/max row | 原子机制 | Compute dense `Binary(kMinimum/kMaximum)`；普通数值数组 | `std::fmin/std::fmax` 风格：单侧 NaN 返回数值，两侧 NaN 返回 NaN；零值分别稳定为 `-0/+0`；完整输入可原位 | Compute scalar loop；Highway `MinNumber/MaxNumber` 未独立承诺 signed zero | `MISSING` | `FIX` |
| byte-bool → f32 value select row | 原子机制 | Compute dense `Select`；批处理数值管线 | 任意非零 byte 为 true；只允许输出完整别名任一 value 输入，其他 overlap 不在合同内 | Compute 逐元素 branch/copy；Highway 可窄载入、同 lane promote 后 mask-select | `MISSING` | `FIX` |
| propagating floating extrema | 原子机制 | Compute dense Reduce；现有公开 search API | 任一 NaN 传播；min/max 的零结果分别保留 `-0/+0`；index 继续把两种零按 IEEE `==` 视为 tie 并返回第一项；空区间 identity 不变 | 当前 Highway `Min/Max/Reduce*` value 路线丢失 signed zero；f32/f64 共用实现 | `PARTIAL` | `FIX` |
| tensor/layout/broadcast/row partition | 应用策略 | Compute provider | SIMD 不吸收 Tensor、axis、broadcast 或 provider policy | Compute 已拥有 | `ROUTED` | `KEEP` |

公开入口冻结为 `reciprocal_sqrt(float*)`、`fmin/fmax(float*)` 和
`select_values(uint8_t*, float*)` 三组 pointer/count overload；名称直接表达数值语义，避免把 pairwise
number operation 与 `min_value/max_value` propagating reduction 混淆。新 row 输出可完整别名任一同型 value
输入，partial overlap 不支持；`select_values` 的 condition 不得与 float ranges 重叠。`n == 0` 接受 null。
不增加 approximate 档、Tensor overload、临时 workspace 或常态 metadata。

### Provider capability map

- locked Highway `18e449b642599b8f51c03ca10173fdc59864e4f2` 提供 portable `Sqrt/Div`、
  `MinNumber/MaxNumber`、窄整数 promote 与 mask select，覆盖 scalar、SSE4、AVX2、AVX-512、NEON、SVE、
  RVV；HYLE 只需 signed-zero semantic adapter；
- Highway `ApproximateReciprocalSqrt` 以及 locked xsimd
  `3b3151ff665514a00a437cc985f47b7de99b772b` 的 `rsqrt` 在 x86/NEON/SVE/RVV 直接使用估算指令，
  不满足 Compute 的 exact precision policy；本轮拒绝该路线；
- xsimd `fmin/fmax` 公开入口委托普通 target `min/max`，其测试未锁定本轮 NaN + signed-zero 全合同；
  它也没有比现有 Highway target TU 更自然的 byte-bool pointer/count 路线。因此三项均由 locked Highway
  lowering 承载，不新增 xsimd 例外或第二次路由；
- `hyle_add_dispatched_kernel(HIGHWAY ...)` 已为 ops/search 生成七档 target TU；Runtime table 是普通
  pointer/count API 的唯一一次 process-cached 间接调用，exact artifact 继续直接绑定单档 symbol。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| S7-F1 | dense f32 rsqrt 仍逐元素 scalar | 在 ops public/runtime/target table 增加 full-precision f32 row；Highway `Div(1,Sqrt(x))` + scalar tail | zero/null、tail、alias、特殊值、逐 runtime differential；Compute dense route 直接复用 |
| S7-F2 | dense f32 min/max 仍逐元素 scalar | 增加 `fmin/fmax` row，以 Highway number op 加最小 signed-zero adapter | 单/双 NaN、inf、subnormal、四种 `+0/-0` 顺序、alias、逐 runtime differential |
| S7-F3 | dense byte-bool f32 select 仍逐元素 branch | 增加 `select_values` row；窄 condition load/promote 后一次 mask select | 0/1/其他非零 byte、tail、true/false exact alias、逐 runtime differential；非 f32 Compute route保持原样 |
| S7-F4 | propagating extrema value 的零符号与 Compute 合同不一致 | 修正既有 f32/f64 shared extrema value/minmax 实现；index 保持 first numeric tie，避免无需求的热路径成本 | signed-zero permutation、first numeric-tie index、NaN precedence、逐 runtime differential |
| S7-F5 | 普通消费者与交付尚未回放 | 更新 SIMD/Compute guide 与 provider ledger；验证 source/static/shared、exact/fat codegen 和三平台 | 无第二次 dispatch、无临时 buffer/heap/lock；Compute dense 与 fallback differential 均通过 |

本表已冻结。未列出的 SIMD family、approximate API、f64 新 row API、Tensor API、线程池和 winner route
不在本轮授权内；新事实要求改变这些边界时退回 `SCOPED/AUDITED`。

### API / usability freeze

- 普通 C++：包含 `hyle/simd/ops.h` 后对一个已验证连续 row 调用一次 pointer/count entry；Tensor 用户继续
  通过 Compute，让 provider 负责 dtype/layout/broadcast 检查；
- semantics：`reciprocal_sqrt` 只有 full-precision 档；`fmin/fmax` 是 number semantics，
  `min_value/max_value` 仍是任一 NaN 传播的 reduction；`select_values` 接受非零 byte；
- lifetime/failure：全为同步 `noexcept` 借用范围，调用返回前完成；无 owner、OOM、cancel 或 teardown；
- steady cost：O(n) 时间、O(1) stack，零 allocation/atomic/lock/syscall，普通构建一次 whole-row function
  pointer hop，exact 构建直接 call；
- 性能限制：只有连续 f32 row 使用这些入口；strided/broadcast Tensor 仍走 Compute scalar fallback，文档不得
  宣称全部 Select/Reduce 都被加速；
- 交付：入口属于既有 `hyle_simd` target 与 export macro，不创建新 component、开关或 vendor runtime
  dependency；source/static/shared 使用同一 public header。

## 实现与候选前检查

S7-F1--F4 已按冻结合同实现：ops 的七档 Highway target body 增加 exact reciprocal-sqrt、number min/max
和 byte-bool select，普通入口仍只有一次 whole-row route；Compute Native 的 dense f32 elementwise 与
all/last-axis extrema reduction 已直接复用，非连续、broadcast 和非 f32 fallback 保持原 owner。search
value/minmax 的 signed-zero 修复没有让普通非零 reduction 在每个 vector block 维护符号 mask：只有最终
数值结果为零时才执行一次 O(n) 符号补扫；index 路径完全不变。

候选前 W1-WIN/MSVC 19.50 回放结果：fat SIMD + Native provider 定向 `3/3 PASS`；独立 exact-AVX2 SIMD
route/codegen/order 套件 `5/5 PASS`。逐 runtime 测试覆盖 19/37 个元素的完整 vector 与 tail、exact alias、
NaN/Inf/subnormal、四种 zero 顺序、非 0 condition byte 和 public zero/null。P0 确认为 O(n) 时间、O(1)
stack、零 allocation/lock/syscall/临时 mask buffer；fixed/exact 是直接 symbol call，fat 仍为一次缓存的
whole-row function-pointer hop。

P1 合同为 [`simd-compute-row-p1-v1.json`](../../../../bench/contracts/simd-compute-row-p1-v1.json)，lock
SHA-256 `deb10651fcb559cc4a08eddb0da8f675af19ea10da569eb268541032d87ebdf1`；语义矩阵 SHA-256 为
`8cfdbb926fd2b6ec1287eaff1596787875fa87b4ae9c7c884db10e65e51705b6`。候选前只运行了 primary 两顺序
smoke：四项 correctness oracle 均通过，但 select 等点值随执行顺序反转，因此没有从 smoke 产生性能
verdict，也没有据此改变产品；正式 P1 只允许在不可变 candidate 上以 15 个独立进程、双顺序和双
placement 执行一次完整 qualification。

## 七问（AUDITED）

1. 第三方：Highway/xsimd 均由中央 lock/cache 管理，Apache-2.0/BSD-3-Clause 归档随安装；本轮只使用
   Highway header-only lowering，版本和拒绝 xsimd/approximate 的原因已在 capability map 固定。
2. 功能：Compute dense row 与普通数组两个纵切共同证明四项缺口；Tensor/broadcast policy 明确仍由 Compute
   拥有，有限集之外没有新 `MUST`。
3. 内部复用：Compute 将复用 SIMD；SIMD 继续复用 Core、Platform 与既有 Highway target 生成机制，不复制
   CPU 探测、runtime route 或 allocation。
4. 性能：全部 O(n)/O(1)，无隐藏资源成本；P1 只在 candidate 后测 Compute-real dense rows，并同时比较同
   语义 scalar oracle 与 direct locked-Highway body，5% 为 material 门而非把小输改名为 parity。
5. FFI：不存在 C/Native FFI；只新增 C++ pointer/count API，未产生跨语言 runtime owner。
6. 构建：既有 ops/search generated target TUs 和 `hyle_simd` package export 承载全部变化；需在 F5 回放三交付
   与 exact/fat artifact。
7. 易用：三组名称、alias、NaN/zero 和 dense/fallback 选型已冻结；guide 必须把 full-precision 与 approximate
   以及 pairwise/reduction 的差异写清。
