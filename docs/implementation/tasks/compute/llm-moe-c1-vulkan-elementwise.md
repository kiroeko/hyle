# C1-Y35：Vulkan 成熟逐元素与门控接入

## SCOPED / AUDITED

- owner `/root`；baseline `1e43875a22efdabed4e445e7ee15bf9823f31031`。
- 当前 owner：`compute_provider_ggml_vulkan`、`compute_route_vulkan`、`graph_route_vulkan`，
  保持 C1 AUDITED，仅对下述有限 FIX 重新定界。CPU Q2 BLOCKED 不变。
- 需求：原矩阵 F03 的残差/广播、稳定变换、SwiGLU、attention sigmoid gate；真实消费为
  本地 Qwen3.5 的 `-exp(A_log) * softplus(a.float()+dt_bias)` 和已登记 DeepSeek 的
  `sqrt(softplus(router_score))`。正交消费为带 stride、rank0/8、三浮点的数学变换。
- 来源：固定 ggml `456172ec733a135778adcd32d00e576a58232e45` 的 Unary、Log、Sqrt、
  Fill/Div、GLU API、Vulkan 分派与 shader；共同数值合同沿 C1-A/G/T，不由新测试发明。
  [源码审计](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-elementwise/source-audit.json)
  是静态输入，不是设备运行或性能 PASS。来源调查到本节具名差异停止。

原版 Softplus 用 `log(1+exp(x))`，F32 的 -40/-20 尾值归零，-15 有显著消去误差；原版
Tanh 的 `1-2/(exp(2*x)+1)` 在 ±1e-8 归零。Log/Sqrt 未显式处理负域和 signed zero。
接口中有 Log/Sqrt/Softplus 和 packed/separate SwiGLU，但 HYLE 当前未绑定这些全部域。
Rsqrt 可由原厂 Fill(1)/Sqrt/Div 组成，不新增求倒根核。动态 Binary Min/Max 未找到同合同
原厂入口，保留既有 fmin/fmax 路径；不拿常量 Clamp 替代双输入 Min/Max。

| FIX | 固定允许改动 |
|---|---|
| Y35-A | typed region 接入 Log/Sqrt/Softplus、原厂三节点 Rsqrt、packed SwiGLU；已有 Abs/Neg/Exp/Tanh/Sigmoid/SiLU、Add/Sub/Mul/Div、separate SwiGLU 继续组成原厂 region。保留原厂融合、allocator、work 与 HYLE 片段边界。 |
| Y35-B | 仅在现有精确 patch 机制中补上述数值合同：Log/Sqrt 的 NaN/负域/±0/Inf，Tanh 近零与特殊值，Softplus 补偿 log1p；Sigmoid/SiLU/SwiGLU 保留稳定负尾与共同 Inf 语义。修改被明确标为语义适配，不能声称原版数学未变；不改版本 pin 或公共构建工具。 |
| Y35-C | 普通入口冷期适配 F32/F16/BF16、rank≤8、合法 stride/起点与广播。复用 Cast/Copy/alias，F32 求值、half 最后一次 RNE；packed 门控保留行内两半，使用原厂 GLU。rank≤4 保留原广播形状；高 rank 必要时物化广播再展平。冷查 F32 容量与原厂 dispatch 限制，空域/设备不适用域及未绑定的 Min/Max/其他激活保留既有合法路径。 |
| Y35-D | 原门下独立 FP64/位级参考；普通 Direct/Graph/Recorder、真实 managed 路径、两槽/变输入/旧结果/关闭、alias/配额/OOM/drain。覆盖大负 Softplus/近零 Tanh、Log/Sqrt/Rsqrt 特殊值、half 最终舍入、scalar/空/tail/奇数 stride/rank8、广播两侧、packed/separate gate 及 published 中间值。 |

## API、成本与七问

普通入口仍为 `vulkan::prepare(context, Unary{...}, desc)`、`GatedActivation` 和相同 Graph
节点；无新公共 API、模型类型、运行期搜索、计时或 Provider 排序。owner、错误、完成、异步
保活、close 和失败传播沿已有 Program/Invocation/Recorder/managed region，不另造执行器。

承载为 `BORROW`：MIT 固定 ggml 的完整程序与融合，HYLE 仅负责语义/布局/资源边界适配。
复用现有 Mem、Vulkan backend、shader 构建、依赖锁和三交付合同，不重开其他 CLOSED 模块。
第三方没有同合同入口的域保留现有实现，升级须重核 patch hash 与本卡数值证据。

P0：逐元素 O(N)，无热期搜索。三节点 Rsqrt 有原厂两个中间张量及三次 dispatch；半精度/
stride 可能增加 F32 整张暂存和转换，高 rank 广播物化额外读写必须计入。低 rank 已合法
region 不因适配拆开；每个公开 half 操作的舍入边界必须保留，不能为融合略过。没有本批性能
赢家声明；新旧路径的完整成本比较、条件默认规则和冗余实现取舍仍在 C1 固定 P1 bundle。

## 验证、预算与停止

- 先 W1 三个受影响 factor 与新增反例，再同候选 W1 static/shared、Air/Mini 适用回归、
  source contract/公开头。Windows DLL 邻 exe、PATH 隔离、禁弹窗；重型串行、parallel2。
- W1/Air 的本地 2B 实际张量稳定门控/路由得分片段必须通过 source/static/shared，独立公式、
  重复执行和资源归零。Y34 的归约片段不能代签这次新接线。
- 保存 selector selected/skipped 理由；依赖锁变化所选工程门在本批固定候选完成，未执行
  不记 PASS。身份匹配的未变证据复用；文档/状态不重建，已通过候选门不重复。
- 产品纠正最多两轮、独立 consumer 编译纠正最多两轮；不放宽数学/资格门，不靠 fallback
  伪装成熟路径通过。来源调查最多60分钟/两个具名来源，当前一个固定源，至上述事实停止。
- 归档后清理本批重复 producer/过期副本，保留共享依赖和活动缓存。本批不运行 P1。

审计裁决：有限 FIX 已固定；产品尚未实施，适用验证尚未运行，C1/共同矩阵/性能未签收。
C1 全部适用验收完成后单独通知用户，在 C2 前等待提问。

## 首个实现候选

审计 `917b758b` 后已实施 Y35-A–D：原厂新操作/门控绑定、精确数值 patch 与普通 route
冷期类型/布局适配，新增独立参考、真实 managed 路径及模型门控 Graph 回归。
patch-tree 为 `3ee02da8a740cc831d6cef427cf8dd9302c9fd3cd931a5586782094bf43f2852`，
原版依赖 pin 不变。五个 source contract 测试通过；实际编译、设备与交付资格尚未运行，
没有 P1。selector 因依赖锁扩为112项 selected、8项未选，未执行门不记 PASS。

首次 W1 构建在新增 Direct 测试缺少 `<algorithm>` 时停止（MSVC C2039/C3861，
`std::ranges::count_if/all_of`）；尚未执行 GPU 测试。补显式头依赖后按同一缓存增量构建，
保留首败日志。产品/算法纠正0，测试编译纠正1；不改变数值门、运行语义或统计工具。

## F1：小量 Softplus 数值纠正（SCOPED / AUDITED）

baseline `1e33a79f`，owner `/root`。W1 的 Provider19项通过，Direct95项中94项通过、
1项失败：严格尾值回归的 Softplus(-15) 在 F32 得到 `3.695349448662455e-7`，独立参考为
`3.059022617435403e-7`；BF16 同样偏大。Graph 尚未运行。
[首败收据](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-elementwise/first-failures.json)
保留编译首败、GPU原始日志与数值，不删除点或放宽门限。

源码定位：补偿 `log(1+e)` 仍依赖设备 log 在1附近的精度；原生 Softplus 也使用这个形式。
既有 `tensor.comp::log_one_plus` 的 LogSoftmax 路径已避开该域：`e<1/64` 用五阶展开，
其余用补偿式；省略项有 `e^6/6` 上界。这是具名已有能力复用，不重审其他数值算法。

有限 F1：ggml Softplus patch 复用该小量表达式；原生 Softplus 复用同一个已有 helper，
将 helper 移出 reduction 条件块，不改变其数学或其他调用者。新增原生 fallback 的同输入
尾值回归，并覆盖阈值两侧；既有 managed 严格门与完整本批适用验收保持。产品纠正消耗1/2，
测试编译纠正1；不启动性能、不改公共仪器、不改变其他 owner 状态。实施前先提交本节审计。

审计 `e7e972cf` 后实施 F1：原生 helper 只移动定义，LogSoftmax 数学保持逐字不变；
Softplus 的两条路径共享其小量处理，严格回归覆盖三浮点及 `exp(-abs(x))=1/64` 两侧。
更新 patch-tree 为 `fb15744d5705ded9d73b045faa0b916e02a22c0e59c69c451811e8e4062a47e5`。
新候选的设备与交付资格尚未运行，不把旧 Provider19项直接代签为新 patch 通过。

### F1 W1 结果与旧结构断言迁移

`117b1c13` 的 Provider19、Direct95项全部通过，包括 managed 与 native Softplus 尾值及
阈值两侧。Graph27项中5项仅旧 native/managed dispatch 数和规划存储断言失败；数值、
输出保活与关闭检查通过，不能把该 suite 记 PASS。按 Y35-D 迁移这五项结构断言，保留原
数值门、生命周期检查和明确的成本断言，不修改产品迎合旧实现形状。

具体账单：half gate→RMSNorm→residual→LayerNorm 改为3个managed region、8个native
dispatch，四个复用块总容量从约10N字节变成20N字节；half unary各保留两次转换。F32 L2
的Rsqrt并入原有region，native/managed分别减为6/2。这些是真实适配开销，不是性能胜出
证据，C1完整成本取舍必须包含。产品纠正仍1/2，测试编译纠正1；其余平台/交付未签收。

### W1 原生模块资格与 F2：Apple 整数类型可移植性（SCOPED / AUDITED）

`6af6bfe4`（产品 `117b1c13`）的 [W1 原生资格](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-elementwise/windows-native.json)
为 static141项、shared140项 PASS，source contract5项 PASS。static Provider19/Direct95
按未变产品与可执行文件身份复用，Graph27完成结构断言迁移；shared为19/95/26。shared
私有runner初始误把缺少的一项归给Provider，已核验并复用实际通过的Provider19，再执行此前
未运行的Direct/Graph；未重复该suite、未改产品/数学门。原始日志、命令hash和DLL身份已核验。

Air同候选首次构建失败，尚未运行GPU测试。[首败收据](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-elementwise/first-air-failure.json)
定位 `vulkan_elementwise.cc:57`：`element_count()` 为 `size_t`，本机底层是 `unsigned long`，
而显式 `uint64_t{512}` 是 `unsigned long long`，`std::min`不能推导同一个类型。两者都为64位，
Windows同底层类型因此未暴露此编译错误；后续all_of诊断来自同一不合法lambda。

F2 baseline `6af6bfe4`，owner `/root`，三个Vulkan owner保持AUDITED。有限FIX仅将此处
常量声明为`std::size_t{512}`，与element_count算术类型一致，数值/范围/设备限制/dispatch
保持。不改shader、API或测试门。先提交本节再实施；产品纠正消耗2/2，之后再有产品失败按
原停止协议裁决，不能继续无预算试改。

F2验证仅增量重编当前W1 static/shared受影响route/Graph并执行其既有95+27/26项；Provider
按固定源码/编译命令/产物身份复用19项，不重跑。Air/Mini执行尚未通过的当前候选原生资格；
真实2B三交付与112项selected工程余门仍需完成，身份匹配者复用。本批不运行P1，CPU Q2
BLOCKED保持。当前统一实现48,943行/198文件，不含测试，C1整包和共同矩阵尚未签收。
