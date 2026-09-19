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
