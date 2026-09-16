# R4 CUDA：成熟激活基线与默认规则收口

## AUDITED / 有限 FIX

- 从属 [R4 主卡](cuda-v1-four-route-foundation.md) C3/C4 的默认规则尾项；产品 baseline
  `a677565959ea5534c5aca4ce90b41d1740a4d1a5`。owner 仍为当前 AUDITED 的
  `compute_provider_cuda`、`compute_route_cuda`、`graph_route_cuda`，不重开共同 Compute/Graph 或 CPU。
- trigger：[Direct](../../../../evidence/performance/compute-cuda-ffn-p1-v2-a6775659-w1-win/README.md)
  与 [Graph](../../../../evidence/performance/graph-cuda-ffn-p1-v2-a6775659-w1-win/README.md)
  均完成 W1 engineering P1，n=15 即裁决，5% 非劣、A/A、全部 control/validator/raw replay PASS。
  未启动 n=31。HYLE 激活没有跨 placement/执行/分配顺序成立的完整链明确优势，按原规则采用合格
  成熟基线；不能因为非劣 PASS 继续把 HYLE 认作赢家。
- 方向保留：Direct M16 有一个 stratum 的 CI 完全低于1；Graph M1 有两个完全高于1，其余跨1。
  这些局部小输/赢如实保留，不把 practical PARITY 写成严格等速，也不按实验布局制造产品规则。
- 主/正交 consumer、MatMul/Q4 精度、两个运行槽、完成/失败/资源边界沿主卡。新增接入仍只为
  当前完整 FFN 的 gated activation，不增加算子目录、模型策略或新的通用执行器。
- 预算：本次有限源码审计已收敛；使用主卡原一次产品算法/规则纠正预算。API/算法/验证夹具固定
  后 Direct/Graph 各一次新候选 P1，原 n=15→31 和5%门不变；失败先查源码，不再无限换候选。

## 能力裁决与固定范围

| ID | 有限 FIX | 实际验收 |
|---|---|---|
| S1 | 将已经编译/oracle/P1 验证的 SGLang precise fragment 窄接入现有 CUDA kernel factor；锁来源/许可 | pin、Git blob、真实 archive、vendored hash、LICENSE/NOTICE、source/static/shared |
| S2 | 明确成熟 Provider 身份；普通 CUDA factory 冷固定成熟合法域，保留 HYLE 必需域 | 用户不排序 Provider、不交 profile；Direct/Graph 实际选择，hot 无硬件查询/计时/重选 |
| S3 | 独立数值、边界和原生寿命检查 | FP16/FP32、向量宽度/多行/特殊值、双槽变化输入、非法对齐/alias、OOM、关闭/完成、真实 capture |
| S4 | 最终候选交付/性能与原生完整边界账单 | 适用普通/host/device 检查、Win/WSL 三交付、CUDA OFF 公共头/依赖投影、P1 与用户文档 |

S1 继续使用 SGLang `06992c92ad4494fe9f9636e75be3e35040512447` 的已审 device body；
不再搜索 peer。保留算术/索引，排除 TVM/Python host glue；私有命名空间改名，避免与独立 benchmark
reference 的 `sglang` 符号发生 ODR 合并。HYLE 维护窄 adapter、升级及完整检查责任，记录 Apache 许可。

S2 在 `compute_provider_cuda` 中复用现有 exact-primary borrow、Prepared、invocation/Completion
机制，增加具名 `cuda_native::sglang_provider()`；原 `provider()` 继续提供 HYLE scalar 能力。
成熟域是 dense FP16/FP32、packed/out-of-place SwiGLU：FP32 SiLU 与乘法、最终一次输出转换，
禁 fast-math、PDL/filter/reuse；FP16 hidden%8、FP32 hidden%4、非空、output count<=INT32_MAX，
输入/输出16-byte alignment。kernel 和 launch 参数 ABI 在 prepare 固定。

普通 CUDA factory 在上述 shape 域优先成熟 Provider；separate 和非向量尾部继续 HYLE。
Graph 已有 cold provider-index 优先规则可消费同一组合，不改共同规划器。描述符准备不能证明尚未
提供的 pointer alignment；运行绑定必须满足已声明16字节合同，不满足时明确拒绝，不能偷偷复制
或 hot 再选。原 HYLE 专家入口继续支持合法2/4字节对齐，不删除有效能力。来源不同不另造 execution owner。

已有冷阶段可访问实际 device/SM/SDK；未来有证据的硬件/版本条件仍可在 owning Provider 匹配。
本轮没有已证明的 HYLE 优势域，因此不虚造 GPU 型号或 driver 阈值；成熟路线作为合法基线，
性能资格仍只签已测 RTX4070/SM89、driver610.62、Toolkit13.3.73，不外推其它硬件为 SOTA。

## 适用验证与停止

- C1/C2/C6、共同失败槽退役、CPU 算法与用户例外不变，复用匹配证据；selector 只按实际 public
  header / package seam 扩展，不能为了本片段重跑 CPU P1 或全库审计。
- 新 kernel 用独立 double oracle；普通 Direct/Graph 测真实默认选择、尾部/separate 保留路径、
  变化输入与 capture；host 插桩只签自有 host 源码，device memcheck 签新完整链，不冒称覆盖整个 SDK。
- Windows/WSL source/static/shared 和 license/relocation；CUDA OFF 在两 Mac 仅签公共头与发布
  投影。Windows DLL 必须物理邻置。重型串行、编译并行2。
- 最终 P1 继续 SAME_CONTRACT：默认路线与锁定独立精确参考采用同样生命周期、数值和完整链。
  原 HYLE 比较证据保留为规则依据；新默认若等价于成熟参考，A/A/NI 不能记作 HYLE 算法胜利。
- 原生成本 side 必须同步使用最终选定的实际数学实现。用统一仪器既有采集/汇总接口、同一
  W1 policy 与两个 shape，执行/分配双顺序各 n=15，一次有限诊断，不扩样；标为 NOT_COMPARABLE /
  NATURAL_ROUTE，只报完整边界成本。裸 API 缺少 admission/owner/失败服务，不签同合同 NI。
- 所有本卡适用检查完成前不授予 CUDA CLOSED；任何未运行项仍未证明。发现范围外具体缺口按
  所属 owner 定向处理，不修改公共治理/性能核心。

## 实施快照（历史 AUDITED 检查点）

- `c5848846` 已接入具名成熟 Provider、冷默认绑定与原 HYLE 能力保留；实际源码归档/四个
  上游文件 Git blob、许可与 vendored identity 已逐项核对。无 TVM/Python runtime。
- Windows MSVC/CUDA shared Release 定向门 6/6 PASS：三个 factor 的真实测试与逐头检查。
  覆盖18个成熟 FP16/FP32 shape 组合、特殊值、双槽变化输入、对齐/alias 拒绝后继续使用、
  原 HYLE 2/4字节对齐能力、OOM/保活/关闭、普通 Direct 与实际 Graph capture 的默认和回退域。
  完整日志暂存 `C:/hyle-scratch/r4-cuda-20260916/mature-rule/`，尚不冒充全部 S4。
- 最终默认 P1 使用 [Direct v3](../../../../bench/contracts/compute-cuda-ffn-p1-v3.json) 与
  [Graph v3](../../../../bench/contracts/graph-cuda-ffn-p1-v3.json)：只更新最终实现身份/claim；
  原 v2 不覆盖，workload、8 strata、control、5%门及 n=15→31 预算原样保留。尚未启动。
- selector 对新增 dependency lock / vendored input 要求 package 闭包检查；沿现有规则执行，
  仅登记 `third_party/sglang_activation/ -> compute_provider_cuda`，不更改 selector 算法。
  三交付、适用 host/device 插桩、两 Mac CUDA OFF 投影、最终 P1 与原生成本诊断仍待完成。

## 最终实现固定：CANDIDATE

`755fb7c76132908e15cc8e2c298c69c227af775a` / tree
`5f91d6d548e6e59406adcdc683c67ee473e59f4b` 为最终不可变产品候选。
[普通/插桩/交付证据](../../../../evidence/correctness/cuda-r4-mature-default/README.md)已完成 S1–S3
及 S4 的普通验收：Windows 全 selector 106 门、增量4门、Win/WSL CUDA-only 三交付、
WSL 四种 host 模式与 device memcheck、两 Mac CUDA OFF 投影均通过。
GCC 曾因测试 gtest 条件宏缺大括号失败，755 仅修测试；产品实现和算法未变。

主卡六个 owner 进入 CANDIDATE。最终 v3 P1 与原生边界成本仍待实际运行，不写 PASS；
文档/状态/证据归档不产生新产品候选，也不重跑已完成的普通门。
