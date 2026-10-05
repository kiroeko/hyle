# CPU ggml：剩余成熟能力批量准入

<!-- MODULE REVIEW: CANDIDATE -->

## Closure card

- owner：`compute_provider_ggml`；baseline `7053e854783c0d9ec42d0b8e80a79a7c02251b14`，Windows canonical。
- trigger：普通 CPU 的现有 Unary/ReLU 与 NumPy batch MatMul 消费。现有 adapter 将原版可执行的
  F32 rank3/4 MatMul 仅限 rank2；原 ReLU 将 NaN 变为零，不能按公共传播合同直接接入。
  [实际无计时探针](../../../../evidence/correctness/cpu-ggml-mature-batch/README.md)使用当前生产优化程序，
  rank4 shared/per-batch weights 与独立 double oracle 一致；激活特殊值和尾部差异已记录。
- 主纵切：模型 batch/heads 的 F32 activation × contiguous-K transposed weight，准备一次，变化输入重复执行；
  ReLU→MatMul 的完整成熟 Graph 保持 region 与原生 pool/slot。
- 正交纵切：批处理 ReLU，三 storage dtype、NaN/±Inf/±0、strides、精确 alias、奇数长度与多线程。
- scope：ReLU 的成熟原核加 NaN 传播接缝；既有 F32 MatMul 的 rank3/4 准入。
  不将原版 SiLU/GELU 的诊断结果签为完整支持；其余成熟候选继续由逐域卡核对。
- non-goals：自研数学/性能优化/裁决，SIMD 清理，GPU，公开 API、默认 Provider 排序、runtime ISA/ABI、
  依赖版本、构建与性能仪器修改。过去宽泛 Native fallback 不作为保留证据。
- 依赖/消费者：已关闭 ggml runtime 的原版整程序、pool/arena/slot；普通 Compute CPU 与 Graph。
- tier：P0。ReLU 原位边界 O(n)，一个 view node、无额外完整 payload；MatMul 只改冷准入条件，
  不改原 kernel、转换、scratch 或热选路。未来公平比较包含接缝、cast、布局转换与完成成本。
- 预算：一次现存源码/实际准入审计、三项 FIX、一批 Windows/Linux 验证；失败最多一轮有界纠正，
  不重复完整模型/GPU/ISA 资格，不启动 Native P1。
- 预声明验证：ggml provider、普通 CPU、普通 Graph 受影响批次；ReLU 三 dtype、有限/非有限值、
  replay、1/4 threads、strided/alias；F32 rank3/4、shared/per-batch/mixed leading broadcast、M1/奇数 K/N、
  未受支持的右大批次拒绝；普通 Direct 与 ReLU→MatMul 完整 Graph 的 Provider/region 与下游实际值；
  Windows/Linux source/static/shared 同一独立消费者；selector、closure/docs/diff、统一实现 LOC。
  重用身份相同且无变更的 loader/ISA/SDK 证据，不重做未受影响矩阵。
- stop：若原版 MatMul batch 布局与公共广播不一致，缩回明确原版可表达域并记录拒绝；
  不能用 flatten 改坏 batch，不能放宽精度或将新未运行验收写 PASS。基础接缝失效在预算内修复或 BLOCKED。

## 能力与形状审计

| 能力 | 当前事实 | 承载与裁决 |
|---|---|---|
| F32 MatMul | `native_shape`/transpose/cont/mul_mat 已保留两 leading axes，rank2 filter 才阻止 batch | BORROW：只开放 dense activation、plain contiguous-K RHS、rank2..4、原 ggml 可广播批次；shape infer 仍先校验 NumPy 语义 |
| ReLU | 原函数有限值/±Inf/±0 正确，NaN 被变成零 | BORROW：原核后原位恢复输入 NaN，冷绑定 callback，输出在下游消费前修正；不写第二套 ReLU |
| SiLU | -Inf 未适配原核为 NaN；-90/-100 的有限负尾部输出零，gate 乘大值可放大损失 | 未资格：不能凭相同名字签全域；既有 SiLU gate 卡不代签这次新增诊断域，逐域卡记录下一批成熟 carrier 调查 |
| GELU-erf | -Inf 为 NaN；float erf cancellation 的尾部误差需按 F32 语义核对 | 未资格：不直接扩大支持、不新写数学 |
| GELU-tanh | 当前原版 F32 使用 FP16 查表，诊断 x=3 偏差约 2.69e-4；gate finite-zero × Inf 分类也可能不同 | 未资格：查具名原版精度配置/成熟 carrier；不裁决 Native WIN，不无限研究 |

## 有限 FIX（AUDITED，已固定）

| ID | 允许改动 | 验收 |
|---|---|---|
| GB1 | ggml Unary ReLU lowering/冷支持，原 kernel 后一个 custom2 inplace NaN 边界 | 三 dtype、特殊值、尾部/stride、alias、重放、多线程，无新数值核 |
| GB2 | F32 `native_f32` 准入从 rank2 延伸到原版已经表示的 rank3/4 | 真实共享/逐批/混合广播与独立 oracle，拒绝原版不支持的左小右大广播；原执行与物理布局保留 |
| GB3 | 将上述能力一批加入 provider/普通入口/整 Graph 与三交付 consumer，记录实际成本与限制 | 不改变模型画像/API/规则优先级；普通用户无需专家胶水；全部适用项真实运行 |

## 冻结形状与七问

普通入口仍为 `cpu::create_context()`、typed `prepare()` 与 `graph::compile()`；没有新增公开选项。
ggml 的外部输入/输出 alias 仍按既有资源合同拒绝；内部原位 view 不代表公共原位支持。
测试明确验证该拒绝，不将 Native 的原位档与成熟 out-of-place 偷换成同一域。
版本/许可证/升级仍由已关闭 ggml runtime 管理，原数学源码与 CPU program ABI 保持；数值边界仅 adapter 拥有。
资源、OOM、close、并发 invocation、completion 复用现有 checked binding 与 ggml pool/slot，
provider 接口不引入新的线程/资源 owner。source/static/shared 的新消费必须实际验收。
F32 MatMul 与 ReLU 是既有公开任务的成熟承载补齐；其它尚未资格 kind 不是 UNIQUE。
P0 成本明确，无热 Provider 搜索或计时；未来 Native 逐域比较须使用本轮真实成本。
没有新的 FFI 或 build factor；用户文档补充真实 rank/dtype/数值接缝。当前未运行的新门保持 UNPROVEN。

## 审计绑定

实际原版程序诊断、`internal::supports`、`native_shape` 与原位 custom2 的 source/slot 合同已核对。
GB1–3 已固定；开始产品修改前 owner 从 SCOPED 顺序进入 AUDITED。新增产品成本只限表内，
其它激活仍未资格、Native 全部冻结。未运行的三交付与测试仍为 UNPROVEN。

## 固定候选与当前验收

候选 `ef9defea07bbc576a4c6bff1ce0d07e4e53acd19`；GB1–3 产品停止变化。Windows Provider 48/48、
普通 CPU/Graph 的具名受影响测试、三 dtype 的普通 Direct 与完整 ReLU→混合 batch MatMul region、
source/static/shared 实际独立消费已通过。Linux 单向校验同步 1,581 项输入，正在同候选验收，
未完成项不签 PASS。新试验曾错误期待外部原位成功，收到既有 unsafe-overlap 拒绝；纠正测试预期，
保留原始失败。内部原位边界不改公开 alias 能力，不扩张资源映射。

用户要求的先专门排查已落入逐域卡 A1–A3；[全族回映与564条冷记录](../../../../evidence/correctness/cpu-ggml-mature-batch/adapter-audit.md)
不代签新产品支持或 Native 胜负。Compute + Graph 当前 **61,050 实现行 / 248 文件**，统一脚本，
不含测试与 vendor。本批 Linux 适用验收尚待完成，不提前 CLOSED。
