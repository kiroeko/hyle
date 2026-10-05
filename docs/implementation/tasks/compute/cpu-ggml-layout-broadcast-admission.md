# CPU ggml：布局、广播与宽标量批量接线

<!-- MODULE REVIEW: AUDITED -->

- owner `compute_provider_ggml`、`graph_route_cpu`，baseline `786a3ec36e4a9b19b6eed572f1e9c96dc93396f5`，Windows canonical。
- trigger：全族审计的 MatMul row-major batch/左小右大、Softmax 非末轴、SDPA BHSD/value depth、
  Concat 宽 scalar 是现存公开 CPU 旅程；当前 adapter 拒绝，原 API/源码已提供相应布局与 byte-copy。
- 主纵切：多 batch/head 广播线性层、两种 attention layout、不同 K/V depth；正交纵切：普通 tensor
  非末轴概率归一化和 mixed-width storage 拼接，准备一次、变化输入、Direct 与完整 Graph。
- scope：下表四组成熟能力批量准入及一批资格；无新数学、API/默认排序/ISA/ABI/pin/构建仪器，
  Native、SIMD/GPU 数学保持冻结。rank>4 MatMul、Bool/log-softmax、mask attention、I64 索引、
  duplicate Scatter、alias/bounded/不兼容布局不借此签完整支持或唯一。
- 资源：既有 arena/pool/slot/checked mappings；原 transpose/cont/repeat 的 payload 冷分配、热复用，
  包含在实际成本。外部 alias 拒绝；索引 math 不能因 widening 存储类型而误准入。
- P0：layout 变换 O(n)、广播必要展开与 RHS repack 的完整字节，attention 原完整程序；
  Softmax 复用上一包安全特殊行，不拆成 HYLE 数学。此批不启动 Native P1，不签 WIN。
- 预算：一次当前原 API/实现审计，LB1–5 有限 FIX，一批 Windows/Linux 受影响测试和相同消费者的
  source/static/shared；最多一轮具体失败纠正。不逐算子重复 SDK、不重复模型/loader/GPU/P1。
- 验证：provider/普通 CPU/Graph；F32 row-major/transposed rank2..4、shared/双向leading广播、变化输入、
  K0非空输出与独立double oracle；三dtype Softmax axis0/1/−2、mask广播/特殊行/恢复；
  BSHD/BHSD、三dtype、causal/noncausal、GQA与不同Dk/Dv；Copy/Concat 全15种plain scalar bits，
  大整数、NaN payload/负零、空部件、strided inputs与变化输入；拒绝未接宽 Gather/index/numeric Cast。
  生命周期/OOM/close/alias复跑现有受影响provider；selector/closure/docs/diff、统一LOC。
- stop：原 kernel 对具体 dtype/layout 不能完整承载则记录具名限制与证据，不增加自研算法或签UNIQUE。

## 已审计固定的有限 FIX

| ID | 允许改动 | 必需验收 |
|---|---|---|
| LB1 | F32 plain RHS dense row-major 的既有原transpose/cont准入；lhs leading broadcast用原repeat；K0用原零identity/repeat | 双向NumPy广播、实际值、成本、K0下游 |
| LB2 | Softmax任意axis用原reshape/permute/cont移动到row轴、复用安全Softmax后逆变换 | 三dtype、mask/特殊行传播、下游/重放、原末轴回归 |
| LB3 | SDPA BSHD/BHSD布局转换与Dk/Dv独立；仅按因果建mask | 三dtype/因果/GQA/独立oracle/两种布局 |
| LB4 | plain scalar native representation供原Copy/Concat使用；同时收紧未提供宽Gather/Cast数学谓词 | 全storage bits、strides、原被接受域不回归、未接算术明确拒绝 |
| LB5 | 一批普通Direct/Graph、三交付、文档/实际成本与剩余ledger；Graph拒绝空输出而非一切空输入，计入LHS原repeat展开 | K0非空输出和空Concat部件保留整region；普通入口自动固定成熟程序；未运行不PASS |

## 审计与边界

原 pin 的 `ggml.c` transpose/cont、repeat_4d 与 mul_mat 保留完整程序；原 mul_mat 只能把 RHS
广播到 LHS，因此 LB1 先用原 repeat 扩展 LHS 到公共 NumPy 输出批次。展开元素超过现有 INT32
资源域保持拒绝；K0 非空输出借原 repeat 零 identity，空输出混合区域不扩大。RHS repack 与 LHS
展开均计入成本。Softmax 原 kernel 仅沿 ne0：LB2 在 scale/additive 后用 reshape/permute/cont
移动选定轴，再逆变换；复用安全特殊行，无新指数或归约数学。每节点 32 tensor 的冷 metadata
预算覆盖此轮最多 16 个 lowering 节点，payload 由原规划器复用。

`ggml_flash_attn_ext` 原输出 ne={Dv,Hq,Tq,B}，CPU 实现 DK/DV 独立；LB3 的 F16 causal 保留 flash，
BHSD 输出用原 permute/cont 恢复。其余原 F32 score/value 程序原输出 ne={Dv,Tq,Hq,B}；BSHD
才恢复排列。causal mask 的 sequence 轴随 layout 冷绑定；公共 infer 已校验 batch/head/depth。
原 dup 同类型走 byte copy，concat_any 对未专化 dtype 走 row memcpy；LB4 的 I8/I16/I32/I64/F64
只是存储载体，不等于 unsigned/FP8 数学。实际 public enum 共 15 种已知 scalar，非 16 种。
Gather 继续仅原 F32/F16/BF16/I32 table、I32 index；不同 public dtype 的宽 Cast 不因原类型映射
相同而准入。宽 Fill、其他 arithmetic/alias/index 域继续具名待资格化。

普通 typed Direct/Graph、原 metadata/payload/worker/完成 owner 不变；原 pin、许可证、交付组件和
依赖方向复用。主模型与非模型纵切、资源/数值/成本/错误/构建/升级七问已有限定界；LB1–5
BORROW/FIX 固定，其余 DEFER。owner AUDITED，适用门尚未运行，均 UNPROVEN。

## 普通 Graph 反例与有限纠正

候选 `7a5dfe49865c5e7f98ac58ef557303c839a8a32d` Windows Provider58/58已通过；普通 consumer 的
K0 MatMul→Copy 得到两个 leaf、零region。源码 `graph_route_cpu/ggml_region.cc::supported_node`
不问输出或原supports就拒绝所有空输入，且region repack仅计算RHS，遗漏本轮LHS广播展开。
据此把当前SCOPED的 `graph_route_cpu` 绑定本卡并AUDITED；LB5仅去除该空输入总拒绝、保留空输出
区域拒绝，把原supports作为语义准入，补LHS展开字节并检查加法溢出。无Graph公共规划/API重写。
这是本批一次产品纠正，测试语法/旧冷拒绝断言迁移另保留原失败收据，不放宽数值门。
