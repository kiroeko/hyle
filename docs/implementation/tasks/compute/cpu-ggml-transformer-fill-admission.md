# CPU ggml：Softmax、非因果 Attention 与 Fill 批量接线

<!-- MODULE REVIEW: AUDITED -->

- owner `compute_provider_ggml`，baseline `be5d18f31`（完整身份由 Git/收据导出），Windows canonical。
- trigger：[已完成全族审计](../../../../evidence/correctness/cpu-ggml-mature-batch/adapter-audit.md)的
  原版 rank4 scaled/narrow/masked Softmax、非因果完整 attention 与 Fill 实际 witness，当前 adapter 拒绝。
- 主纵切：多 batch/head 的 scaled additive Softmax 和非因果 SDPA，普通 Direct/Graph，准备一次、变化输入重放。
  正交纵切：浮点 Fill 和非模型概率归一化，三 dtype、尾部、空输出、非有限值、外部 alias 拒绝。
- 原数学、pin、ISA/runtime/ABI、pool/slot 保持。无新公开 API、默认排序、Native/SIMD/GPU 数学或性能仪器修改。
- 数值：Softmax 最后轴、F32 accumulation；scale 后 additive mask，无有效有限值或有 NaN/+Inf 的行输出 NaN。
  原版 debug Softmax 对上述行有 assert，必须在原 kernel 前安全替换为有限诊断行，之后恢复 NaN，
  不新增 exp/归一化数学。非因果 Attention 的 scores 使用相同边界；原 causal F16 flash 档不改。
- 资源：原 metadata arena、pool、prepared immutable recipe、独立 invocation；外部 alias 仍拒绝。
  数值接缝增加 O(n) 扫描和原生临时 payload，均在冷规划计入、热重放复用；未来比较必须包含。
- 预算：一次当前源码定界，四项 FIX，一次 Windows/Linux 受影响批次与 source/static/shared；
  一轮具体失败纠正；P0-only，不跑 Native P1，不为单函数重复整套交付。
- 适用验证：provider unit、普通 CPU/Graph 具名消费、独立普通三交付 consumer；三 storage dtype、
  rank1/2/4、axis 正负等价、scale、additive 广播、NaN/Inf/all-masked/零scale、重放、多线程、
  Fill 空/标量/三dtype bits、拒绝 Bool/log-softmax/其它axis/外部alias；SDPA 非因果多head/GQA、
  q_len>k_len、causal 不变、双精度 oracle、replay；现有 OOM/slot/close 验证复用并重跑受影响 provider。
  docs/closure/diff、selector selected/skipped、统一 Compute+Graph LOC 必需。
- stop：不能证明同合同的 mask/layout/其它 dtype 保持具名拒绝和 ledger，不用拒绝签 UNIQUE。
  本卡不承诺一次完成全 CPU 接线；其它普通 tensor、index/layout 候选仍由总卡逐批处理。

## 有限 FIX（已审计固定）

| ID | 有限改动 | 验收 |
|---|---|---|
| TF1 | 最后轴 floating Softmax 的 rank/dtype/scale/additive broadcast 准入；原 scale/add/softmax 与特殊行边界 | 有限 oracle、undefined row NaN、debug 原断言输入安全、下游前修复、外部 alias 拒绝 |
| TF2 | 非因果 SDPA 原 score→Softmax→value 完整程序；只 causal 时建/施 causal mask；非因果 F16 采用原 F32 cast 程序 | 三dtype、GQA/变化输入、q_len>k_len，原因果回归 |
| TF3 | floating Fill 用冷存的精确 native scalar bits→原 `repeat_4d`，不借输入、不经数值 cast；区域参与 | scalar/empty/NaN payload/Inf/−0/三dtype，真实 Graph 与无输入 Direct |
| TF4 | 一批 provider/普通入口/独立三交付验证和用户文档、成本/ledger | 有限门真实运行；不签 Native WIN |

## 审计结论与七问

原 pin `456172ec733a135778adcd32d00e576a58232e45` 的 `ops.cpp:5584` 显示 Softmax debug 会
assert NaN 与 sum>0；只放开 rank/dtype 不完整。TF1 使用原 scale/add 形成真实 F32 scores，
两个无 exp/归一化数学的 custom row 边界包住原 Softmax，特殊行在 kernel 前变为零、之后恢复 NaN。
代价是一个 F32 安全 score 缓冲与两次 O(n) 检查，masked/scaled 另有原 score 临时值；因公开特殊值
合同必要而显式记录 scale/mask 不再融合，不能拿 raw fused 调用签同成本。布局 materialization 全由原库。
TF2 非因果采用原 score 程序，因果 F32/BF16 共用安全 Softmax，F16 causal flash 不扩范围。
原 `repeat_4d` 总会产生独立输出，`ops.cpp:1789` 的 F16/BF16 repeat 按 16-bit 原样复制；
TF3 比 Fill→cast 更符合 exact ScalarValue（含 NaN payload），无自研 Fill 循环。
原 `supports` 的 inputs.empty 需仅为 Fill 开放，原 compile 的 available/output 证明可无输入执行。
metadata 的每节点 32 个 tensor 预算足够本轮 <=10 节点 lowering；原 allocator/OOM/slot/replay 承载。

公开 `cpu::create_context()`、typed prepare 与 Graph add/compile 无变化；资源和完成无新 owner，
pin/许可证/升级/loader 复用已关闭 runtime；C++ 三交付需实际新消费，原 C/FFI 无新增；
用户文档写清数值、成本和拒绝域；P0 无新热选路、数学、系统调用或按轮分配。
七问与主/正交纵切已定界，TF1–4 BORROW/FIX 固定；其它族 DEFER 到下一批，不裁决 Native。
未运行门 UNPROVEN。只有本 owner 进入 AUDITED 后才改产品。
