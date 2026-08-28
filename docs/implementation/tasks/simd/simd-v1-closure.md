# simd-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `simd` 当前唯一 closure card。此前 `simd-01` 至 `simd-25`、旧
> `simd-module-closure.md`、旧 performance 表和 architecture 中的“终态/完成”只作历史事实与审计输入，
> 不授权产品修改，也不提供当前性能证书。机械状态只看
> [`module-status.json`](../../module-status.json)。

## Closure card（`SCOPED`）

- component / owner / round ID：`simd` / CPU SIMD owner / `simd-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`29b77897c0a14c92dd1baa49dccb41033dc835cd` /
  `278527f61607682396637d44d594e9dd65b75dd0`
- 进入原因与 demand/defect ID：
  - `SIMD-CAP-01`：38,024 行现有实现来自多轮 benchmark/peer 补齐，尚未按真实用户任务重新证明
    capability family 完整性；当前 guide 已明确留下 narrow sort 等未实现边；
  - `SIMD-API-01`：whole-buffer、runtime/exact/native/fat-kernel、D-tag 与 narrow surface 很宽，
    普通用户面、expert source seam 和 internal selector 需要重新分层，而不是把 387 个名字视作同类 API；
  - `SIMD-PERF-01`：机械提取当前为 387 个 public/selector 名、344 个 benchable、314 个曾被 bench
    source 覆盖，但当前 report matrix 为 0；历史点值和“全部打平/SOTA”不能继承；
  - `SIMD-CORRECT-01`：scalar/SSE4/AVX2/AVX-512/NEON/SVE/RVV、partial/tail、mask、NaN、narrow、
    alias 与 dispatch safety 的证据来自不同旧轮，必须绑定同一候选；
  - `SIMD-DEP-01`：直接底层依赖是 Core/Platform；半成品 compute 调 whole-buffer，container 通过
    build-private `simd_vec_headers` 使用两个 target-bound kernel，修改公开/专家 seam 会影响真实边；
  - `SIMD-DOC-01`：architecture/user/history 仍混有旧 C ABI、过期 LOC、provisional 与互相冲突的
    “终态”叙事；
  - `SIMD-DELIVERY-01`：fat、exact、native/fat-kernel 与 source/static/shared 的实际可组装性尚未在
    当前 build/package 合同下统一验收。
- 真实 consumer task / normal workload：推理/工程计算对连续 tensor/buffer 做 mixed-precision dot、
  dtype 搬运、elementwise/activation/softmax、scan/search、sort/top-k/select、bitpack 与批量非加密随机；
  source 用户在固定 target TU 写有 partial tail 的 transform/gather/compress/widen-dot kernel；预编译
  用户发布一个 fat binary，已知硬件用户选择 exact target。基准 size strata 固定为适用操作的
  `n=31/4096/1048576`；sort 另用 `n=4096,k=10`，bitpack 用 `n=4096,bits=11/13`。不以旧 bench
  中存在的行决定 API 或 workload。
- 范围：`include/hyle/simd/**`、`src/simd/**`、`tests/simd/**`、`bench/simd/**`；simd component、
  exact/fat/native build contract、package/source delivery；Core/Platform 直接依赖；container 的两个
  build-private vec consumer 与 compute 的代表性 whole-buffer consumer；architecture/user/performance/
  maintenance 文档。下游模块只做受影响回归，不借本轮宣告其收口。
- 非目标：GPU/accelerator backend、矩阵乘/卷积/attention/model operator、完整 BLAS/FFT/linear algebra、
  container 的 UTF-8/probe policy、密码协议、C API/Native FFI、稳定 C++ ABI、任意编译器混用、per-ISA
  DLL、为了 API 数或 benchmark 点值新增常态 metadata、Formal release 证书。
- 直接依赖 / 直接消费者：`PUBLIC core,platform`；产品消费者为 `compute`（PRIVATE batch API，未完成）
  和 `container`（PRIVATE_BUILD `simd_vec_headers`，不拉完整 simd product）。不存在 math/error 直接边；
  是否应新增只能由 audit 中的真实公开 include/失败合同证明。
- tier 与理由：`P1`。batch kernel 与 target-bound primitive 预期高频，但尚无完成的推理引擎 profile
  把某行提升到 P2；本轮签正常 replay 的工程非劣，不宣称端到端 token/s。
- material margin / resource budget：同合同 5% non-inferiority；单线程采样固定最高 performance core，
  编译最多 4 jobs，不运行无关全库 benchmark，不并发两个性能 workload。小于 5% 的点差只有可归因机制
  且能传播到正常 consumer 才能驱动产品。
- 总调查预算 / 性能预算：6 小时 / 90 分钟。3 小时内必须冻结 capability ledger；若范围不能收成一个
  有限产品批次，记录具名 blocker/follow-up，而不是继续扩表。
- 最多复现、机制诊断、产品修改次数：每个 material finding 一次反序/复现和一次 codegen/profile
  机制检查；整个 frozen `FIX` 集一个合并产品批次；一次完整候选重跑。仪器问题单独登记，不在 SIMD
  轮次重写统计核心。
- machine profile：W1-WIN、W1-WSL、M1-MAC。W1-WIN 是 P1 qualification machine；M1-MAC 提供
  NEON 正确性与补充同合同数据；WSL 提供 GCC/sanitizer。AVX-512 SDE 与 SVE/RVV QEMU 只证明执行
  correctness/隔离，不伪装真硅性能。
- instrument profile：`closure-perf-v1`。
- strong peers（每项 capability 最多两个，均为本地只读 checkout）：
  - portable D-tag/native/dispatch/sort/bitpack：Highway
    `239071390b18ca70f8fadde90ebefa6d8b12cb69`；
  - vector transcendental：xsimd `3b3151ff665514a00a437cc985f47b7de99b772b` 与 SLEEF
    `7623d6cfa2712462880fa63a4d0f0b5f775d1a83`，先资格化精度合同再比较；
  - 推理 dtype/batch/dispatch 形状：llama.cpp
    `7584430716ee229751771ed0d6bbcb780d105eeb` 与 oneDNN
    `5dc67b7f37b1215d8c1f054e99c1b7cfd40317a9`，用于任务/算法形状，不做不同合同的点值胜负。
- 无同合同 peer 时的预算及禁止声明：以 scalar oracle、复杂度/bytes moved/分配/dispatch/codegen 和
  预冻结 consumer replay 裁决；不得声称未测 ISA、未测 dtype/op、整个模块或端到端推理 SOTA。
- 适用 correctness / delivery / performance gates：public-header self-contained；zero/null-if-zero、
  partial/tail/guard-page、alignment/alias、integer overflow、NaN/Inf/signed-zero/ULP/total-order、RNG state、
  sort/select/top-k/bitpack property/differential；runtime init reentrancy；ASan/UBSan；x86 scalar/SSE4/AVX2
  与 macOS NEON 真机，AVX-512/SVE/RVV 执行门；component/source/relocated static/shared、fat/exact/native
  consumer；container/compute affected regression；P1 A/A、known-delta、order/placement 与 5% NI；当前
  architecture/user/performance/maintenance 文档；独立红队。
- 预期输出：capability-family map；全部 public header/symbol 的普通/expert/internal trace；与 peer/
  north-star 的有限 gap ledger；冻结 `FIX` 集；不可变 candidate；三平台/backend/delivery/consumer
  manifest；有资格的分能力性能 verdict；当前 architecture/user guide；红队、LOC 与 terminal receipt。
- 停止条件 / 可接受限制：所有 `MUST` 为 `COMPLETE/ROUTED`，不存在 `PARTIAL + KEEP`；全部适用 gate
  有诚实 verdict。无真硅的 AVX-512/SVE/RVV 可以只保留 correctness 与明确的未签性能限制，但不得对它们
  宣称非劣；任何适用 P1 row 未运行、仪器无资格或 material regression 未解都使聚合性能
  `INCONCLUSIVE/FAIL` 并进入 `BLOCKED`。Native FFI 和模型算子保持 negative space。

## SCOPED baseline facts

- immutable baseline LOC：全库 306,880；自实现含测试 124,275；自实现不含测试 83,119；SIMD 实现
  38,024（108 files；`include/hyle/simd + src/simd`）。LOC 只描述规模，不证明缺口或完成度。
- 产品外部语法消费：compute 的 `add/dot/saxpy/exp`；container 的 bitset/string target-bound vec
  kernel。其余调用是 tests/bench/docs。
- `scripts/simd-op-coverage.py` 在 baseline 自检通过：344 benchable 中 source 覆盖 314（91.3%），
  30 个均已分类；但没有当前历史 matrix row，所以只能作为 audit inventory，不能继承 performance verdict。
- 当前 C++ guide 自报 narrow sort 未实现，architecture/user 顶部自报 `NOT_REVIEWED`；这是 audit
  seed，不是已经冻结的 `FIX`。

## Capability-family coverage map

尚未裁决。下一状态 `AUDITED` 前必须从真实任务与 peers 正向建立 family map，再把 whole-buffer、
runtime/exact/native/fat-kernel、D-tag/backend 和 internal selector 全量反向归类。旧 task 的勾选不继承。

## Capability / gap ledger

尚未裁决。进入 `AUDITED` 时冻结有限 `FIX/BORROW/ROUTE/REJECT` 集；在此之前不修改产品代码、公开
API、layout、算法或常态 metadata。

## Candidate

尚未产生。

## Verification manifest

尚未执行；SCOPED baseline inventory 不是 gate `PASS`。

## 独立红队

只在 candidate 冻结后执行，30 分钟上限。

## Closure decision

尚未裁决。
