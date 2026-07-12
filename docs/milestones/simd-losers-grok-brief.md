# SIMD 剩余真输施工专题报告（grok brief，2026-07-12）

> **给谁**：下个 session 的次模型（grok）先做一遍，opus 后续校对难点。
> **目标**：把三平台符号检验 + objdump 确认的**剩余真输**逐个照 Highway 追平（"有 Highway 参考，HYLE 最差只能平；输=还没照抄到位"）。
> **本 session 已追平（勿重做）**：SaturatedNeg/Abs、MulHigh32、TwoTablesLookupLanes、LoadInterleaved2；LoadNOr 9.6×→1.5×（部分）。

---

## 0. 必守方法学纪律（踩过的坑，违反=白干）

**A. 四道关判"真输 vs 伪影"**（缺一不可，本 session 血泪）：
1. **配对符号检验**：`scripts/bench-stat.py <exe> -n 15 --pair HYLE Highway`，看 `[sign]` 列（不是 `[spread]`）。同机 HYLE/Highway 噪声相关，绝对 spread 会把真输判平；符号检验看 N 次 ratio 是否一致同侧（k/N，p<0.01）。
2. **隔离单-op bench**：完整 bench（190 op 连跑）有相邻 op 串扰。可疑项抽出单独 bench 复测。
3. **objdump 终审**：`g++-14 -O3 -std=c++23 -DHYLE_SIMD_TARGET_AVX2 -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul -I include -isystem third_party/highway -w -S -o x.s probe.cc` 后 `sed -n '/hyle_fn:/,/ret/p' x.s`。**HYLE 与 Highway codegen 逐指令相同 = 伪影，不要改**；不同才是真输。（实例：MaskedApproxReciprocal 三平台 sign 都 WIN 但 objdump 逐指令相同=纯伪影。）
4. **布局偏差阈值 ~2–4%**：`≤1.05×` 的一边倒 HYLE 慢极可能是布局偏差（HYLE/Highway 不同函数符号落不同地址，且 bench 宏 HYLE 恒先定义），除非 objdump codegen 真不同否则不碰。只有 `≥~1.10×` 且 codegen 不同才值得。

**B. 三平台错开跑**（用户 2026-07-12 拍板）：
- 每个性能结论要 **WSL(GCC avx2/sse4) + MSVC(avx2) + Mac(NEON)** 三平台各 sign 验，方向一致才算。
- **WSL 和 Windows MSVC 是同一台物理机**：跑 bench 时**必须错开**（不能 WSL bench 和 MSVC bench 同时跑，会互抢 CPU/缓存）。Mac 是独立机可并行。
- **GCC 追平 ≠ MSVC 追平**：本 session 发现一批 mask op 在 GCC 追平但 MSVC codegen 没优化到位（Set*First/OrderedDemote2MasksTo MSVC 仍 5×）。追平必须每编译器各验。

**C. stale .o 假 PASS 铁律**：改了 header 后 ninja 可能服旧 .o → 测试假 PASS。**每次改后强制删**：`find build-wsl -name 'vec_test.cc.o' -path '*avx2*' -delete` 再 build。测试数不变、明明该变的没变 = 立即怀疑 stale。

**D. 正确性优先于性能**：每个改动先跑对应 vec_test（`--gtest_filter=`），四环境（WSL avx2/sse4 + MSVC avx2/sse4 + Mac NEON）全绿，再谈 sign 追平。带边界的（饱和/溢出/NaN）必须红队测极值。

**E. 提交纪律**：每项独立 commit（正确性+sign 都过后）。commit 信息写清 objdump 前后指令数 + sign ratio 前后。

---

## 1. 环境速查

- **WSL 权威 repo**：`/root/hyle`（ext4）。同步：`rsync -a --delete --exclude "build*" --exclude ".deps" --exclude "_bench_out" /mnt/c/public/project-hyle/hyle/ /root/hyle/`。build 目录 `build-wsl`，`CXX=g++-14 CC=gcc-14 cmake --build build-wsl --target <t> -j`。
- **bench exe**：`build-wsl/bin/hyle_op_parity_{full_bench,bench,widen_bench}_avx2`（sign 用 `_bench_out` 持久目录存 exe，`/tmp` 会被清）。
- **MSVC**：scratchpad 里 `msvc_verify.bat`（VsDevCmd + 编 avx2/sse4 test + 跑）；`msvc_sweep.bat`（编 bench + python sign sweep）。Windows 有 python3.12。
- **Mac**（独立机，`ssh kiro@172.22.10.198`，`export PATH=/opt/homebrew/bin:$PATH`）：`~/hyle` 单向覆盖（tar+scp，rsync 缺）。target `hyle_simd_neon_test` / `hyle_op_parity_full_bench_neon`。
- **op 定义**：`include/hyle/simd/backend/{avx2,sse4,neon}.h`；泛型层 `include/hyle/simd/vec.h`。
- **bench**：`bench/simd/op_parity_{full_bench,bench,widen_bench}.cc`。
- 剩余真输完整清单 + 已追平记录：`docs/milestones/simd-signtest-losers.md`。

---

## 2. 施工卡（按批，含难度与是否留 opus）

### 批 A：MultiRotateRight u64（最慢，2.87ns / 2.9×）——★★难，可先做但难点留 opus 校对
- **位置**：`avx2.h:1944` MultiRotateRight（现 store+reload + N×8 双重标量循环）。bench `op_parity_bench.cc:775`。
- **根因**：纯标量。Highway generic（`third_party/highway/hwy/ops/generic_ops-inl.h:8484` 起，V_SIZE>8）用 `DupEven + TableLookupBytes + Shr<u16>(变量) + ConcatEven` 全向量。
- **子问题**：Highway 那步 `Shr(u16 变量移位)` 在纯 AVX2 无 `_mm256_srlv_epi16`（=AVX512BW），Highway fallback 用 `MulHigh(v, Pow2(16-bits))`（见 `x86_128-inl.h:10594` u16 右移；Pow2 见 `x86_128-inl.h:10352`）。**必须先实现 AVX2 的 u16 变量右移向量化**（同本 session 修 sse4 变量移位的 `v*Pow2` 手法，注意 GCC 把 float→int 溢出折叠成饱和的坑，用内联汇编 cvttps2dq 规避——见 sse4.h `Pow2_u32`），再组装 Highway 的多步。
- **验证**：vec_test 有 MultiRotateRight 测试（搜 `MultiRotateRight`）；bench sign 三平台。
- **难度**：成体系移植（子能力 + 多步）。grok 先搭 u16 变量右移 + 照抄 Highway 结构，**跨 128-lane 的字节 gather 正确性留 opus 校对**。

### 批 B：sub-128 精确宽度二分基础设施 → 解 LoadNOr 剩1.5× / Compress 3× / Gather 2.6×——★★★最重，建议 opus 主导
- **共性**：这三个都需要 Highway 的"按 num_lanes/count 精确宽度 partial load/store"（4+2+1 lane 的 LoadU/StoreU + Combine），HYLE 定宽后端目前缺 sub-128 partial load/store 能力。
- **LoadNOr**：`avx2.h` 已加 native maskload 版（9.6→1.5×）。Highway avx2（`generic_ops-inl.h:2846`）用递归二分（满宽 LoadU 半个 + ConcatUpperLower / Combine），无 maskload（maskload 硬件延迟>普通 load）。追平需实现该二分。
- **CompressStore**：`avx2.h:2544`，HYLE 契约**更强**（精确写 CountTrue 个，不触碰尾部）≠ Highway（写满向量）。**这是主动设计选择**——先与 opus/用户确认是否要改契约对齐 Highway 才追得平；不改契约则记录为"契约优先，保留"。
- **GatherIndexN/NOr**：变长 gather harness，同需精确宽度处理。
- **难度**：这是一整套基础能力，牵扯 tag 系统 / partial 寄存器语义。**grok 不要单独动，留给 opus 设计 sub-128 能力后统一做**。grok 可先只读 Highway 三处实现写调研笔记。

### 批 C：128-cmp 族（Eq128/Ne128/Lt128/Lt128Upper/Min128/Max128，1.02–1.06×）——★中，低优先真输
- **objdump 已定性真差**（非伪影）：Max128 HYLE 11 insn vs Highway 9。差在"把 128-bit 比较结果广播到整 128-lane"：HYLE 用 `vpermq+vpermilpd+vpsrad+vpshufd`（4 条），Highway 用 `vpslldq(8)+vpunpckhqdq`（2 条）。参考 `x86_256-inl.h` 的 Lt128/Max128。
- **改法**：定位 HYLE 128-cmp 底层的"比较结果广播"helper（在 vec.h，搜 `Lt128`/`Eq128`），把 permq+permilpd+psrad+pshufd 换成 pslldq+unpckhqdq。一组 6 个共享底层，修一处连带。
- **风险**：128-bit 大整数高低 64 位组合逻辑易错，exhaustive/red-team 测各种高低位组合（相等/仅高/仅低/进位）。
- **难度**：ratio 小但逻辑易错。grok 可试，**正确性由 opus 校对**（128-cmp 语义微妙）。

### 批 D：66 覆盖缺口（补 bench 对拍，非修输）——★易，grok 主力
- 见 subagent 审计（`docs/milestones/simd-signtest-losers.md` 未列全，重跑审计或见本报告附录）。真缺口：**block swizzle 族**（ConcatLowerLower/Upper、ZipLower/Upper、Combine、BroadcastBlock、ExtractBlock、InsertBlock、SwapAdjacentBlocks、ReverseBlocks、OddEvenBlocks、Interleave*Blocks）、**byte/lane shifts**（ShiftLeftBytes/RightBytes/LeftLanes/RightLanes）、**Masked FMA**（MaskedMulAdd/MulAddOr/NegMulAdd）、**ConvertInRange/PromoteInRange 族**、**SAD/sums**（SumsOf2/4、SumsOf8AbsDiff）、**bf16 dot**（Bf16EvenToF32/OddToF32、MulEvenAdd/OddAdd）、**Interleave3To/4To/From**（HYLE 自造 lowering，最值得对拍）。
- **做法**：在 `op_parity_full_bench.cc` / `op_parity_bench.cc` 加对拍行（`hs::Op` vs `hn::Op`，仿现有 UNARY_I/BINARY_I 宏），跑 sign。**先补 bench 揭示是否有隐藏真输**，是真输再进上面的修复流程。
- **难度**：机械补 bench 行，grok 主力。每个补完先 sign 看是否又揪出真输。

### 批 E：MSVC 独有大输（GCC 已平 MSVC 没有）——★★中，需 MSVC objdump
- Set{Only,Before,AtOrAfter,AtOrBefore}First 5×、OrderedDemote2MasksTo 4.6×、Reverse2/4/8 2.8×、Dup128MaskFromMaskBits 5.4×（**这些 WSL GCC 已追平**）。
- **根因待查**：同一份 C++，MSVC codegen 没 GCC 优化到位。需在 **MSVC 上 objdump**（`dumpbin /disasm` 或 MSVC `/FA`）对比 HYLE/Highway，看 MSVC 在哪一步没内联/没优化，可能要给 MSVC 加显式 intrinsic 或调整写法。
- **难度**：MSVC 工具链 + codegen 分析。grok 可先 dumpbin 收集证据，**修法留 opus**（MSVC 特定优化易反复）。

---

## 3. grok 边界（明确交接）
- **grok 主做**：批 D（补 bench 覆盖，机械+高价值揪隐藏真输）、批 A 的 u16 变量右移子能力搭建、批 C 试改 + 批 E 的 MSVC dumpbin 证据收集。
- **留 opus 校对/主导**：批 B（sub-128 基础设施，牵扯 tag 系统）、CompressStore 契约决策、批 A 跨 128-lane 字节 gather 正确性、批 C 128-cmp 语义、批 E MSVC 修法。
- **每项完成必做**：四道关判真输 + 三平台错开验 + 强制删 .o 防 stale + 独立 commit（写 objdump 前后 + sign 前后）。
- **遇到 codegen 逐指令相同 / ≤1.05× 一边倒**：记为伪影，**不要硬改**，写进 losers doc 的"伪影"区。

---

## 4. 起手句（给 grok）
"按 `docs/milestones/simd-losers-grok-brief.md` 施工：先做批 D（补 block-swizzle/byte-shift/masked-FMA 等 66 覆盖缺口的 Highway 对拍 bench 行，每补一批用 `bench-stat.py -n15 --pair HYLE Highway` 看 `[sign]` 揪隐藏真输），再搭批 A 的 AVX2 u16 变量右移向量化子能力。严守四道关（符号检验/隔离/objdump/布局阈值）+ 三平台错开 + 强制删 .o。codegen 逐指令相同或 ≤1.05× 一边倒记为伪影不改。sub-128 基础设施/CompressStore 契约/128-cmp 语义留 opus。"
