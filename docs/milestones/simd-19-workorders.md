# simd-19 分派施工手册(工单制,面向中上能力 AI 执行)

> **这份文档是什么**:把 `simd-19-partial-vector-repr.md` §5.1(常绿垫片迁移)拆成可独立执行、可独立
> 验证的工单卡,交给中上能力 AI 流水线执行。设计原则:**认知密度前置**(设计判断全部在卡片里做完),
> **执行零判断**(每步只有"照做 + 过闸门 + 记台账"),**错误不可累积**(每卡独立提交,闸门不过即 revert)。
> **怎么用**:先由高能力方完成 W0 批次(见 §5),然后每次把 §8 的交接 prompt + 一个卡号交给执行 AI。

## §1 分工原则:什么不可下放

| 工作 | 谁做 | 为什么 |
|---|---|---|
| W0:traits + 垫片 + 包裹脚本 + `static_assert` 矩阵 + 映射表 + 逐卡核对现状代码 | **高能力方**(最好的 AI 或用户) | 类型系统错一处全线崩;包裹脚本的正确性决定全局;卡片里的 before 代码必须与真实代码逐字一致 |
| W1~W3:热点 op 升 native-D(~12 卡) | 执行 AI | 形状全同、映射表给定、闸门客观 |
| W4:half↔full 边界 op(先补 oracle 再改) | 执行 AI(**盯紧**) | 最易出正确性 bug,靠"oracle 测试先行"卡死 |
| W5:收口(全量夜跑/三平台/文档回写) | 执行 AI 跑,**高能力方审** | 跑是机械的,判读不是 |
| 一切 escalation 裁决、任何触碰 vec.h/契约的决定 | 高能力方 | 铁律 §2-1 |

## §2 铁律(执行 AI 的八条约束,违反任一 = 该卡作废)

针对的是中级 AI 的八种典型失败模式,逐条对应:

1. **白名单之外只读**。每张卡列出允许触碰的文件与函数区间;`tests/`、`include/hyle/simd/vec.h`、
   其他 backend(`sse4/avx2/avx512/scalar/sve/rvv.h`)、`bench/` **一律只读**。diff 越界 = 拒收。
   (防:范围蔓延、顺手重构、契约漂移。)
2. **intrinsic 只准用卡片映射表里的**。表里没有 → STOP,写报告,不准凭记忆发明。不确定某 intrinsic
   是否存在,用 §3-5 的存在性探针验证,探针不过 = 表有误 → STOP 上报。(防:幻觉 intrinsic。)
3. **gtest 红 = revert 本卡全部改动 + 报告**。严禁改测试、改 oracle、改期望值、加 `GTEST_SKIP`。
   测试是仲裁者,不是障碍物。(防:改测试过测试。)
4. **同一错误最多修 2 次**。第 2 次修复仍失败 → revert + 报告,不准"再换个思路试试"。(防:卡住时
   创造性发挥,越挖越深。)
5. **完成的唯一定义 = 五闸门原始输出全部粘贴在报告里 + 台账已更新 + 恰好一个提交**。没有原始输出的
   "已通过"不算数。(防:跳过验证、谎报。)
6. **一卡一提交**,消息格式 `perf(simd): s19-<卡号> <op> D-form`(W4 用 `fix`/`test` 前缀,卡片注明)。
   开工前 `git status` 必须干净;结束时要么 1 个提交要么 0 个(revert 干净)。禁 amend、禁 rebase、
   禁并行做多卡。(防:大爆炸提交、状态污染。)
7. **构建/测试命令逐字用 §3 的**,不自创、不"优化"。禁止在 Mac 上跑全量 bench(jetsam 会 SIGKILL,
   见 §3-4 只跑单 op)。(防:环境坑。)
8. **报告诚实优先**:闸门失败、数字变差、没做完,都直接写。任何"我觉得应该没问题" = 未验证 = 没过。
9. **开工树不干净 → `git stash` 后在干净基线开工,或 STOP 上报**;禁止在来历不明的未提交改动上继续
   (2026-07-15 C3 批复补充)。
10. **控制变量生效要自检**:环境变量/开关类控制,必须用**输出特征**验证生效(如换序 = HWY# 行在前),
    命令执行成功不等于控制生效(2026-07-15 C12 env 名教训)。

铁律 6 修订(2026-07-15 批复 #2):代码+台账状态 = 1 个实质提交;允许再加 1 个纯 hash 回填的 docs
提交。其余不变。

## §3 环境与命令(全部可复制粘贴,在 Windows git-bash 执行)

权威源 = Windows `C:\public\project-hyle\hyle`;Mac(`kiro@172.22.10.198`,免密)= 可整体覆盖的执行镜像。

**3-1 同步(Windows → Mac,单向覆盖,保留 Mac 的 build-mac/.deps):**
```bash
tar -C /c/public/project-hyle/hyle -cf - --exclude=build --exclude=.git --exclude='*.obj' \
    include tests bench src CMakeLists.txt cmake third_party/highway \
  | ssh -o BatchMode=yes kiro@172.22.10.198 'tar -C ~/hyle -xf -'
```

**3-2 构建 + 单元测试(闸门 G1/G3 用):**
```bash
ssh -o BatchMode=yes kiro@172.22.10.198 \
  'export PATH=/opt/homebrew/bin:$PATH && cmake --build ~/hyle/build-mac --target hyle_simd_neon_test -j \
   && ~/hyle/build-mac/bin/hyle_simd_neon_test'
```
构建后**必须**确认二进制是新的(stale-exe 教训):
```bash
ssh -o BatchMode=yes kiro@172.22.10.198 'ls -la ~/hyle/build-mac/bin/hyle_simd_neon_test'
# mtime 必须晚于你本次同步时间,否则 = 没编进去,STOP 排查,不准拿旧二进制的绿说事
```

**3-3 单 op bench(闸门 G4,仅卡片带 bench 字段时):**
```bash
ssh -o BatchMode=yes kiro@172.22.10.198 \
  'export PATH=/opt/homebrew/bin:$PATH && cmake --build ~/hyle/build-mac --target hyle_op_parity_bench_neon -j \
   && HYLE_BENCH_ONLY=<卡片给定的过滤串> ~/hyle/build-mac/bin/hyle_op_parity_bench_neon'
```
**禁止**不带 `HYLE_BENCH_ONLY` 跑(全量 2227 行会被 jetsam SIGKILL);**禁止**多个 bench 进程并发。

**3-4 全量 bench(只有 W5 收口卡用,夜跑):**
```bash
ssh -o BatchMode=yes kiro@172.22.10.198 'export PATH=/opt/homebrew/bin:$PATH && nohup sh -c \
  "~/hyle/build-mac/bin/hyle_op_parity_bench_neon > /tmp/s19_full_1.txt 2>&1" &'
# 完整性判定:wc -l = 2227 才算完整,短的是被 kill 的,作废重跑
```

**3-5 intrinsic 存在性探针(对映射表有疑问时,唯一合法的验证方式):**
```bash
ssh -o BatchMode=yes kiro@172.22.10.198 \
  'printf "#include <arm_neon.h>\nauto f(int16x4_t a,int16x4_t b){return vadd_s16(a,b);}\n" > /tmp/p.cc \
   && clang++ -fsyntax-only -std=c++20 /tmp/p.cc && echo EXISTS'
# 把函数体换成待验证的 intrinsic;输出 EXISTS = 存在;报错 = 不存在,STOP 上报
```

## §4 验证闸门(每卡统一)

| 闸门 | 内容 | 通过标准 | 失败动作 |
|---|---|---|---|
| G1 编译 | §3-2 构建 | exit 0,且二进制 mtime 新 | 按铁律 4 处理 |
| G2 类型 | W0 埋的 `static_assert` 矩阵(随 G1 编译自动执行) | 编译过即过 | 同上 |
| G3 正确性 | `hyle_simd_neon_test` **全绿**(不是只跑相关 case——防连带回归) | 0 failed | **revert + 报告**(铁律 3) |
| G4 性能 | 仅卡片带 bench 字段时,§3-3 单 op | **只记录数值,不做判定**;数值写进台账 | 无(判定统一在 W5 收口做,防执行 AI 为凑数瞎调) |
| G5 台账 | 更新 `simd-19-ledger.md` 本卡行 + 提交 | 报告含台账 diff | — |

W4 额外闸门 **G3.5 穷举 oracle**:边界 op 的专项测试必须先于实现改动存在(见 W4 卡序);oracle 用**标量
参考实现**逐 lane 对拍,严禁照抄向量实现自证。

## §5 工单批次与卡片清单

> 卡片正文(before 代码、精确 diff 形状、白名单行区间、本卡映射表条目)由 W0 生成,追加到本文档 §6 之后。
> 下面是批次结构与卡目录。执行顺序严格按卡号;**W3 完成即收口 PromoteMaskTo——奖品时刻放中间,尽早证明
> 路线,不押到最后**。

**W0(高能力方,先行,~1-2 天)——树绿基线**
- `RawNeon<T,N>`/`MaskRawNeon<T,N>` traits + `to_q`/`from_q` 垫片 + 全 T×N `static_assert` 矩阵;
- 机械包裹脚本(所有 op 体 `x.raw`→`to_q(x)`、赋值→`from_q<T,N>(…)`),跑完全树编译绿 + gtest 绿;
- 生成 C1~C19 卡片正文(逐卡核对 neon.h 现状代码)+ 台账初始化 + 打 tag `s19-w0-green`。

**W1(执行 AI)——最机械的 4 张**
- C1 `Set/Zero` D-form;C2 partial `LoadU`(`vld1_*`,注意:partial 只准读 N×sizeof(T) 字节,不准满宽读);
  C3 partial `StoreU`(同上,只写 N 宽);C4 `Iota/Dup` 族(若卡片核对后确认存在 partial 路径)。

**W2(执行 AI)——比较与 mask 桥**
- C5 比较族(`Eq/Ne/Gt/Ge/Lt/Le`,D-form:`vceq_*`/`vcgt_*`/…);C6 `MaskFromVec`/`VecFromMask`/
  `MaskFromNormalizedVec`;C7 mask 逻辑族(`And/Or/Xor/AndNot`)。

**W3(执行 AI)——本里程碑的靶心**
- C8 `PromoteMaskTo` 本体(D 入:`vshll_n`/sshll 一条);C9 `DemoteMaskTo` 本体(`vmovn` 方向);
  C10 `PromoteLowerTo/PromoteUpperTo` 输入侧;C11 `DemoteTo` 输出侧;
- **C12 收口卡(带 bench 字段)**:`HYLE_BENCH_ONLY=PromoteMaskTo` 6 遍(双序),数值贴台账。参考:
  现状 0.575,目标进入 gate 0.06 的 TIE 带。判定由高能力方做。

**W4(执行 AI,盯紧)——half↔full 边界(先测后改)**
- C13 补穷举 oracle 测试:`LowerHalf/UpperHalf/Combine/ZeroExtendVector/Concat*`(标量参考逐 lane,
  全 T×N 组合;此卡只动 `tests/simd/` 的**新增**文件,是白名单唯一例外,卡片单列);
- C14~C18:上述五族逐个从"满寄存器存低 64"改真 D↔Q(`vget_low`/`vcombine`)。每卡先确认 C13 测试
  对该 op 是绿的(旧约定下也必须绿),改完仍绿。

**W5(执行 AI 跑,高能力方审)——收口**
- C19:全量 bench 夜跑 ×3(双序)+ `hyle_simd_neon_test` + Windows 侧 avx2/sse4/scalar gtest 复跑
  (partial traits 动的是 neon.h,x86 理论无关,但要证明)+ cloc 增长统计。产物全部附台账,
  判读与矩阵/文档回写由高能力方做。

## §6 卡片模板(W0 按此生成;下为**示意卡**,真实卡内嵌真实代码)

```markdown
### C5 比较族 D-form
- 目标:neon.h 中 Eq/Ne/Gt/Ge/Lt/Le 六个函数的 partial 分支摘垫片,换 D-form。
- 白名单:include/hyle/simd/backend/neon.h 第 NNN~NNN 行(六个函数体);其余只读。
- before(现状,垫片形态,以 Gt<i16> 为例):
    m = from_q<...>(vcgtq_s16(to_q(a), to_q(b)));
- after(目标形状):
    if constexpr (kIsD<T,N>) m.raw = vcgt_s16(a.raw, b.raw);   // D
    else                      m.raw = vcgtq_s16(a.raw, b.raw);  // Q(原样)
- 本卡映射表(唯一合法来源):
    vceqq_s8→vceq_s8  vceqq_s16→vceq_s16  vceqq_s32→vceq_s32  vceqq_f32→vceq_f32 …
    (64-bit lane N=1:vceq_s64 存在;f64 N=1:vceq_f64 存在——已用 §3-5 探针验证)
- 闸门:G1→G2→G3→G5(本卡无 bench 字段)。
- STOP 条件:某类型的 D-form 不在映射表且探针失败;需要触碰六个函数之外的任何代码。
- 预算:≤1.5h;同一错误修复 ≤2 次。
```

## §6a W0 卡片(高能力方执行;每步一个提交、每步全绿)

### W0a 表示层 + 垫片(✅ 已完成,提交 `6414273`,2026-07-15)
- 已落地:`RawNeonD/MaskRawNeonD` 10 型特化、`kIsDBand<T,N>`、`RawNeonNT/MaskRawNeonNT` 分档 alias、
  33+ 条 `static_assert` 矩阵、`to_q(vec/mask)`/`from_q<V>(q)` 垫片(终态形式,现仅恒等分支实例化)。
- 闸门实录:Mac 构建 exit 0、二进制 mtime 新、`hyle_simd_neon_test` 220/220 绿。行为零变化(未切换)。
- 注意:垫片签名与计划文档的 `from_q<T,N>` 不同,实际是 **`from_q<decltype(x)>(expr)`**(目的类型化,
  机械包裹时不需要局部类型知识)。后续卡片一律按此写。

### W0b 机械包裹(脚本 + 人审 diff;完成后树仍绿、行为仍零变化)
1. **先做 `.raw` 全量清点**(改代码前,产出清单存 oplab):
   ```bash
   grep -n "\.raw" include/hyle/simd/backend/neon.h > oplab/s19_raw_sites.txt   # ~650 处
   grep -n "&[a-zA-Z_]*\.raw\|sizeof(.*raw" include/hyle/simd/backend/neon.h    # 危险点:取址/sizeof
   grep -n "VecNeon<[^>]*>{\|VN<D>{\|MN<D>{" include/hyle/simd/backend/neon.h   # 绕过 .raw 的聚合初始化
   ```
   **三类危险点必须逐条人工定处置**(flip 后 `sizeof(raw)`/取址语义会变!):取址→改为
   `auto q = detail::to_q(x); memcpy(…, &q, …)`;`sizeof(x.raw)`→按旧语义写死 16 并加注释;聚合
   初始化→改 `detail::from_q<V>(…)`。
2. 写变换脚本 `oplab/s19_wrap.py`,两条规则:
   - R1(读):表达式位置的 `X.raw` → `detail::to_q(X)`(X 为标识符;函数已在 detail 内也写
     `detail::to_q`,查找会命中封闭命名空间,两处都合法)。
   - R2(写):语句 `X.raw = EXPR;` → `X = detail::from_q<decltype(X)>(EXPR);`(先匹配 R2 再 R1,
     同一行读写共存时 R2 的右侧再走 R1)。
   - **匹配不上的 `.raw` 一律不改,输出到例外报告**;人工处理后重跑,直到例外报告为空。
3. 例外:`to_q`/`from_q`/`VecNeon`/`MaskNeon` 定义本身、W0a 新增区域,脚本跳过(行号白名单)。
4. 闸门:①Mac 构建 + 220 gtest 绿;②**代码生成同一性抽查**(此步 raw 类型未变,垫片是恒等,二进制
   应功能等价):`objdump -d` 抽 3 个热 op(Add/Gt/PromoteMaskTo 实例化处)与包裹前 diff,只准有
   符号地址差,不准有指令差;③`HYLE_BENCH_ONLY=PromoteMaskTo` 单 op 数值记台账(应与现状同)。
5. 提交:`refactor(simd): s19-w0b route all neon raw access through evergreen shims (no-op)`。

### W0c 翻转(一行级 diff,风险集中点)
1. `VecNeon<T,N>::raw` 类型改 `detail::RawNeonNT<T, N>`;`MaskNeon<T,N>::raw` 改
   `detail::MaskRawNeonNT<T, N>`。
2. 构建。预期两类残错:W0b 例外清单漏网的 `.raw` 直用;跨 T/N 传 raw 的边界 op。**只准用两个垫片修**,
   禁止就地写 `vget_low`/`vcombine`(那是 W1+ 卡的事,W0c 只求"全部经垫片、语义等价")。
3. 闸门:①220 gtest 绿(**这是 flip 的真闸门**);②Windows 侧 avx2/sse4/scalar gtest 复跑(理论无关,
   要证明);③`HYLE_BENCH_ONLY=PromoteMaskTo` 数值记台账(允许移动,记录即可,不判定);④打 tag
   `s19-w0-green`。
4. 提交:`feat(simd): s19-w0c flip VecNeon/MaskNeon raw to N-banded types`。

### W0d 生成 C1~C19 卡片正文 + 台账
1. 逐卡打开 neon.h 现状(post-W0c)函数体,把**真实代码**贴进卡片 before 栏;写 after 形状
   (`if constexpr (detail::kIsDBand<T,N>) … D-form … else … 原 Q-form …`,摘掉该函数内的垫片调用)。
2. 每卡映射表逐条过 §3-5 存在性探针(尤其 64-bit lane 的 D-form 与 `vqtbl`/reduce 类,别凭记忆)。
3. 初始化 `simd-19-ledger.md`(§7 格式,C1~C19 全 `todo`);卡片正文追加到本文档 §6 之后。
4. 提交:`docs(simd): s19-w0d generate work cards C1-C19 + ledger`。

## §6b 卡片正文 C1~C19(W0d 生成,2026-07-15;行号锚以 `s19-w0-green` tag 为准)

通用规则(每卡适用,不再重复):白名单 = 指定函数体;改法 = 在函数内加
`if constexpr (detail::kIsDBand<T, N>) { …D 分支,直接用 x.raw(已是 D 类型)… } else { …保留现有
包裹形… }`;D 分支**不得**调用 `to_q/from_q`(直接 raw 对 raw);Q 分支不动。闸门 G1→G2→G3→G5
(§4);G4 仅 C12/C18。映射表疑问用 §3-5 探针。**发现测试红一律 revert,严禁改 tests/ 已有文件。**

### C1 Set/Zero(neon.h:252/271)
- before(Set):`v = detail::from_q<decltype(v)>(vdupq_n_f32(…))` 等 10 分支;Zero 转发 Set。
- after:kIsDBand 时 `v.raw = vdup_n_f32(…)` 等(q 后缀去掉)。
- 映射:`vdupq_n_{f32,f64,s8,u8,s16,u16,s32,u32,s64,u64}` → `vdup_n_*`(全部存在,含 vdup_n_f64)。

### C2 LoadU/StoreU 边界护栏测试(先行;白名单 = tests/simd/ 新增文件 vec_partial_boundary_test.cc)
- 内容:①StoreU 尾部护栏——对每个 partial 形状 (T,N),在 N*sizeof(T) 后填哨兵字节,StoreU 后断言
  哨兵未被改写;②LoadU 只读边界——把源放在页尾(mmap PROT_NONE 下一页,或 asan 下堆尾),LoadU 不崩。
- **预期:当前实现下①必红**(既有满宽写)。此卡验收 = 测试**存在且红**(红名单写进台账),C3/C4 修
  复后转绿。这是全手册唯一"以红为过"的卡。
### C3 LoadU partial 窄读(neon.h:279)
- after:kIsDBand 且 `N*sizeof(T)==8` → `vld1_*`(8 字节);`<8` → `detail::RawNeonD<T>::type tmp =
  vdup_n_*(0); std::memcpy(&tmp, p, N*sizeof(T)); v.raw = tmp;`。
- 映射:`vld1q_*` → `vld1_*`(10 型全存在)。C2 的②转绿。
### C4 StoreU partial 窄写(neon.h:298)
- after:`==8` → `vst1_*`;`<8` → `std::memcpy(p, &v.raw, N*sizeof(T))`。C2 的①转绿。
- 映射:`vst1q_*` → `vst1_*`。

### C5 比较族(neon.h:1578-1658 六函数)
- 映射:`vceqq/vcgtq/vcgeq/vcltq/vcleq_*` → `vceq/vcgt/vcge/vclt/vcle_*`;Ne = Eq+Not 时用
  `vmvn_*`(u64 无 vmvn_u64:用 `veor_u64(x, vdup_n_u64(~0ull))`,探针验证)。
- 64-bit lane D-form(vceq_s64/vcgt_s64/vceq_f64 等)AArch64 存在——逐条探针后再写。
### C6 MaskFromVec / MaskFromNormalizedVec / VecFromMask(neon.h:1856/1883 附近/1835)
- 纯 reinterpret + (MaskFromVec 的) 归一化移位:`vreinterpretq_*` → `vreinterpret_*`;
  `vshrq_n/vcltzq` 类归一化 → `vshr_n/vcltz_*` D 形。
### C7 mask 逻辑族(neon.h:1938-1958)
- 映射:`vandq/vorrq/veorq/vbicq/vmvnq_u{8,16,32,64}` → `vand/vorr/veor/vbic/vmvn_*`
  (vmvn 无 64-bit:同 C5 的 veor 反转)。

### C8 Promote 族 partial 输入 D 路(neon.h:1094 PromoteLowerTo 及其被 vec.h PromoteTo 调用的路径)——**靶心**
- 现状(包裹后):`vmovl_s16(vget_low_s16(detail::to_q(v)))` 形。
- after:输入是 D band 时直接 `vmovl_s16(v.raw)`(= sshll,输入本来就是 D!)——去掉 to_q+vget_low。
  输出宽形若仍 partial(如 i8×4→i16×4)则结果也是 D:`vget_low(vmovl_*(…))` 或用 `vmovl` 后 from_q,
  卡内按 (SrcT,DstT,N) 逐组写死,不留自由发挥。
- 完成后手跑 `HYLE_BENCH_ONLY=PromoteMaskTo` 记台账(应显著向 1.0 收敛——链条 ldr d + sshll + cmgt.4s)。
### C9 DemoteTo partial 输出 D 路(neon.h:1123)
- 现状:`vqmovn/vmovn` 产 D 后走 zext-Q 约定 + from_q。after:输出 D band 时直接 `r.raw = vqmovn_*(…)`。
### C10 向量逻辑族(neon.h:754-780 + Not:2763)
- 同 C7 映射,向量版(含 f32/f64:按位用 vreinterpret 到整型再回,照现有 Q 分支形状抄成 D)。
### C11 lane 存取(GetLane/ExtractLane/InsertLane,grep 定位)
- 映射:`vgetq_lane_*` → `vget_lane_*`;`vsetq_lane_*` → `vset_lane_*`(lane 索引 <N 不变)。

### C12 收口采集(无代码)
- `HYLE_BENCH_ONLY=PromoteMaskTo` 双序各 3 遍(§3-3;两序 = 正常跑 + **`HYLE_BENCH_HWYFIRST=1`**
  ——注意无第二个下划线,engine.h:83;换序生效的自检:输出里 HWY# 行应在 HYLE# 前),原始输出全贴
  台账。判定高能力方做。(2026-07-15 教训:本卡初版写错 env 名致执行方三遍换序无效;env 生效性要用
  输出特征自检,不能只看命令跑没跑成。)

### C13 half/combine 穷举 oracle(白名单 = tests/simd/ 新增 vec_half_combine_oracle_test.cc)
- LowerHalf/UpperHalf/Combine/ZeroExtendVector/InsertIntoUpper × 全 (T,N) 合法组合,标量参考逐 lane
  对拍(参考实现手写数组语义,禁止调被测 op)。**当前实现必须全绿**(这是改前基线),然后 C14-C16
  改动全程保持绿。
### C14 LowerHalf/UpperHalf native D(neon.h:3026-3043)
- after:输入满宽时 `out.raw = vget_low_*(v.raw)` / `vget_high_*(v.raw)`(直接产 D,零指令/一条);
  输入 partial(N≤半宽)保留包裹形(冷路径)。
### C15 Combine/ZeroExtendVector/InsertIntoUpper native(neon.h:3045-3060)
- after:两输入均 D 时 `out.raw = vcombine_*(lo.raw, hi.raw)`;ZeroExtendVector = `vcombine_*(lo.raw,
  vdup_n_*(0))`。输出必满宽(N*sizeof(T)==16 才有意义)——静态断言现有约束不变。
### C16 Concat{LowerLower,LowerUpper,UpperLower,UpperUpper} partial 路径(grep 定位)
- 输入 partial 时经 C14/C15 的 D 原语组合;满宽路径不动。改前先确认矩阵中这些 op 现为 TIE,改后
  `HYLE_BENCH_ONLY=Concat` 冒烟无回退。
### C17 ResizeBitCast 审计(neon.h:3062 附近,低优)
- 现状已翻转安全(Q 临时量)。审计项:partial→partial 小尺寸时可换 exact-byte memcpy 省一次 Q 物化;
  不值得就台账记 `no-change` 关卡。

### C18 全量收口采集(高能力方判读)
- 同步后夜跑:三个 NEON bench 目标,双序各 3 遍,`nohup` 串行(§3-4;`wc -l` 验完整);gtest 220;
  输出文件名 `s19_<tu>_<order>_<n>.txt` 全部留存。
### C19 文档回写(高能力方)
- 矩阵 PromoteMaskTo 行、simd-19 doc 状态/验收节、cloc 增长、本台账收尾。

## §7 台账(`docs/milestones/simd-19-ledger.md`,W0 初始化)

```markdown
| 卡 | op | 状态 | G1 | G3 | G4(数值) | 提交 | 备注 |
|---|---|---|---|---|---|---|---|
| C1 | Set/Zero | done | ✓ | 220/220 | — | abc1234 | |
| C2 | LoadU | blocked | ✓ | 3 failed→revert | — | — | 上报:… |
```
状态只有四种:`todo / done / blocked / reverted`。`blocked/reverted` 必须带上报摘要。执行 AI 每卡
结束时更新本文件(台账更新与代码同一个提交)。

## §8 交接 prompt(逐字粘给执行 AI,替换 <卡号>)

```
你是 simd-19 施工 AI。唯一任务:执行 docs/milestones/simd-19-workorders.md 中的工单卡 <卡号>。
先读该文档 §2 铁律、§3 命令、§4 闸门、你的卡片正文,再开工。规则(违反任一 = 本卡作废):
1. 只改卡片白名单内的文件/行区间;tests/(除卡片明示的新增文件)、vec.h、其他 backend、bench/ 只读。
2. intrinsic 只准用卡片映射表里的;表里没有→用手册 §3-5 探针验证;探针失败→停,写报告。
3. gtest 红→revert 本卡全部改动并报告;严禁改测试。
4. 同一错误最多修 2 次,仍败→revert+报告。
5. 完成 = 五闸门原始输出全部粘贴 + 台账更新 + 恰好 1 个提交(格式见铁律 6);缺任一项不算完成。
6. 开工前 git status 必须干净;不并行做卡;不 amend;构建/测试命令逐字用 §3,禁全量 bench。
7. 报告诚实优先:失败/变差/没做完直接写。
产出格式:①做了什么(3 行内)②五闸门原始输出 ③提交 hash ④台账 diff ⑤(若有)STOP 上报。
```

## §9 Escalation 与收口

- 执行 AI 的一切 `blocked` 上报、G4 数值判读、W4 的任何正确性疑点、文档/矩阵回写 → 高能力方处理。
- 收口标准 = `simd-19-partial-vector-repr.md` §六(0.575→TIE 校准判定、三平台 gtest、全量复跑无回归、
  cloc 回写、行数警戒线确认)。
- 若 W1~W3 全部 done 而 C12 判定未达 TIE 带 → 高能力方按矩阵的证据阶梯(逐指令 diff 起步)接手,
  执行 AI 不参与性能归因。
```
