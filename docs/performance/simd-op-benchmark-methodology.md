# SIMD op 对拍方法论——如何科学地 benchmark 一个 SIMD op

> 面向后续给 op_parity 新增测试条目、复跑数据、或判定"某 op 是否真慢"的贡献者。本文是**操作规程**;
> 每条规则背后的实验证据链在 [`../debug/op-parity-harness-artifacts-hunt.md`](../debug/op-parity-harness-artifacts-hunt.md)(三层测量伪影的定位与仪器重建全史),
> 当前权威数字在 [`../milestones/simd-orderswap-matrix.md`](../milestones/simd-orderswap-matrix.md)。
> 基础设施:`bench/simd/op_parity_engine.h`(测量引擎)、`bench/simd/op_parity{,_full,_widen}_bench.cc`(三个对拍 TU)、`scripts/op-parity-report.py`(聚合+校准闸门)。
>
> **一句话版本:亚 ns 级 SIMD 循环的单次读数不是数据;只有"锚点校准通过 + 多副本相位采样 + 双序一致"的聚合读数才是数据;而一个 LOSE 判决要变成"真算法输"的定论,还需要过一条证据阶梯。**

---

## 0. 为什么不能"随便写个循环测一下"

这套流程的每个部件都对应一类被实测坐实的伪影,量级全部**大于常见的真实差距**(±10%~40%,而我们要分辨的差距常在 5%~15%):

| 伪影 | 量级 | 机制 | 引擎对策 |
|---|---|---|---|
| P/E 核调度迁移 | ±10~30%/次 | 13900K 混合架构,不钉核时两侧测量落在不同核/频率 | 进程内 `pin_and_boost()`(P 核亲和+高优先级;mac 无亲和性 API,用 QoS+8× epoch) |
| 代码放置运气 | ±10~15%,尾部 ~30% | 同一二进制里**逐指令相同**的两个循环因 uop-cache/取指几何差异稳定不同速;每次重编译重掷 | 每侧 4 个独立实例化副本 + `stagger(rep)` 确定性相位错位(0/16/32/48)+ 防 ICF 唯一立即数 |
| 副本相位相关 | 单 op ~10% | 同尺寸副本被编译器等差排布,步长为 64 倍数时 4 个副本共享同一相位=同一次掷骰 | stagger 强制两侧都扫 4 相位 |
| 闭包栈-堆 4K 别名 | ~25~40%,随启动器变 | kernel 每迭代重载捕获链,闭包栈地址由环境块大小决定 | kernel 内指针外提(`auto* p_X = X.data()`) |
| 超小循环前端超敏感 | ±50% | ≤5 指令的循环体对放置极端敏感 | 宏形状循环 4× 展开 |
| mac 环境定时器 | err% 4~28% | ~1.6ms epoch 撞驻留应用的 ms 级定时器爆发 | `__APPLE__` epoch 8×(env `HYLE_BENCH_EPOCH_SCALE`) |

**判断仪器是否可信的唯一标准是校准闸门**:报告首段的锚点 op(源码级验证两侧同指令,真值=1.00)
必须全部落在闸门内(x86 ±4%,mac ±6%),否则整表拒绝引用——不要看任何后面的数字。

---

## 1. 新增一个测试条目

### 1.1 kernel 的强制形状

放进 `HYLE_RUN_PAIR(bench, <HYLE kernel>, <Highway kernel>)`,两个 kernel 都必须是这个形状:

```cpp
[&](auto rep) {
  hyle_bench::stagger(rep);                 // ① 必须是第一条语句(相位错位)
  auto* p_src = src.data();                 // ② 所有 .data() 外提为局部指针(斩断捕获链)
  auto* p_dst = dst.data();
  for (std::size_t i = 0; i + w <= kN; i += w)   // ③ 主循环(见 1.2 的形状规则)
    hs::StoreU(hs::TheOp(hs::LoadU(d, p_src + i)), d, p_dst + i);
  nb::doNotOptimizeAway(p_dst);             // ④ 防 DCE
  hyle_bench::mark(rep);                    // ⑤ 必须是最后一条(防 ICF 折叠副本)
}
```

规则与理由:
- **`(auto rep)` 泛型参数不可省**——引擎靠它实例化 4 个不同地址的副本;写成普通 `[&]{}` 会静默退化成单副本(放置运气一次定生死)。
- **`stagger(rep)` 第一、`mark(rep)` 最后**,两者都不可省。省 mark → 链接器 ICF 把副本折叠回同一地址;省 stagger → 副本相位相关(9b 节漏洞回归)。
- **指针外提**:kernel 里出现任何 `X.data()+i` 内联访问都算错——MSVC 无法证明 store 不改写闭包,会在热循环里每迭代重载捕获链,并暴露在栈-堆 4K 别名下(同一 exe 换启动器数字就变)。
- 两侧 kernel 除 `hs::`/`hn::` 与各自 tag 外**文本必须对称**——循环骨架、buffer、sink 全部一致。不对称的骨架测的是骨架不是 op。

### 1.2 循环形状规则

- **走宏优先**:UNARY_I/BINARY_I/UNARY_F/BINARY_F/TERNARY_F/CMP_I 已内置 4× 展开和全部形状规则,
  能套宏就套宏(full_bench 的主体)。
- 手写 kernel 时,**若 op 本体 ≤ ~5 指令,必须 4× 展开循环体**(超小循环处于前端放置超敏感区,
  副本散布会大到判决全变 UNSTABLE)。
- kN=4096(L1 驻留,隔离计算吞吐)。改 kN 前想清楚:变大=测内存墙,变小=测循环开销。
- 有第三方对照行(xsimd/store+reload 基线)用 `hyle_bench::run_third(bench, "label", [&]{...})`——
  单实例、无判决,只做信息展示。

### 1.3 对拍必须契约对等

**pair 里的两个 op 必须是同一契约**,否则测的是语义差价不是实现差价:
- HYLE 默认版语义更强时(如 TableLookupBytes 的 &15 全定义),pair 用契约对等的变体
  (TableLookupBytesInRange vs Highway),强语义默认版放 run_third 行让标价可见。
- 语义有分歧先做决策(参考 DemoteToNearestInt 案:上游自己都不统一的语义不值得付钱模拟,
  见矩阵演进段),再让 pair 逐字节可比。
- 前置条件类差异(in-range/nonzero/对齐)要么两侧同置,要么分家成 InRange 变体
  (命名沿用 Highway 官方词汇 `*InRange`)。

### 1.4 新锚点的资格

想把某 op 加进校准锚点集(`scripts/op-parity-report.py` 的 `ANCHORS`):
- 必须**源码级逐 intrinsic 验证**两侧完全同指令(不是"看起来都是一条 vpaddd",是打开两边源码
  逐参数核对;bench 注释里的"MUST tie"不算证据——TableLookupBytes 曾因此混入,实际多一条 vpand)。
- 首次入列后跑一轮看读数——锚点本身持续偏离 1.00 就说明资格审查有漏。

---

## 2. 如何跑(逐平台规程)

### 通用铁律

- **同一台物理机上 Windows 与 WSL 的跑数必须错开**(WSL 是同机 VM);mac 是独立机器可并行。
- 每平台**双序各跑一遍**(默认序 + `HYLE_BENCH_HWYFIRST=1`),两份输出一起喂给报告脚本——
  判决要求双序一致,不一致自动降级 UNSTABLE。
- **构建后核对 exe 时间戳晚于源码修改时间**再跑(stale-exe 假数据烧过不止一次);
  cmake 输出 "no work to do" 时尤其要查。
- 跑数期间不要在同机做重负载(构建/rsync 大文件)。

### Windows / MSVC

```
cmake --build build --target hyle_op_parity_full_bench_avx2   # 及 _bench/_widen
build\bin\hyle_op_parity_full_bench_avx2.exe > hf.txt
set HYLE_BENCH_HWYFIRST=1 && build\bin\...exe > hwf.txt
python scripts/op-parity-report.py hf.txt hwf.txt
```
- exe 自己钉核(默认 LP2,env `HYLE_BENCH_CPU` 可换),**不需要包装器**;若一定要用外部包装器,
  切记 PowerShell 设 `Process.ProcessorAffinity` 会**重置** exe 内已设的线程亲和性/优先级。
- 自建 scratch 复验 TU 时 cl 必须带 `/DNDEBUG /O2 /arch:AVX2 /bigobj`——漏 NDEBUG 会让 Highway
  断言全开,锚点飙到 1.7~2.5(校准闸门会抓住,但你会白跑一轮)。

### WSL / GCC

- 严格走 rsync→`/root/hyle`(ext4)再构建;`/mnt/c` 上的跑数一律无效。
- rsync 必须 `--exclude='build*' --exclude=.deps`(带过去的 Windows FetchContent 缓存会毁掉配置)。

### mac / NEON

- 无亲和性 API,列分辨率天然较低:**至少 4~6 次运行合并**(双序×2~3),报告用 `--gate 0.06`;
- 只有 full 形状集有 NEON 目标;跑前确认机器空闲(刚解压大目录会撞 Spotlight 索引)。

### 隔离复验(objdump/单 op 探针)的 flag 陷阱

- 单独编一个 TU 做 objdump 对照时,**必须带 bench 的全套 ISA flag**
  (GCC:`-mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul`;并且带 `-DHYLE_SIMD_TARGET_AVX2`)——
  缺 flag 时 Highway 静态 target 退 128-bit、HYLE ScalableTag 退化,得到的是假对照。
- **不要用小 TU(<10 op)做"隔离复核"下判决**:每 op 放置方差过大,锚点必然出闸=读数一律不可信。
  历史上的"隔离复核"仪器自身锚点读 1.19~1.38,其全部免罪/定罪后来都被推翻过。

---

## 3. 如何读报告

```
== calibration anchors ==            ← 先看这里
  ... paired=1.00 best=1.00 ... ok
CALIBRATION PASS                     ← 没有这行,后面全部不可引用

op    ratio_paired  ratio_med  spreadH  spreadG  verdict
```

- **数字 = HYLE/Highway 配对中位比值**,>1 = HYLE 快。三估计器(best-of-replicas/median/逐轮配对)
  必须一致越线才发 WIN/LOSE;双序或多运行判决不一致 → UNSTABLE(**UNSTABLE 一律不算输赢**)。
- **spread 列是该 op 的自报误差棒**(副本间放置摆动):spread 大 → 判决线自动放宽 → 别对着一个
  spread 1.3 的 op 的 0.95 做文章。
- **0.90~0.96 的"边缘带"**:一批 op 会在 LOSE/UNSTABLE/TIE 间随轮次进出(MSVC 列尤甚)。
  单轮进了 LOSE ≠ 定罪,连续多轮稳定 LOSE 且 spread 小才值得立案。
- run_third 行只做参考,无判决。

---

## 4. 一个 LOSE 如何定罪/翻案(证据阶梯)

历史教训:**14+ 轮修复中,被草率"定论"过的项(地板/伪影/形状红利)后来几乎全部翻案**。
关闭一个真输项前,按阶梯走完:

1. **确认判决稳定**:双序一致 LOSE、spread 小、且至少两轮重跑仍在——排除边缘带徘徊。
2. **规范化逐条 disasm diff**(不是签名级!):把两侧循环提出来,mnemonic+操作数形状规范化后
   逐条比。**向量指令签名相同不等于逐指令相同**——FindLastTrue 案里 6% 就是签名级对比漏掉的
   一条标量 movslq(int 域算术混 64 位 sink 的符号扩展;标量尾巴统一用 ptrdiff_t/size_t 域)。
   MSVC 用 `dumpbin /disasm`,GCC 用 `objdump -d`(flag 陷阱见 §2)。
3. **指令确实少/同却慢** → 查端口分布(vpermd/vperm2i128 是 port-5 独占;内存源广播在 load 口
   免费)、依赖链(循环不变量能否外提——AddSub 案:形态 A 两算术都依赖 a,b 无法外提,链式场景
   24× 差)、以及 bench 形状红利(load→op→store 形状会奖励"能折叠内存操作数"的序列;链式/寄存器
   输入场景可能反转优劣,两种场景都测再下结论)。
4. **多形态尝试,一次一个**。已验证有效的修复模式库:
   - store+reload 兜底 → 原生 shuffle/cvt(历史最大宗:ReverseLaneBytes 0.05、Broadcast 0.46);
   - 向量域运算 → bits/标量域(CompressNot:`mbits^0xFF` 复用索引表,消向量 Not);
   - 重复提取 → 提一次共享(CompressStore 的双 movemask);
   - 组合出的掩码/比较 → 直接喂硬件语义(IfNegativeThenElse:blendv 本按 MSB 选择,免 vpcmpgtd);
   - 手搓序列 → 组合进已有 native op(NEON RoundingShiftRightSame 转发 vrshl,计数处理全外提);
   - 免分支代数变形(RoundingShr:`bias=(1<<s)>>1`,s=0 自然恒等,免 Gt/blend——可能伴随契约收窄,
     负 s 收窄为 unspecified 与 Highway 对齐);
   - **intrinsic 写法敏感**:同语义不同写法,编译器待遇天壤之别(MSVC 折叠 peephole 认
     `inserti128(v,low128(v),1)` 自 dup、不认 `broadcastsi128(cast)`,后者栈 spill 0.47)。
5. **GCC 反赢陷阱 → 编译器分叉,不要统一**:一侧输时另一侧常在反赢(BroadcastLane GCC vpermd
   1.13W / SetBeforeFirst GCC scalar-bits 1.30W / reduce256 GCC 128-domain 1.17W)。正确处置是
   `#if defined(_MSC_VER) && !defined(__clang__)` 分叉,让 MSVC 追平**且不牺牲** GCC 的反赢;
   把两侧改成同一形态往往等于把赢的那侧拉回 1.0。
6. **改完必须全表重跑,不只看目标 op**:复合上下文回归只有全表能抓
   (ConcatLowerLower 案:自己的 kernel TIE,却把 Slide1Up 打到 0.099)。修共享原语时给已知
   复合消费者留金丝雀(Slide1Up/SlideMask1Up 之于 ConcatLowerLower)。
7. **契约维度**:语义比 Highway 强且要为此付钱时,先问"上游自己遵守吗、有消费者吗"
   (DemoteToNearestInt:Highway 三个 target 三种 NaN 行为,强语义无人消费 → 收窄对齐);
   语义被 scalar oracle 固化、可能有依赖时保默认、加 InRange 变体分家(TableLookupBytes)。
8. 阶梯走完仍无解才允许写"接受的地板",且必须附:双序读数、逐条 diff 结论、已尝试过的形态清单。
   **签名级对比和单一形态的否定实验不足以盖棺。**

## 5. 修复后的收尾清单

- 正确性:动了后端就跑全套闸门——MSVC avx2/sse4/scalar + WSL 同套 + SDE avx512
  (`sde64 -future --`)+(动了 NEON)mac neon test;动了 sve/rvv 加 qemu 探针。
- 性能:改动平台全表双序重跑,CALIBRATION PASS 后核对目标 op 与金丝雀;
- 文档:更新 `simd-orderswap-matrix.md`(该文件**手工维护**,不要用任何脚本整篇重生成)——
  真输清单条目、演进段加一轮、全量表对应单元格;
- 非平凡的调查过程按惯例补进猎人日志或新开 `docs/debug/*-hunt.md`(结论先行、诚实记录误诊)。

---

## 附:速查——症状 → 第一嫌疑

| 症状 | 第一嫌疑 | 快速验证 |
|---|---|---|
| 锚点出闸 | 机器状态/构建 flag(NDEBUG?)/仪器回归 | 重跑一遍;查 exe 时间戳与 flag |
| 同 op 读数换一次编译就大变 | 代码放置(正常,靠副本聚合吸收) | 看 spread 列 |
| 同 exe 换启动器读数变 | 指针外提被绕过(新 kernel 漏了?) | 检查 kernel 是否有内联 `.data()` |
| 稳定的单编译器 5~6%,"指令看着相同" | 一条标量指令(签名级 diff 的盲区) | 规范化逐条 diff,盯标量尾巴与寄存器宽度 |
| 抄了 Highway 序列反而暴跌 | intrinsic 写法不被该编译器的 peephole 认 | dumpbin 找 spill/extract;换等价写法 |
| 修了 A,别的 op 突然暴跌 | 复合上下文回归(A 是共享原语) | 全表重跑;grep A 的内部调用者 |
| GCC 输而 MSVC 赢(或反之) | 编译器形态偏好差异 | 分叉,别统一;两侧各自 dumpbin |
| mac 列大面积 UNSTABLE | 机器不空闲/运行次数不够 | 查 top;合并 6 次运行 |
