# SIMD op 对拍方法论——如何科学地 benchmark 一个 SIMD op

> 面向后续给 op_parity 新增测试条目、复跑数据、或判定"某 op 是否真慢"的贡献者。本文是**操作规程**;
> 每条规则背后的实验证据链在 [`../debug/op-parity-harness-artifacts-hunt.md`](../debug/op-parity-harness-artifacts-hunt.md)(三层测量伪影的定位与仪器重建全史),
> 当前权威数字在 [`../milestones/simd-orderswap-matrix.md`](../milestones/simd-orderswap-matrix.md)。
> 基础设施:`bench/simd/op_parity_engine.h`(测量引擎)、`bench/simd/op_parity{,_full,_widen}_bench.cc`(三个对拍 TU)、`scripts/op-parity-report.py`(聚合+校准闸门)。
>
> **一句话版本:亚 ns 级 SIMD 循环的单次读数不是数据;只有"锚点校准通过 + 执行核审计 + 双
> layout-bank 多副本采样 + 按完整运行分簇的 bootstrap CI + 双序一致"的聚合读数才是数据;而一个 LOSE 判决要变成
> "真算法输"的定论,还需要过一条证据阶梯。**

## 本轮收口顺序（不可颠倒）

这里的 **303** 是 benchmark 标题机械映射到的公开 op 名（其中 299 属于 perf-benchable universe，
另 4 个 `BitCast/Set/Zero/ZeroExtendVector` 虽被对拍，但按构造/retag 从性能分母剔除），不是 GTest
case 数，也不是报告行数。一个 canonical op 可能有多个类型、参数和 kernel 形状；三个 TU 的标题行
去重前多于 303，反过来“某名字测过一次”也不证明它的所有 overload/形状都测过。
历史上的 **41** 是同一覆盖树中尚未写独立 benchmark 的名字，并不等于 41 个都该有 L1 nanobench；
它们现已由 manifest 全量分类并闭环。

1. **先重测 303**：simd-22 双轨重构和 Highway/xsimd 内化后，在 Windows/WSL/macOS 从新鲜
   二进制重跑现有 parity/full/widen；先修仪器与不对称测试形状，不能一看到 0.93 就改算法。
2. **再狩猎重构回归**：把新 303×三平台 orderswap 全表及 HYLE 独有 benchmark 与重构前证据比较；
   interface shape、vendor 内化和机械搬运造成的下降必须先恢复。
3. **再狩猎真 LOSE**：只有排除回归和仪器伪影后，稳定 LOSE 才进入 §4 的反汇编证据阶梯；
   首个历史候选是 NEON `ScatterIndexN`。
4. **最后闭环 41 并做第二轮 Highway 落后项狩猎**：逐名分类；native 补对拍，paired/info 修账本，
   alias/synthetic 补正确性和零开销证据，algorithm 用真实工作集 benchmark。合成 op 不为凑数进入
   nanobench 分母。

2026-07-16 已完成上述 1–4：九份报告 union 303/303，320 个旧 case 的三平台明确 LOSE 均为 0；
41-name manifest 中 15 个 native 新行在三平台各自可用 target 上的第二轮狩猎同样为 0 LOSE。最大的新增回归
`SumsOfShuffledQuadAbsDiff`（Windows 0.08x、Mac 0.11x）已从 scalar spill 改为寄存器 recipe，WSL
认证为 1.43–1.49x WIN，Mac 两种 shuffle 中心值约 0.99、认证为 TIE。完整计数与证据见 orderswap matrix/simd-23。
C++ facade 随后单独落地，没有与底层修复混成第三套 kernel。

---

## 0. 为什么不能"随便写个循环测一下"

这套流程的每个部件都对应一类被实测坐实的伪影,量级全部**大于常见的真实差距**(±10%~40%,而我们要分辨的差距常在 5%~15%):

| 伪影 | 量级 | 机制 | 引擎对策 |
|---|---|---|---|
| P/E 核调度迁移 | ±10~30%/次 | 13900K 混合架构,不钉核时两侧测量落在不同核/频率 | 进程内 `pin_and_boost()`(P 核亲和+高优先级;mac 无亲和性 API,用 QoS+8× epoch) |
| 核类型/绑定静默失败 | 可改变输赢方向 | 只设置 affinity 不检查返回值，或不同 run 落入不同 efficiency class | metadata 记录 `pin_ok/priority_ok/cpu/core_class`；混核或绑定失败时 `--certify` 抑制 verdict |
| 睿频/温度/功耗状态 | 数个百分点到两位数 | 长表前后频率漂移，Windows 与 WSL 同机并跑，或散热尚未稳定 | H/G replica 紧邻交替、双序反转、每文件 anchor 归一；同机串行、AC/性能模式、热稳定后多轮跑 |
| 代码放置运气 | ±10~15%,尾部 ~30% | 同一二进制里**逐指令相同**的两个循环因 uop-cache/取指几何差异稳定不同速;每次重编译重掷 | 每侧 4 个独立实例化副本 + `stagger(rep)` 确定性相位错位(0/16/32/48)+ 防 ICF 唯一立即数 |
| 副本相位相关 | 单 op ~10% | 同尺寸副本被编译器等差排布,步长为 64 倍数时 4 个副本共享同一相位=同一次掷骰 | stagger 强制单序扫 4 相位；双序选择 R/R+8 两个 layout bank，共 8 个代码实例/侧 |
| 闭包栈-堆 4K 别名 | ~25~40%,随启动器变 | kernel 每迭代重载捕获链,闭包栈地址由环境块大小决定 | kernel 内指针外提(`auto* p_X = X.data()`) |
| 超小循环前端超敏感 | ±50% | ≤5 指令的循环体对放置极端敏感 | 宏形状循环 4× 展开 |
| mac 环境定时器 | err% 4~28% | ~1.6ms epoch 撞驻留应用的 ms 级定时器爆发 | `__APPLE__` epoch 8×(env `HYLE_BENCH_EPOCH_SCALE`) |

**判断仪器是否可信的唯一标准是校准闸门**:报告首段的锚点 op(源码级验证两侧同指令,真值=1.00)
的 raw paired 必须落在闸门内(x86 ±4%,mac ±6%),否则报告把整表 verdict 抑制成
`UNCALIBRATED`——不要看任何后面的数字。`best-of-replicas` 和 spread 仍显示用于诊断，但 best
不再参与 anchor gate：它本身对固定代码放置敏感，与 v4 把 placement 当 nuisance parameter 的模型
冲突。parity/full/widen 每个 TU 必须自带至少一个合格锚点；不允许再用“兄弟 executable 已 PASS”
替本 TU 背书。op verdict 的 hierarchical CI、双序与 `place_u` 门槛不因此放宽。

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
  副本散布会大到判决全变 INCONCLUSIVE)。
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

### 1.5 覆盖 manifest：不是每个公开名都要独立 benchmark

`scripts/simd-op-coverage.py --check` 与 `scripts/simd-op-manifest.json` 是覆盖口径的共同真源。一个
未独立对拍的公开名必须且只能按下列证据处理：

| class | 独立 L1 parity | 必需证据 |
|---|---:|---|
| `native` | 是 | 对等 Highway row；三平台可用腿做 focused 双序认证 |
| `alias` | 否 | `covered_by` 指向已对拍目标 |
| `synthetic` | 否 | 正确性 + 组成目标已对拍；禁止为凑分母重复测 |
| `paired_info` | 否 | 已在 pair/run_third 中出现，修账本 |
| `algorithm` | 否 | 真实工作集/吞吐 benchmark，不用 L1 单寄存器循环 |

CI 必须拒绝三种漂移：新增 uncovered 未分类、native 没有 source row、alias/synthetic 被误加独立 row。
新增合成接口应在同一提交更新 manifest，不能让“待补 benchmark”数字先虚增再靠文档解释。

`Stream` 是标准反例：它的非临时写合同必须用大于 LLC 的缓冲区、交错顺序和必要 fence 测 GiB/s；
把它塞进 `kN=4096` 的 L1 harness 只会测到缓存写，不是该 op 的产品语义。

---

## 2. 如何跑(逐平台规程)

### 通用铁律

- **同一台物理机上 Windows 与 WSL 的跑数必须错开**(WSL 是同机 VM);mac 是独立机器可并行。
- 快速诊断可双序各跑一遍；**权威再认证必须双序各至少两遍**(默认序 +
  `HYLE_BENCH_HWYFIRST=1`)并给报告加 `--certify`。环境变量还会切换 R/R+8 layout bank，不只是交换
  时间顺序；两序两轮形成 4 个完整 run cluster，每簇含 4 个**相关的** placement replica。不得把
  16 个 replica 当成 16 次独立运行来缩窄 CI。
- `--certify` 只接受带执行审计字段的当前 metadata；旧 v2/v3 文件仍可读作历史线索，但会被标为
  `INSUFFICIENT`，不能混入当前 HEAD 证书。
- **构建后核对 exe 时间戳晚于源码修改时间**再跑(stale-exe 假数据烧过不止一次);
  cmake 输出 "no work to do" 时尤其要查。
- 跑数期间不要在同机做重负载(构建/rsync 大文件)。
- 后台控制器重启前先确认没有旧 `hyle_op_parity_*` 子进程；“新旧两批各自都钉同一 CPU”会直接
  让它们在同一逻辑核上争用。发现重叠时两批全部作废，不从中挑好看的 run。
- 不用瞬时“CPU MHz”去事后修正比值：现代睿频是每核/每负载动态量，读数本身会扰动且不能代表热循环。
  控制方法是同核、紧邻配对、双序、多轮、anchor；仍对频率/热状态敏感的结果降为 `INCONCLUSIVE`。

### Windows / MSVC

Windows 性能树必须用**单独的新 build 目录**，并在 configure 与每次 build 都固定同一输出语言，推荐
`VSLANG=1033`。曾出现 build tree 在中文 `cl /showIncludes` 下配置、之后在英文 DevShell 下构建，
导致 Ninja 的 `msvc_deps_prefix` 与真实输出不匹配：头文件依赖完全没记下，改 `vec.h` 后仍报告
“no work to do”，旧 exe 继续产出看似通过校准的假数据。权威跑数前必须同时满足：

- configure/build 使用同一 `VSLANG`，不得复用语言不明的旧 performance tree；
- 目标 `.obj` 与 `.exe` 时间戳晚于相关 benchmark 源和 SIMD 头；
- 构建日志确实出现目标 TU 编译，不能只看旧 exe 仍可运行。

```
set VSLANG=1033
cmake --build build-bench-v3 --target hyle_op_parity_full_bench_avx2   # 及 _bench/_widen
build-bench-v3\bin\hyle_op_parity_full_bench_avx2.exe > h1.txt
build-bench-v3\bin\hyle_op_parity_full_bench_avx2.exe > h2.txt
set HYLE_BENCH_HWYFIRST=1 && build-bench-v3\bin\...exe > g1.txt
set HYLE_BENCH_HWYFIRST=1 && build-bench-v3\bin\...exe > g2.txt
python scripts/op-parity-report.py --certify h1.txt h2.txt g1.txt g2.txt
```
- exe 自己钉核(默认 LP2,env `HYLE_BENCH_CPU` 可换),**不需要包装器**;若一定要用外部包装器,
  切记 PowerShell 设 `Process.ProcessorAffinity` 会**重置** exe 内已设的线程亲和性/优先级。
- 自建 scratch 复验 TU 时 cl 必须带 `/DNDEBUG /O2 /arch:AVX2 /bigobj`——漏 NDEBUG 会让 Highway
  断言全开,锚点飙到 1.7~2.5(校准闸门会抓住,但你会白跑一轮)。
- Windows PowerShell 5.1 的 `>` 会把 stdout 写成 UTF-16LE；报告器会按 BOM 自动识别 UTF-8/UTF-16。
  不要因为文本编辑器看着正常就假定解析到了 metadata；报告必须显示非零双序计数和 execution audit。

### WSL / GCC

- 严格走 rsync→`/root/hyle`(ext4)再构建;`/mnt/c` 上的跑数一律无效。
- rsync 必须 `--exclude='build*' --exclude=.deps`(带过去的 Windows FetchContent 缓存会毁掉配置)。

### mac / NEON

- 无亲和性 API,列分辨率天然较低:**至少 4~6 次运行合并**(双序×2~3),报告用
  `--gate 0.06 --certify`;
- 当前 parity/full/widen 三个形状集均有 NEON target；必须串行跑，跑前确认机器空闲（刚解压大目录会
  撞 Spotlight 索引）。

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

op  ratio_paired  ci95_lo  ci95_hi  ratio_med  spreadH  spreadG  place_u  verdict
```

- **数字 = HYLE/Highway 配对中位比值**,>1 = HYLE 快。每文件先用多个真值锚点的 paired 中位数消除
  残余全局偏斜；bootstrap 先抽完整 run cluster，再在簇内抽 replica，不能把同一固定二进制的代码
  放置样本伪装成独立运行。
- `LOSE/WIN` 要求 CI 和两个 order/layout bank 的中位数都越过 0.97/1.03 实用等价带，且越过
  `place_u=max(3%, (max(spreadH,spreadG)-1)/2)` 的代码放置不确定度。只有**整个 CI** 落入
  0.97–1.03 且 `place_u` 没有宽过 3% 才是 `TIE`；CI 只是碰到/跨过等价带，或被 place_u 否决，
  都是 `INCONCLUSIVE`。
  后者不能偷换成“基本持平”。
- **spread/place_u 是该 op 的自报误差棒**：spread 大会自动放宽 LOSE/WIN 判决线。逐指令相同的
  `Not i32` 曾在 MSVC 固定二进制稳定读 0.962；旧的 pooled-replica CI 误报 LOSE，新口径因
  CI/placement 证据不足降为 INCONCLUSIVE，反汇编再确认两侧热循环完全相同。
- **0.90~0.96 的"边缘带"**:先看 CI；只有上界仍低于 0.97、双 order 都低于 0.97 才进入 LOSE
  候选。单轮 0.93 或 `INCONCLUSIVE` 不定罪；连续多轮稳定 LOSE 才值得进入反汇编阶梯。
- run_third 行只做参考,无判决。

---

## 4. 一个 LOSE 如何定罪/翻案(证据阶梯)

历史教训:**14+ 轮修复中,被草率"定论"过的项(地板/伪影/形状红利)后来几乎全部翻案**。
关闭一个真输项前,按阶梯走完:

1. **确认执行与判决稳定**：`execution audit` 通过（同逻辑 CPU/同 core class；mac 为 QoS 成功）、
   双序一致 LOSE、spread/place_u 可接受、且至少两轮重跑仍在。混 P/E 核、睿频/热状态漂移和边缘带
   徘徊先降为 INCONCLUSIVE。
2. **规范化逐条 disasm diff**(不是签名级!):把两侧循环提出来,mnemonic+操作数形状规范化后
   逐条比。**向量指令签名相同不等于逐指令相同**——FindLastTrue 案里 6% 就是签名级对比漏掉的
   一条标量 movslq(int 域算术混 64 位 sink 的符号扩展;标量尾巴统一用 ptrdiff_t/size_t 域)。
   MSVC 用 `dumpbin /disasm`,GCC 用 `objdump -d`(flag 陷阱见 §2)。
3. **指令确实少/同却慢** → 查端口分布(vpermd/vperm2i128 是 port-5 独占;内存源广播在 load 口
   免费)、依赖链(循环不变量能否外提——AddSub 案:形态 A 两算术都依赖 a,b 无法外提,链式场景
   24× 差)、以及 bench 形状红利(load→op→store 形状会奖励"能折叠内存操作数"的序列;链式/寄存器
   输入场景可能反转优劣,两种场景都测再下结论)。
   具体 op 的 canonical 名只说明“至少一种形状被覆盖”，不代表所有形状。候选项必须补一个真实消费者
   形状。MSVC `Mod i32 invariant` 是反例教材：旧寄存器 sink 既有逐 vector 外部调用的测试形状问题，
   又确实藏着 `Divisor` 布局和 sign-vector 活跃区造成的额外 ymm copy；只补 load→Mod→store 会看到约
   0.98，却会过早放过真实 codegen 差距。修正 sink、字段布局与不变 sign 分支后，Div/Mod-accumulate/
   Mod-store 分别达到 1.153/1.154/1.119。结论必须同时解释测试形状和最终逐条汇编。
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

### `<0.98` 全扫规则

全量报告生成后，无论 verdict 是什么，都机械列出 `ratio_paired < 0.98` 的行并至少完成一次源码/形状
检查。这个阈值是**猎人入口，不是 LOSE 线**：

- 有额外 store/reload、调用、标量往返、不同算法或 Highway 明确编译器特化时，进入汇编阶梯并修复；
- 两侧逐条相同或最终都是同一 intrinsic，记录为 code placement 候选；
- `spreadH/spreadG` 大、CI 跨等价带的行记录为高噪 INCONCLUSIVE，不靠加宽/改名统计口径消失；
- 修共享原语后重跑受影响 suite 的双序四文件，并再次全扫，防止复合消费者回归。

2026-07-16 的 v4 复测由该规则额外抓出 MSVC `SetOnlyFirst` 和 `IsFinite`；它也明确阻止把
instruction-identical 的 `Not`、单 intrinsic 的 IfThen zero 家族以及高 spread 的 Concat/MulOdd
误修成平台特化。具体证据见 `../debug/op-parity-harness-artifacts-hunt.md`。

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
| mac 列大面积 INCONCLUSIVE | 无 affinity、机器不空闲或运行次数不够 | 查 top；合并双序各 3 次；仍宽就诚实保留 INCONCLUSIVE |
