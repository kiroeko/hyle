# simd 复杂度机会调研任务书(可单独派出的研究型 AI,一次一题)

> **任务性质**:对 `simd-complexity-opportunities.md` 的每个机会产出**证据包**,回答"值不值得做、
> 怎么做、预期赚多少"。**调研不施工**:产品代码零改动,原型只准放 `oplab/` 或 `/tmp`。
> 结论质量标准与性能战一致:**每个论断要么带数字、要么带 file:line,二者皆无的句子不准写**。

## §1 铁律

1. `include/`、`src/`、`bench/`、`tests/`、既有 docs **一律只读**;唯一可写处 = 自己的报告文件
   `docs/performance/research/simd-r<编号>-<slug>.md`(新增,**不提交**,高能力方审后定去留)+ 原型代码
   (`oplab/scratch_r<编号>/`)。
2. 禁止只读文档下结论——现状证据必须来自**真实代码**(贴 file:line + 摘录)与**真实测量**。
3. 原型测量用快探针模式(best-of-30、`clock_gettime`/`__rdtsc`、正确性对拍标量)并**明确标注
   "方向性证据,非校准判定"**;需要校准级结论时写"待校准仪器复核",不准冒充。
4. 对照物调研必须读到**对方源码**(本仓 `third_party/` 已有 highway/xsimd/ggml 相关;没有的给出仓库
   与 commit/文件路径),"据说/一般来说"不算证据。
5. 时间盒:每题一个会话;盒内答不完的问题列为"未决",不准硬凑。
6. 诚实优先:原型比现状慢、机会不成立,都是**合格结论**,照实写。

## §2 报告模板(每题一份)

```markdown
# R<编号> <题名> 调研报告
## 结论先行(做 / 不做 / 缓,一句话理由 + 预期收益数字)
## 现状证据(file:line + 关键代码摘录 + 实测基线数字)
## 对照物(每个:源码路径 + 算法要点 + 它为什么快/慢)
## 原型(若做):做法、数字、正确性对拍结果、与基线比值
## 若立项:里程碑草图(阶段/验收/预估工作量)+ 契约与风险
## 未决问题
```

## §3 题目卡(按优先级派发,一次派一题)

**R1(P0)softmax 融合**:`src/simd/activations.cc:64-71` 现为 4-5 遍扫描(max/exp/sum/div…),
vocab ~128k 是推理真热路。必答:①现状每 elem 周期数基线(quick-probe);②单遍/两遍融合形
(online-softmax,flash-attention 的 max-sum 单遍递推)原型数字;③ggml `ggml_soft_max` 与
llama.cpp 实际调用形怎么做的(源码);④数值稳定性对拍(vs 双精度参考,最大相对误差);⑤融合后
契约是否需要误差档位说明。

**R2(P0)AVX-VNNI 通道**:`avx2.h:957` 注释断言过时(cpuinfo 已有 avx_vnni),`avx512.h:798`
SumOfMulQuadAccumulate 是标量循环洞。必答:①本机/目标机 CPUID 实测(哪些 VNNI 形可用);
②`vpdpbusd` 走 avx2-VNNI 与 avx512-VNNI 的 dispatch 路径设计(对照现有 AES/CLMul 硬件门控先例);
③int8 点积微基准:现状组合 vs vpdpbusd 原型比值;④Highway 的 `SumOfMulQuadAccumulate` 各 target
怎么发的(源码)。

**R3(P0)归约误差档位**:pairwise/Kahan 分档。必答:①现状 `ReduceSum`/`Dot` 的误差实测
(1e6+ 元素病态分布 vs 双精度参考);②pairwise 与 Kahan 的速度税各多少(原型);③档位 API 形
(契约命名对齐 InRange 家族先例);④消费者盘点(activations/norm 哪些需要哪档)。

**R4(P1)Scan/PrefixSum 族**:Highway 没有此 op(差异化机会)。必答:①log-step 向量内 scan +
块间进位的原型数字 vs 标量;②消费者盘点(radix sort 直方图前缀、compaction、CAE 装配);③op 面
设计(InclusiveScan/ExclusiveScan/ScanAcross?)与各后端可行性(SVE 有天然形?)。

**R5(P1)radix sort / argsort**:对照 Highway `contrib/sort`(vqsort)。必答:①vqsort 源码要点
与本仓可借度;②8/11-bit radix + SIMD 直方图原型 vs `std::sort`/vqsort 数字(u32/f32 key,1e6);
③argsort(key+index)形态消费者(top-k、CAE 网格重排)。

**R6(P1)流式 top-k**:采样热路(logits 128k 取 k≤64)。必答:①现状路径盘点(llama.cpp 怎么做);
②SIMD 阈值滤 + 小根堆/寄存器网络原型数字;③与 R5 的边界(k 大小分界点实测)。

**R7(P2)f64 数学函数**:表驱动 range-reduction + Estrin。必答:现状 f64 exp/log 精度与速度基线、
sleef/Highway math-inl 对照、值不值得(消费者:CAE 双精度求解器)。

**R8(P2)math InRange 快变体**:exp/log 限域跳边界处理。必答:域界设计、省多少(原型)、契约
命名对齐 InRange 家族。

**R9(P3,合并一题)机会型盘点**:histogram(conflict-free 加宽)、gather→LUT 重写点、复数
split 布局、Philox 向量化。每个只答:消费者存在性 + 预期量级 + 是否立项,不做原型。

## §4 交接 prompt(逐字粘给研究 AI,替换 <R 编号>)

```
你是 HYLE simd 调研 AI。唯一任务:完成 docs/milestones/simd-complexity-research-brief.md 中
题目卡 <R 编号>,产出该卡要求的报告(模板见 §2,路径规则见 §1-1)。
铁律:①产品代码只读,原型只放 oplab/scratch_r<编号>/;②每个论断带数字或 file:line;③原型数字
标注"方向性";④对照物必须读到源码;⑤答不完列"未决",不硬凑;⑥慢了/不成立照实写。
产出:报告文件 + 3 行内的 chat 摘要(结论先行)。不提交任何 git 变更。
```

## §5 判读与去向

报告由高能力方审:通过 → 报告提交入库,"做"的题立项为新里程碑(进 backlog-map);"不做"的题把
结论一行写回 `simd-complexity-opportunities.md` 对应条目(防止重复调研)。
