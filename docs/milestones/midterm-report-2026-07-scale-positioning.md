# 中期报告：体量、进度与定位（2026-07-17）

> **历史快照**：本文数字停在 2026-07-17，不再作为当前进度信源。2026-07-19
> 的 67.2k 生产体量、70–80% 功能判断带和模块冻结规则见
> [`pre-release-readiness.md`](./pre-release-readiness.md) 与
> [`../architecture/hyle-scope.md`](../architecture/hyle-scope.md)。

> **结论先行**：HYLE 已不是「clay 体型的几万行小加速库」。自有生产 **~53k**（simd 独占 **~34k / ~64%**）。旧完成投影 **~28–40k 作废**。纪律收口终态 **~65–80k 生产**；做满守界 **~80–100k**。若做到 **~100k 生产**，定位是 **中型、多支柱混合计算底座**（附带产品级 SIMD 子系统），不是小型「基础加速胶水库」，也还没到 Seastar/全栈运行时帝国。

权威数字与终态表：[`../architecture/hyle-scope.md`](../architecture/hyle-scope.md) 规模节。
能力补缺：[`simd-capability-fill-list.md`](./simd-capability-fill-list.md)、[`simd-25-scalable-product-parity.md`](./simd-25-scalable-product-parity.md)。

---

## 1. 测量口径（先统一再争论）

| 名称 | 命令/范围 | 2026-07-17 约 |
|---|---|---:|
| **生产** | `cloc src include` code | **53.3k** |
| 测试 | `cloc tests` | 22.4k |
| bench | `cloc bench` | 12.2k |
| docs | `cloc docs` | 14.2k |
| **自有工程** | `cloc . --vcs=git --exclude-dir=third_party,.deps` | **105k** |
| **git 全跟踪** | `cloc . --vcs=git` | **187k** |
| 再加 ggml 等 | 依赖树足迹 | 200–350k+ |

**常见误读**

- `cloc --vcs=git` **仍含** 已 commit 的 third_party（本仓 simdutf ~60k、mimalloc ~15k…）→ 187k **不是**「手写 187k」。
- 里程碑 ✅ 计数 **≠** 能力完成度（颗粒不等重）。
- LOC 进度 **≠** 差异化卖点完成度（simd 加厚会抬分子，不自动填 compute/graph 洞）。

---

## 2. 今天站在哪（模块）

| 模块 | 生产约 | 体感完整度 | 备注 |
|---|---:|---|---|
| **simd** | **34.4k** | 定宽+主 batch 厚；变宽主产品核与公开派发已覆盖 | 库中库；A1/C 尾/真机未完 |
| io | 5.0k | 大面落地 | shmem 等尾；HTTP2 后置 |
| compute | 3.8k | **仍简陋** | 借 ggml；Vulkan/自造 kernel 欠账 |
| sched | 3.1k | 主干里程碑收口 | 栈 guard 等边角；值得人工通读验收 |
| container | 2.7k | 相对成型 | |
| platform | 1.4k | 基质可用 | |
| mem | 0.7k | **偏瘦** | NUMA/大页等占位；malloc 在 mimalloc |
| graph | 0.6k | **薄** | 01/02 有破土，未厚 |
| 其余基质 | ~1.6k | 可用 | |

**结构事实**：生产代码重心已是 **SIMD 产品 + 若干支柱**，不是均匀的「七模块小库」。

---

## 3. 预期为何失真

| 旧心智 | 现实 |
|---|---|
| clay ~3 万行小 infra | clay 无真 SIMD / 无 fiber 调度 / 无 proactor / 无 GPU 层 |
| HYLE 完成 ~28–40k | 仅 simd 已 ~34k |
| simd 净新 ~5–8k | 七后端 × ~387 op + math/dtype/sort/bitpack/双轨 → **产品级 ISA 层** |
| 加速库 = 薄封装 | 自研后端 + 零 vendor 产品核 + 对拍仪器 |

失真主因是 **品类升级**（混合计算底座 + 自研 SIMD），不是 cloc 虚报。

---

## 4. 终态行数剧本

| 剧本 | 生产 | 何时 |
|---|---:|---|
| **A 纪律收口（主预期）** | **65–80k** | simd 补缺冻结扩面；compute 薄适配+必要核；mem/sched/io/graph 补设计项 |
| **B 做满守界** | **80–100k** | simd-25 真机签字；compute Vulkan+若干 kernel；graph-02 完整 |
| **C 失控** | **120k+** | 镜像 Highway contrib / 算子帝国 / 协议吸入 |

相对今日 ~53k：纪律路径大约再 **+12–27k**；做满路径 **+27–47k**。
tests 常随生产 ×1.4–1.6；自有工程（+bench+docs）A 约 130–160k。

**弹性最大旋钮**：**compute 野心**（+3k 薄层 vs +15k 自造）与 **simd 是否再扩面**。

---

## 5. 能力进度（与 LOC 分开）

**已厚**

- sched 主路径、io 大面、container/platform 地基
- simd：定宽面、双轨、主 whole-buffer 核、产品零 Highway；simd-25 使 SVE/RVV **主路径有真实产品 TU**，SVE partition 已吃满物理 VL（基础 network 保持 cap；仍缺真机性能）

**仍薄 / 未完**

- mem 设计尾（NUMA/大页等）
- compute 相对愿景简陋
- graph 编排未厚
- close-01 C 发布面
- simd：能力面与 NEON KV64 性能已收口；仍缺 SVE/RVV 真机性能签证

**主观功能完成度**约 **55–70%**（视是否把「SIMD 对标 Highway 产品层」算进必达）。
**生产 LOC ÷ 终态 A 中值**约 **~74%**——数字好看于能力，因 simd 已超买。

---

## 6. 若做满 ~100k 生产：算什么规模？

### 6.1 一句话定位

**中型、多支柱的高性能系统底座（systems infra）**，内含 **一个已达「独立 SIMD 库」体量的子系统**；服务 AI/工程计算管线的 **调度 + 内存/容器 + 网络 IO + CPU SIMD + GPU 原语 + 轻量编排**。
**不是** 微型「加速 helper」；**不是** 应用框架；**默认也不该是** 自研完整 GPU 算子生态（算子继续借 ggml 时）。

### 6.2 和参照物比（量级感，非对标声明）

| 参照 | 量级感 | 相对 ~100k 生产 HYLE |
|---|---|---|
| clay 自有 | ~3 万 | HYLE 约 **3×**，且多整根支柱 |
| 单一用途小库（某容器/某 JSON） | 通常 &lt;2 万 | 大一个数量级档 |
| **产品级 SIMD 库**（Highway 量级自研面） | 很大 | HYLE **simd 子集已占 1/3～1/2 故事** |
| 网络/运行时（asio 用法库 / 中型 runtime） | 数万～十余万 | **同档可对话** |
| Seastar 级共享-nothing 运行时 | 很重、意见强 | HYLE 更 **多支柱拼底座**，默认更薄、更可嵌入 |
| 完整推理引擎 / 训练框架 | 更大 + 领域层 | **上层消费者**，HYLE 停在 L0 |

### 6.3 定位标签（对外怎么说）

| 可以说 | 不要说 |
|---|---|
| 通用高性能 **C++ infra / 混合计算底座** | 「只有几万行的小加速库」（过时） |
| **CPU SIMD 产品面** + 统一调度 + proactor IO + GPU 原语 | 「又一个 hashmap 库」 |
| 算子 **可换供应商**（ggml） | 「我们实现了完整深度学习框架」 |
| 中型系统、长期维护资产 | 「toy / 周末项目体量」 |
| 若 100k 且纪律在：完整 L0 | 「已经对标全部 SOTA 运行时每一维」 |

### 6.4 结构风险（100k 时仍要盯）

1. **simd 占比过高** → 像「SIMD 公司附带 runtime」。要用 graph/compute/io **真实消费**证明 ROI。
2. **compute 若从 4k 冲到 20k+ 且脱离 ggml** → 品类滑向算子库，scope 需明示。
3. **发布面**（close-01）若不做，100k 仍是「强内部库、弱产品」。
4. **足迹 30 万+**（+ggml）是正常依赖树，汇报时与 **自有 10 万** 拆开。

### 6.5 规模判定口诀

```text
~30k 生产     → 小型 infra / clay 级
~50–70k       → 中型底座成形（HYLE 今日～纪律终态）
~80–100k      → 中型偏满的多支柱底座（做满守界）
120k+ 自有    → 警惕 scope 膨胀或第二产品线混入
+ggml 足迹    → 仓库「很大」，不等于自研「很大」
```

**做到 100k 且落在剧本 B**：
称 **「完整向的中型高性能基础加速与运行底座」** 合适——**基础**指 L0 原语与机制，**加速**指 SIMD/GPU/内存路径，**运行**指 sched+io 一体；**不要**缩回「小库」叙事，也 **不要**夸成全栈引擎。

---

## 7. 建议决策

1. 文档与对外统一：**生产今日 ~53k；终态纪律 ~70k 中值；做满 ~90k 中值**。
2. **simd 预算视为已基本花完**；只走补缺清单。
3. **compute 单独设行数/能力预算**，防止静默进入 C。
4. 下一阶段用 **真实消费者**（graph Body、compute 热路径）消化 34k simd，而非继续齐名。
5. close-01 与合同页决定「内部强」能否变成「可交付产品」。

---

## 8. 维护

- 每季度或生产 ±5k 时，更新 hyle-scope 规模节与本节「当前」表。
- 历史错误投影不在正文堆叠；以 git 与本节「作废」句为准。
- 能力状态仍以各里程碑文档与 backlog-map 为准，本文不替代 milestone ✅。
