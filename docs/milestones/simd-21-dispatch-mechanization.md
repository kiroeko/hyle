# simd-21: 双腿分发机制化(静态宏腿不动,动态 cpuid 腿从手抄样板变成机制)

> **状态**:规划(2026-07-16)。**触发**:用户担忧"hyle simd 到处必须用宏、没有动态分发,不像 Highway
> 能两条腿走路"。**事实修正(先说结论)**:两条腿都已存在——
> ①静态腿 = header 级 `hs::` op 面,per-TU `HYLE_SIMD_TARGET_*` 宏,零间接零损失(章程主路径:
> "编译期优先,运行期零 dispatch");
> ②动态腿 = `src/simd/{ops,math,sort,bitpack,dtype,scan}` 六模块的
> "目标无关 `kernel_impl.h` + 薄 per-ISA TU(`kernels_{scalar,sse4,avx2,avx512,neon}.cc`)+
> `dispatch_ops.cc` 按 `best_isa()` 一次冷跳选路"(simd-04 建成,hyle_simd.dll 即此物,面向
> "产品发单一二进制、客户 CPU 未知"场景)。
> **真正缺的**:动态腿是六份手抄样板,没有可复用机制、没有消费者接入故事、没有文档——以至于项目
> 作者本人都没意识到它在。本里程碑 = 把它机制化,不是从零发明。

## 一、现状盘点(证据)

| 腿 | 形态 | 位置 | 状态 |
|---|---|---|---|
| 静态(宏) | 每 TU 定义 `HYLE_SIMD_TARGET_<ISA>` + 对应 `-march`,vec.h 阶梯选后端头,全内联 | `include/hyle/simd/vec.h` 目标阶梯;bench 三 TU、tests 变体即用法 | ✅ 主路径,零损失,不动 |
| 动态(cpuid) | 目标无关 impl 头被 N 个薄 TU 各以不同宏+旗标编译,dispatcher 按 `best_isa()` 填函数表,调用 = 一次间接跳 | `src/simd/ops/{kernel_impl.h,kernels.h,kernels_*.cc,dispatch_ops.cc}`,math/sort/bitpack/dtype/scan 同构 | ✅ 能用,❌ 六份手抄、无机制、无文档 |

关键既有资产:ODR/每 TU 宽度隔离、per-ISA `-march` 旗标(`cmake/HyleModule.cmake` 的
`hyle_set_highway_isa` 等)、`best_isa()`/`cpu_features()` 单源检测——**所有难题已在六个模块里趟平
六遍**。机制化 = 提取公因式。

## 二、设计原则(回答"动态怎么不拖累内联")

1. **分发粒度 = kernel/batch 级,永远不是 per-op 级**。动态腿的间接跳只发生在"一批工作"的入口
   (一次冷跳,首调解析后缓存函数指针);进了 kernel 内部,代码是该 ISA 静态编译的 `hs::` 全内联形
   ——与静态腿逐指令相同。Highway 的 `HWY_DYNAMIC_DISPATCH` 同理(它也是函数级不是 op 级)。
   所以两条腿的内循环性能**同源同形**,动态腿的代价只有入口一跳 + 二进制体积 ×N 份 kernel。
2. **不 vendor Highway 的 `foreach_target.h` 宏体系**(章程)。Highway 用预处理器把用户代码对每个
   target 重复展开;HYLE 用**构建系统**干同一件事:薄 TU 由 CMake 函数在 configure 期生成——
   这正是六个模块手写薄 TU 的自动化,不引入宏森林。
3. **静态腿零改动**:机制化只服务动态腿;不需要动态的场景(自家引擎按架构重编)照旧宏路,零间接。
4. `best_isa()`/`cpu_features()` 保持唯一真源;分发表解析一次(静态 init 或首调 + 原子缓存)。

## 三、方案(M0 机制 → M1 试点 → M2 消费者故事 → M3 收敛 → M4 度量)

**M0 — 机制本体(高能力方,~1 天)**
- `cmake/HyleDispatch.cmake`:`hyle_add_dispatched_kernel(<name> IMPL <impl.h> FNS <sig 列表>
  TARGETS scalar sse4 avx2 avx512 neon)`——configure 期生成薄 TU(每个 ~5 行:定义目标宏、include
  impl、实例化并注册)+ 按 ISA 挂旗标 + 生成 dispatcher TU(函数表 + `best_isa()` 解析)。
- `include/hyle/simd/dispatch.h`(~100 行):注册表模板、一次性解析、调用桩。x86 基线 = sse4
  (以现状 dispatch_ops 的既有基线为准),scalar 永远兜底;arm64 基线 = neon。
- 生成物与现有手写形对拍:同一 impl 走新机制 vs 手写薄 TU,objdump kernel 应同形(闸门)。

**M1 — 试点收敛(可交执行 AI,工单模式)**
- 选最小模块(`scan/` 或 `dtype/`)迁到 M0 机制;gtest 全绿 + kernel objdump 同形 + bench canary
  无回退(校准仪器)。此卡是机制的实弹验收。

**M2 — 消费者接入故事(高能力方,半天;这是"彻底"的关键)**
- 文档 `docs/architecture/simd-dispatch-guide.md`:两条腿怎么选(决策树:自家可重编→静态宏;
  发布单二进制/客户 CPU 未知→动态);写一个动态 kernel 的完整三件套示例
  (impl.h 契约:只写 `hs::` 泛型、禁 `#if` 目标判断、入参出参 POD/指针;CMake 一行;调用点)。
- 给北极星消费者立样板:推理侧一个真实 fused kernel(可与 R1 softmax 调研合流)走此机制。

**M3 —(非紧急,逐步)其余五模块收敛到机制**
- 每模块一张工单卡(执行 AI);不改行为只换接线,闸门 = gtest + objdump 同形 + canary。
  存量模块能用就不急,机制在即可。

**M4 — 度量入档**
- 冷跳成本 bench:direct vs dispatched 同 kernel,batch 大小摊销曲线(给消费者一个"batch 多大
  起动态腿开销 <1%"的数字);纳入校准 canary 防复合回归。

## 四、"彻底快改"的最小路径

M0 + M2 ≈ **1~2 天高能力方工作**——因为六个既有模块已把 ODR、每 TU 宽度、march 旗标、cpuid 检测
全部趟平,这是提取不是发明。M1/M3 用现成工单制交执行 AI。没有大爆炸:静态腿与六模块现状在机制
落地前后行为不变。

## 五、验收

1. 新增一个动态 kernel 只需三件:impl.h + CMake 一行 + 调用点;不复制任何样板。
2. 静态腿零改动(git 可证)、零性能变化(canary)。
3. M1 试点:gtest 全绿、kernel objdump 与手写形同形、bench 无回退。
4. 摊销曲线数字写入 guide;两条腿的决策树文档化。
5. 章程对齐:编译期优先不变;动态腿即 motivation.md 的"运行期一次冷跳兜底",不 vendor 宏体系。
