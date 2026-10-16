# G06 metadata/decode 有限终审与整项验收

状态：**G06 ACCEPTED，2026-10-16**。本记录关闭 G06 已冻结的 M/S/L/D 工作，整 goal
完成项为 G01/G02/G03/G06，共 **4/10**。FlashInfer 的 G06 D6/D7 已验收；该组件仍为
**CANDIDATE**，由[当前 G04/G05 卡](inference-g04-g05-cache-attention-domain.md)承接原 CUDA
kernel-only 性能未决门。其余十个 G06 owner 本轮 CLOSED。此裁决不验收 G04/G05 或完整引擎。

baseline 为 `10129b90bbfc4d85c25e1595c823962b5a9c8146`；
完整产品快照为 `669121418669b31f6849c74f64362aaf8d6eb919`。
Graph 最后冻结在669，其余十个 owner 的产品输入保持各自
`0a29d119426b2975d48c238c37a2994675e0ba9f`，按实际输入匹配复用证据。
[施工卡](inference-g06-metadata-decode.md)保存合法 scope/audit、有限 FIX、所有失败和费用；
[最终裁决证据](../../../../evidence/correctness/inference-g06-final-qualification/validation-result.json)
保存当前输入、七门和本轮状态变更。候选后仅补测试、文档和证据，产品已停止变化。

## 范围与公开旅程

主消费者是 Qwen attention 的单/多请求 Q1 与既有 mixed prefill/decode；正交消费者是两个
真实 attention node 共享不可变布局、各持原 workspace 的数值 DAG，以及页重排、增减请求、
shared physical pages 和独立旧结果。固定 SGLang7224fce1、vLLM98f72111、llama.cpp cbe28bb3
的原始来源、SHA 与成熟形状见[成本审查](../../../../evidence/architecture/inference-g06-final-cost-review/p0.json)。

自然 C++ 入口仍为 `BatchSchema::version → BatchInvocation::update_batch → PreparedBatch::run
→ checked Completion → owning output → close/drain`；Graph 通过完整 `NodeLayout` 集准备，
再 bind/submit。Apple/Vulkan ragged 通过 `update_ragged_batch` 后重新 bind 原容量输入。
host-known 直接使用 owned 布局；device-produced 来源经原 checked producer/F2 边界。
库拥有通用准入、依赖、错误和实际退休；请求选择、bucket、页政策与模型调度由引擎拥有。

| 能力族 / 公开入口 | 归属与有限结果 | 实际证据 |
|---|---|---|
| M：version/host/metadata/indices、prepare/update/report | SELF active immutable snapshot、显式缓存全 blob、common delta 和 native per-bank delta；保旧版、inactive 零区、预算和导出并发 | [common](../../../../evidence/correctness/inference-g06-common-source-windows-attempt-4/receipt.json)、[native metadata](../../../../evidence/correctness/inference-g06-native-pending-air-attempt-5/receipt.json)、最终SDK共同66项 |
| S：common Graph NodeLayout/prepare/report | SELF 在首次上传前共享合格同输入镜像；每 node 独立计划/workspace与必要等待。native Apple/Vulkan 各 controller 的 index banks 按 M 单独计费 | common 两个 dependent attention node、反序 NodeLayout、三 episode 数值与上传断言；对应源码与三交付 |
| L：ready/pending paged bind/run/submit | BORROW 原 F1 unique/ordered read；完整 block/generation/permission 分组，Direct/native/pending及两个Graph准入层 O(G log G)，真实退休保所有权限 | [256-block Graph](../../../../evidence/correctness/inference-g06-graph-page-admission-windows-source/receipt.json)、三host/六nativehost及原pending风险 |
| D：CPU/FI update、native paged/ragged update、explicit prepare_replay | BORROW 原数学与队列，SELF 必要状态桥；四路线实际 topology/runtime 复用，真实 CUDA/FI capture 和拒绝域 | [CPU paged](../../../../evidence/correctness/inference-g06-cpu-runtime-q1-source/receipt.json)、[CPU ragged](../../../../evidence/correctness/inference-g06-cpu-ragged-runtime-source/receipt.json)、[native lengths](../../../../evidence/correctness/inference-g06-native-runtime-lengths-source/receipt.json)、[native ragged](../../../../evidence/correctness/inference-g06-native-ragged-runtime-source/receipt.json)、[FI schedule](../../../../evidence/correctness/inference-g06-fi-schedule-runtime-source/receipt.json)、[capture](../../../../evidence/correctness/inference-g06-capture-replay-source/receipt.json) |

上述四族均为本轮 COMPLETE。F1/F2/Queue/Completion/native math 为 BORROW；没有新增模型
scheduler、通用 planner 或 benchmark-only 公开入口。API、默认值和原语分解沿施工卡有限表。

| 语义覆盖 | 正常/边界 | failure/OOM/overflow | ownership/teardown/并发 | 主/正交消费者 | 平台/交付 |
|---|---|---|---|---|---|
| M | PASS：空、grow/shrink/reorder、generation、full-wire独立oracle | PASS：导出失败重试、旧版/预算不变 | PASS：8线程同缓存、独立寿命、旧reader | PASS：64步与F4导出 | PASS：四路线source、八SDK、适用宿主风险 |
| S | PASS：同版、新版、反序node与foreign/resource拒绝 | PASS：失败前缀与冷OOM | PASS：原跨queue依赖、独立workspace/旧结果 | PASS：两个实际attention节点 | PASS：适用common CPU/FI Graph；native banks不冒充共同镜像 |
| L | PASS：同backing不同block、重复/重排/两组重绑 | PASS：closed/poison/permission/pending冲突 | PASS：partial/unknown/多native前缀实际退休 | PASS：256 blocks及Direct/Graph/joint/pending | PASS：当前三host、六nativehost、source/SDK实际调用 |
| D | PASS：稳定tail/跨页、请求join/end、mixed17+1、schedule边界 | PASS：capture录制/launch/post-record OOM、旧slot拒用/新slot恢复 | PASS：idle capture释放页权限、旧output/在途/关闭拒更新 | PASS：单/多请求及独立非恒定V/poison尾行oracle | PASS：四native与三CPU+GPU实际路线；CPU capture明确Unsupported |

## 七门与七问

| Gate | Verdict | 实际依据 |
|---|---|---|
| scope | PASS | 四族有限 FIX、主/正交消费者、原始peer和全部当前准入域；G04/G05原计时门独立保留 |
| architecture | PASS | 原F1/F2、QueueSequence、native worker/fence、owned结果与actual retirement；无第二执行owner/公共构建改动 |
| correctness | PASS | 上表逐族语义；Windows当前Graph428、原完整source569；CUDA24故障suite、40 pending模式与fresh-process capture；当前WSL三san各291/15模式及Air六配置 |
| delivery | PASS | [八份迁址SDK](../../../../evidence/correctness/inference-g06-final-relocated-sdk/receipt.json)：CPU91/CUDA110/Apple111/Vulkan108每配置，23独立头联合、真实native/joint/capture、installed-only及shipping无观察hook |
| usability_docs | PASS | [批次指南](../../../user/batch-attention.md)、[Apple](../../../user/apple-compute.md)、[Vulkan](../../../user/vulkan-compute.md)的调用、重绑、生命周期、费用与资格；本轮docs/closure机械检查 |
| red_team | PASS | 独立阶段自审，非第二审查人；Graph漏去重修在669，原生平台san补六配置，分配/计划/capture误读与旧node校验成本均已裁决 |
| engineering_performance | PASS | [P0结构审查](../../../../evidence/architecture/inference-g06-final-cost-review/p0.json)八行账与实际report/allocator/oracle；本轮无新计时排名或P1非劣声明 |

1. **第三方**：沿原依赖lock、许可证与factory交付。baseline后未改依赖lock、third_party或
   native数学；固定peer只证明成熟形状。Windows原生CUDA batch继续具名不支持。
2. **功能与组合**：上表回映全部本轮新增/调整入口，主/正交旅程涵盖完整创建、更新、消费、
   结果与关闭；无用引擎glue代替已准入MUST。
3. **复用**：原F1检查不变，F2只证明dirty范围；冷零初始化加成功delta才证明完整mirror。
   原数学、scheduler、native对象和result slots保持唯一owner。
4. **性能**：原初合同为P0/必要P1；有限新承诺由bytes、上传/等待/重规划/分配事实和复杂度裁决。
   peer自然tensor owner不具有同一F1/不可变版本/错误退休合同，本轮不签计时parity或speedup。
   原G04/G05 kernel-only5%门及[15,31]历史不变，仍UNRESOLVED。
5. **FFI**：本轮为C++23，沿原typed native桥；无新C ABI或跨语言runtime需求。
6. **交付**：八SDK均实际安装搬迁、独立公开头与消费者运行；CPU/CUDA各17共同头，Apple/
   Vulkan各加自身3头。未选路线头按中央split规则不安装；空guard不作native资格。
7. **易用**：只选hyle::compute/hyle::graph；明确ready与pending、何时update/replan/recapture、
   成功update后native ragged重绑、旧结果/allocator寿命和原取消拒绝。源码与安装入口一致。

## 成本、资格与停止

每版仍分配active backing/control、做generation CAS/active拷贝/refcount；显式全blob导出
首次全容量分配/清零，缓存访问仍锁。native controller每prepare创建的小version owner可经
recipe allocator分配；另一invocation allocator无新增attempt不证明整步零分配。
common按active extent扫描、至多五段delta；native按各bank真实旧值更新。共享只省合格镜像
上传，原计划/workspace与等待各自计费。页GlogG不代表整个Graph prepare线性；既有node-set
最坏立方校验在baseline10129已存在，单独记账，无当前material trace支持扩FIX。

CPU fixed-Q Q1保原graph/pool/scratch，FI保实际chunk/CTA/window等价schedule；跨界重新规划。
Apple变长度仍managed建图，Vulkan变参数仍重新编码。CUDA捕获须显式冷准备、稳定storage、
原计划合格；改地址显式recapture，真replan失效。CPU未准capture，原生Apple/Vulkan的managed/
Recorder合同各自保留。driver/private费为UNKNOWN；未知硬预算按原资格拒绝。

[当前Air六组host风险](../../../../evidence/correctness/inference-g06-final-native-host-sanitizers/receipt.json)
每配置九目标，Apple20/Vulkan17个GTest及六standalone消费者；TSan原refcount/Metal边界
模型只签对应宿主facet，不签LSan/shader/driver/private pool。Vulkan仅在完整GLSL输入、
feature与生成命令相同时借152个shader数据文件，151个wrapper仍按各san编译。原宏、链接、
预构建guard、封存器与磁盘失败、有限追加、重叠时间及UNKNOWN费用全部留在原记录。

[统一不可变LOC](../../../../evidence/architecture/inference-g06-final-cost-review/loc.json)：669全库
3718683、自实现不含测试180623、11 owner唯一实现联合54333行/228文件。联合值是当前实现
总量，并非本轮新增或推理引擎工作量预测。已验SDK/san/LOC不因本文状态改变而重跑。

只有复现的当前M/S/L/D语义、生命周期、成本事实或交付缺陷，或新的具名消费者需求，才按
reopen规则重新定界。完成本轮有限集合后停止G06产品施工；G04/G05原性能未决、G07–G10、
完整引擎token/s与多GPU/多机继续沿原goal，不由本记录代签。
