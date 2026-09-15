# R3：CPU 默认计算与 Graph 完整纵切

<!-- PROJECT UNIT REVIEW: SCOPED -->

## Closure card

- baseline：`032b1dc7`；R1/R2 已完成，Compute/Graph 的产品共同语义候选为 `4e9f74a0`。
- owner：CPU 默认 Direct/Graph 组合、Native、oneDNN 与 oneDNN region。新默认组合有不同依赖闭包，
  分别登记 compute_route_cpu、graph_route_cpu；普通用户仍只选择 Compute/Graph 与 CPU feature。
- 真实缺口：compute/cpu.h 只装 Native，用户必须手排 oneDNN；普通 Graph 缺少自动接入成熟 region 的
  CPU 入口。cpu.h 又归 Native factor，不能直接塞入未声明的 oneDNN 或反向 Graph 依赖。
- 主纵切：变化输入的 F32 RMSNorm + gated FFN、解码/批处理；正交纵切：普通非模型 Direct 运算、
  有界动态输入/独立 Invocation、caller worker 上的嵌套调用与资源退出。
- 复用：现有 CPU Device/Context/Queue、PreparedOperation/Executable、Native 19-family、oneDNN
  BRGEMM/region、公开 Sched bulk。各自真实 owner/生命周期保留，不另造万能 executor 或线程池。
- 非目标：CPU Q4 kernel、其它算子目录、模型 CPU+GPU、ggml CPU blocker 解除、Sched 产品重审、
  GPU runtime 新实现或公共性能仪器改造。
- 预算：定向审计 45 分钟；有限 FIX 后实现/普通验证最多六小时；API/规则固定后一次适用 P1，
  最多一次有源码机制依据的性能纠正。只处理实际反例，不追加整体架构研究或无界探针。

## Owner 和交付范围

新增两个 factor 先登记 QUEUED，再 SCOPED/AUDITED；当前不创建 target/option 或声称能力存在。
Native/oneDNN/Graph-oneDNN 的旧 R1 AUDITED 只保留旧范围，在本卡完成有限审计后重新绑定 AUDITED。

新默认入口改变 Compute/Graph 产品聚合闭包。现有 selector 把变更前后的产品 roots 都作为 owning
contract，因此 Compute、Graph、CPU/CUDA backend 需要合法 SCOPED→AUDITED 的**交付边界**绑定。
这些 owner 本轮仅处理聚合/头归属与直接消费，R2 的机制结案不撤销；不因此重审 CPU/CUDA backend
算法、重跑 CUDA P1 或提前施工 R4。公共治理脚本保持不变。

## 已限定的工作面，审计后固定签名和规则

| ID | 动作 | 完成判据 |
|---|---|---|
| C41 | 固定 CPU Direct 和 Graph 普通入口及真实 factor 边，迁移 cpu.h 归属 | 普通用户不排列 Provider；没有 Compute→Graph 回边，CPU OFF 不泄漏新头/库 |
| C42 | 接入既有 Native/oneDNN 单算子与完整 region 的开发期默认规则 | 准备只匹配/绑定；不试跑用户输入、不搜 timing profile；缺席/拒绝有合法基线或明确错误 |
| C43 | 按需求盘点既有 19-family 并保留有效算法；真实 FFN/批处理组合 | 不以目录数代替实际数值、packing/alias/预算与失败验证；Q4 schema 不冒充 CPU kernel |
| C44 | 将公开 Sched 的既有真实消费接到普通默认入口 | 1/2 worker、变化输入、嵌套/并发独立 scratch、关闭与真实 owner 回收；不漏算/重复执行 |
| C45 | 相同普通 C++ 消费、适用平台与三交付，迁移指南/样例/受影响测试 | 源码/静态/动态入口与实际选择一致；Windows/WSL/两 Mac 按真实支持域取证 |
| C46 | 固定 API/算法后的默认入口 P1 与终局 | 同实现 direct 包装税及最强适用完整片段；同数值/格式/线程/资源/完成，方向与 5% 工程门分开 |

## 定向证据入口

- 四路线计划的 R3 与 Compute/Graph/compute-routes 长期合同，不重读全历史队列。
- cpu.h、Native build_plan、oneDNN select_route、Graph oneDNN match_pattern、中央 factor/product 图。
- [已登记 public Sched 消费](cpu-executor-foundation-probe.md)：有效且保持 open 的 runtime、body 恒成功、
  无 cancel 的同步 bulk；部分 admission 减少并发而排空 cursor，零 admission 才返回错误。样例检查
  Status，不能吞掉失败或重跑已接纳前缀。借用期限到所有 Compute consumer 完成后。
- bench/compute/route_p1_bench.cc 与 bench/graph_provider_onednn/region_p1_bench.cc 的现有同合同
  direct/完整片段机制；旧强制 Provider 结果只作研发证据，不能签新默认入口。

## 预声明验证

先 selector 选择 owner 与真实直接消费者；新 component/public header/聚合 seam 才扩展相应交付。
Windows、WSL 的 Native+oneDNN；Apple Silicon 的 Native/Sched，当前 x64-only oneDNN 不伪称可用。
普通默认入口覆盖合格规则的边界/留出 shape、oneDNN OFF、非法/不支持输入、冷 OOM、并发独立资源、
Context/runtime 关闭、packing 持有与重复执行。GPU backend 只回放本次交付边界真正受影响的门。

P0 先列默认组合、冷查询/JIT/packing、每槽 scratch、executor/fan-out/wait 与 metadata；不把 prepare
后的 allocator/转换/完成成本排除。已有公共性能 instrument 负责 A/A、双顺序、placement、n=15 和
5% material margin。先完成相同合同与极短 preflight，未通过就按有限源码原因裁决，不换姿势补 PASS。
API/规则未固定前不启动正式 P1。各 native 路线的性能不留给 R7 首次接线。

## 身份登记阶段验证

本次中央 manifest 仅追加两个 QUEUED 声明；没有边、split header、product root、target、源码或编译
选项变化。CMake 的 QUEUED 合同本身拒绝任何 target metadata；selector 已成功解析新旧图，closure
通过 41 个 component 的 registry/状态迁移，docs/diff 通过。

selector 对中央 manifest 保守生成全量计划（外部 r3-register-plan.json）。该调度不是产品风险 PASS：
本步骤没有产品输入或可构建组件变化，ordinary build/header/delivery/P1 对身份登记不适用，未执行。
这遵循 change-impact 文档中“适用风险及最终裁决仍以当前 closure card 为准”的边界；不改 selector，
也不把全量 selected 写成已通过。实际 C41–C46 的依赖/源码落地后，再执行本卡声明的适用 CPU/Graph
及交付门。没有以 nominal identity 启动全库重审。
