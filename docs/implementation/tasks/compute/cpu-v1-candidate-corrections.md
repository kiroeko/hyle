# R3 候选的有限交付与激活纠正

<!-- PROJECT UNIT REVIEW: BLOCKED -->

## Closure card

- owner：compute_route_cpu、graph_route_cpu、graph_provider_onednn，从 CANDIDATE 退回 AUDITED。
- baseline：`e4a81e88`；产品/benchmark 候选 `79e96ceb`。
- 父卡：[R3 CPU 默认计算与 Graph](cpu-v1-four-route-foundation.md)。公开 API、默认选择、能力目录、
  生命周期及该卡其余 owner 不变；本卡只固定下列已发生的反例和明确缺项。
- 正常消费：同一 CPU Direct/Graph/Sched source/static/shared；连续 gated FFN 与分列并行 FFN。
- P1：W1-WIN 原 graph-cpu-route-p1-v1 合同，5% 门、原 shape/双顺序/placement/n=15→31。
  本卡使用父卡已预留的唯一一次有源码原因的性能纠正；不增加性能纠正或采样预算。
- 预算：定向源码审计 30 分钟；纠正、适用回归及不可变候选 P1，不重复整体架构研究。

## 已知反例和固定 FIX

| ID | 实际证据 / 成本机制 | 唯一允许改动 | 验收 |
|---|---|---|---|
| C45a | Air AUTO 正确禁用 oneDNN，source 19/19 与 relocated shared 3/3 通过；static EXPORT 引用未导出的 OPTIONAL_ABSENT provider 而失败 | 两个新 route 的本地 CMake 将缺席 oneDNN 的静态安装边裁去，保留 source build 的 manifest 依赖；不改 HyleModule、selector 或 manifest 协议 | Air/Mini Native 三交付；Windows oneDNN OFF static；x64 oneDNN ON 三交付继续保持真实依赖 |
| C46a | Graph 首次工程 P1 资格/controls PASS，11/12 非劣门 PASS；串行 17×129×257 的 Native peer 在全部分层快约 4%～5%，两层区间跨过 5% 门 | region 激活的完整连续 column range 合并为一次既有 SIMD batch，分列任务继续逐行；同实现 direct 和自然序列 comparator 同步采用连续 batch | 全图 double oracle、SIMD 尾数/变化输入/资源回收和原两 worker consumer；一次修正候选的原 Graph P1 |
| C46b | 首轮只有公共 arithmetic A/A，未接入本卡要求的实际 Graph/peer workload A/A；工具 QUALIFIED 不替代这个模块门 | 仅 Graph benchmark 增加 selected/peer 两侧同实现的 A/A 模式及模块 validator；复用既有 nanobench parser、Direct route 已有 5% factor-median / 10% gross preflight 规则，不改公共仪器 | 原 serial/two-worker shape、双顺序、allocation 顺序与两 placement；preflight 失败就停止正式采样 |

C46a 的源码依据是 apply_swiglu 即使覆盖整个连续矩阵，仍按行重复分派和处理尾部。合并连续
pointwise 区间不改变每个元素的数学任务，减少重复行尾；它不是按 benchmark 数字添加 shape
阈值。分列任务不连续，不能跨行吞并别人的区间。数量由原 dense tensor/workspace 检查保证。
Native peer 保留真实产品实现；不删除或故意弱化它。此前的差距仍如实保留，不能称已经解决。

首轮完整包：外部 scratch `p1-graph`，qualification QUALIFIED，performance INCONCLUSIVE，
n=31 已达上限。停止对旧候选重采样；最后两项 two-worker 完整路线显示 WIN，其余通过项只签
仪器统计非劣。模块 workload A/A 缺项未签 PASS，因此首轮也不能完成本卡的整体性能验收。
完整数据已归档，不能仅保留有利分层。C46b 在修正候选冻结前定界；不增加第二次产品性能纠正。

## 冻结与适用验证

公开头、签名、对象 layout、错误/关闭/OOM 合同、Provider 顺序与资格条件保持冻结；无新 op、
公共 option、依赖或 runtime。C45a 只投影缺席实现的安装边；oneDNN ON 的实际链接不裁掉。
C46a 使用已依赖 SIMD，不修改 SIMD/Native/oneDNN BRGEMM 内核或通用 Graph planner。

先 selector；Windows/WSL 回放 region 与普通 Graph route/真实 Sched consumer；W1 off-static
与两 Mac 的 source/shared/static 验证 C45a。已通过且投影未变化的 Direct/CPU/Native 机制结果
复用。WSL 定向 sanitizer 及剩余三交付沿父卡完成。API/算法固定到新 commit 后才运行修正 P1，
原比较输入、预算、所有权、线程、完成点和统计规则不变；没有合格最终门就不能关闭父卡。

七问沿父卡：oneDNN 版本/许可不变；同一完整消费者；复用原 SIMD；完整 peer 和包装税比较；
无新 ABI；补足缺席依赖的三交付；普通入口不增加用户选型负担。Native、Mem、Sched 的功能
范围与原终局不因此重开。纠正仍不能使必须的门通过时，保留实际结果并按协议裁决，不再迭代。

C46b 的 A/A 保留原三组 fixture 同时存活的顺序，只在 cold setup 替换两侧 route 类型。
模块 validator 复用公共 affinity policy，使父进程实际 mask=0x5、caller=2、workers=0,2；
保存所有输出。公共 runner 对 command validator 失败仍可启动诊断子进程，因此本模块 benchmark
在 instrument nonce 存在时先检查同一 build 的 A/A 结果；失败即在任何计时前退出。这个 cold
检查与结果文件只属于 Graph benchmark，不改变公共仪器、统计规则或产品 runtime。

初轮定向结果：Windows oneDNN ON/WERROR 的 7/7 route/region/Sched 回归通过；oneDNN OFF
static 的 source 与 installed 三个消费者各 3/3 通过，缺席 oneDNN 安装边检查通过。新增 A/A
fixture 通过 MSVC /W4 /WX 语法检查；validator 通过 Python 语法检查，尚不代表 A/A 性能通过。

## 不可变修正候选

`d9712e68d2b812449a170f474d71a347a27b0ba1` 已固定上述产品与 benchmark；三个纠正 owner 进入 CANDIDATE。
完整 P1 和剩余适用验证尚待裁决；父卡其余 owner 的未改实现仍复用 `79e96ceb` 投影。

## 当前裁决：Graph 性能资格 BLOCKED

- blocked_from：CANDIDATE。
- 当前 owner：graph_route_cpu、graph_provider_onednn。compute_route_cpu 的 C45a 产品修正固定后，
  回归父卡管理剩余 Direct P1 与三交付；这不把 Graph 缺项转移给 Direct 或 R7。
- blocker：`d9712e68` 的实际 workload A/A 16 进程 / 192 对比较中，selected 双 worker 的
  Native comparison 位置有一个同实现 ratio=0.9027134979，超出预声明 [1/1.10,1.10]。
  其余 gross 门和所有 factor-median 5% 门未见越界；单个点仍必须执行已经固定的停止规则。
- blocker_owner：R3 Graph 模块 benchmark 的双 worker 工作负载资格；不是公共仪器改造任务，
  也尚不是 Graph 产品性能退化结论。源码原因未被证明，不把随机重跑当恢复条件。
- resume_condition：独立、有限、预先审计的模块 benchmark 稳定性工作形成可复现的原因与合法
  纠正，并以同一完整 workload、线程/资源/完成合同通过已固定 A/A 门；再依法 SCOPED/AUDITED
  恢复未完成的 Graph 性能裁决。若需要改变预声明实验合同/预算，须先明确该决策。
- performance：BLOCKED；聚合 R3 验收 INCONCLUSIVE。产品计时因 cold guard 在首个 child 的
  开头拒绝而没有开始；不存在修正候选的 n=15 或 n=31 产品结果。没有放宽门或追加采样。

证据 `graph-correction-aa-stop.zip` 保留全部 16 次 A/A 原始输出、失败诊断、拒绝计时日志、
固定合同/命令和构建 cache。普通交付与 sanitizer 尚按父卡完成；它们不能代替性能资格。
