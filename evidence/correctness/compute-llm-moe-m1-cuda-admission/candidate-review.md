# M1-C 候选复核

候选：`43732241adf81ad2bad9f64594e72d752b900b37`。实现冻结后单独复核本卡C1–C6，
不重审其它模块，不研究冻结自研算子。下列是源码/交付复核；host插桩的实际结果独立记录，
不能由本页代签。

| 检查 | 候选事实与验证锚点 | 裁决 |
|---|---|---|
| 公开范围 | 三个工厂只暴露HYLE类型；普通CUDA工厂组装库，Graph扩展复用Compute程序；`program.h`不安装 | PASS |
| 原生资源 | 每invocation独立backend/stream/pool/cache，metadata由已有arena持有；外来buffer为non-owning，输出保活到Completion | PASS |
| 准备与执行 | `initialize`冷建描述/内部资源并上传一次mask；`submit`只重绑已验证端口/派生view并执行原版图，未提前运行用户payload | PASS |
| 完成与失败 | 两个原生event连接HYLE Queue；出口失败先drain原版stream；失败槽退役，新槽可恢复；部分构造由RAII回滚 | PASS |
| Graph边界 | 先验证DAG可收缩再提交F32 region；无关cuBLAS节点不拆碎片段；attention仍复用整个Compute原版程序 | PASS |
| 数值与准入 | SM89有限域、shape/类型/scale/索引容量冷检查；BF16避免上游缩至F16的域，保留F32原版组合与score存储成本；不降低容差 | PASS |
| 动态库边界 | Windows调用栈/map反例已定位并修复每DLL计时器初始化；独立SDK实跑普通入口；Linux无此条件，原证据身份保留 | PASS |
| 安装与依赖 | Windows仅build interface保留未准入FlashInfer占位边；安装不要求其头/库；三交付隐藏producer、搬移prefix后运行，原版C符号/头不作为公开SDK | PASS |
| 冻结与成本 | 原版数学未改；没有P1或旧自研比较。managed首次pool/handle/capture/cache成本明确，不能声称prepared零查询或性能赢家 | PASS |

P0账单：冷阶段创建Context/recipe，每invocation创建metadata、原版backend和临时tensor资源；
F32边界无payload重打包，FP16 attention含Q升F32/输出cast和一次mask上传；BF16包含完整F32
score矩阵及cast。每次submit有两个event边界，原版内部stream/capture和managed维护由上游拥有。
Graph跨库组合的同步与转换均属完整成本；M2还须完成相同合同的成熟默认路径性能验收。

本次复核没有发现需要新增产品FIX的反例。三交付发现的Windows导出/初始化缺陷和Mac测试
声明条件问题都保留原始失败。剩余适用验证仍以各自真实收据和最终terminal record为准。
