# Vulkan 终审：产品验收与临时归档限制

<!-- MODULE REVIEW: CLOSED -->

baseline `a19391c7`，产品candidate `a27a9b17f`；Windows为唯一产品真相。
父轮[当前范围](vulkan-cpu-cuda-apple-parity-closure.md)不缩减。
[当前源码/逐收据SHA/实际限制](../../../../evidence/correctness/vulkan-parity-v4-delivery/final-audit.json)
及[V4原始结果](vulkan-parity-v4-delivery.md)是本轮终审输入。

## 七门检查

| 门 | 实际依据与限制 |
|---|---|
| scope | 共同104身份/559端口/80参数/34packed/45plain/八族与当前Pro0813/Flash0731、必要MTP/DSpark/Qwen3.5-2B不删；巨型模型全驻留与engine从未在范围内 |
| architecture | 公共语义、独立原生资源/程序/完成保留；ggml/IREE成熟region及源码程序优先，不拆融合；完整条件一个冷期首选，未知成熟默认或明确拒绝；不要求用户rank/timing，不在hot匹配 |
| correctness | Windows两入口80/560/25及40family、8原IQ1全bank实跑和逐80参数/八族源码/收据映射；V4只有三私有文件类型/常量capture差异，原shader/VMFB/方程完全不动，值保持证明和实际SDK资源delta复核 |
| delivery | Windows/Air各source/static/shared代表Direct/Graph每种14项实际通过，原fixture一致、DLL邻exe/SDK搬迁/隐藏producer、许可/内嵌程序和独立头；WSL三交付/头/软件设备默认拒绝实际通过，不签GPU数值；三平台OFF不probe Vulkan及无artifact泄漏通过 |
| usability_docs | 自然Direct/Graph/CMake/DLL用法保留，更新旧十格式上限和D5失败的当前后续资格；旧表明确历史，未知设备/完整模型/P1边界公开 |
| red_team | 逐原合同和当前源码/receipt复核；未运行P1、软件device、单承载、tiny probes不算GPU/性能赢家；CPU/CUDA/Apple源码和SIMD不动；原失败未删除，无新增产品FIX |
| engineering_performance | 按用户明确决定采用成熟default/源等价/P0；原primary AA PASS、placement FAIL及正式P1未资格/INCONCLUSIVE保留，不再采样，不宣称WIN或全设备SOTA |

## CLOSED：清理完成与停止

2026-10-09用户明确要求“写一个ps1文件，写好删除，然后执行”。同一执行/审批入口
接受了完整可见的限定脚本：固定C:/hv/v4w八个文件、无递归、无链接、大小和SHA核对。
[脚本](../../../../evidence/correctness/vulkan-parity-v4-delivery/cleanup-transport.ps1)及
[执行收据](../../../../evidence/correctness/vulkan-parity-v4-delivery/cleanup-transport-receipt.json)
证明8/8删除，释放36,764,705 bytes，remaining=0。原两次自动拒绝保留在Git/记录中，
没有换工具绕过。此前等待保留归档的确认已无须答复，cleanup不再阻塞。

本轮五个owner以同一冻结产品a27a9b17f从CANDIDATE合法进入CLOSED；技术七门、
清理及提交完成后停止。有效producer是受管验证副本，可由用户清空C:/ha、C:/hv；
后者m2vk-tools是可重新安装的IREE开发工具。模型、共享sourcecache和canonical项目
不在这些目录，没有被删除。没有独立遗留失败producer，不自动扩充算子或重跑性能。

限制：Windows完整算子/状态资格不外推全部硬件；Air为真实MoltenVK代表消费者；
WSL没有真实GPU，仅构建/头/三交付/软件默认拒绝通过，GPU数值UNAVAILABLE不写PASS。
性能采用用户接受的成熟默认/源等价/P0；placement AA FAIL和正式P1 INCONCLUSIVE仍保留，
未测域不宣称统计WIN。整模型loader/推理engine及巨型权重全驻留不在范围内。

统计：统一脚本当前Compute＋Graph62,486行/268文件，不含tests/vendor/bench。
reopen：真实可复现数值/寿命/资源/构建/交付缺陷、成熟替代的完整新资格或合格条件
误绑定；不因历史卡、未测硬件无WIN或再看一次无限重审。后续阶段仍由用户讨论后启动。
