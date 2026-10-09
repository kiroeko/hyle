# Vulkan 终审：产品验收与临时归档限制

<!-- MODULE REVIEW: AUDITED -->

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

## 仍需明确的归档限制

技术验收已经齐备；状态仍未CLOSED，待最终合法candidate/terminal步骤及以下取舍。
自动审批拒绝删除本轮临时传输归档；随后更安全的单Windows脚本、固定文件/路径/大小/
哈希核对与只读dry-run准备也被拒绝，理由只有`blocked by policy`，未执行删除。
Windows `C:/hv/v4w` 的8个同步bundle/tar合计36,764,705 bytes（约35.1MiB）暂留。
有效producer每机保持受管复用，失败都在同producer中纠正，没有遗留的独立失败producer；
未触碰模型、共享cache或用户文件。是否接受这些小型归档保留，需要用户明确收口取舍，
不能假写cleanup PASS或假称goal已完成。

统计：统一脚本当前Compute＋Graph62,486行/268文件，不含tests/vendor/bench。
reopen：真实可复现数值/寿命/资源/构建/交付缺陷、成熟替代的完整新资格或合格条件
误绑定；不因历史卡、未测硬件无WIN或再看一次无限重审。后续阶段仍由用户讨论后启动。
