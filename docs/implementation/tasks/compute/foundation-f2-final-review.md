# F2 固定候选终审输入

状态：REVIEW_IN_PROGRESS；本页由[F2活动卡](foundation-f2-batched-transfers.md)纳入，不是terminal。
产品snapshot为5b3efe2cce3fcab2bc55cf749af2d862a786ef25；compute/Apple/Vulkan的03162add0
owner scope不变，只有CUDA按[终态delta](foundation-f2-cuda-terminal-cleanup.md)重新固定。
候选后的测试维护与证据不授权修改产品。F3仍READY_FOR_SCOPE，未登记或实施。

## 七问与源码审查

2026-10-11，reviewer为当前维护agent；与实现阶段分离的候选后源码/证据复核。
本遍检查现有公开入口、原生接线、失败/退休、成本、普通样例与指南，不增加peer调查或产品FIX。
工程裁决仍以[当前资格矩阵](foundation-f2-verification-matrix.md)及剩余实际门为准。

| 问题 | 已核对的边界与证据 | 当前裁决 |
|---|---|---|
| 第三方/版本/许可证/fallback | [实施前审计](foundation-f2-source-audit.md)固定原vLLM/SGLang来源；产品继续使用锁定CUDA SDK、MLX/Metal及ggml Vulkan，不增加dependency pin或vendor副本。CUDA固定PFN_v13000/冷proc解析，缺席沿原same-stream memcpy，失败不换路线重试 | 源码与实际batch/absence门匹配；不外推旧driver实机 |
| 真实消费者与完整旅程 | 量化权重/状态与正交资源分组搬运使用同一有限byte/phase/lease合同；[普通样例](../../../../samples/package/graph/batched_transfers/main.cc)两组/两槽/三轮，原BF16/Q8或Apple affine Q4搬运后由原Direct/Graph数学消费 | 四路线实际数学已有证据；完整模型/文件管线不计为F2完成 |
| 下层复用与owner | common helper只拥有range/permission/control；CPU/CUDA原QueueSequence，Apple原Runtime/Job，Vulkan原Run/Device/transfer handoff分别执行；private util_core直接边由中央声明投影 | 未发现第二runtime、通用native executor或新copy/math kernel |
| 复杂度与同合同成本 | [当前P0账](foundation-f2-verification-matrix.md)与copy_ranges/copy_control/三个controller源码相符：排序alias/root而非R²比较，G≤64；所有slot容量冷固定，native calls/依赖/完成/host wait分账 | 不宣称P1赢家、vendor零分配或零host wait；最终P0聚合待影响门完成 |
| C/FFI清洁度 | 新入口为C++ program/invocation/submission及值类型；原CUDA/Metal/Vulkan仅在所属private adapter，普通公共batch头没有vendor函数表或C schema | 本批无Native FFI交付 |
| source/static/shared与依赖 | 当前矩阵有适用三平台四路线source及迁址static/shared、ON头/缺席诊断、Windows DLL与Air dyld/WSL ldd；中央component-contract已PASS | selector扩展factor/header/delivery逐项仍待裁决，不能以SDK数学包代签全部格 |
| C++易用性/指南/限制 | [指南](../../../user/compute.md)与公共batch头、普通graph façade样例核对：独立槽、显式phase与producer、last-write逻辑覆盖、旧结果/失败slot、borrow寿命与drain | 没有要求普通消费者选择内部factor/provider；关闭/未知成本限制保留 |

## 具名finding与处理

| ID | 严重度/类别 | 事实及处理 | 结果 |
|---|---|---|---|
| R-F2-01 | 产品correctness | CUDA真实producer非法地址后owned free的host fatal已反证03162add0；仅CUDA依法退回AUDITED并修原终态释放边界，再固定5b3efe2cc，原失败保留 | Windows/WSL实际反例及正常回归PASS；native physical free仍UNKNOWN |
| R-F2-02 | 证据缺口 | Vulkan原fence/drain未知状态必须继续保活；补正常持有/放弃分支，真实提交后观察OOM，恢复原观察才退休且保留首错 | [Windows/Air10项copy](../../../../evidence/correctness/foundation-f2-native-risk/receipt.json)PASS |
| R-F2-03 | 测试fixture | 首unknown测试误将blocking wait当作报错立即返回，120秒TIMEOUT；改成实际等待线程握手与pending output合同，未改产品 | 首TIMEOUT和修正PASS同收据保留 |
| R-F2-04 | 证据缺口 | Apple原GPU prefix晚失败必须阻止原FIFO中已排队copy发布，旧成功ready不丢失 | 同故障收据8项Apple copy PASS；不宣称覆盖全部vendor硬故障 |
| R-F2-05 | 环境/工具资格 | WSL实际只有llvmpipe；CUDA设备可用不能代签Vulkan GPU | [真实枚举](../../../../evidence/correctness/foundation-f2-wsl-route-qualification/receipt.json)NO_QUALIFIED_GPU；GPU运行未运行，适用交付单独通过 |
| R-F2-06 | 环境/fixture | Windows旧MSVC中文include prefix与英文输出不匹配；WSL继承Windows CUDA_PATH；scratch cache恢复遗留说明导致parse FAIL | [Windows SDK](../../../../evidence/correctness/foundation-f2-windows-vulkan-sdk/receipt.json)与[component合同](../../../../evidence/correctness/foundation-f2-component-contract/receipt.json)保留有界恢复、真实deps/工具链与全部失败；公共工具/产品未改 |
| R-F2-07 | 尚未完成的工程门 | selector中央图扩展结果包含factor、逐头与其他产品delivery；当前仅部分门有对应实际证据 | UNPROVEN；继续原门，不写CLOSED，不提前F3 |

源代码复核没有把首错、unknown、逻辑metadata归零或旧证据改写为全局成功。当前未发现新增
公共语义/算法/布局改动的依据；如果剩余实际门产生具体产品反例，仍按owner协议裁决。
本页不以测试数量或文档齐全替代工程关闭。最终terminal需补完整风险矩阵、七门、统一LOC与
selected/skipped裁决，并在registry依法从CANDIDATE进入CLOSED后才启动F3。
