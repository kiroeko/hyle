# Vulkan TopK 字节起点与设备对齐纠正

产品 `7f4a3fb6`；测试/三交付快照 `5f4df55b`。完整记录见 [result.json](result.json)。

| 验证 | 实际结果 |
|---|---|
| Windows strict static / shared | 51 Direct + 20 Graph / 51 Direct + 19 Graph |
| Air / Mini static | 各51 Direct + 20 Graph；Mini只签正确性/可移植性 |
| Windows / Air source、relocated static、relocated shared | 各6个普通消费者；包含rank8捕获起点1/4/7、TopK双输出和mixed FFN |

合法dense View仍可有非对齐字节起点；冷降秩别名必须计入常量根native offset，且直接进入
原版Argsort的输入满足设备descriptor对齐。其他偏移感知算子继续四字节合同。Graph测试覆盖
root=1与alias=alignment−1相加后重新对齐，断言不多复制。没有新排序算法、公开API或热期搜索。

Windows测试与交付使用DLL邻置、PATH unset与非交互错误模式；
W1/Air日志未见VUID/同步错误。最终两factor配置保留已运行Windows static EXE的精确hash，
因此复用其测试记录，不重复计数或运行。构建/断言失败和fixture修正日志保留于原始归档。

已核对归档后清理Windows/Air producer与展开源码，共1,933,074,359逻辑字节；保留SDK和
consumer。统一实现LOC为47,202行/193文件，不含测试、bench和第三方。

15个通用交付门现已在固定 `5f4df55b` 验证副本通过，共45条source/static/shared路径；
[独立交付回执](remaining-deliveries-receipt.json)与[清理回执](remaining-deliveries-cleanup.json)
保留各自身份，未把早先96个native门及1个组件合同改签成同一候选112门。
归档核对后删除45个重复producer目录，共9,492,056,710逻辑字节。C1完整真实权重片段、索引/状态接入、
固定优选与实现淘汰、性能和矩阵签收仍未完成；CPU oneDNN性能资格BLOCKED不变，没有P1。
