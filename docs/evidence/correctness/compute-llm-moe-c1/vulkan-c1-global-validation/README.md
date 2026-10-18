# C1-Y32 Windows 所选工程验收

不可变候选 `c733aa93a8c19e1e9d5747ee54c292db6075c5a1`；selector 产品 baseline 为
`4b3ca598e7dacb752923658b3ff17bdaa4281b1c`。Vulkan 产品仍为 `55fe0b8a` 的实现，
Y31 的真实 BF16 状态片段及 Windows/Air 六交付不重复运行。

[执行结果](selected-gates-result.json)与[原始证据](selected-gates-raw.zip)保留全部所选命令、
原始输出、退出码、CTest 明细、工具链/离线来源及 hash：96 项 maintenance/configure/
原生 correctness/headers、1 项 component-contract、15 项默认产品交付，共112项 PASS；
未选项及理由原样保留。Windows 的 Apple OFF 公开头不能代签 Apple runtime。

[交付收据](remaining-deliveries-receipt.json)与[交付原记录](remaining-deliveries-raw.zip)证明
15 个组件各有 source/static/shared 三条真实消费，共45条；包括安装重定位、独立消费者、
包卫生检查，其中 header-only 项签接口包消费。重型步骤串行、并行度2；Windows
HYLE/vendor/CRT DLL 邻置、PATH 隔离、禁错误弹窗。本组默认交付证明包可消费，
不替代各路线算子或模型的数值验收。

component-contract 首次失败来自本轮离线夹具：在未初始化的空 cache 中先放了锁定 archive，
产品正确拒绝非空且无标记的 cache。原失败保留；改用现有 HYLE 初始化函数准备空 cache，
再放同一锁定 archive，仅该门重跑一次通过。产品及公共工具均未改，96 项原生门没有重复运行。

[清理收据](remaining-deliveries-cleanup.json)记录归档核验后删除45个重复 producer 目录，
释放 9,492,224,026 逻辑字节；保留重定位 SDK、独立消费者、共享来源 cache 和证据。
[统一实现 LOC](all-implementation-loc.json)覆盖全部 Compute/Graph include 和 src 组件，
排除 tests/bench/vendor，使用统一脚本并自检。

本节没有 P1、默认性能赢家或自有实现淘汰裁决；CPU oneDNN Q2 资格仍 BLOCKED。
四路线完整成本优选、C1 整包与64行/256格矩阵仍未签收，C1 完成通知尚未触发。
