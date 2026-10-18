# F5 ragged多请求实际append/F4首批

AUDITED consumer证据，不是candidate/全F5。Windows/WSL CPU Direct+Graph、WSL RTX4070
原FlashInfer Direct+Graph、Windows RTX4070 Vulkan、Air Mac16,12 M4/MLX及M4/MoltenVK
typed Direct+原native Graph全部PASS。每CPU/native profile F16/BF16×Direct/Graph×old/new/
old12episode；两个CUDA journey各6，共72episode，每个独立验证两request的全部有效Q。
七raw archives保准确身份/命令/inputs/history、原FAIL和同根累计预算，无预算重置。

实际数据操作：old ragged KV offsets0/3/5；first request追加一行后new0/4/6，因此second
request两行从old位置3/4搬到new4/5，不能只有indptr更新。Qcap3/active2、KVcap8、GQA4:2/
D64、F16/BF16。旧readonly K/V保活，new独立F1容量writers，原F2 phase0 clone有效
prefix与tail初始化，phase1明确依赖append；完整新容量成功ready才F4 dirty替换K/V/layout/
expected-base发布，clean query原block继续分享，old版本仍可实际消费。

独立uniform SDPA oracle：first request old/new/old分别0.25/0.375/0.25；second始终0.625。
原CPU固定Direct/Graph、CUDA原native plan/update/kernel/原Graph region、Apple/Vulkan
typed native controller与原GPU Program真实执行，未以CPU代GPU。CUDA Graph同identity
schema与原invocation，consumer上下文保持原provider admission，不造第二executor。

原F2只export成功写过的logical范围，不能把未写tail伪装whole ready。本测试显式copy
ready NaN padding source到new unused两行：clone K/V有效prefix2560bytes、append512、
tail init1024，total4096。whole-capacity clone再append会是4608；本事实不称ragged独立
页CoW，不隐去初始化或readonly整buffer使旧版本保留期间必须另分配buffer的成本。
数学只读同metadata有效区，NaN padding不进入结果；copied spans完整覆盖new readonly容量。

最初fixture默认保留paged extents而被原ragged schema拒绝的FAIL保存；改显式pages/refs0、
page_size1。原MSVC pair常量窄化/WX编译FAIL与CUDA TU漏include MaterializedBatchProgram
编译FAIL也完整保存；改明确float/u16数组与自含test header，不改产品validator/API/maths。
public源码与头均未变，common33/九头及原controller矩阵复用fc9e425bf签证。没有回放旧
模块或算子测试，也没有因反例新造“产品缺口”重开旧F1/F2/F4。

新增--ragged-append-only测试入口；最终七profile命令只运行本轮ragged旅程，旧矩阵结果
来自匹配产品代码/合同的前轮证据。前段完整fixture运行也保留，但不需要为末次参数
改变重跑所有old cases。库按available memory j4/j2/j1、test1，重型跨机器串行，原tree/
cache继续；不是installed static/shared/loader/san/P1或单机吞吐证书。

仍待完整producer/accepted-prefix/error/alias/OOM/cancel/unknown/close风险、18适用SDK
格/san/P0/LOC/guide/redteam/legal final closure；整个A/F4/F5范围和用户末轮foundation/
高性能单机engine代码量评估保留，不能由本72episode签整个goal。
