# F5 candidate common33 sanitizer

固定candidate77df1c10c，actual WSL/GCC14.2/shipping/static。ASan/LSan、UBSan、TSan各
实际33项通过：容量/metadata identity/owned版本、OOM/overflow/并发generation、materialize
indices/mask/GQA/有效区、精确page guards/F4组合与F2页复制。没有回放原F4或算子目录。

Windows Git candidate内容80个变化输入经hash核对单向复制到discardable WSL sanitizer
source，保留原F4 compiler/configuration trees/cache以节省不变依赖；另立预声明F5新factor
san gate累计，未把F4记录改成F5。每mode configure300/build1800/tests120，consumer1，
库按实时内存j4/j2/j1，跨机器重型串行。全部命令/source/build/输入hash/时间见
[receipt.json](receipt.json)与raw.zip。

TSan首次exit66在测试前报unexpected memory mapping，保留FAIL；同tree/flags只对重试
进程setarch x86_64 -R关闭ASLR，原累计不重置，无suppression、global kernel或产品改动。
重试33项PASS。GCC旧fixture misleading-indentation warning保留，未因此改candidate。

这里只签common33的三个san facet，不代签CPU Direct/Graph控制器、native host-control、
GPU shader/driver或private heap；这些适用性/验证仍随候选卡核对。goal active、非CLOSED。
