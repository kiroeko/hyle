# F5 公开CPU消费者与迁址static SDK原型

固定产品输入c092c5ba3，当前状态仍AUDITED。新增consumer只include公开Compute/Graph/CPU/
simd头，无src私有头、trusted detail桥或测试产品宏，普通链接hyle::compute/hyle::graph。
Win MSVC19.50/shipping/static源码原型和cpu,vulkan SKU的迁址安装均通过四域：paged/
ragged×F16/BF16，每域Direct/Graph独立mean=.25 oracle，GQA4:2、Q有效1/物理2、非整tail、
乱physical页2/0、padding与old-output拒复用。不是native Vulkan数学证书。

安装消费者原cpp实际复制到隔离consumer-source，hash一致；find_package只给新prefix，
安装前缀实际staging→relocated rename，cmake metadata无canonical src/build/旧staging路径。
六新前端公开头独立TU、未含cuda/apple的find_package请求具名拒绝、dumpbin依赖与干净
runtime PATH运行通过，完整命令/输入/hash/原profile树/累计见[receipt.json](receipt.json)。
没有从canonical include或原build加载HYLE头/库；静态HYLE仍保原第三方CPU DLL合同。

源码原型沿原graph-windows consumer root/预算追加；SDK是预声明F5-06的首次
cpu-vulkan-static gate，复用身份匹配F4 producer tree/cache，建立独立F5 cumulative记录。
后续同格candidate验证不重新取预算，不把原F4资格改写为F5；CMake再生成的simd生成文件
prerequisite重编译和所有真实stage保留，没有全库/旧算子回放或公共工具改动。库按实时内存
j4/j2/j1，consumer1，与其他机器重型串行。

当前只是public CPU SDK前置：实际Vulkan两条消费、新detail/split头/negative symbols、
immutable candidate、完整18格、san/P0/guide/红队未由此签，完整SDK格仍NOT_RUN。
有限最终清单见[最终计划](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/foundation-f5-final-validation-plan.md)，
变化选择见[validation-plan.json](validation-plan.json)。goal保持active，完整原目标不缩水。
