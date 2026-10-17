# 构建与组装：选择、事实与交付

构建系统把用户意图投影为真实target/依赖闭包；它不决定模型运行放置或每个算子的Provider排名。
源码、静态SDK和共享SDK保持相同产品入口和计算语义。操作步骤只见
[初始化指南](../developer/setup/README.md)与[交付指南](../user/distribution.md)。

## 一份内部图，两种消费层

[`HyleComponents.cmake`](../../cmake/HyleComponents.cmake)声明factor直接边、公开头例外、trace消费和测试夹具。
同一声明推导闭包、拓扑顺序、真实内部target link、产品聚合和安装集合；模块不再手写第二份内部图。

用户通过公开`hyle::<product>`消费。内部factor只为源码、类型或交付边界而拆分，
不把解环的小模块变成新产品。维护时可以精确选择factor，父工程和安装消费者仍使用产品。
没有职责的bridge/facade不因架构图对称而存在。

生产闭包与测试夹具闭包分开：测试可复用普通factor，但不能因此制造新公开产品或改变SDK声明。
普通tests可与install共用构建；改变ABI的测试插桩不能作为生产SDK导出。

## 配置只表达无法推导的选择

| 配置层 | 责任 |
|---|---|
| 用户意图 | 选择产品、CPU/GPU/hybrid模式、GPU API、交付及需要的协议能力 |
| 开发诊断 | tests、bench、sanitizer、trace、精确ISA与内部factor隔离 |
| 工具和获取路径 | 已安装compiler/SDK、工具路径、锁定源码cache或原始archive override |
| 内部事实 | 目标平台、选中factor、实际可用实现、capability与需要传播的宏 |

[统一初始化](../developer/setup/README.md)将模式/API转换为后端集合，底层CMake仍只有一个
`HYLE_COMPUTE_ROUTES`集合，不增加CPU＋GPU或逐Provider布尔选择。
路线、公开专家入口与混合桥接由内部装配描述推导，factor依赖仍只在中央DAG声明。
未选后端不探测/下载；已选的完整产品缺必要依赖就明确失败，不静默形成lite版本。

普通能力用ON/OFF；确实允许探测降级的能力才用AUTO/ON/OFF，ON缺件失败、OFF不探测。
从平台、产品或target已能推导的事实不得变成第二个cache输入。
SDK root/archive override仅改变获取位置，不改变版本、能力或锁身份。

## Target承担使用要求

公共头实际需要的语言、include、编译语义、link及presence定义由owning target传播，
私有实现设置不泄漏给普通用户。`HYLE_*_API/BUILD/STATIC`由构建负责，用户不手动同步宏。
公共对象布局不能因消费者自行定义feature而改变。

Core-only不链接无关profiler、测试runtime或GPU SDK。更高ISA的实现TU不抬高普通consumer floor。
目标架构使用compiler/显式目标选择归一化，不能用Visual Studio或macOS宿主架构代替实际target。
一棵build目录对应一种原生目标/工具链，源码cache则是跨build资源。

父工程保有自己的编译器、flags、输出目录、sanitizer和包策略。
第三方的generic options在局部作用域设置，不FORCE改写父cache；upstream不能遵守时，窄适配保存其真实边界，
而非再建立通用配置语言。

## 依赖身份与原生接入

[`HyleDependencyLock.json`](../../cmake/HyleDependencyLock.json)记录来源、revision/archive/hash与许可证。
模块请求具名依赖，不另写pin或下载器。原文源码cache保持不可变，补丁在build-local目录按明确身份生成，
binary/subbuild与共享source cache不互相包含。离线使用相同身份与既有禁网语义，缺件具名失败。

CUDA等machine SDK由系统提供，不把SDK/driver冒充FetchContent源码。
Provider接入由owning factor描述原生target、必要options、usage requirements、资源和许可；
尽量使用厂家target，而非复制完整源文件清单。host工具与目标runtime分开准备，不能复制整套SDK代替发现。

## 按需安装与运行资源

安装metadata从生产target/资源图生成，按请求组件先导入必要外部依赖，再导入对应闭包。
请求Mem不应探测CUDA或Torch；重复find可补充同一SDK的产品，不混用不同安装前缀的同名实现。
源码与安装target提供同样的公开使用要求，内部detail targets只是链接闭包，不成为用户选择项。

静态库的必要runtime链接要求仍需传到最终应用；共享库隐藏内部vendor符号。
RPATH与loader路径须可迁移，构建机器的SDK绝对路径不能写进安装合同。

`hyle_stage_runtime`沿实际链接owner处理DLL和显式额外资源，不能按“应用似乎只调用CPU”省掉已链接GPU依赖。
普通内部imported依赖交给CMake原生闭包；Hyle入口和额外资源owner需要在调用处可见，
不能偷偷创建跨目录全局SDK注册表来透明支持任意scope。按实际配置选择artifact，保守保留同artifact依赖。

许可文本与native资产同属交付闭包，不能只安装头和archive；规则见[许可](licensing.md)。
ABI、数值、资源寿命、平台限制仍是工程约束，构建成功不代替真机计算或性能结论。
