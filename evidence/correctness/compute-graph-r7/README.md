# R7 验收证据（进行中）

当前不可变候选为 `87ae583407cbb5ed13108c3837b020e6e343a3ca`。R7 仍在验收，不能由本页已完成的
项目代替剩余 Windows/WSL TRACE OFF/ON 三交付与 Windows fixture 检查。

## 已完成的结构与配置检查

- [P0 源码证明](p0-source-inheritance.json)：四份运行源只增加静态 trace hooks；十五个算法、
  路线和依赖输入保持不变。冷报告按节点线性复制，没有新增运行共享状态、锁或原子。
- [关闭跟踪的生成代码](off-codegen-final.json)：真实 MSVC Release shared 对象中的 29 个函数 /
  COMDAT 保持指令和调用目标；仅消除已逐项证明的匿名 namespace 路径 hash、同一 fatal assert
  的源码行号与路径差异。这是 P0，不是新的耗时测量，也不证明整个 binary placement 不变。
- [B1 配置结果](b1-configure.json)：九个实际 configure 正反例通过；absent factor 不泄漏实现、
  私有依赖或 Tracy。配置 probe 只验构建条件，不充当任何真实计算路线。
- [输入与原始归档身份](p0-configure-identity.json)保存 source archive 创建记录、脚本、配置日志和
  全部 hash；[原始归档](p0-configure-raw.zip)已逐条重算 hash。

产品代码来自 `e84a7ad3`。之后三个提交只修正测试：MatMul 原有语义是 declared-type accumulation；
custom operation descriptor 必须 trivially copyable，寿命测试把 shared owner 放在实际 lowering
callback 中。`src/include/cmake/samples/bench/scripts` 在两个候选间逐树一致。生成代码记录保留
当时的候选身份，未把旧二进制冒充重新构建的 `87ae5834`。

本轮没有新增 P1 采样。历史资格仅按已证明不变的计算路径及原机器、版本、格式和 shape 继承；
TRACE ON 的 profiler 成本不属于 OFF 性能证据。原 TSan 不记作本轮重跑。

[候选后的有限源码复核](source-review.md)已完成，未新增产品 FIX；报告 ownership、失败清理、
冷热成本、依赖传播和比较资格的检查与运行门分开记录。

统一 [LOC 原始结果](implementation-loc.json)统计整个 Compute + Graph 为 **37,035 行 / 130 个
代码文件**，不含测试、bench 或 vendor。它包含所有已登记 backend/provider，不是完成百分比。

## 待完成

Windows broad selector、Mac host sanitizer、Mac 三交付与 F5 fixture 已完成下述覆盖。Windows/WSL
TRACE OFF/ON 三交付与 Windows F5 fixture
检查继续按 [R7 当前卡](../../../docs/implementation/tasks/compute/four-route-foundation-final-review.md)
执行；未执行项保持待验，六个 owner 尚未 CLOSED。

## Windows 111 门覆盖完成

[覆盖身份](windows-selected-identity.json)逐项列出全部 111 个选定门及七个 skipped reasons，
[原始归档](windows-selected-raw.zip)保留原 selector、两个失败续跑与最终成功续跑的结果、
命令、日志、receipts、context 和执行脚本，228 个条目已逐一重算 hash。

最终去重覆盖来自 39 个原批 PASS、9 个第一轮修正测试后的 PASS、63 个最终续跑 PASS。公共
docs/closure/configure 以较新的实际记录计一次；这不是把原来的 BLOCKED 整批改成 PASS。

三次失败均属于新增测试：Generic Graph 错将 declared-type F32 累加预期为另一枚举；native custom
寿命 fixture 把 shared_ptr 放进要求 trivially copyable 的 descriptor；native FFN 又出现同一枚举
预期错误。三个具名测试提交修正了假设，shipping 输入逐树不变；失败原始日志没有删除。

主 factor/header build 是 MSVC 19.50 / Ninja Release。续跑的 package smoke 没有继承
`CMAKE_GENERATOR=Ninja`，因此实际采用相同 MSVC 的 Visual Studio/MSBuild Release，显式
`--parallel 2`；记录按实际 generator 保留。全部十五个产品的 source/static/shared 消费通过，
不把这些 OFF 普通门冒充后续 TRACE ON 或 Apple 原生验证。

## Mac host ASan / UBSan 完成

[插桩身份](air-host-identity.json)与[原始归档](air-host-raw.zip)记录 Air 的 Release static、TRACE OFF、
CPU Native + Apple MLX + Vulkan/MoltenVK。两种 sanitizer 各覆盖 49 个具名插桩编译单元、7 个测试
程序 / 395 个 GTest；六个公共头门也已通过。原始命令、编译选项、日志、执行脚本与各自制品 hash
独立保存；复用 build 切换 sanitizer，不把第二次制品冒充第一次制品。

Apple/Vulkan 新增报告的 owner 释放、复制后读取、reset、metadata OOM 与失败后重试用例都实际通过；
Vulkan 的借入常量去重、owned/borrowed 字节边界及 quota 回收也有行为检查。Generic Graph 原有的
逐分配点失败回滚检查同时通过。

范围限 HYLE host C++；MLX vendor、loader/driver 与 GPU 指令没有因此获得 sanitizer 资格。
未重新运行 TSan，也没有新增 P1；TRACE ON 和三交付仍由独立后续门裁决。

## Mac OFF/ON 三交付与 fixture 完成

[交付身份](air-delivery-identity.json)和[原始归档](air-delivery-raw.zip)记录同一完整 source ZIP 上的
TRACE OFF/ON 两种配置，每种均实际运行 source、迁移后的 static SDK、迁移后的 shared SDK 的
Compute/Graph 消费者，共十二条消费旅程。SDK 消费时原 source/build 已隐藏；真实 compiler 定义、
linkage、报告与完整 FFN/custom 结果均检查。OFF 没有 Tracy client，ON 的 static/shared 各只有一个
实际 client 制品。

[fixture 身份](air-fixtures-identity.json)和[原始归档](air-fixtures-raw.zip)记录八个 primary/placement
程序构建、四个 Apple/Vulkan Direct/Graph 的数值 oracle 及退役入口负检查。没有运行 nanobench
执行采样；保留的 cold prepare 日志不承担新的性能裁决。

第一轮 fixture 配置把 public-product build 改成 developer-factor roots，被现有互斥门正确拒绝，
未进入编译。验证脚本改为独立的 fresh maintenance build 后通过；失败脚本、配置日志和 BLOCKED
记录与修正后的 PASS 均保存。没有修改公共构建门，也没有重跑已经通过的 Mac 交付。
