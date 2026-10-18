# CPU标准构建与原版SDK交付

候选 `eb7f97b8ff8d82071ce65aa3f6fcd89715f7b174`。产品真相为Windows；
[身份与输入SHA](identity.json)、[原始日志/驱动与收据](validation-logs.tar.gz)、
[selector原计划](selector-plan.json)完整保留。本包没有新增C++数学或变更原SDK内容。

| 实际门 | 结果与范围 |
|---|---|
| 普通`compute-cpu` preset | 配置成功，CPU/oneDNN/ggml/Torch AUTO启用，其余三路线OFF |
| SDK取得接缝 | Windows十项实际检查：锁定本地archive、离线命中/miss、坏archive、坏root、CUDA包拒绝、OFF、不支持AUTO/ON、缓存manifest不能被Torch_DIR绕过 |
| Windows source/static/shared | 每种交付14个GLM、14个DeepSeek官方Direct/Graph测试通过；原runtime七DLL的SHA与已验SDK一致，安装消费者只给HYLE prefix |
| Windows runtime helper | 真实DLL、增量、幂等、空闭包、ALIAS、条件边、不相关owner隔离通过；最终复核1.906秒 |
| 显式SDK定位 | 错误Torch_DIR/Caffe2_DIR/TORCH_LIBRARY/c10_LIBRARY缓存同时存在，显式root仍绑定三个正确imported library |
| 组件/交付合同 | `HyleCiComponentContract.cmake`实际通过（266.985秒）；导出不含producer SDK绝对路径 |
| WSL GCC14.2 CPU source/static/shared | 同一实际FP8投影：普通CPU Direct和Graph各两次重复执行通过；安装只给HYLE prefix，未设置LD_LIBRARY_PATH |
| WSL SDK | 原锁定archive与缓存复用；十二个`.so`文件SHA与原已验SDK相同 |

默认在线HTTP下载未重新执行：复用已经下载并取得完整数学验收的锁定原archive，避免另一份大文件。
在线分支只签源码检查，不声称实际在线下载PASS。WSL交付测试是FP8真实执行接缝，不冒充Windows
28项完整套件；原D5数学和P1依其未变C++/SDK身份复用，不新签胜负。

## 失败与纠正边界

Windows首次runtime脚本空extra-list解析失败，随后修正并通过；static消费者第一次误用旧Hyle_DIR，
清除旧配置后通过。WSL shared独立消费者真实发现`libtorch_cpu.so`缺失；只给exe加RPATH仍失败，
原因是Native adapter的DT_RUNPATH阻止祖先RPATH查找。固定为owner runtime登记、应用SDK路径和
Native自身DT_RPATH，最终shared启动通过。static最终完整输入也重新构建并启动通过。

另外的WSL失败为夹具使用空DEV_FACTORS、错误Graph方法/未传caller queue、错误exe位置和只构建
单target后安装缺Time库；它们不代表数学缺陷。收据保留首次非零exit code；部分同名重跑覆盖原raw
日志，因此本档不声称拥有每次失败的逐字完整日志。现有最终原始日志和成功套件可核对。

## 验证范围

selector对新增preset、SDK锁附属元数据和opt-in runtime helper保守选择120门。它只是影响计划，
没有完整`run`结果，不得把它写成全库PASS。本卡对这三个接缝逐项定界：旧preset和source dependency
记录未变；未登记runtime文件的其它owner没有新SDK路径；公共C++头、数学、调度、GPU和其它库
实现均未变。新增接缝由实际组件/runtime合同、CPU主consumer、反偏置空闭包/条件owner/失败和
三交付验证。其它factor数学复测不属于本卡风险，保持未运行记录；不伪造selector的REUSED/签署。
docs/closure另以当前状态提交记录，不因说明变化重建或重跑P1。

P0：首次解包/完整文件SHA检查在配置期；CPU context/prepare/submit执行路径不变。每个SDK身份只
存一份完整SDK；Windows每个exe只邻置必需runtime，Linux引用唯一SDK的loader目录，不复制整个SDK。
Compute + Graph统一脚本统计为60,866实现行/244文件，不含测试、bench、vendor；归档内`loc.json`。

本档只关闭CPU构建接缝。自研算子逐精确域与路由裁决仍未完成；ggml优化ISA资格也不能由本档代签。
