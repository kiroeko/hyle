# F5 固定host Graph控制器首批

AUDITED有限checkpoint，未签CPU Direct/native controller/candidate/全F5。Windows MSVC19.50/
STATIC和WSL GCC14/shipping原source树：原CPU Direct/Graph八域16oracle与新控制器八域
8oracle，共24/profile；另F16/BF16页闭环各old/new/OOM-preserved/old四oracle共8/profile。
原native CUDA Graph两域12oracle与逐页六oracle定向复核PASS。common33项/八compute头、
两个Graph头独立TU、modified Graph owner TU/WX和新graph_batch目标/WX通过。

MaterializedInvocation冷生成自己的immutable fixed recipe Graph并调用原compile/leased
invocation，不接外部可篡改证明。第一档host/system-resource；原Context/Queue/原math与
completion/permission/alias/retirement全复用，没有第二executor、GPUfallback或新kernel。
prepare_batch生成owned同版本indices/mask；paged只接exact same-layout frames，ragged
为ordinary ready readonly K/V。调用者不能换metadata，submit消耗当前prepared episode，
下一run显式prepare/bind；warm不生成索引/编译/规划。此API没有代替完整CPU Direct控制器。

private bind_gather核同一Graph identity，capacity唯一logical路径为reshape→axis0 checked
Gather并用指定indices；root/reshape不能published或有其他logical读者。实际physical plan
只允许其Gather operation/原qualified region及非materializing metadata View，插入整块
copy/其他alias读取拒绝。原bindings保存actual page guards，原F1 use batch统一保到原
Graph prefix实际退休。五个实际图案例：合法、额外Copy、错误indices、published reshape、
foreign proof；只有合法图通过。原native G-PAGE资格保留，无普通v19/Invocation字段税。

冷output backing cap包括63bytes最大padding；借用allocator/ResourceBudget按原wrapper
与CompileOptions各自合同。独立output backing owner只保recipe/allocator/budget，不反持
pool；分配writer并接owner后冷detach原pool root。private take_write只从成功idle、无旧
result/completion的caller output取回现有writer供reset，不创建新权限。未提交重复prepare、
成功run未取output也可prepare；外部storage lease/native-use阻止原F1覆写。关闭并释放
控制器后escaped output仍可原F1 native-read，最后guard释放后metadata quota0。

actual paged cap5页、old2/3→new2/1无page0，Qcap2/active1/GQA4:2；NaN tail由owned
indices排除。保旧页时new page1可写，原F2显式clone→append：clone1024+append512bytes，
whole K/V arena5120未复制。F4完整K0/K1/V0/V1/layout目录dirty替换、expected-base成功
发布，clean原block共享；new结果0.375，old重读0.25。wrong/duplicate bind不能运行混合
episode；old result拒绝prepare；update metadata OOM保原成功输出，exported storage拒绝
复用且旧bytes不变；cold output byte cap/metadata OOM、close/escaped native guard通过。

原MSVC误用GraphBuilderOptions/CompileOptions allocator字段的编译FAIL完整保留，改用
真实metadata_allocator接口后成功，没有修改旧API。精确命令/身份/hash/input history/
原所有FAIL与累计预算在四raw archives。原tree/cache/root预算复用、库按available内存
j4/j2/j1、test1，无旧算子测试或全库回放。一次调度偏差如实记录：common八头编译尾段
与WSL CUDA定向增量门短暂重叠；两门均终态PASS，后续重型仍要求跨机器串行，不据此
伪称全程串行或重置预算。installed/迁址/loader/san尚未执行。

仍待CPU Direct精确页控制器、Apple/Vulkan原native完整同版本控制器、ragged实际append、
其他producer/error/unknown/alias/OOM/cancel/关闭完整风险、18 SDK适用格/san/P0/LOC/
guide/redteam/legal closure。整个A/F4/F5范围保留，本checkpoint不签整个goal。
