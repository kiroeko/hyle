# 已知 TensorView 对齐的冷选择修复

<!-- MODULE REVIEW: AUDITED -->

owner `compute`，baseline `c8dd92ac`，父任务[CUDA best-of](cuda-best-of-completion.md)。
这是新的实际反例，重新定界旧CLOSED；原CPU规则/数学和宏验收不重开。

## 反例与有限边界

Windows static普通CUDA入口，以F16 Log `[1,2048]`、device Buffer的2-byte偏移view准备：
冷入口丢弃已知offset，只保留描述符/resource，选ggml的较强alignment，然后submit拒绝。
CUDA规则新增的256约束也必须在已有view冷选择中生效，不能在执行时重搜Provider。
原失败保留于CUDA C4 static首次日志；不是数值或性能FAIL，不重采P1。

## 有限 FIX 与验收

FIX-A：仅私有 `prepare_for_mode`/`prepare_views_for_mode`传递已知view，对具有明确resource
base对齐保证的view，按offset排除不满足proposal alignment的候选。descriptor入口不猜地址；
不解引用opaque handle、不读payload、不改公开API/对象/hot submit或公共治理。
FIX-B（原CUDA owners）：从原native target读取真实编译器配方facts而非依赖兄弟目录变量；
ggml版本在已有冷factory按真实Provider identity检查。仅测量候选把原native binding alignment
数据提升至256，复用原执行表/检查，descriptor先准备后换不合格view必须拒绝，无热重路由。

有限验证：原Compute单元门；Windows/WSL普通CUDA bound-view偏移与descriptor后换view拒绝；
23规则命中/失配、原25数学/资源与7 Graph消费者、两OS source/static/shared。旧模型559合同
和未变路线不重跑；shader/原完整成本结构未变的WIN复用，未跑不PASS。
一次具名修复、原C4 static失败同ID最后一次重试；不追加AA/P1/n或另起ID重置预算。
其余C4首次步骤仍按原预算，先冷审计AUDITED再写Core产品；完成回到父轮终审。

源码审计已确认：Mem view有offset/resource的不可变冷事实，原CUDA resource保证base256；
私有candidate已有各端口minimum_alignment，新增过滤无需新metadata/ABI/Backend查询。
原CUDA tensor已保存input/output alignment；Norm output与input0同dtype，用同一已保存
alignment即可。修改冷数据及复用原pointer检查，不加逐kernel vtable或新热搜索。
FIX-A/B冻结；同ID失败纠正理由为此具名cold alignment及配方作用域修复。

纠正构建已证实配方flag启用，Core/CUDA冷数据修改编译通过；route跨factor私有头缺少其
API宏owner显式include，原build-2 C4430保留。只加该既有公开头include，不改数学/配方。
原static configure/build同ID累计允许最后3次（包含初次PASS），不许第四次；其余未运行步骤
预算不变，static数学首次FAIL仍只许原ID第二次。配置输入只应是CMake/参数，原task-local
宽source目录导致不必要失配；本次不改公共治理/统计，也不重新做P1。

最终编译PASS。第二次relocate因首次未资格SDK目标仍存在而拒绝（未执行move），此setup失败
原样保留。核对C:/hv内明确SDK路径后移走首次目标，原relocate同ID只允许最后第三次；不重跑
已PASS的configure/build/install，数学仍仅原ID第二次。后续三交付使用一次新prefix，避免复用
初次失败目标导致无关重建；不把移动/脚本步骤计成算子完成。

最终consumer build仅测试缺少`<algorithm>`（新增guard核对的`std::all_of`），原build-2日志
保留。显式加标准头，原consumer build同ID最后第三次。测试头include不改变consumer CMake、
SDK或参数，已PASS的consumer configure-2按此具体配置等价证明复用，不再因宽源码hash重配。
不改公共runner，build依赖以这条原配置证明代替宽hash；数学仍只有最后第二次，失败即停止。

## static 最后数学批次终态及退回冷选择修复

原29项第二次结果28 PASS/1 FAIL：23完整赢家实际命中、变输入输出、偏移view回退与descriptor
后换违规view拒绝均PASS；managed回退失败。整批保留FAIL，原数学批次预算结束，不跑第三次。
已确认 `prepare_candidate`资源失配按标准返回Unimplemented，而上层Direct候选循环直接
返回该错误，未继续后续合法候选。新增有限FIX-C只在这条冷候选循环把Unimplemented视为
decline；InvalidArgument/Internal/OOM等仍传播，不改数学/ABI/执行。Core仍处于AUDITED，
先独立提交此审计后修源码。随后按新缺陷的独立窄正确性口径复核managed/偏移/非法资源，
不伪重命名原29项预算；保留原28项数学证据及原FAIL。其余source/三交付仍待，不能CLOSED。
