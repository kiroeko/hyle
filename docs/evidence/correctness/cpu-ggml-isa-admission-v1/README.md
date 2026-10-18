# CPU ggml优化程序：当前实际证据

这是CPU优化程序接入卡的证据，不是完整CPU算子/路由关闭记录；其它路线旧F35待验仍保留。

数学/资源代码候选`bf4027e7dde9371b984a0322275dcaf6444648a2`；三交付安装路径纠正候选
`d0b774c503b800dfa66df84a4d0af63945456df9`只改CMake私有shim路径，回归数学身份沿用前者。
当前固定候选`129f36a9f0c5c31799728cc32afb8fc74f8d3eb8`追加的修改仅在测试接缝：GCC警告括号与
各直接CPU消费者声明同一runtime helper；没有新增数学、公开API或另一执行队列。

## 已运行

- Windows低并行2构建，串行CPU回归：8/8 CTest目标，383项GTest通过。完整输出为
  `validation-logs.tar.gz`中的`cpu-regression-LastTest.log`与`tests.xml`。
- Windows源码、静态SDK、动态SDK：真实普通入口F16 MatMul及完整RMSNorm→MatMul Graph，
  每种交付各重复两次；Direct显式核对实际Provider是ggml。安装consumer只给HYLE安装前缀，
  由现有helper将优化程序DLL放到exe旁，没有手工设置vendor目录或手工补DLL。
- Linux/WSL GCC14.2：CPU runtime/Provider 5/5 CTest、51项GTest通过，涵盖资源故障、
  narrow与实际GGUF算术；源码、静态、动态的普通F16 Direct/Graph消费者均各重放两次通过，
  未设置`LD_LIBRARY_PATH`。Linux同步清单绑定Windows原文件SHA，不反向传产品代码。
- 两平台静态/动态SDK均实际搬移前缀、重新发现与链接，普通F16 Direct/Graph通过，随后恢复
  原验证前缀。Windows缺程序/错ABI/必要回调缺失的负向消费者得到预期失败；这些具名负向项
  退出1是预先声明的成功条件，不能与初次接缝失败混淆。
- CPU OFF在故意无效Torch root下配置通过；检查无CPU优化target/资产registry/活runtime。
  冷ISA单元删除每个必需能力、空事实和非x86事实均不启用AVX2程序；没有物理非AVX2主机验证。
- `receipt.json`逐项保存候选、命令、退出值与耗时，日志归档保留失败及修复后的结果。
- 统一LOC脚本WORKTREE统计60994实现行/248文件，排除测试、vendor与CMake；
  C++产品内容对应以上候选，统计时仅Native裁决文档处于未提交状态。

## 失败与有限纠正

首次运行误用tests ON/install ON配置，后改为既有协议要求的tests ON/install OFF。
交付fixture曾误用公开接口，修正fixture后源码计算通过；未修改产品API。
一次新建静态验证目录触发不必要的完整oneDNN重编译，主动终止后改用已资格化producer，
保留该次部分日志，不能算完成或产品失败。一次脚本遗漏sys导入，在已成功配置处恢复，
没有重复编译或回归。

真正产品缺陷为私有SDK shim查bin而MODULE实际安装在lib；纠正后静态/动态安装consumer通过。
这是本卡第1次产品根因纠正。DLL仍通过helper放在Windows exe同目录启动。
Linux首次Provider测试因缺少helper没有RPATH，而资源测试与绝对路径原module加载成功；补充
测试接缝声明后通过。GCC三处GTest条件断言要求显式括号也已修正。均未放松数值或编译警告门。

## P0与证据边界

原CPU target的SOURCES/INCLUDE/DEFINITIONS/LINK依赖生成完整私有优化程序；每份CPU初始化与pool/
plan/整图compute/free属于同一绑定程序，slot借用同版POD图与workspace。原数学和fallible资源源码
没有复制或改写；CPU/OS事实复用Platform缓存。探测/模块加载/匹配只在冷prepare，重放不搜索或计时，
不拆成熟融合图。进程保留程序模块，slot按生命周期释放其pool/workspace；其它平台保留原基线。

尚未运行Native与新优化程序的性能裁决，不宣称任何自研WIN或CPU整体完成。
selector保守全库计划未执行全量gate，本证据只描述上面的实际CPU验收，不把计划标成PASS。
原标准CPU构建已验证的通用component/runtime资产合同按未改工具身份复用；本卡不重跑全库或GPU。
`validation-logs.tar.gz`同时保存Windows与Linux原始日志/receipt/fixture驱动，负向和修正历史均保留。

801个canonical产品/构建/脚本输入已与Windows和Linux验证副本逐文件核对，没有额外产品输入。
全部C++与头文件原本一致；发现的差异只有未用的preset/仪器/合同脚本、Windows构建不执行的Linux
link/staging分支，及Linux helper注释。已从Windows单向补齐，最终完整SHA清单一致。
这些已源码证明不影响该次执行的差异不触发重新构建或P1；保存前后身份记录，不能冒称此前整树SHA一致。
