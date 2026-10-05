# CPU ggml优化程序：当前实际证据

这是仍在AUDITED轮中的分段证据，不是完整CPU或runtime关闭记录。

数学/资源代码候选`bf4027e7dde9371b984a0322275dcaf6444648a2`；三交付安装路径纠正候选
`d0b774c503b800dfa66df84a4d0af63945456df9`只改CMake私有shim路径，回归数学身份沿用前者。

## 已运行

- Windows低并行2构建，串行CPU回归：8/8 CTest目标，383项GTest通过。完整输出为
  `validation-logs.tar.gz`中的`cpu-regression-LastTest.log`与`tests.xml`。
- Windows源码、静态SDK、动态SDK：真实普通入口F16 MatMul及完整RMSNorm→MatMul Graph，
  每种交付各重复两次；Direct显式核对实际Provider是ggml。安装consumer只给HYLE安装前缀，
  由现有helper将优化程序DLL放到exe旁，没有手工设置vendor目录或手工补DLL。
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

## 未运行/未裁决

Linux优化程序和三交付、完整冷ISA/ABI失败与CPU OFF隔离尚未完成；不得标PASS。
尚未运行Native与新优化程序的性能裁决，不宣称任何自研WIN或CPU整体完成。
selector保守全库计划未执行全量gate，本证据只描述上面的实际CPU验收，不把计划标成PASS。
