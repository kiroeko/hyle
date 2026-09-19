# Y26 受检 Gather：Windows static 首轮验证

身份：baseline `858b035b` + 固定 implementation patch SHA-256
`7ce22084d7628072c81c2198edec1a05e15d4f263c70c2c748d6c25a4d258ac9`；完整源码 patch、配置、命令和原始结果见 [raw.zip](raw.zip)。
[结果](result.json)记录各 EXE/CMake cache/日志 hash；未使用性能仪器或 timing 判断。

- 19 provider + 60 Direct + 21 Graph = **100 用例通过**，无 disabled/skip。
- 新增 9 Direct/生命周期 + 1 Graph 先做定向运行，不重复计入100。
- 原版源码合同、实际 MASM parent composition、两个公开头检查均通过。
- 六种原始存储、I32/I64、任意轴/rank8/stride/byte origin、空域与129尾块；
  坏索引保持 Recorder output，失败槽退役、新槽恢复、独立结果及常量寿命；
  非coherent分支、flush/invalidate失败、取消/丢弃、部分提交drain与错误优先级。
- Windows DLL 邻置、实际 EXE 的 PATH unset、SetErrorMode0x8003；Vulkan validation
  开启，GTest与CTest XML/log均未出现VUID/同步错误。编译类测试保留所选编译器环境。

shared、Air/Mini、真实模型片段及source/static/shared实际消费尚未运行。
本记录不签Y26整体验收、C1矩阵或性能优选；没有更改ggml算法或vendor patch。

归档SHA-256：`7c4fe4452476b7c34f38a17849e86172e28eb82ed74dc5e26c9e9d2f0e3d1c85`。
