# R4 失败执行槽退役：产品验证

Owner `compute` / `graph` 的[当前施工卡](../../../docs/implementation/tasks/compute/r4-failed-invocation-retirement.md)
现为 CANDIDATE，产品冻结于 `a116280b`。本目录覆盖该有限修复的跨平台验证，不签完整 CUDA 路线。

## 已运行

- Windows MSVC 19.50 Release、shared、Werror、instrumentation OFF；Ninja 并行度 2。
  Compute、CPU backend、Native Provider、Graph 的九项选定 CTest 最终全部通过。
  首次运行 8/9 通过；一个旧 internal delegated 用例仍复用已取消槽，按新失败合同迁移后，
  只重建并重跑该 backend 测试，51 个用例通过；未重复其它八项已经通过且未变化的测试。
- 新增/迁移用例检查同步接纳失败、接纳前拒绝可重试、异步失败/取消、丢弃 Completion 后的
  真实终态、Provider 自带 completion owner、Graph command/capture、独立槽与重新创建槽。
  失败后不能再进入 Provider；Graph bind/reset/prepare_replay 也不能恢复失败 scratch。
- 修复后的 Compute 被真实 cuBLASLt 和 Marlin shared Provider 消费，两者各 2/2 CTest 通过，
  包括 GPU 数值/重复输入/双槽/寿命用例。受控失败 fixture 只证明共同协议，没有伪造设备丢失。
- 所有 Windows exe 的运行库实际邻置；SetErrorMode(0x8003)、非交互 CTest、串行、120 秒 timeout。
- P0 源码/生成代码显示复用原状态字节，同宽/同对齐由 static_assert 约束；Direct admission
  CAS、现有引用计数和 terminal store 数量不增加。terminal 增加成功/失败状态解码；不声称零指令
  增量。异常 attachment/aggregate 失败会排空已经接纳的工作，再返回没有 Completion 的错误。

## 证据与未完成项

`source-hashes.json` 绑定改动的产品和测试，`p0-codegen.json` 保留精确 dumpbin 段及成本限制。
`windows-initial.zip` 包含原始成功/失败日志、实际 DLL hash、完整生成代码、selector 初始计划与 LOC。
`sha256.json` 绑定本目录产物。P0 不把旧 CPU P1 换成新候选；没有追加 CPU P1 或改变已获批例外。

## 四环境普通验证检查点

产品与测试冻结在 `a116280b`。Windows 原共同层九项通过后，仅补其余直接 CPU consumer 和公开
source 样例 14/14，通过项合计 23 项；CUDA backend 的三项（含实际 late failure）也通过。
WSL 23/23、MacBook Air 19/19、Mac mini 19/19；四环境 relocated static/shared 的 Direct、Graph、
公开 CPU executor 样例均各 3/3 通过。Air/Mini 为 Native-only，不能把缺席的 oneDNN 行记为执行过。
原始 logs/CMakeCache/LastTest/候选与完整 archive hash 保存在三份 `*-ordinary.tar.gz` 和
`windows-delivery.zip`。所有 native 构建并行度 2；重型验证串行。

Windows source 配置首次同时指定 tests/install，构建合同正确拒绝；修正 harness 分为 validation
和 SDK 配置。安装 consumer 的两个 runtime staging target 曾同时向同一 exe 目录复制相同 DLL，
一次 COPY_FILE 失败；仅将 consumer build 串行化后通过。保留初次日志，没有修改产品/构建治理，
没有以 PATH 替代 DLL 邻置。后续同目录 runtime staging 保持串行。

`ordinary-projection.json` 确认 C7 产品/测试仍与该候选相同；其后的 C3 SwiGLU 是独立改动，不把
C3 加入本卡的完成证明。适用 WSL ASan/UBSan/TSan 正在串行运行，尚未记 PASS；本卡未结案。
C7 候选 Compute + Graph 全实现 **32,252 行**；C3 检查点后的当前全实现 **32,562 行**，均不含
测试/第三方，统一脚本 self-check PASS；代码规模包含既有资产。

## 最终 sanitizer / 红队

WSL ASan、UBSan、TSan 各 12/12 runtime CTest 通过，使用同一个不可变完整源 archive；重型串行、
Ninja 并行度 2。TSan 依既有环境设置 `setarch -R`；外部 oneDNN/JIT 未插桩范围如实保留。
`wsl-sanitizers.tar.gz` 包含三套完整日志及候选/配置身份，`windows-cuda-backend.zip` 保存三个
CUDA backend 回归。全部预声明门已完成，红队/P0 限制和精确命令由当前施工卡与本目录日志记录。
本次状态记录不重建，不启动 CPU P1；R4 普通入口/完整 FFN/交付/性能仍由其独立 owner 完成。
