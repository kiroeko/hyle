# R4 失败执行槽退役：初始产品验证

Owner `compute` / `graph` 的[当前施工卡](../../../docs/implementation/tasks/compute/r4-failed-invocation-retirement.md)
仍为 AUDITED。本目录保存本次实际产品改动的初始 W1 验证，不签完整 CUDA 路线或跨平台交付。

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

WSL/Air/Mini、适用 sanitizer、完整直接 CPU consumer 和四环境三交付仍未运行，不记 PASS。
当前是实现检查点；后续仅对这些预声明范围补验。Compute + Graph 当前全实现 **32,252 行**，
不含测试/第三方，统一脚本 self-check PASS；代码规模包含既有资产。
