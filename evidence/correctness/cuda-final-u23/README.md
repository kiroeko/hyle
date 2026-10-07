# CUDA U2/B2 与 U3/G1：Windows 接线批次

产品候选 `45a1d4a8b0950d4ae7d680aea526cb8055fffab8`，Windows canonical。
现有 `C:/hv/m2b1-cuda` producer，shared、CPU+CUDA、SM89、Toolkit13.3、MSVC19.50；
验证形态 tests ON／install OFF，不冒充 shipping SDK。

本批实现真实 driver release/API 独立冷事实与稀疏 unknown、原 stock GGUF 八类型映射及目标八格式 embedding。
原数学／库 pin、硬件数值资格、资源／失败与热执行保持；DLL 在 exe 邻目录后才启动测试。

## 实际结果

- 六 factor、七 CTest 进程集中 build j2 / test j1：原主批 **441 个 GTest 通过**，14个用例因
  未传 fixture 路径跳过。CTest 的7/7进程成功没有被当作零SKIP完整验收。
- 找到现存 `.scratch/d5-qwen35-six-profile-fixture`，只补原缺项：Direct9项／Graph7项零SKIP通过，
  其中14个是原跳过项、2个是同一过滤器选中的直接同族用例；没有重建或重复529秒主批。
- backend 两项 snapshot/sparse matcher 零SKIP通过；XML记录实际 driver release `617.14`，
  原 native NVML差分也为617.14，CUDA API整数独立。原 Context move/close、unknown事实匹配与数组复制均有检查。
- 主批的新 stock MatMul／grouped expert／embedding 与 Graph replay均通过，包括原codec字节、
  变化 activation／IDs、坏ID失败前输出保持；只证明本批所选接缝及原已有消费者。
- 源码／静态／动态安装消费、Linux、当前104制品全部精确合同／新版分支、IQ1三扩展、non-GGUF、
  全条件 best-of、清退与P1仍待后续批次，不能把本批写为CUDA收口完成。

原 configure/install flags 冲突及 helper namespace 编译失败均保留在[批次归档](windows-batch.zip)，
具名修正后 build/test完成。没有P1、没有噪声赢家或全平台SOTA结论。

统一 Compute+Graph 非测试实现：**60,273 行／251文件**，`--self-check`通过。
