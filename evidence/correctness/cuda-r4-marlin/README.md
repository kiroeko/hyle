# R4 Marlin 初始产品检查点

Owner 为 `compute_provider_marlin`，当前卡仍是
[R4 CUDA](../../../docs/implementation/tasks/compute/cuda-v1-four-route-foundation.md)，
状态 AUDITED；本目录不签完整路线、三交付、默认 Provider 规则或性能。

## 已运行

- W1 Windows、RTX 4070 / SM89、CUDA 13.3.73、MSVC 19.50.35728；shared、Werror、
  CUDA ON、独立 Marlin factor，Ninja 并行度 2。配置的 component closure 不含 CPU/oneDNN/Graph。
- 最终 CTest 2/2 PASS：公开头独立编译，以及五个真实 GPU 测试。数值测试覆盖 M=1/7/9/16/17/33/64、
  K=N=1024；另有 K=128、N=4096/2816/64 的 tile/最小边界，共 11 组形状，每组双槽、三次变化输入。
  oracle 独立计算 FP16 scale 与 dequantization 后的 double dot product，核对 FP16 输出。
- weights/scales 压缩常驻、move 后空 owner、非法 q/scale/format/shape/资源、workspace 大小、
  native 32-bit 索引边界均覆盖；受控 pending Completion 保留释放后的 artifact/lane。
- host scratch OOM，以及 cold packing、prepare、invocation 所用 metadata allocator 的逐分配失败
  回滚通过；所有 metadata allocation 最终归零；预留 Completion 后重复执行不增加 metadata allocation。
  未主动制造设备丢失或 CUDA OOM，不能把 host/metadata 注入当作这些设备错误的证明。
- 45 个现有 selector 工具单测通过；只增加新 factor 的 CUDA ON 配置及 vendored owner 映射。

CUDA TU 采用上游要求的 relaxed constexpr。构建仍有上游 scalar descriptor 中两个 MSVC C4244
窄化警告；它们发生在编译期类型描述，HYLE C++ adapter 的 Werror 构建通过。未改 device 算术，未
把整个 vendor TU 宣称为零警告。原始日志完整保留。

## 证据与边界

`windows-shared.zip` 保存 configure/build/CTest/LastTest、实际 exe 邻置 DLL hash、selector 计划、
工具检查 receipt 与统一 LOC。启动前复制 cudart 与 MSVC runtime 到 exe 目录，父进程使用
SetErrorMode(0x8003)，CTest 非交互、串行、120 秒 timeout。
`source-hashes.json` 绑定本次实现、测试、依赖锁和构建输入；`sha256.json` 绑定本目录证据。
上游 Git blob、实际 archive SHA-256 和改造后的目录 hash 分开登记，不混同来源和 HYLE 适配内容。

中央 component manifest 和 dependency lock 改动让 selector 扩大选择其它 native/package gates。
这些项目尚未执行、没有记 PASS，留在 R4 C5 最终实际闭包集中裁决。当前没有 R4 CANDIDATE，没有 P1。

已确认后续依赖修复：Compute/Graph 当前 terminal callback 在失败后也清除 in-flight 标记，而
Marlin 的成功完成才保证 locks 可重用。当前公开 header 要求失败后销毁 invocation；仍须独立有界
修复共同层/Graph 的失败槽准入，之后才签完整 R4 生命周期验收。本次没有修改这些 CLOSED owner。

统一脚本当前 Compute + Graph 全实现 **32,183 行**，不含测试及第三方，self-check PASS；
其中包含既有资产，不能换算为已完成路线比例。
