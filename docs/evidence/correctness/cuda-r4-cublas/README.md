# R4 C1：cuBLASLt 产品接入检查点

当前 owner 保持 AUDITED。本包证明第一个真实 CUDA 计算 Provider 的 Windows shared 接入，
不代签完整 R4 默认路线、Graph FFN、三交付或性能。

- W1-WIN / MSVC 19.50 / CUDA 13.3 / RTX 4070 SM89；Werror，构建并行度 2。
- CTest 2/2 PASS：独立公共头，以及四个 GPU 测试用例。
- FP32 左右输入的四种行列布局组合；FP16 M=1/17、K=128、N=64 的普通/转置权重；
  每项三次变化输入、两个独立槽，以 double oracle 检查实际结果。
- 拒绝 host-only resource 和未支持整数类型；受控 stream gate 确認 Completion 尚 pending 后，
  释放 prepared/invocation 并关闭 Context，继续等待真实终态及核对结果。
- 句柄、描述符和算法冷准备；独立槽 workspace；FP32 输入使用 pedantic compute，
  FP16 禁止以 output type 存储 split-K 中间和。SDK 私有 storage 不假报零字节。
- 实际 HYLE、CUDA Runtime、cuBLASLt 与 MSVC DLL 均在 exe 同目录；非交互 CTest，timeout 120 秒。

`source-hashes.json` 锁定产品/测试/接线文件；`windows-shared.zip` 包含 configure、初次测试源码
编译错误和修正、最终 build/test、DLL hashes、CMake cache、selector 与统一 LOC。
LOC 为 Compute + Graph **31,486 行**，不含测试，self-check PASS；源码规模不等于验收百分比。

新增 factor 的 CUDA configure 条目沿用现有 selector 配置，45 个工具单测 PASS，没有修改选择器核心。
原始 selector 还因中央组件声明变化选出了全库 native/package gates；本次只运行 C1 实施检查，
其余选择项尚未执行、不记 PASS，随 R4 C5 的最终组件/交付闭包一次裁决。owner 未进入 CANDIDATE。
原 gate 的五项 skipped reasons 保留在原始计划中，没有将未执行项伪装为 skipped/PASS。

边界：plain rank-2、匹配 FP16/FP32、FP32 accumulation、16-byte tensor alignment；普通 MatMul
output 为 dense row-major，输入支持 row/column-major。workspace 最多 32 MiB，256-byte alignment。
未测设备、WSL、完整 packaging、Graph capture、OOM 专项及 P1 均未由本包签发。
