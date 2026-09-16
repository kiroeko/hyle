# R4 C3：HYLE CUDA SwiGLU 初始产品验证

Owner `compute_provider_cuda` 仍由 [R4 CUDA 卡](../../../docs/implementation/tasks/compute/cuda-v1-four-route-foundation.md)
绑定为 AUDITED。本检查点实现真实中间计算片段；完整 FFN、普通入口、Graph capture、最终三交付及
P1 仍待本包接线，不用单个 kernel 签完整路线。

## 实现与已运行

- 独立 `compute_provider_cuda` factor 与 `compute/providers/cuda.h` 专家工厂，公共头无 SDK 类型。
  CUDA OFF 的 absence 不探测 SDK；ON 时只编译该 Provider 与必要 Compute/CUDA backend 闭包，
  本次 W1 配置未包含 CPU、oneDNN、Graph 或 cuBLAS/Marlin。
- FP16/FP32 dense SwiGLU，packed halves 或 separate；FP32 中间计算，输出一次转换；非空且输出
  元素数不超过 INT32_MAX。shape/type 在冷阶段固定；单个 native launch，没有搜索、分配或同步。
  不支持输入输出重叠；submit 在记录 kernel 前检查实际 binding。没有额外 native invocation 状态
  或 caller workspace；Compute 管理真实执行槽。driver/code image 全局存储没有被声明为零。
- W1 MSVC 19.50、CUDA 13.3.73、RTX 4070/SM89、shared/Werror、并行度 2 构建通过。
  含 SM80 PTX 与 SM89 native image，实际 GPU 资格只给 SM89。
- 公共头与四个 GPU 测试最终通过：48 个 type/layout/shape 组合，双槽、三次变化输入，覆盖
  M=1/7/17、H=1/127/256/513、尾部、NaN/Inf/零及独立 double oracle；非法资源/shape、alias；
  pending Completion 保留已释放的 prepared/slot 与关闭 Context；逐 metadata allocation OOM 回滚。
  Completion 预留后的三次 replay 没有新增 HYLE metadata allocation，销毁后 live count 为零。
- 初次四个测试中三个通过；alias 测试误把运行期 binding 检查放到 prepare 断言。核对共同合同后
  只修改该测试，验证 submit 在进入 Provider 前拒绝重叠并保持槽空闲；仅重跑此失败用例且通过。
  其它已通过用例与产品代码未变，没有为测试修改公开合同。
- selector 中仅增加新 factor 的 CUDA ON 配置；现有 45 个工具单测全部通过，没有改统计或仪器核心。

## 边界与证据

数值、寿命和分配测试均在真实 GPU Provider 上运行；没有制造 hang/device loss，也没有以 host
OOM 注入冒充 CUDA 设备 OOM。没有 P1、性能胜负或完整 FFN 结论。
`windows-shared.zip` 保留初次/最终日志、构建、DLL 邻置 hash、selector 计划、工具单测与 LOC。
启动前复制实际 cudart/MSVC runtime 到 exe 同目录；SetErrorMode(0x8003)，CTest 非交互、串行，
120 秒 timeout。`source-hashes.json` 绑定源码，`sha256.json` 绑定本目录产物。

中央 component seam 扩大的其它 native/package gates 尚未运行，不记 PASS，随 R4 C5 最终闭包
完成。本次统一脚本 Compute + Graph 实现 **32,562 行**，不含测试/第三方，self-check PASS。
