# Windows runtime DLL 增量 staging

## 维护卡 / SCOPED

- owner：build/package；baseline `1644bdc23a6a94a98ac65ccc8bb94fbf64897756`。
- 反例：CUDA 私有函数改变 DLL 但不改变 import library；Ninja 不重链接 source sample EXE，因此
  `hyle_stage_runtime_dependencies()` 的 POST_BUILD 不执行，样例加载旧 DLL。
  `custom-execution-v10-source-consumer` producer/staged SHA256 分别为 `133d8c83…`/`fe600192…`；
  更新到相同 producer 后，原本失败的真实跨 stream consumer 通过。
- 范围：源码与 installed CMake helper 的增量 runtime staging；不改变产品 API、组件图、工具链、
  CUDA discovery、性能仪器或模块状态。
- 审计/纠正预算：一个表示、一个实现、一次纠正；先确认 CMake/Ninja 依赖语义，再改公共 helper。
- 验收：同一已有 EXE，在只改变依赖 DLL 实现而 import library 不变时，增量 build 必须更新 runtime；
  source/shared-install 两种 helper 都覆盖，首次构建和无改动构建保持正确；Windows 代表性 consumer、
  docs/closure/diff。非 Windows helper 仍是无操作，不重跑 P1 或全库。
- 临时操作：构建后从本次 producer/SDK 显式 `cmake -E copy_if_different` 到该 consumer runtime 目录，
  核对 hash 后才运行；不把旧 staged artifact 的失败误判为新候选结果。
