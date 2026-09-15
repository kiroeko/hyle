# Windows runtime DLL 增量 staging

## 维护卡 / AUDITED

- owner：build/package；baseline `1644bdc23a6a94a98ac65ccc8bb94fbf64897756`。
- 反例：CUDA 私有函数改变 DLL 但不改变 import library；Ninja 不重链接 source sample EXE，因此
  `hyle_stage_runtime_dependencies()` 的 POST_BUILD 不执行，样例加载旧 DLL。
  `custom-execution-v10-source-consumer` producer/staged SHA256 分别为 `133d8c83…`/`fe600192…`；
  更新到相同 producer 后，原本失败的真实跨 stream consumer 通过。
- 范围：源码与 installed CMake helper 的增量 runtime staging；不改变产品 API、组件图、工具链、
  CUDA discovery、性能仪器或模块状态。
- 审计/纠正预算：一个表示、一个实现、一次纠正；先确认 CMake/Ninja 依赖语义，再改公共 helper。
- 验收：同一已有 EXE，在只改变依赖 DLL 实现而 import library 不变时，增量 build 必须更新 runtime；
  source/static-install/shared-install 都覆盖，首次构建和无改动构建保持正确；Windows 代表性 consumer、
  docs/closure/diff。非 Windows helper 仍是无操作，不重跑 P1 或全库。
- 临时操作：构建后从本次 producer/SDK 显式 `cmake -E copy_if_different` 到该 consumer runtime 目录，
  核对 hash 后才运行；不把旧 staged artifact 的失败误判为新候选结果。

## 有限裁决（实现前固定）

- BRS1：source/package 使用同一份受安装的 helper；consumer 依赖一个仅作 staging 的 custom target，
  后者依赖实际 DLL 文件。每次构建该 consumer 都检查并只复制不同文件，不强制重链接 EXE，也不依赖
  import library 的时间戳。`TARGET_FILE_DIR` 在项目 CMake 3.25 floor 的 CMP0112 NEW 下不会反向依赖
  consumer；不制造 cycle。依据为 CMake 本机 `--help-policy CMP0112` 和
  [`add_custom_target`](https://cmake.org/cmake/help/latest/command/add_custom_target.html)。
- BRS2：按 consumer 实际 DLL 闭包处理，不看 HYLE_STATIC。静态 HYLE 用户也可能链接自己的 shared
  kernel。空闭包合法；同一 consumer 重复 opt-in 幂等；不重写 parent output directory。
- BRS3：生成每配置的短 copy script，使用 CMake 3.25 已有的 `file(COPY_FILE ... ONLY_IF_DIFFERENT)`；
  旧 `copy -t` 要求 3.26，不能继续与项目 floor 矛盾。非 Windows 不生成 staging target。
- 验证用极小 DLL consumer：第一次运行、只变 DLL 实现后 EXE 不变但结果更新、无改动 build 不改文件、
  空闭包与重复 opt-in；再回放已有真实 CUDA source/static/shared，禁止手动复制掩盖 helper 错误。
  不重跑 provider/算子测试或 P1。CUDA v2 与本维护卡分别裁决，不复制模块状态。
