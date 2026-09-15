# Windows runtime DLL 增量 staging

## 维护卡 / 完成

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

## 结果（2026-09-15）

`8b6c2560` 实现 BRS1--3；`aaaf96f7` 把无改动 timestamp 检查精确到微秒（不改变产品）。源码与安装
配置均 include 同一 `HyleRuntimeStaging.cmake`，package 安装该文件。Windows MSVC Release/Ninja 上：

```text
cmake -DHYLE_SOURCE_DIR=C:/public/project-hyle/hyle -DHYLE_SMOKE_ROOT=C:/hyle-scratch/runtime-staging-v1-source -P cmake/HyleCiRuntimeStagingContract.cmake
cmake -DHYLE_SOURCE_DIR=C:/public/project-hyle/hyle -DHYLE_SMOKE_ROOT=C:/hyle-scratch/runtime-staging-v1-shared -DHYLE_STAGING_PREFIX=C:/hyle-scratch/custom-execution-v10-shared-sdk -P cmake/HyleCiRuntimeStagingContract.cmake
cmake -DHYLE_SOURCE_DIR=C:/public/project-hyle/hyle -DHYLE_SMOKE_ROOT=C:/hyle-scratch/runtime-staging-v1-static -DHYLE_STAGING_PREFIX=C:/hyle-scratch/custom-execution-v10-static-sdk -P cmake/HyleCiRuntimeStagingContract.cmake
```

三条全部 PASS：初次 build、只改 DLL 后 EXE/import library SHA256 不变而结果从 1 变为 2、无改动 build
不重写 staged DLL、空 runtime 闭包和重复 opt-in。之后再次直接执行三个 `runtime_probe 2`/`empty_probe`
均成功。source/shared/static 的真实 `hyle_custom_cuda_sample` 也全部 PASS，不使用手动 copy。
源码/两个 installed SDK helper SHA256 均为 `f94b029efee2df6d87ea7fb69025a00d98bd7ae932074ad572b518aea62d8a87`。

候选后复核：staging 依赖真实 DLL 文件而非 import lib/EXE；每配置脚本生成、重复调用幂等、空列表合法；
非 Windows 返回，未改父工程 output 目录、产品 ABI、组件图或代码生成。maintenance gate PASS。
selector `2c24e638 -> aaaf96f7` READY，但对 common CMake seam 保守选择全闭包；实际执行范围遵守本卡
实现前固定的 Windows staging 风险，未运行的全库 factor/跨平台矩阵不标 PASS，也没有为此改 selector。
docs/closure/diff 为终局固定门；无新增性能采样或自动 GitHub CI。
