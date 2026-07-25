# 00-scaffolding：项目脚手架（CMake 结构 + 测试/基准/sanitizer 框架 + CI + 宏骨架）

| 字段 | 值 |
|---|---|
| **阶段** | P-1（万物之前）|
| **模块** | 构建系统 / 全局 |
| **Tier** | — |
| **LOC 估** | 1-2k（多为 CMake/CI 配置）|
| **骨架?** | ✅（walking skeleton 第 0 步）|
| **状态** | ✅ 完成（commit 3c8150d，后续 da95bfb 补构建配置）|

> 本页保留首轮脚手架的历史名称与 checklist；当前 core 宏/cache 形状已由
> [`core-module-closure`](./core-module-closure.md) 重审，`HYLE_CACHE_LINE_SIZE`、
> `HYLE_ALIGNED`、`HYLE_PAD` 和 `scaffolding_test` 不再是现行 API。

> **cloc 留档（完成时实测，`cloc --vcs=git`）**：本步贡献约 **294 行**（非文档）= core/macros/visibility/platform 头 + version.h/cc + scaffolding_test + 大部分 CMake/cmake 模块。CMake 占比高属正常（脚手架以构建配置为主）。无 stub。

## 前置依赖

- 硬前置：无（这是第一步）。
- 决策依据：[`engineering-charter.md`](../architecture/engineering-charter.md)（C++23/GoogleTest/nanobench/TSan/CI 矩阵/覆盖政策）、[`coding-conventions.md`](../architecture/coding-conventions.md)（HYLE_ 宏、std 分层、include 纪律、能力 dispatch 三层）、[`hyle-scope.md`](../architecture/hyle-scope.md)（模块划分与 Tier）。

## 目的（为什么做这一步）

**所有里程碑都在它上面编译、测试、跑基准。** 脚手架不规划好，后面每个模块各搞各的构建、测试散落、sanitizer 没接，回头返工成本极高。一次把"骨架 + 工具链 + 纪律"立起来，之后每个里程碑只是往既定结构里填模块。

clay 的最大负债是**零测试**——HYLE 第 0 步就把测试/基准/sanitizer/覆盖率全接上，杜绝"先写后补测试"。

## 在架构中的位置

依赖图的根之下、`error/` 之前。它不产出 runtime 代码，产出的是**项目骨架 + 构建/测试/CI 基础设施 + 全局宏头**。后续 `error-01` 起的每个里程碑都 `add_subdirectory` 进来、复用这套测试/基准约定。

## 具体事务（做什么）

### 1. 目录结构（模块化，对应 scope 的模块划分）

```
hyle/
├─ CMakeLists.txt              # 顶层：项目、C++23、选项、子目录
├─ CMakePresets.json           # 构建预设（debug/release/asan/tsan/ubsan/coverage）
├─ cmake/                      # 复用的 CMake 模块
│  ├─ HyleModule.cmake         # hyle_add_module() / hyle_add_test() / hyle_add_bench() 函数
│  ├─ HyleWarnings.cmake       # 统一告警(-Wall -Wextra -Werror + -Wpadded 等)
│  ├─ HyleSanitizers.cmake     # asan/tsan/ubsan 开关
│  └─ HyleCoverage.cmake       # llvm-cov/gcov 接入
├─ include/hyle/               # 公开头（按模块）
│  ├─ macros.h  visibility.h  platform.h   # 全局宏骨架（本里程碑产出）
│  └─ <module>/...             # 各模块公开头（后续里程碑填）
├─ src/                        # 内部实现（按模块）
│  └─ <module>/...
├─ tests/                      # 单测（镜像 src/ 结构）
├─ bench/                      # 微基准
├─ third_party/                # vendored（mimalloc/xxhash/highway/xsimd/simdutf + fcontext asm 等）
└─ .github/workflows/ci.yml    # CI 矩阵
```

每模块是一个 CMake target（`hyle_<module>`），`hyle_add_module()` 统一处理：公开 include、可见性宏、告警、链接。**模块间依赖在 CMake 里声明**（编译期强制依赖图无环——roadmap 第一节那张图变成 `target_link_libraries` 关系）。

### 2. 顶层 CMake 要点

- [ ] `cmake_minimum_required(VERSION 3.25)`、`project(hyle LANGUAGES C CXX ASM)`（ASM 给 fcontext）。
- [ ] `CMAKE_CXX_STANDARD 23` + `CXX_STANDARD_REQUIRED ON` + 工具链门槛检查（GCC14+/Clang18+/MSVC19.43+，低于则 `FATAL_ERROR`）。
- [ ] 默认 SHARED（带 C-ABI），`HYLE_STATIC` 选项切静态。
- [ ] **修掉 clay 的反模式**：不硬编码 `-march=native`（设为可选 `HYLE_NATIVE`）、不硬编码 Clang `-g -O0`。
- [ ] 选项：`HYLE_BUILD_TESTS`/`HYLE_BUILD_BENCH`/`HYLE_SANITIZER=off|asan|tsan|ubsan`/`HYLE_COVERAGE`。

### 3. 测试框架（GoogleTest）

- [ ] vendor googletest（submodule 或 FetchContent，iron 已有可参考）。
- [ ] `hyle_add_test(<module> <files>)`：自动链 gtest + 该模块 target + sanitizer flag，注册到 `ctest`。
- [ ] 约定：`tests/<module>/<feature>_test.cc` 镜像 `src/`。
- [ ] 冒烟测试：一个 `scaffolding_test.cc` 验证测试框架本身跑通。

### 4. 微基准（nanobench）+ sanitizer + 覆盖率

- [ ] vendor nanobench（单头）。`hyle_add_bench()` 同理。
- [ ] `HyleSanitizers.cmake`：asan/tsan/ubsan 各自独立构建预设（互斥，charter 定）。
- [ ] `HyleCoverage.cmake`：llvm-cov（Clang）/ gcov（GCC）生成覆盖报告 + 按 Tier 设阈值门槛（T0≈100%/T1≥90%/T2-3 80%+）。

### 5. CI 矩阵（.github/workflows）

- [ ] 三编译器 × 三平台（Linux/Win/mac）编译 + ctest。
- [ ] sanitizer 构建（TSan/ASan/UBSan 分 job）。
- [ ] 覆盖率报告 + 阈值门禁。
- [ ] （留位）AVX512/GPU 真机 runner——没有就标记为已知覆盖缺口（charter 覆盖政策现实 #2）。

### 6. 全局宏骨架（本里程碑唯一的 runtime 头）

- [ ] `include/hyle/macros.h`：`HYLE_FORCE_INLINE`/`HYLE_LIKELY`/`HYLE_UNLIKELY`/`HYLE_CACHE_LINE_SIZE 64`/`HYLE_ALIGNED`/`HYLE_PAD`/`HYLE_NODISCARD`/`HYLE_DISABLE_COPY_AND_ASSIGN`——**属性宏必带 `#else` 兜底**（conventions 铁律），删 clay 的 MSVC<1900/GCC5 退化分支。
- [ ] `include/hyle/visibility.h`：三态导出宏（static/dllexport/dllimport），`HYLE_<MOD>_API` 模式。
- [ ] `include/hyle/platform.h`：平台/架构探测宏（`HYLE_OS_*`/`HYLE_ARCH_*`/`HYLE_COMPILER_*`）。

### 不做（留给后续里程碑）

- 任何实际功能代码（error/mem/... 都是后续里程碑）。
- 能力 dispatch 的探测器实现（simd-04）——这里只立 CMake 多变体编译的脚手架占位。

## 交付物

- 完整 `CMakeLists.txt` + `CMakePresets.json` + `cmake/*.cmake`。
- 三个全局宏头（macros/visibility/platform）。
- 一个空 `hyle_scaffolding` 模块 + 其冒烟测试 + 冒烟基准，证明 `hyle_add_module/test/bench` 三个函数都通。
- 跑通的 CI 配置。

## 验收标准

- **编译**：三编译器 × 三平台空模块全过；低于工具链门槛时 CMake 明确报错。
- **测试框架**：`ctest` 跑通冒烟测试；`HYLE_SANITIZER=tsan/asan/ubsan` 三个预设都能构建并跑冒烟。
- **基准框架**：nanobench 冒烟基准能跑、出数。
- **覆盖率**：覆盖报告能生成、阈值门禁能触发（用冒烟模块验证管道，不是验数字）。
- **CI**：push 后矩阵全绿（真机缺口标记为已知）。
- **宏骨架**：macros/visibility/platform 三头在三编译器下编译干净（`-Wall -Wextra -Werror` 无警告）。
- **硬指标**：新建一个空模块 `hyle_foo` + 一个测试 + 一个基准，**只写业务三行、其余全由脚手架函数搞定**——证明后续里程碑的"填入成本"已降到最低。

## 参考

- `engineering-charter.md`：C++23 工具链门槛、测试栈、覆盖政策、CI 矩阵。
- `coding-conventions.md`：HYLE_ 宏骨架（属性宏兜底）、三态可见性宏、能力 dispatch 三层（CMake 多变体编译脚手架）。
- clay `macros.h`/`visibility.h`/`CMakeLists.txt`（`slate-render/slate/infra/`）：宏与可见性的继承源，但删退化分支、改反模式。
- iron 的 vendored googletest：测试框架复用参考。
