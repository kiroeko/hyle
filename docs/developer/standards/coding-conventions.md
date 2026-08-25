# HYLE 编码规范

> 这里规定如何写代码才不破坏模块、生命周期、性能和构建合同。为什么选择某项能力属于
> architecture；一次改动做了什么属于 implementation。

## 1. 命名与文件

- namespace 使用 `hyle`；只在确属私有实现时使用 `hyle::detail`。
- 类型使用 `PascalCase`，函数与变量使用 `snake_case`，宏使用 `HYLE_*`。
- 公开头位于 `include/hyle/<module>/`，实现位于 `src/<module>/`；不要从公开头 include `src/`。
- 新符号先确定 owner module 和 demand ID，再确定文件；不要建立 `misc` 或 benchmark helper 公共面。
- 注释解释合同、原因、平台约束和不明显的不变量，不逐行翻译代码。

## 2. C++ 与标准库

- 主体源码使用 C++23。能用 `constexpr`、concept、RAII、`<=>`、`std::span` 等语言/标准能力时，
  不保留旧编译器 polyfill 或 SFINAE 仪式。
- 原生 C++ 头使用 `<cstddef>`、`std::size_t` 和 `std::byte`。只有内部确实由 C compiler 解析的
  外部/系统边界才使用 `<stddef.h>` 与全局 `size_t`。
- std 类型可以出现在自然的 C++ source API；但 owning 容器进入 public signature 前要审查
  allocator、layout、compile cost、共享库工具链合同和 HYLE 是否已有更合适的 owner/view。
- 已有同合同 HYLE 热路径类型时，不在产品路径退回更弱的 `std::unordered_map` 等实现；测试可以
  用标准库做 oracle。需要有序或 node-stable 语义时也不能错误替成 flat container。
- 产品 runtime 避免 iostream/locale/stringstream；诊断走 obs 或小型格式化/`fprintf` 边界。

没有 Native FFI 时，不为假想跨语言 ABI 禁止所有自然 C++ 类型。若未来 FFI 准入，相关规则在
[Native FFI policy](../../architecture/native-ffi-policy.md)单独生效。

## 3. Include 与公开面

- 公开头 self-contained，直接 include 自己使用的声明；能局部前向声明时避免重实现头。
- 使用 `#pragma once`，不要同时叠加 include guard。
- 不建立全模块上帝头、巨大 `decls.h` 或强制函数表；只有多头真实共享的小型定义才放 `*_defs.h`。
- template shell 只保留类型安全与必要 inline；重算法放 compiled implementation，避免每个用户 TU
  重编和膨胀。
- public/internal 的判断由消费者合同决定，不因测试或 benchmark 需要而公开。

## 4. 宏、属性与可见性

- 只有 C++23 无法表达的代码生成、传播宏、编译器属性和平台分支使用宏。
- 宏参数全部括号化、只求值一次；未知编译器分支提供安全 fallback。
- 布局约束使用 `alignas`、真实字段与 `static_assert`，不用 `#pragma pack` 猜布局。
- `HYLE_<MODULE>_API` 只标真正跨 shared-library 边界的函数/方法。含非平凡私有成员的类避免
  整类导出，尤其在 MSVC 上防止 C4251 与错误 ABI 暴露。
- `HYLE_ASSUME` 只表达已由逻辑保证的不变量，不能代替检查；违反 assume 是 UB。
- `HYLE_RESTRICT` 只在实际不别名合同清楚且可测试时使用；`HYLE_UNROLL` 由 codegen 和 workload
  证明，不为 benchmark 强行展开。

## 5. 错误处理

- 可恢复失败返回 `Status`/`Result<T>`，使用 `HYLE_RETURN_IF_ERROR` 与
  `HYLE_ASSIGN_OR_RETURN` 传播。
- 只有增加用户语义时才 `WithContext`；不要每层重复拼接同一句上下文。
- OS/backend translator 保留 operation、原生错误类别/数值与必要路径；不要只返 `false/-1`。
- HYLE 自身不变量用 `HYLE_FATAL_ASSERT` 等 fail-fast，不能 catch 后继续。
- 外来异常只在 adapter 边界捕获并立即转成 `Status`；不要让不同模块发明私有 error enum、
  error string 或 handler。

## 6. Ownership 与内存

- 类型或命名明确 owner、view、borrow 和 invalidation；裸 pointer 不应同时暗示拥有与借用。
- move 后状态、destroy 幂等性、callback context、异步保活、取消与 quiescence 写在 public declaration
  附近并测试。
- 默认路径使用模块登记的 allocator/resource；custom allocator、arena、quota、NUMA 等显式 opt-in。
- 由一个 allocator/runtime 创建的内存由同一边界释放。跨线程 free 是否允许必须由具体类型声明。
- intrusive object 必须在脱链/引用归零前满足其 membership 与并发不变量；不能让裸 accessor 越过锁
  或 epoch 的有效期。

## 7. 并发与平台

- 每个共享类型声明 `not thread-safe / thread-compatible / thread-safe / lock-free` 及 progress 保证。
- 原子 memory order 以不变量证明为依据；无证明先用安全顺序，再通过测试与 profile 收紧。
- cache-line padding 只用于真实持续写争用的 control state，不为微基准给所有对象加常态空间税。
- OS、CPU、topology、VM、thread 与动态库机制走 platform；模块不复制 CPUID/syscall probe。
- 可分发 ISA probe 使用 baseline、独立 TU、必要时禁 LTO；运行时 dispatch 与 exact/native build 是
  不同交付合同。详见 [SIMD dispatch guide](../maintenance/simd-dispatch-guide.md)。

## 8. 性能代码

- 先写清复杂度、allocation/copy/syscall/contention，再优化指令。
- batch 与 scalar API 都由真实使用决定；不要把一次自然任务拆成 benchmark-friendly calls。
- fast path 与 fallback 必须语义一致并差分测试。
- 任何新增常态字段、缓存、索引、padding 或 indirection 都记录受益消费者与资源成本。
- 最终判断使用 release artifact、反汇编/counter 和已资格化 benchmark；源码形状不是性能证据。

## 9. 构建与依赖

- 模块内源文件和外部依赖由 `src/<module>/CMakeLists.txt` 拥有；内部组件边只在中央 component
  manifest 声明，由构建函数投影真实 link edge。
- 公开头暴露依赖类型时使用 `PUBLIC`，仅实现使用时用 `PRIVATE`。不要用全局 include/link 修漏链。
- feature option 表达用户能力，不表达某台维护机器是否恰好安装依赖；auto-detection 要 fail closed
  或明确报告。
- 新依赖同时覆盖 pin/hash、online/offline、install/export、license 和受影响 CI。
- source/static/shared 都必须由独立 consumer 验证，不能从 producer build tree 偶然取文件。

## 10. 测试与提交

- 测试覆盖正常、边界、失败和生命周期；并发/解析/算法按风险加 sanitizer、stress、fuzz、property
  或 differential oracle。
- benchmark 不提供测试专用 public API；性能 harness 使用正常入口或明确 internal seam。
- 修改底层模块同时运行直接消费者与 package closure，不能只跑本模块单测。
- 提交前运行 `git diff --check`、受影响测试、`python scripts/check-docs.py` 和适用机器合同。
- 保留用户无关的既有修改；提交只包含本轮能够解释和验证的文件。
