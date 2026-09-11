# Native FFI 准入策略

> 决策状态：`ACCEPTED`，2026-08-26。当前产品状态：`N/A`。

## 决策

HYLE 的默认产品是 C++23。完整源码、headers + static libraries、headers + shared libraries 都是
C++ 交付形式；shared library 的存在不自动产生稳定 C++ ABI，也不自动要求外语接口。

项目没有原生 C 客户，也没有当前要交付的 Java、.NET、Python 或 Rust binding。Native FFI
无限期延后，直到出现一个已命名的外语消费者、可运行 journey、负责维护的 owner 和明确收益。
预发布阶段不为旧 C-shaped surface 保留兼容性。

早期 `include/hyle/c/`、公开 `c_*.cc` bridge、C surface/export fixture 和 managed probe 已从当前
产品树移除。它们曾是未经消费者验证的历史实验，不是保留 API，也不构成未来 FFI 的设计起点。
入口文档、包 manifest 和版本说明不得重新把这些名字宣传为可用能力。

## 为什么不是“把全部 C++ API 导出一遍”

C calling convention 只能解决函数调用和简单布局，不能自动表达模板、重载、RAII、异常、STL
对象、allocator、借用关系、回调线程、取消或异步完成。机械追求 1:1 parity 会产生第二套难用 API，
增加动态符号、常态元数据和兼容成本，却没有真实用户验证。

外语用户通常需要的是少量、任务级的运行时操作，而不是调用任意 C++ 模板函数。未来的边界应
围绕这些 journey 设计；各种语言的惯用 wrapper 属消费方或单独 integration 项目，不属于每个
基础模块的日常收口门。

## 准入门

只有同时满足以下条件，某个 FFI slice 才能从 `N/A` 进入设计：

1. 有已命名语言、运行时、应用和 owner，而不是“以后可能有人用”。
2. 有端到端任务：create、正常调用、失败、并发/回调、cancel、destroy 和 unload。
3. 已证明进程内 source/static/shared C++ 接入不能解决该任务。
4. 边界可以比 C++ API 更窄，且不要求暴露模板、STL、异常或 C++ 对象布局。
5. shared artifact 的加载、版本协商、内存释放、线程和生命周期可以自动验证。
6. 维护者接受该 slice 独立版本化、兼容和安全审查的长期成本。

任一条件缺失即保持 `N/A`，而不是先做一个 speculative C API。

## 未来允许的形状

若准入，优先使用运行时类型擦除：

- 固定宽度整数、明确对齐的 POD、`pointer + length`；
- zero-invalid opaque handle；创建者负责销毁，禁止跨 allocator 释放；
- `function pointer + context` callback，并明确调用线程、重入、取消和晚到完成；
- owning error handle 或调用方 buffer，并明确长度、编码和释放函数；
- batch-first 的粗粒度操作，避免逐元素跨边界；
- 每个 artifact 自己的 capability/version query，不假设 core 能替其他库背书。

禁止把以下内容放进边界：C++ 模板、STL 类型、异常、引用、虚表、`std::function`、隐式 TLS、
编译器相关类布局，或让外语 allocator 释放 HYLE 内存的合同。

## `FFI=N/A` 的完成定义

模块按新标准收口时，若 FFI 判定为 `N/A`，候选产品应当做到：

- 没有该模块的公开 C schema、bridge source 或导出符号；
- 没有只为 FFI 存在的 CMake option、target、package metadata 或 ABI version；
- 没有把 C fixture 写成用户 sample 的文档；
- 没有为假想 FFI 增加常态字段、间接调用或所有权限制；
- architecture/implementation 可以保留决策和清理记录，user 文档不出现不存在的入口；
- shared artifact 的实际 export table 不泄漏退役 `hyle_*` C 入口、vendored allocator API、raw 汇编
  procedure 或私有 signal trampoline。编译器修饰后的显式 C++ exports 仍是 C++ SDK，不是 FFI。

内部与操作系统、汇编、GPU kernel 或第三方库交互所需的 `extern "C"` 不等于产品 FFI，可以在
清楚的私有边界继续存在，但必须在 shared artifact 中保持隐藏。该要求由安装后 PE/ELF/Mach-O
扫描守护；header-only 产品没有虚构的 shared binary。

## 重新开启

真实消费者出现时建立独立实施任务，从 journey 和威胁模型开始，不复活整套旧 C API。该任务
必须明确哪些能力导出、哪些仍只属于 C++，并用目标语言从精确 shared-library 路径完成正负
旅程。已删除的旧符号和名称没有兼容或保留承诺。
