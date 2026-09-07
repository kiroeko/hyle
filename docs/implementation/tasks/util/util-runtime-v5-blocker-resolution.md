# Util-runtime v5：有限 blocker 恢复

<!-- MODULE REVIEW: CANDIDATE -->

## Scope and retained evidence

- baseline：`3ed2a2cef5ffc0c5acbcaff6421e568a5546f77a`，即 `util -> util_runtime` 身份迁移及三种
  Util 交付已验证后的干净提交；
- resume authority：v4 terminal card 已用独立机制复现 RT-01，并把 RT-02/03 定位到 util 私有测试；
  本轮复用其 consumer/capability/dependency 审计，不重新扫描引擎或发明新功能；
- scope：只消除 Registry 类型擦除层的额外按值构造，固定 Windows 私有 fixture 的 SDK include
  先决顺序，并让 native 长符号测试遵守“平台 demangler 失败则原样回退”的公开合同；
- non-goals：新能力、`util_core` 或任何 FROZEN 依赖修改、unregister/unload/drain、Native FFI/C API、
  P1 计时、公共 selector/closure/build-system 修改和全库测试。

## Budget and stop rule

审计阶段只复核三条调用链并冻结有限 FIX；实现最多一个产品批次（Registry RT-01 与 demangle
RT-03）和一个 test-fixture 批次。
验证只跑 `util_runtime` factor、两张 runtime 头、Util source/static/shared consumer、三平台与 WSL
ASan/UBSan/TSan、P0/codegen/export 及候选后一次红队。若出现第四个独立产品 MUST，或既定修复需要改变
依赖/能力边界，立即裁决 BLOCKED，不扩建公共仪器。P1 因没有已准入热 consumer 而不适用。

## Audit result and frozen FIX

| ID | retained fact | frozen correction and proof |
|---|---|---|
| RT-01 | `Registry::create -> EntryView::Creator -> Entry::invoke` 以及 `FrozenView::create -> Creator` 都复制了按值 signature；`Creator` 本身还有一层 | 三个公开调用入口各自只形成其声明所需的 `Args...` 参数；内部公共 helper 借用这些局部参数，erased invoker 只收 `remove_reference_t<Args>*...` 地址槽，最终以 `std::forward<Args>(*slot)` 恢复原类别。测试 named rvalue 的入口构造数、borrow-only factory 零内部 move、调用前 throwing move 可传播，以及现有 move-only by-value factory |
| RT-02 | `dbghelp.h` 依赖 `windows.h`，普通 include sort 会把前者移到前面 | 私有 fixture 用局部 `clang-format off/on` 固定 `windows.h` 后 `dbghelp.h`，不改生产 header 或全局 formatter |
| RT-03 | GCC 可合法拒绝人为构造的 1500 字节符号；Windows 定向测试又证明 DbgHelp 在输出填满 caller buffer 时可能返回非零截断长度，旧实现误当完整结果 | 平台 integration 分支验证“成功时确含 identifier，否则必须精确等于输入”，两条分支都并发复放；Windows 只有在结果未触及容量边界时才接受，否则扩容重试，达到上限仍不明确则回退输入；fake adapter 以非零截断返回证明重试、1500 字节成功结果、OOM 与释放路径 |

审计同时确认：Entry lifetime 已由 Registry-owned immutable chain 保证，地址槽不会逃逸调用；引用、
const 引用、值与 rvalue-reference 的折叠均保持原签名，空参数 pack 合法；factory 的最终按值参数若需要
一次 move，仍由既有 `is_nothrow_invocable_r` 准入覆盖。本轮只执行以上三项，不修改布局外的常态
metadata、复杂度、依赖或能力表。

## Implementation batch

有限实现已锁定为 candidate `7019998034f8025890770fad04406dc5ef2d6194`、tree
`a0f5b8d9ae5af99b9e15d60678ec6dac7d566327`：Registry 三个入口复用内部
`find_entry/invoke_entry`，erased
invoker 仅接收地址槽；具名 throwing-move 回归证明三个入口各只有公开边界的一次 move，且调用前异常
不被 `noexcept` 误终止，既有 move-only by-value factory 仍通过。Windows 私有 adapter 固定 SDK include
顺序，并模拟 DbgHelp 非零截断；实现只接受留有明确余量且相对上一轮确实增长的结果，否则继续扩容或
回退原始名字，避免把 1022/1024 的部分结果冒充完整 owning string。Windows util-runtime 三个测试
程序在候选前的工作树验证通过；三平台候选验证与终态签证仍待执行。

### 候选后的私有 fixture 校正

首次 W1-WSL ASan 运行在 `hyle_util_runtime_demangle_fault_test` 启动 GoogleTest 时报告
alloc/dealloc family mismatch。该隔离程序为验证 public owning string 的 `bad_alloc` 传播而重载了普通
`operator new/delete`，却没有一并重载标准库会使用的 `nothrow` new/delete；于是 GoogleTest 的
`nothrow new` 走 ASan C++ allocator，释放却落入 fixture 的 `free`。校正仅补齐同一私有 executable
中的 scalar/array nothrow 配对，不修改产品头、实现、API、算法、依赖或正常 workload。失败保留为
验证历史，受影响的 ASan 与三平台私有 fixture 必须在终态前重跑。

## Candidate verification

验证执行 commit/tree 为 `342dbcfd10c2ab985aab023991db2c7367c3719e` /
`2edd6da353a43cde252282772cf6d6910436bdb4`。它相对产品 candidate 只增加上述私有 fixture 配对和本卡
记录；`git diff 70199980..342dbcfd -- include/hyle/util src/util_runtime` 为空，因此产品 API、实现、
算法、正常 workload 与 P0 身份没有漂移。

| Gate | W1-WIN | W1-WSL | M1-MAC | Verdict |
|---|---|---|---|---|
| factor tests 与两张 runtime 头 | MSVC 19.50，3/3 | GCC 14.2，3/3 | AppleClang 21，3/3 | PASS |
| Util source / relocated shared / relocated static | 1/1、1/1、1/1 | 1/1、1/1、1/1 | 1/1、1/1、1/1 | PASS |
| sanitizer | 本卡指定 WSL | ASan 3/3；UBSan 3/3；TSan Registry/头正常通过，fault executable 在既定 `setarch x86_64 -R` 运行合同下 3/3 | 本卡指定 WSL | PASS |
| compiled exports | 仅两个 decorated C++ demangle symbols | 同两个 C++ symbols | 同两个 C++ symbols | PASS |

TSan 的 fault executable 首次普通 CTest 在进入 `main` 前由 GCC runtime 报 `unexpected memory mapping`；
该程序在项目文档既定的 WSL `setarch -R` 合同下全部通过。并发 Registry 主程序无需该绕行即通过，且
没有 sanitizer 报告；这不是把未运行项写成 PASS。

P0 forwarding probe 复用 v4 的独立复现源码，在 WSL/GCC 14 与 M1-MAC/AppleClang 21 的 `-O3`
candidate headers/artifact 上，普通调用和启用 throwing-move 两种运行都输出
`moves inside borrowed creator: 0` 并正常退出；Windows 的同合同回归在三个公开入口各观察到恰好一次
公开边界 move，且调用前异常可传播。Registry 继续保持 register
O(key+help)+expected-amortized O(1)、freeze O(1)/零分配、frozen lookup
O(key)+expected O(1)/无锁无分配；erased hop 只传参数槽地址，未新增常态 metadata、分配、锁或
runtime route。demangle 仍为 O(input+output) 冷诊断路径。

候选后红队复核了值/引用/rvalue-reference/空参数折叠、entry 与 allocator 生命周期、替换后借用保活、
freeze publication、callback 重入以及 Windows DbgHelp 截断/串行化边界。没有未决 correctness、
ownership、concurrency、异常、交付或易用性 finding；API 中没有 benchmark 的固定 N、地址、执行顺序、
peer、fast/slow mode 或常态测量 metadata。`util_runtime` 继续复用已冻结的 Core/Error/Mem/Container/
Container-text 路线，没有复制 allocator、hash、text 或 error authority，也没有要求上游模块重开。

### Aggregate gate verdict

| gate | verdict | basis |
|---|---|---|
| scope / architecture | PASS | 三项固定 blocker 关闭；能力、依赖和产品 façade 不变 |
| correctness | PASS | 三平台、边界/故障/并发测试及 WSL ASan/UBSan/TSan |
| delivery | PASS | source + relocated shared/static × 三平台；exports 无旧 C/FFI 面 |
| usability / docs | PASS | 自然 owning factory、借用 Creator 与参数边界已写入用户指南 |
| red team | PASS | fixture 校正后无 material product finding |
| engineering performance | PASS | P0 复杂度、分配/锁结构及 forwarding probe 通过；P1 不适用 |

不可变产品候选 LOC：全库代码 480,987；自实现不含测试 77,069；util-runtime 实现 544。registry v1
仍要求的 `self_with_tests=122059` 只作为旧 schema 兼容字段登记，不向 owner 常规报告，也不参与完成度。
