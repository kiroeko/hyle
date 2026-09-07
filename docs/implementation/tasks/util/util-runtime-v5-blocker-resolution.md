# Util-runtime v5：有限 blocker 恢复

<!-- MODULE REVIEW: AUDITED -->

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

有限实现已完成但尚未锁定 candidate：Registry 三个入口复用内部 `find_entry/invoke_entry`，erased
invoker 仅接收地址槽；具名 throwing-move 回归证明三个入口各只有公开边界的一次 move，且调用前异常
不被 `noexcept` 误终止，既有 move-only by-value factory 仍通过。Windows 私有 adapter 固定 SDK include
顺序，并模拟 DbgHelp 非零截断；实现只接受留有明确余量且相对上一轮确实增长的结果，否则继续扩容或
回退原始名字，避免把 1022/1024 的部分结果冒充完整 owning string。Windows util-runtime 三个测试
程序当前通过；这不是三平台或终态签证。
