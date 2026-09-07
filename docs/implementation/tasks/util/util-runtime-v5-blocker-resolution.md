# Util-runtime v5：有限 blocker 恢复

<!-- MODULE REVIEW: SCOPED -->

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

审计阶段只复核三条调用链并冻结有限 FIX；实现最多一个 Registry 产品批次和一个 test-fixture 批次。
验证只跑 `util_runtime` factor、两张 runtime 头、Util source/static/shared consumer、三平台与 WSL
ASan/UBSan/TSan、P0/codegen/export 及候选后一次红队。若出现第四个独立产品 MUST，或既定修复需要改变
依赖/能力边界，立即裁决 BLOCKED，不扩建公共仪器。P1 因没有已准入热 consumer 而不适用。

## Audit input（SCOPED，尚未冻结 FIX）

| ID | retained fact | intended proof |
|---|---|---|
| RT-01 | `Creator/create/Entry::invoke` 的连续按值 hop 可在 `noexcept` 内触发额外 throwing move | 值参数只在公开边界构造一次；类型擦除用地址槽传递，最终 factory 仍得到声明的 `Args...` 类别 |
| RT-02 | `dbghelp.h` 依赖 `windows.h`，格式化后次序会复发 | 私有 fixture 以不可重排 include block 编译通过 |
| RT-03 | 合法 ABI 实现可拒绝构造的超长符号，公开语义是原样回退 | integration 接受成功 demangle 或精确 fallback；确定性 fake adapter 继续证明长结果/释放路径 |

完成调用链和现有测试复核后先进入 AUDITED，再允许修改产品代码。
