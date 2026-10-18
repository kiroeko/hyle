# R6 最终候选 host ASan

产品 `8e45b10423686a0753aaa9d461c4ef3b117412c5`，验证快照
`a41d43d531365a1fb8617d60113b37924e9a9864`；MacBook Air Mac16,12 / Apple M4。
[产品输入身份](../vulkan-r6-product-delivery/p0-unchanged.json)确认测量快照与产品候选的 shipping
输入相同。重型验证在两项 Air 原生诊断完成后串行执行，build parallel 2 / CTest parallel 1。

## 实际覆盖

- Release/WERROR、`HYLE_SANITIZER=asan`；实际 12 个 HYLE 产品/生成/测试编译单元均含
  `-fsanitize=address`，原始 compile commands、cache 与两个 executable hash 保留。
- 两个 CTest 程序中的 14 个 Compute、3 个 Graph GTest 全部通过，合计 **17 项**。
  包含新增 packed 常量原 owner 释放后的真实 MatMul、结果保活、取消/关闭、OOM/观察失败、
  两线程槽与完整 FFN/custom SPIR-V；不是空 target 或只 copy 检查。
- ASan 只覆盖 HYLE host；外部 loader、MoltenVK/driver 与 GPU shader 没有插桩。
  本次禁用外部 validation layer；实际 Khronos synchronization validation 由
  [修正后的两系统 native 记录](../vulkan-r6-f32-corrected/README.md)单独证明。

## 既有 UBSan / TSan 的继承范围

[早期 host 插桩](../vulkan-r6-air-native-host/README.md)分别在 `ecdd92a7` 实际执行 UBSan/TSan
和当时的 16 项 native GTest。这些 receipt 继续绑定原候选与数量，不改写为本次重新执行。

从 `ecdd92a7` 到 `8e45b104`，Vulkan 产品差异仅在 `vulkan_operations.cc` 与
`vulkan_program.cc`：前者是冷期实际 BK32 资源账单、合法完整 tile 和 specialization；后者在
全部 lowering 结束后释放最终 dispatch 不再引用的常量 owner。tile 始终为非零的 32/64/128，
资源算式有固定上界；没有新增 host 重解释访问。常量处理发生在尚未发布的冷准备期，新增
寿命风险由本次 ASan 和对应真实计算用例覆盖。

原生分配/传输、提交/观察、queue 锁、quarantine 和 Graph lowering 文件未变，因此这些
UB/并发路径沿原记录继承；不能将其解释为新路径重新跑过 UBSan/TSan，也不外推 GPU 检测。
物理非一致 memory type、故意 hang/device-loss 及未知设备仍未取得实机资格。

## 完整性

[receipt.json](receipt.json)保留原始字节；[identity.json](identity.json)给出候选与验证范围。
[raw.zip](raw.zip)含 configure/build/CTest/LastTest、全部与筛选后的编译命令、cache、receipt
和执行/归档脚本，共 10 个具名文件，另含 archive-manifest。归档前核对 executable、编译命令、
日志 hash 和 14+3 的实际 GTest 数量；Windows 接收后全部条目再次校验通过。

raw.zip SHA-256：`2dfb13231ffb896b6d00a73685b204fe119b2b85e0cafa2e5edf5ca31e37523f`。

本证据完成最终 ASan 项；最终三交付、selected gates 和 R6 terminal 仍需独立验收。
