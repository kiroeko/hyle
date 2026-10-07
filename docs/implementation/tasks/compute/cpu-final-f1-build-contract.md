# F1 CPU构建、软件资格与预编译合同

<!-- MODULE REVIEW: SCOPED -->

- 当前owner：compute_provider_native、compute_ggml_runtime、compute_route_cpu、graph_route_cpu；baseline `167cb5ff6f9edac8a9aaa77c41e6720bd3c377d2`。
- 总合同：[CPU最终收口](cpu-final-closure-plan.md)；用户新增三平台/全重叠域与构建一致性要求构成新的公开交付trigger。
- 已确认源码反例：Native默认WIN宏只查peer，`qualified_default_environment()`只查编译器/Release/硬件，
  不查有效flags/LTO；自定义优化recipe仍能继承旧证书。CPU-only新ggml与mixed旧ggml能力不同；Mac ARM64不提供现ATen接入。
- 主consumer：普通CPU模型计算Direct/Graph及installed SDK；正交consumer：同语义tensor批处理与定长预编译replay。
- 非目标：不改数学、公共Graph动态模型、GPU资格、SIMD、性能统计或closure工具。

## 本包有限审计项

| ID | 问题 | 审计后允许的有限动作 | 验收 |
|---|---|---|---|
| B1 | flags/LTO/相关feature变化仍可能启用旧Native WIN | owning构建投影精确性能recipe；不匹配只撤资格，保持合法成熟默认 | matching/flags/IPO/Debug/多config及peer失配规则投影 |
| B2 | 标准CPU及mixed SDK实际core/格式能力不一致 | 固定同进程runtime/core ABI与能力合同，优先同一CPU能力；不可保持时具体决策，不静默降级 | 单/混合路线配置、能力说明、实际新格式负/正消费 |
| B3 | 预编译SDK与Graph变体容易被误称全部best-of | 区分构建期存在集/冷规则/Invocation尺寸选择，投影真实已编入程序与身份 | safe baseline/ISA、规则失配、变体命中/回退、source/static/shared |
| B4 | 无特化规则时只用全局Provider顺序，默认依据不完整 | 已测优选→逐op/format/precision/resource/region成熟默认→合法兜底；规则低成本冷绑定 | 未知硬件/版本/线程、feature裁剪、完整region与普通叶、错误不伪fallback |

scope只授权审计；B2的ABI/源码反证与F2/F3能力冻结后才写该部分产品。
构建获取/安装复用现有中央锁与package registry，不另造第二构建系统。普通用户仍只选CPU，不给timing或Provider顺序。
公开对象/ownership/失败/并发/完成不变；提交前owner必须AUDITED且有限FIX固定。

## 预算与验证

一轮本地源码/已归档SDK身份审计、一次实际机器工具链盘点；每ABI疑问最多两个具名现有接缝候选，
不能借本包重签全部GPU。B1一次参数化配置投影检查、匹配旧成本的普通consumer/owning测试；
B2/B3稳定后集中三交付/selected package/相应CPU直接消费者。P0，无本包赢家P1。
docs/state只closure/docs；未运行不签PASS。涉及公共package seam再定向选择，不能扩大治理工具。
完成后CANDIDATE→七门CLOSED；若B2依赖未接齐，保留实际未完成，不能以B1已修关闭本包。

当前：SCOPED，B1–B4均待AUDITED；产品改动0，新验收0。
