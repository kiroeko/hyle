# Vulkan V3：性能资格边界与工程取舍

<!-- MODULE REVIEW: CLOSED -->

本页只记录[V3](vulkan-parity-v3-defaults-best-of.md)的性能资格条件，不暂停其它交付工作。
产品候选 `09410e32c`，普通API/算法/default已冻结；原math与两端完整service相同。

## 已执行证据

- 两完整消费者：本地Qwen3.5-2B BF16 FFN和DeepSeek混合hash-MoE，原节点/精度/输入
  cycle及全部checks/conversions/native完成/owned lease保持。Graph与explicit native Plan
  的prepare产物都执行同一Program/Invocation；原Recorder不等责路径已排除。
- verify-only正确性、重复执行和资源预算通过；两端dispatch/managed-region数量与
  cold/warm resident bytes完全相同。
- primary唯一8process paired A/A：原5%因素/10%单process门PASS。
- neutral placement首次8process：数值/资源/原native环境与telemetry均通过，peer自身
  DeepSeek allocation=1因素ratio1.0649869809947092，order=1为1.050915475875156；
  固定5%因素门FAIL。全部单process仍在10%门内。
- 该比较是相同实现自身，不是Graph对native-plan胜负，不能从它认产品LOSS。
  完整[原始receipt](../../../../evidence/performance/vulkan-v3-owned-fragments/owned-placement-aa-receipt.json)
  与raw/nonce/SDK/二进制/实际P核policy均保留。

## 未执行与停止

正式n15未执行，P1为INCONCLUSIVE/未资格，不能写PASS/WIN。唯一预检预算已结束，
不升31、不换layout/目录重测、不要求用户关闭更多程序或改系统性能模式。
源证据证明两端steady-state使用同一执行和资源管理代码，当前没有具名产品优化钥匙。
本次不能唯一归因于GPU频率、缓存、driver或桌面竞争，不能把推测写成已证原因。

这里的6.50%指两侧计时比偏离相等；不是数值误差。peer A/A两侧均是同一显式native
Plan service、相同输入，不是Graph/native算法比赛。因此它证明这项测量条件未达5%
比较资格，不能单独证明产品更慢或有自研算法落后；用户尚未接受工程取舍。

## 两个具体取舍

1. 工程收口：接受source等价、当前完整correctness/resource/default和P0成本证据；
   保留本次P1未资格/INCONCLUSIVE及原失败，不宣称统计性能赢家。V4平台/三交付和
   其它七门仍须完成；性能局限明确写入terminal。以后出现实际可修复回归再定向重开。
2. 保留原P1硬门：V3性能保持BLOCKED，V4独立推进；只有具名测量/环境改变后另行
   固定新scope与预算，不能原条件下继续采样。本页不自动启动下一性能轮。

用户决定前不能凭本页将P1改PASS或整个goal标完成。构建/语义/交付MUST不受例外豁免。

## 用户明确决定（2026-10-09）

用户在解释计时偏差后答复“就这样吧，一般我倾向删除自己算子，用市面成熟的”。
据此采用选项1：源等价/成熟默认/正确性资源/P0工程裁决，原placement AA FAIL与
正式n15未运行/P1 INCONCLUSIVE保持，不重测、不制造WIN。独有能力仍需逐项明确，
成熟完整替代无明确优势的self数学按其偏好删除。V4所有构建/数值/资源/交付MUST
没有豁免，整个goal仍未完成。本性能政策问题结束；route按协议恢复本轮V4范围。
