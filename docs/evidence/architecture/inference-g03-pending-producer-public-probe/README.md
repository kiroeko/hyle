# G03 已接纳 producer 到 F2 的真实公开反证

固定产品b428、2026-10-14 WSL/RTX4070、tiny1024byte；[公开consumer](public-probe.cc)、
[actual final](final.log)、[receipt](receipt.json)与[raw](raw.zip)。本次无产品修改。

真实CUDA stream checkpoint→cudaMemsetAsync数值producer，原LeaseUseBatch.write/attach
持F1 writer guard。producer未退休时，F2 bind_mutable成功保其原block，随后bind_producer/
prepare/submit建立原native排序；checkpoint未释放时无ready output/returned writer。
释放后copy实际wait成功，返回writer/publish，1024byte全为49，reset释放source/producer
keepers。没有producer中间host wait，只有最后copy observation。

初始“F2只能绑定unique ready writer”的推断被源码和实际结果推翻：原bind_mutable借
LeaseAccess::ordered_view，不是WriteLease.view。无需新增重复F2 pending-source API。
前两次fixture错误期待refusal，首early-return未显式观察原producer而导致resource live
assert；后续又在copy还持source/producer bindings时要求全部uses消失。保全部FAIL/log
与累计，按原寿命/成功reset修fixture，不改F1/F2。最终actual数据与guard证明通过。

这只反证F1 producer→原F2自然路线，并非完整producer→attention：F2 successful output
和PagedArenaRead/PreparedBatch的ready门仍在；当前最后copy wait不能自动消失。下一
审计定位这一精确owner seam，不添加已经存在的机制，不把ready copy代签整个G03。
