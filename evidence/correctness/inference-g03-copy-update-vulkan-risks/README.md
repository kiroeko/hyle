# G03 Vulkan CopyBatchUpdate 权限与六个生命周期

2026-10-14，按[活动卡](../../../docs/implementation/tasks/compute/inference-g03-copy-update-owner.md)
验证candidate bcaec010b；[consumer](consumer.cc)、[实际日志](final.log)、[累计](receipt.json)、
[源码字节](source-identity.json)、[全部输入/FAIL/编排](raw.zip)。MacBook Air Mac16,12
arm64 / macOS27.0.1 / AppleClang21，非software Vulkan真实QueueSubmit/fence；4jobs串行。

六场景均PASS：成功、native suffix拒绝但inline prefix已接纳、未知fence观察错误、closed、
取消、未知退休时放弃两controls。保原mutable root/geometry/last-write，源-only/坏ID
拒绝；成功writer回同generation/exact frame；失败/closed/cancelled无ready writer，
control回收保原Status。未知不提前退休，实际WaitForFences/成功GetFenceStatus后才释放
原source guard；abandon没有隐藏host wait，原Runtime实际排空才释放所有borrowed资源。

未知边界按原Run合同：一次OOM观察被记录，但尚无实际fence退休时Completion仍pending，
owner.observe为Unavailable；trusted input/prefix仍是借用，public writer view/return拒绝。
实际退休后wait保最初OOM，input/dependency/ready全部拒绝。fixture初版错误地要求未知
observe立即返回OOM且拒绝geometry，实际FAIL保留；修正fixture解释，不改candidate。
abandon case只屏蔽positive GetFenceStatus，不伪造工作或GPU长shader；真实native fence
与WaitForFences未替换，验证的是未知退休的保活。source组件不代签pending attention数学。

累计46.477秒，包含原错误fixture FAIL；hostsan/SDK仍继续。
