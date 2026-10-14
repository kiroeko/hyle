# G03 原 pending copy 到 attention 的 receiver 反证

2026-10-14；compute_batch SCOPED 审计，产品固定 bcaec010b，source terminal
944734ec1。没有产品修改。[裁决](verdict.json)、Apple [原型](apple-prototype.cc)/
[结果](apple-final.log)/[累计记录](apple-receipt.json)/[实际编译](apple-compiled-inputs.json)/
[原始编排](apple-raw.zip)，Vulkan [原型](vulkan-prototype.cc)/
[结果](vulkan-final.log)/[累计记录](vulkan-receipt.json)/
[实际编译](vulkan-compiled-inputs.json)/[原始编排](vulkan-raw.zip)。

MacBook Air Mac16,12 / arm64 / macOS27.0.1 / Apple Clang21。两个原 native route
均实测成功来源和取消来源两种情况：原 CopyBatchUpdate 取消状态为 CANCELLED，后继
原 attention 仍发布 SUCCESS，输出描述为 2x8x64。原 Q/metadata/未变页/values 的
read guards 与 dirty page 的原 exclusive ordered-read admission 均由原 F1 保持；
源与后继是同一原 Runtime/queue，不创建第二执行器。最后只 wait attention，无
producer/copy 中间 host wait。冷 metadata 上传在 source 提交之前完成，不能算作
warm 无等待的证明。

这是 trusted source 审计原型。原型为 unchanged page2 建立真实 ready frame，仅借其
full-arena address capacity；dirty page0 的 permission 另外从原 update 获得，没有
伪造 ReadLease。当前公开 controller 仍要求 ready frames，原型不证明普通用户可以
安全直接组合。输出 shape 检查不代替独立数值 oracle，也不证明 late native producer、
64 步、两 slots、七路线、取消导致 GPU work 被抢占或整个 G03 完成。

Apple 用原 GPU checkpoint 留住 queued copy。Vulkan 实际 QueueSubmit/fence/wait
未替换；fixture 仅遮住 GetFenceStatus 的成功观察以维持 unknown retirement，不能
声称真实 GPU shader 被阻塞。来源的请求取消在后继提交之前发生；接纳已知失败应
拒绝后继，接纳之后才确定的失败则须保 actual retirement 并阻止发布成功结果。

Apple 前两次错误 target 实际运行旧 copy-owner fixture，不计 receiver PASS；第二次
实际 source-target 审计失败保留。第三次 `_v2` 实际编译本原型才复现反例。Vulkan
第一次 fixture ResultLease 解引用编译失败，第二次调用未导出内部函数导致 shared
链接失败；最终 private source fixture 链接已有 bca static artifact 成功，
[原产品 receipt](vulkan-static-product-receipt.json)保留。它不是公开 SDK 验收，不
以 export 内部函数修测试。原始 FAIL/旧 fixture 记录均保留。

累计 Apple 39.474 秒、Vulkan 52.752 秒；最多四 jobs，heavy 串行，无全模型/P1。
本证据只授权活动主卡继续界定 receiver 与必要下层 owner，CLOSED source owner
不因该 fixture 自动重开。
