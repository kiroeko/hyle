# F4 candidate 独立终审输入

固定29d3c7a3e，产品src/include/cmake对当前HEAD无diff。本次先重新从公开合同、实际owner和
负路径读候选，不把原24项PASS或原F1/F2证书当新F4组合完整性证明；最终裁决等剩余交付格。

- Schema/Entry/Version/Identity：opaque entry持schema owner，不允许裸ordinal重建；version持
  Store并使record.refs存续，generation检查饱和前拒绝，无指针身份ABA。跨schema/store/base
  错误拒绝；empty schema可完整import/finalize，资源零字节和metadata错长度拒绝。
- 导入：先完整IDs/重复/描述/约束，forecast metadata根/字节/记录后才做owned snapshot。
  分配失败不消费输入，局部incoming清理；collect只移除唯一anchor，既有version不被修改。
  metadata payload沿原mem资源，Store目录allocator及ResourceBudget只在其承诺范围计费。
- Immutable共享：begin/finalize只拷ReadLease handles。Version成员无writable owner；允许任意
  借来的外部仍可写alias冒充只读会违反原F1导入前提，guide已明确，没有新全局写探测承诺。
- Head：expected identity来自同Store；store monitor内检查closed/身份交换handle，原previous
  放在外面drop。临时identity引用不可能成为末引用（head/expected仍存续），不存在递归lock。
  错误Status可分配，不承诺所有路径allocation-free。current关闭后允许保留旧readonly版本。
- Clone：同Store合法route/entry、schema资源和byte-addressable编码预检，只有目标成员复制。
  F2真实端点按实际source资源声明；Apple逐对象readonly import不能用memory.ref冒充，
  普通consumer单独冷计划验证。Opaque可readonly分享，raw-clone Unimplemented。
- Writer：ticket→writer转移唯一draft引用；pending成员阻止finalize。只original block/generation
  的成功F1 publish/ready可接回。改掉writer handle、未结束use、producer原错误都拒绝；
  no-op readonly任务失败不是payload被写坏，不虚造状态恢复/任意rollback。
- Cancel/close：candidate标失败，拒绝未开始suffix，不撤销实际prefix。CUDA无抢占明确保留。
  Store close只停止自身接纳，不关闭callerContext；Store drain在monitor之外等原F2，slot
  observers保留显式wait，防collect破坏正在观察的submission。TSan覆盖Store新并发；CUDA/
  Apple实际pending+concurrent wait/close覆盖native组合。Vulkanunknown在原观察恢复前不能
  取writer/rearm或回收live record；实际GPU待验不由CPU测试代签。
- Abandon：F4无native回指Store循环。原F2实际native descriptors/LeaseUse保留源/目标，所有
  F4 controls可被释放；源last-use直到actual retirement。Apple的LeaseUse回调先于native
  owner丢retired Job，metadata最终余额需显式owner join后检查，不能把早期余额当泄漏。
- 成本：V/D/E/B/C固定容量，backing去重收全pool容量、escapedreader/native use继续计费。
  冷ledger有二次扫描，默认B=128，不适合作无界paged hotpath声明；F5须独立规划。有界多
  slots不代表无限任意模型状态。无新worker/reaper/kernel/global cache/所有Tensor字段。
- 普通public source和installed consumer互相不代签；native-only尤其要证明缺CPU和runtime
  loader，不可把CPU/native混合包成功当原native-only安装资格。剩余SDK/WSL证据未齐前
  维持CANDIDATE，不先写最终red_team/delivery PASS。

目前这遍源码审查未得到要求退回AUDITED的产品反例。后续只能按活动卡补证据，真实反例
才退阶段修候选。SDK缺格、WSL普通source、terminal风险总矩阵仍是未完成工作，不抹掉。
