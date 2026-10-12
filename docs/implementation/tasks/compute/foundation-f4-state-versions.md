# F4 显式不可变资源集合版本

## Closure card

- component / owner：新compute_state。已在1f2bd596b登记QUEUED；共同schema/集合版本/容量/发布由
  compute_state拥有，复制计算/原native退休保持F1/F2与各原route。不让Compute/Mem反向依赖state。
- baseline：bc9a689f983bfa5bf9da9a30019ff58f033378f0；A已CLOSED，不重开其原签范围。
- trigger：[固定peer研究](../project/foundation-f45-reference-research-2026-10.md)、
  [设计F4-01..06](../../../architecture/compute-state-paged-design.md)及用户明确活动goal授权。
- 正常consumer：显式readonly资源集合→多个有界draft→未改成员分享/pure final直接replace→
  只对dirty共享成员沿原F2 clone→成功ready全部成员→expected-base发布；旧版本/reader/native
  使用者保活，失败/取消不发布，不伪rollback实际已写prefix。
- 主纵切：recurrent FP32、conv window、KV块/布局和不可变host sidecar组成的显式完整状态。
  正交纵切：异构渲染资产集合候选更新/保留旧集合及分段矩阵共享/dirty更新。
- public journey：schema/opaque entry→StateStore有限目录→StateVersion/StateDraft→原ReadLease/
  WriteLease/typed F2 invocation→可观察CloneTicket/独占writer→ready Version→StateHead。
- scope：仅F4六FIX，冷容量/元数据与readonly完整成员；不把collection版本嵌到所有Tensor。
- non-goals：模型/RNG/token/checkpoint含义、任意过去rollback、业务事务、模型格式、page/LRU/
  admission策略、F5数学、万能Queue/runtime、新copy/math kernel、worker/reaper、公共工具重写。
- 依赖：compute/F1/F2、compute_ops、mem、util_core/error。native types只从其原factory进入
  typed clone adapter；不得把MLX/Vk opaque layout raw-clone，转换由consumer显式选择/计费。
  直接消费者只有新state fixture、普通四route state consumers及其必要package seams。
- 性能：P0，目录O((V+D)E)、分享零payload copy、dirty O(bytes)、已ready token发布O(1)短monitor；
  backing根去重、clone pools、host metadata深复制与native未知成本分账。无自动P1。
- 机器：Windows CPU/CUDA/Vulkan，WSL CPU/CUDA与适用sanitizers，Air CPU/Apple/Vulkan；
  Air先核对实际Mac16,12硬件/toolchain，Mini不用。WSL软件Vulkan仅构建/包。
- 预算：两轮有限owner审计；六FIX；每gate configure300/build1800/test120/consumer180秒；
  sanitizer build1800/test120。原失败累计，不因日志/提交/remote copy重置。重型门串行，
  库j4≥12GiB/j2≥8GiB/j1≥4GiB、test1，复用source cache但binary不跨配置共享。
- 预声明验证：schema/ID/不同store/base/完整成员/描述编码/overflow/容量/metadata OOM；
  share/replace/host-owned snapshot/old versions/导出reader及native-use容量；dirty clone仅实际
  dirty bytes、原ready/reset/独占writer/producer；冲突、失败prefix/cancel/abandon/未知/
  未启动suffix、不提前回收/close/collect/drain，不关闭caller Context。
  主/正交public consumers，四route实际source/static/shared、公开头/缺路线/真实迁址/loader。
  新store-local publication/record回收并发适用ASan/UBSan/TSan；原GPU数学不由san代签。
  每轮docs/closure/diff，registry明确previous-ref/生成status；不用queued identity触发全库。
- stop：预算到达或原权限/退休不能满足该旅程则BLOCKED/INCONCLUSIVE；系统合同新事实退回
  SCOPED/AUDITED；固定scope外新feature不顺手加入。目标完成后再做用户要求的剩余基础库统计。

## 当前阶段

SCOPED。本卡冻结有限需求与机器/预算，尚未授权产品实现。QUEUED身份尚无target/header/
dependency；实现必须在独立AUDITED checkpoint后进行。

## 能力初审与已确认复用

| 能力族 | 层级/来源 | 当前覆盖 | 有限动作 |
|---|---|---|---|
| 成员readonly/独占writer/native-last-use | 原子；F1实际闭环 | COMPLETE于原F1范围 | BORROW ReadLease/WriteLease/LeaseUse，不复制引用或fence协议 |
| 批量dirty bytes copy/producer/actual retirement | 复合；F2及peer copy source retention | COMPLETE于原四F2范围 | BORROW四typed Program/Invocation/Submission与原reset/drain |
| schema/不透明成员身份/完整immutable集合 | 复合；SGLang conv/temporal/sibling，资产集合 | MISSING | SELF有限schema与owned目录/host sidecar |
| 多branch分享/替换/发布冲突/旧版本 | 复合；vLLM CoW、llama有限snapshot，资产更新 | MISSING | SELF bounded draft/version/head，不解释模型恢复 |
| 集合容量/backing去重/clone生命周期 | 复合；两消费者重复glue | PARTIAL | SELF目录/clone记录；原F1实际identity/原native退休 |
| token/position/RNG/MTP接受/页表选择 | 应用策略 | OUT_OF_SCOPE | APPLICATION consumer显式列入成员或业务key |

第一轮owner输入：ReadLease::adopt只消费成功且禁止未跟踪writable alias；Readonly导入不可
promote为writer。WriteLease::publish要求所有admitted uses成功退休。
F2 output只覆盖成功last-write范围；取writer需要get output→drop submission→原reset→try_write。
LeasePool析构会关闭domain，新native use/try_write拒绝；clone pool owner必须随F4拥有对象保活。
LeaseAccess::backing返回真实pool根/resource/fullcapacity；同pool不同entry不能按entry重复计费。
collection版本generation不同于单F1 block generation；opaque IDs不能用裸ordinal跨schema伪造。
原native owner/alias合同不允许F4把外部仍可写的array/Vk allocation改名immutable。

## 有限FIX（实施前冻结API后补验收）

| ID | 工作与验收 | 当前 |
|---|---|---|
| F4-01 | schema/opaque身份/有限目录与完整成员；host sidecar拥有，不引用caller临时span | UNPROVEN |
| F4-02 | 分享/pure output替换/原F2 dirty clone；编码/alias/跨资源非法拒绝；不会全集合copy | UNPROVEN |
| F4-03 | 全成员成功ready、expected-base冲突不改current、旧版本/native use保活 | UNPROVEN |
| F4-04 | producer/partial error/cancel/abandon/unknown/未启动suffix，有限记录/backing真实回收及close/collect/drain | UNPROVEN |
| F4-05 | recurrent+conv+paged布局与正交资产集合public consumer，显式snapshot/恢复限制 | UNPROVEN |
| F4-06 | 四route自然typed入口、三交付/新头/缺席/迁址、P0已知/未知账/guide及红队 | UNPROVEN |

七门目前UNPROVEN；本轮target/公开形状、default容量、不可变成员导出与clone注册/退出合同
尚待第二轮审计冻结。主/正交body都须沿普通public C++ API，不以private测试hook代签。
