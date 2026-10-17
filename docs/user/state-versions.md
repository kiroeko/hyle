# 显式只读集合版本与 dirty clone

入口在[State API](../../include/hyle/compute/state_versions.h)及
[typed clone API](../../include/hyle/compute/state_clone.h)，产品target为hyle::compute。
与Graph组合的普通consumer选择compute和graph两个产品及实际route。

StateSchema冷声明一个完整集合，成员是有描述的资源租约只读资源或确切长度的host metadata。
集合完整仅指这个schema；哪些recurrent/conv/KV、位置/RNG或资产字段构成有效snapshot由
consumer决定。StateEntryId持schema owner，不能以ordinal跨schema复用。

```cpp
auto schema = StateSchema::create(specifications).value();
auto state = schema.entry(0).value();
auto store = StateStore::create(schema, limits).value();
auto initial = store.import(complete_members).value();
auto head = StateHead::create(store, initial).value();

auto base = head.current().value();
auto draft = store.begin(base).value(); // 分享readonly backing，不复制集合
draft.replace(state, successfully_ready_pure_final).IgnoreError();
auto candidate = draft.finalize().value();
auto published = head.publish(base.identity(), candidate);
```

示例省略生产代码的Status处理。import先检查ID/描述/遗漏/重复/容量，不消费caller资源。
resource成员使用原ReadLease权限，不把未停止的writable aliases变成不可变；metadata深复制
成自有readonly bytes，库不解释其格式。版本/identity共享handle O(1)，每个保留identity仍占
版本容量；draft/finalized writer等拥有句柄也要明确销毁。head冲突不修改current或candidate。
全成员必须成功ready且没有pending writer才能finalize；已经失败的draft不能把实际prefix
伪rollback后继续发布。取消、放弃或关闭均不改变旧版本。

必须修改共享成员时才dirty clone。先按批量搬运 prepare资源与容量，cold factory只借稳定native
endpoint（CPU/CUDA的Queue，Apple/Vulkan原Context/Program角色），不要捕获Store/Version
owning handle造成反向环：

```cpp
auto route = StateCloneRoute::prepare(store,
    {.resource = destination_resource, .capacity_bytes = arena_bytes, .slots = 2},
    original_copy_program,
    [queue_ptr](const CopyBatchProgram &p) noexcept {
      return CopyBatchInvocation::create(p, *queue_ptr);
    }).value();
auto ticket = draft.clone_for_write(state, route).value();
auto writer = ticket.wait().value(); // 显式host等待；poll()可非阻塞观察
// 同步写或原native producer，通过原WriteLease/LeaseUse/native接口。
writer.publish().IgnoreError();
```

Apple/Vulkan factory调用它们各自的CopyBatchInvocation::create(program)，不需要通用Queue。
clone只复制该完整byte-addressable成员，保留原编码；不对opaque布局做raw clone。不同资源/
设备/原数学转换不是同一件事，需使用批量搬运接口支持的资源组合和已有成熟转换。pool/resource/Context、
borrowed Queue与allocator活到全部版本、writer和实际native use退休。

ticket内部遵循批量搬运 output→drop submission→reset→ReadLease::try_write的独占规则。Writer
不能发布被换成其他block的mutable handle。如果原typed native操作消费了WriteLease，
用complete(ready)接回同一block/generation的成功readonly结果；不同root或不完整extent
拒绝。Native异步writes必须资源租约跟踪，publish未ready返回Unavailable，已接受失败保留原错。

固定copy program的source endpoint必须是真实ReadLease.view().resource()。Apple原生array
readonly import的资源owner是逐对象身份，与普通MemoryResource::ref不同：pure final优先
readonly replace/分享；应用维护的conv/KV dirty域可以放在资源租约稳定资源池。若确需clone一个
已导出的native dense结果，冷prepare时声明它的实际source资源并保活该source owner通过
整个copy route。不能只拿目标MemoryResource的ref冒充所有原生source。完整例子和独立数值
参考见[公开消费者](../../tests/compute_state/public_state_consumer.cc)。

取消停止candidate发布，不保证GPU抢占已经接受的复制；native request_cancel可以明确返回
Unimplemented，此时仍等真实退休。批量搬运取消会关闭该invocation，确认退休后显式冷rearm才
重新创建原copy slot；unknown不能rearm。close不关闭caller Context，旧readonly版本仍可
消费；collect非阻塞处理已放弃copy与retained backing，drain仅显式等待本store的copy工作。
最后状态版本 owner析构不隐藏host等待，资源租约与批量搬运继续保活已经接受的native工作，borrow寿命仍由caller满足。

容量包括版本/draft/clone slots、所有仍导出的资源租约 block、完整backing根及host snapshots。
同pool不同entry只按完整pool capacity观察一次，资源租约原预算不再重复reserve。单独导出的reader/
native use可能比版本目录更晚释放，因而释放version不一定立即降低backing占用。容量满返回
Unavailable，collect不会驱逐活版本。默认上限可冷配置，不是整个进程/driver显存的硬上限。

StateCloneRoute::report返回原每slot最近episode和shared program元数据快照；native work或
host observer活跃时Unavailable，rearm后是新的slot身份。克隆bytes、更新upload、数学、
metadata复制、host等待和opaque native私有bytes分别计费，不拿这些接口证明零复制/capture
或整引擎吞吐。paged 布局作为 state 元数据不会自动执行分页 attention；计算接口见[批次 attention](batch-attention.md)。
