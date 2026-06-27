# container-05：SmallVector + intrusive list/slist

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 1-2k |
| **骨架?** | |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台，container 36→45 case）|

> **cloc 留档（`cloc small_vector.h intrusive_list.h small_vector_test.cc`）**：**255 行** = 149 头(两个) + 106 测试。header-only 模板,非 stub。
> **clay 没有这两者**——设计标"新增,抄 folly 设计(non-code)"。无 clay 同名文件可 diff。
> **SmallVector<T,N>**：inline storage 前 N 个不碰堆,超 N spill 到 Allocator 缓冲(memcpy 搬迁)。`inline_storage()` 可查;move 时 inline 复制元素、heap 偷缓冲(测试分别验)。T 限 trivially-copyable。给通用热路径(参数列表、临时集合)。
> **IntrusiveList<T, &T::hook>**：节点内嵌 `IntrusiveListHook`,O(1) push/pop/**remove(任意节点不遍历)**、**零分配**。环形 sentinel。`node_of` 用 container_of 偏移技巧从 hook 还原 T(boost.intrusive 同款;UBSan 会标 null-ptr-member 偏移,可接受、已注)。给 sched 的 waiter 链/run-queue 溢出。
> **自审**:SmallVector spill 后数据完整(100 元素跨 spill 验)、move inline vs heap 两路径分别测;IntrusiveList O(1) 中间删除 + hook 非首成员时偏移正确还原节点。无并发(单线程结构,不需 TSan)。