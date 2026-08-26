# container-05：SmallVector + intrusive list/slist

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 1-2k |
| **真实 LOC** | 255（149 头×2 + 106 测试） |
| **骨架?** | 否 |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台，container 36→45 case）|

> **历史迁移注记（2026-08-26）：** 本页保留原 `container-05` 合并任务及当时验收事实，不是
> `container_intrusive` 的 v2 签署。侵入式链表现已从完整 `container` 拆成独立、header-only 的
> `hyle::container_intrusive` 组件，公开头仍为 `<hyle/container/intrusive_list.h>`，类型仍在
> `hyle::container` namespace。当前候选边界见
> [`container-intrusive-design`](../../../architecture/container-intrusive-design.md)，使用方法见
> [`container_intrusive` 用户指南](../../../user/container-intrusive.md)；当前状态是
> `CANDIDATE / 复审中`，不能从本页的历史“完成”或旧 LOC/test 数推导为 `SIGNED`。

## 前置依赖

- 硬前置：`mem-02`（两者显式注入/不需分配器，但 SmallVector spill 走 `Allocator`）。

## 目的（为什么做这一步）

两个 std 没有、sched 又强依赖的结构：
- **SmallVector**：小集合零堆分配——通用热路径（参数列表、临时集合）省掉 malloc。
- **IntrusiveList**：节点内嵌 hook，O(1) 任意节点摘除 + 零分配——sched 的 waiter 链 / run-queue 溢出链刚需（`std::list` 每节点一次分配会拖垮调度热路径）。

clay 无这两者，设计标"新增，抄 folly 设计（non-code）"，无 clay 同名文件可 diff。

## 在架构中的位置（原任务）

原任务把两种类型都记为 T0 container、header-only 模板。当前归属已经细化：SmallVector 仍属于完整
container；侵入式链表由 `container_intrusive` 独立交付，并由 time 及完整 container 通过真实 target
依赖消费。sched waiter 后来因 detached wake-link 状态复用保留专用链；这一当前裁决不改写本页历史验收。

## 具体事务（做什么）

- **`SmallVector<T,N>`**：inline storage 前 N 个不碰堆，超 N spill 到 `Allocator` 缓冲（memcpy 搬迁）。`inline_storage()` 可查；move 时 inline 复制元素、heap 偷缓冲。T 限 trivially-copyable。
- **`IntrusiveList<T, &T::hook>`**：节点内嵌 `IntrusiveListHook`，O(1) push/pop/**remove（任意节点不遍历）**、零分配，环形 sentinel。`node_of` 用 container_of 偏移从 hook 还原 T（boost.intrusive 同款）。

## 交付物

- `include/hyle/container/small_vector.h` + `intrusive_list.h`（共 149 行）。
- `tests/container/small_vector_test.cc`（106 行）。

## 验收标准

- **编译**：✅ 双平台。
- **正确性**：✅ container 36→45 case。SmallVector spill 后数据完整（100 元素跨 spill）、move inline vs heap 两路径分别测；IntrusiveList O(1) 中间删除 + hook 非首成员时偏移正确还原节点。
- **诚实边界（历史实现）**：当时的 `node_of` 偏移技巧会被 UBSan 标记 null-ptr-member，本页曾将其
  记录为可接受。当前候选不再求值 null member expression，而是只解码已支持的 MSVC / Itanium
  data-member-pointer representation，并对未知/复杂 representation 编译期 fail closed；现行边界以
  独立架构页和本轮 sanitizer 证据为准。

## 对抗式自审

SmallVector 的 inline↔heap 迁移是错误高发区——分别测了"未 spill move 复制 inline"和"已 spill move 偷 heap 缓冲"两路径。IntrusiveList 的 `node_of` 在 hook 非结构体首成员时偏移必须正确，专项测试验证从中间 hook 还原节点地址。

## 参考

- folly `small_vector` / `IntrusiveList`（设计参考，非抄代码）。
- [`container-design`](../../../architecture/container-design.md)（完整 container 聚合摘要）。
- [`container-intrusive-design`](../../../architecture/container-intrusive-design.md)（侵入式组件当前归属与边界）。
