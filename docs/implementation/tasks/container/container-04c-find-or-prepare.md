# container-04c：SwissTable insert find_or_prepare 融合（单趟探测）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 0.05k |
| **真实 LOC** | +50（swiss_table.h shell） |
| **骨架?** | 否 |
| **状态** | ✅ 完成（双平台 11/11 含 oracle 对拍 50k/200k） |

## 前置依赖

- 硬前置：`container-04b`（type-erased core + erase 空槽回收）。

## 目的（为什么做这一步）

红队点名的最大"白白丢性能"点：insert 新键原走 `find()`（1 哈希 + 1 探测查重）再走 `find_insert_slot()`（再 1 探测找空槽）——**两趟探测、两次哈希**。学 abseil 融合成 `find_or_prepare`：单趟探测中同时查 key 命中 + 记录首个 empty/deleted 槽位，**哈希只算一次**。grow 才二次探测（冷路径，摊销罕见）。

## 在架构中的位置

纯 `swiss_table.h` 热路径内联层优化，不动 type-erased core 的 DLL 边界（container-04b 已定）。

## 具体事务（做什么）

- `find_or_prepare`：单趟探测同时（a）查 key 命中（H2 match + key eq）、（b）记录探测序列里首个 empty/deleted 槽位。命中返回既有槽；未命中返回首个可插槽，哈希全程算一次。
- **附带修 `growth_left` 双扣 bug**（并发红队挖出）：原 insert 无条件 `--growth_left`，但 erase 打 tombstone 不动 `growth_left` → 复用墓碑时同一净占用槽被扣两次（grow 偏早）。改为**只在插入 empty 槽时 `--growth_left`，复用 tombstone 不扣**，精确反映空槽预算。

## 交付物

- `include/hyle/container/swiss_table.h`：find_or_prepare 融合 + growth_left 记账修正（+50 行）。

## 验收标准

- **编译**：✅ Win MSVC 19.50 + Linux GCC14.2。
- **正确性**：✅ 双平台 SwissTable 11/11，含两 oracle 对拍（50k 混合 + 200k churn）——融合或 growth 记账错会立即发散。
- **诚实边界**：算法级 op 削减（新键路径 1 哈希+1 探测 vs 原 2+2），**未跑 before/after microbench**（削减是结构性的，bench 留作 profile-guided 任务）。

## 对抗式自审（正确性论证）

探测序列确定 → `find_or_prepare` 返回的首个 empty/deleted 槽**正是 later `find()` 会停下处**，插那里保持所有探测链完整。7/8 装载因子保证空槽恒 ≥ cap/8 > 0，probe 必遇 empty lane 终止（不死循环）。growth_left 修正后空槽预算精确：只有真正消耗 empty 槽才扣，tombstone 复用是零净增、不扣。

## 参考

- abseil `raw_hash_set` 的 `find_or_prepare_insert`（融合探测来源）。
- `container-04`/`04b`（SwissTable 探测层 + type-erased core）。
