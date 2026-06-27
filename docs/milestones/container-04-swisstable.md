# container-04：SwissTable 探测层（控制字节 + SIMD 探测，clay 只 TODO 过）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 3-4k（大）|
| **骨架?** | |
| **状态** | 进行中 |

## 前置依赖

- 硬前置：`container-02`(XXH3 hasher)、`mem-01`(Allocator)。
- 决策依据：`container-design.md` §"SwissTable SIMD 探测"(控制字节值精选、三角数二次探测、三套 group SSE2/NEON/SWAR、tier 倒置 probe.h 自包含);`hyle-scope.md` 行77。

## 目的

flat hashmap 是 container 焦点 #5 与 13 处 clay 依赖的底座。clay 只有**线性探测 + C 宏泛型**,且注释里明说"控制字节数组没实现、待测"(`hashmap.h:109-131`)。本步**C++23 concept 模板重写 + 实现 clay 只 TODO 的 SwissTable**:控制字节 + 16 路 SIMD 探测 + 三角二次探测。

## 关键设计决策

### D1：abseil SwissTable 控制字节(值精选,别自改)
- 与 slot 平行的 `int8_t ctrl[]`,每字节:`0x80`(空)、`0xFE`(删/tombstone)、`0xFF`(哨兵 kSentinel)、`0hhhhhhh`(full,存 hash 高 7 位 H2)。这些值为 SIMD 精选(abseil `hashtable_control_bytes.h`)。
- H1 = hash 全宽做 group 起点;H2 = hash 高 7 位填 ctrl。

### D2：三套 group(probe.h 自包含,不依赖 simd/ Tier1)
- **x86 SSE2**(group 宽 16):`_mm_set1_epi8(h2)` + `_mm_cmpeq_epi8` + `movemask` → 16 位候选位图。
- **SWAR 标量兜底**(abseil GroupPortableImpl):非 x86/ARM 必需,否则编不过。本步**先做 SSE2 + SWAR**(双平台 x64 都能跑/验)。
- **NEON**(ARM,group 宽 8):写但**无 ARM 机不验证**,标注。
- probe.h 内联 intrinsic,**不拉整个 simd/**(保 container T0 不依赖 simd T1)。

### D3：三角数二次探测(非线性)
- `offset += (index += Width)` 的 group 跳跃,非 clay 的线性探测。空槽终止;erase 用 tombstone,能证明不在任何探测链的删除位降级为 empty(避墓碑堆积)。

### D4：correctness-first 模板;type-erased core 显式延迟(防"悄悄漏层")
- 本步做**正确的 concept 模板 `SwissTable<K,V,Hash,Eq>`**,对拍 `std::unordered_map` 验正确性 + 双平台。
- **type-erased core**(`detail::*_internal(void*, item_size...)` 防 per-T DLL 膨胀,conventions 行80/clay 范式)**本步不做,登记为 container-04b** —— 它是减膨胀优化、需 measure;先把探测/resize/erase 的正确性立住。**显式登记,绝不假装已做**(mem-03b 教训)。

## 具体事务

- [ ] `include/hyle/container/probe.h`:控制字节常量 + `Group`(SSE2 / SWAR;NEON 未验)`match(h2)`/`match_empty`/`mask_*`。
- [ ] `include/hyle/container/swiss_table.h`:`SwissTable<K,V,Hash,Eq>` —— ctrl+slot 数组(Allocator)、find/insert/erase/rehash、三角探测、负载因子 7/8。
- [ ] **不做**:type-erased core(container-04b)、string key 特化(container-06)、并发(SwissTable 单线程,clay flat 同)。

## 验收标准

- **编译**：Win x64 MSVC + Linux x64 GCC14.2(SSE2 + SWAR 两 group 都编/跑)。
- **测试**：**对拍 `std::unordered_map`**——随机 insert/find/erase 序列结果一致(几万次操作);负载因子触发 rehash 后数据不丢;erase tombstone 后仍能查到其他键;迭代覆盖所有键;碰撞键(同 H1)正确区分;空表/单元素边界。SWAR 与 SSE2 路径结果一致(同输入同结果)。
- **硬指标**：10 万随机键对拍 unordered_map 零差异;rehash 跨越多次扩容数据完整;SSE2 与强制 SWAR 编译两版结果逐位一致。
- **基准(留 bench 阶段)**:对标 absl::flat_hash_map / std::unordered_map 查找吞吐。

## 参考

- abseil Swiss Tables 设计笔记 + `hashtable_control_bytes.h`(控制字节值)。
- `container-design.md` §SwissTable;clay `hashmap.h`(线性探测,SwissTable 仅 TODO)。
