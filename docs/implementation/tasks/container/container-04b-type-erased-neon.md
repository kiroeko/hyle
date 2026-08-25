# container-04b：SwissTable type-erased 冷核 + erase 空槽回收 + NEON

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **状态** | ✅ 完成（Win x64 MSVC 19.50 + Linux x64 GCC14.2 双平台 11/11 SwissTable；container 132→133；**aarch64 GCC14 NEON 编过+指令验证、无 ARM 机未跑**）|

> **范围**（container-04 §D4 + coverage-map 登记的三项补漏）：① type-erased 冷核(防 per-T DLL 膨胀,conventions §7) ② erase reclaim-to-empty(abseil 优化) ③ NEON group(ARM,无机)。
> **cloc**：新核 `swiss_table.cc` **68 行**；`probe.h` +62(NEON+clz+BitMask helper)；`swiss_table.h` 净 +84 重构;测试 +20。

## ① type-erase 冷路径(非热路径)——关键设计决策

container-04 文档 §D4 登记 type-erased core 时注明"需 measure"。**对最被依赖的容器做去模板膨胀手术,正确的边界是只 type-erase 冷路径**:

- **abseil 实证**:`raw_hash_set` 把 find/insert/erase **保留在模板内联**(热路径),只把 resize/policy 走类型擦除。**热路径若 type-erase,每次探测都要间接调 hash/eq**——对 `int` 这种廉价键是灾难。
- **HYLE 做法**:把 `grow()` 的重逻辑(alloc + memset + rehash-reinsert 循环 + free)抽进 `swiss_table.cc` 的 `detail::swiss_resize`/`swiss_free`,**每个 `<K,V>` 实例化只剩一个三行调用**;hot find/insert/erase **逐字节不动**,零间接、零回归。
- **类型擦除接口**:`SwissStorage` POD(ctrl/slots/cap/mask/size/growth_left) + `swiss_resize(s, new_cap, slot_size, slot_align, alloc, hash_slot_fn, hash_state)`。核心不知 K——靠 `hash_slot(slot, state)` 函数指针 thunk 重算槽哈希(键在 slot 偏移 0,thunk 把 `void*` reinterpret 成 `K*`)。`find_insert_slot`/`set_ctrl` 是纯控制字节逻辑,核内自带一份。
- **收益**:大 `grow()` 体(~35 行)离开每份实例化 → DLL 防膨胀(§7 兑现);**热路径性能 0 影响**(measure 顾虑由"不碰热路径"直接绕开)。

## ② erase reclaim-to-empty(abseil 优化)

container-04 的 erase 只打 tombstone(靠 grow 的 same-size rehash 回收)。本步加 abseil 回收规则:erase 时若**该槽所在 group 与前一个 group 都有空槽、且跨该槽的非空 run 短于一个 group 宽**,则没有任何键探测**穿过**该槽 → 可直接降为 `kEmpty`(还 `growth_left`),否则才打 tombstone。`erase_at(idx)`:`empty_before && empty_after && (empty_after.trailing_zeros() + empty_before.leading_zeros(kGroup) < kGroup)`。效果:churn 下 tombstone 更少、rehash 更少、容量更紧。**正确性**:只在"没键穿过"时降空,find 不受影响。

## ③ NEON group(ARM64,编过+指令验证、未跑)

`probe.h` 加 `GroupNeon`(8 宽):`vld1_u8` 载 ctrl,`vceq_u8`(match/match_empty)/`vtst_u8`(empty_or_deleted)→ 每命中字节 0xFF → `pack` 折成 **1-bit/lane 掩码**,满足与 SSE2/SWAR **同一个共享 BitMask 契约**(迭代/lowest/leading_zeros 全通用)。abseil aarch64 用 byte-mask + /8 移位;HYLE 折到 bit 以**不分叉 mask 类型**(多几条指令,正确性等价)。`Group` 别名加 `#elif HYLE_ARCH_ARM64`。

**诚实边界**:本机无 ARM 机 → **无法运行测试**。但装了 `aarch64-linux-gnu-g++-14` 交叉编译器,`swiss_table.cc`(核用 Group)+ 一个强制实例化 `GroupNeon::match/match_empty/erase_at` 的 TU **均 `-c` 编过零错**,`objdump` 确认真发 `dup`/`cmeq` NEON 指令。"写不验" → **"写 + 真 aarch64 编过 + 指令验证、仅未跑"**。

## 对抗式自审（六类，CHECKLIST#4）

- **① 正确性**:type-erase 后热路径逐字节未改;冷核 `swiss_resize` 忠实镜像原 grow(growth_left=new_cap*7/8-size 等价原"初始化后逐次减");两 oracle 测(50k 混合 + 200k churn)= reclaim 的强对拍——若降空降错致键丢,必发散;双平台 11/11。
- **② 性能**:热路径 find/insert/erase **零间接**(核只接管冷 resize/free);reclaim 减 tombstone/rehash;NEON 给 ARM 8 宽 SIMD(替 SWAR 标量)。
- **③ 规范**:type-erased core+typed shell 正是 §7 范式;`HYLE_CONTAINER_API` 导出核;无 std 容器泄漏。
- **④ clay 覆盖**:本步是 container-04 显式登记的补漏(§D4),非 clay 新表面;clay hashmap.h 本就只 TODO SwissTable,HYLE 此处是超 clay 的升级。
- **⑤ 残留脚手架**:无。container-04 文档/coverage-map 的"type-erased 延迟 04b"登记**本步兑现转 ✅**。
- **⑥ 过度复杂**:只擦冷路径(克制),没把热路径也擦(会回归);reclaim 是 abseil 算法忠实移植非自造;NEON pack 复用 SWAR 同款 top-bit 折叠。
- **type-erase 真省膨胀?**:大 grow 体从每份 `<K,V>` 实例化移入 DLL 单份;留在头里的只是小 hot 函数(本就会内联)。≥2 个不同 `<K,V>` 时净省。

## 交付物

- `include/hyle/container/probe.h`(+GroupNeon + detail::clz32 + BitMask::trailing_zeros/leading_zeros)
- `include/hyle/container/swiss_table.h`(grow/free 转调核 + hash_slot_thunk + erase_at reclaim;hot 路径不动)
- `src/container/swiss_table.cc`(detail::SwissStorage + swiss_resize + swiss_free,新)
- `tests/container/swiss_table_test.cc`(+1:EraseReinsertChurnStaysBounded;原 10 全绿)

## 参考

- abseil `raw_hash_set.h`(EraseMetaOnly reclaim 规则、policy 类型擦除热/冷分界)、`GroupAArch64Impl`(NEON)。container-04(探测层/控制字节/三角探测)。conventions §7(type-erased core+typed shell)、§10(SIMD 分层)。
