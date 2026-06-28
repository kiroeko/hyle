# container-06：string_ref + string_buf 驻留池

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **Tier** | 0 |
| **LOC 估** | 1-2k |
| **骨架?** | 否（类型核心完整，高层工具显式拆 container-06b） |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台 + WSL TSan；container 93→95 case，新增 StringRef×20 + StringBuf×13）|

> **cloc 留档**：**生产 363 行** = string_ref.h 269+string_buf.h(头计入) / string_buf.cc 83 / swiss_table.h for_each +11；**测试 341 行**。
> **预测 1-2k vs 真实 363**：不是 stub，是**精瘦 + 复用红利**——① string_ref 是 ptr+len+null 位的 16B POD 视图，算法都是字节扫描，本就不该膨胀；② StringBuf 复用 container-04 的 SwissTable 做反查，.cc 仅 83 行（clay 是 363 行虚函数抽象类+自带哈希）；③ clay cstring.h 1033 行的大头是**高层字符串工具**（score_similarity/path_match/format_si/find_path_matches），属 `string_utils` 层，显式登记 container-06b，**非略过**。

## clay API diff（cstring.h + string_buf.h）

逐条对照 clay 同名文件的**类型核心** API 表面：

- **string_ref（cstring.h）全覆盖**：构造(ptr+size+nt / const char* / 泛型 char range) · `[]`/front/back/begin/end/empty/c_str · substr · find(char/ref)/rfind(char/ref) · find_first_of/find_last_of/find_first_not_of/find_last_not_of · starts_with/ends_with/contains/count · trim · copy_to · `operator==`/`<=>` · equal_ignore_case · 自由函数 c_str(alloc)/dup/concat/slice/split/rsplit · `_sr` 字面量 · `Hash<string_ref>`。
  - **对抗自审补漏（本里程碑内修）**：首版漏了 `rfind(string_ref)`/`find_last_of`/`count`/`rsplit`——diff clay 时发现，**当场补全并加测**，未留到下个里程碑。
  - **相对 clay 的现代化**：① 不委托 `std::string_view` 实现 find 族（clay 全委托），自己写 constexpr 字节扫描——string_ref **就是** std 视图的替代，公开头委托回 std 不干净；② 公开头不 include `<string>`/`<string_view>`/`<iostream>`；③ `operator<=>` 替 clay 6 个比较运算符；④ `_sr` 字面量替 `CLAY_STR` 宏。
- **string_buf.h**：`StringBuf` 类(add/retain/remove/lookup/intern/size) + make/destroy 工厂全覆盖。
  - **改进**：clay 是 `virtual` 抽象类 string_buf_t；HYLE 用具体类 + pimpl（SwissTable+mutex 藏 .cc，无 C4251，无 `<mutex>` 泄进公开头）。hash 命中时 **memcmp 校验内容**——XXH3 碰撞（天文数字级）走 fail-fast 而非像 clay 那样信任哈希静默别名。
  - **登记待补（container-06b）**：clay 的**全局单例** `intern()`/`usprintf()`/`lookup(hash)`（进程级字符串池）未做。

## 对抗式自审（六类）

- **① 正确性**：`find(string_ref)` 循环边界改减法判断（`i+needle.size` 在 4GiB 串会 uint32 回绕，已修）；rfind/find_last_* 的 `--i` 下溢用 `if(i==0)break` 守卫；substr 越界钳制 + null_terminated 只被"延伸到末尾的尾巴"继承。**lookup 返回指针在锁外**——契约同 clay：调用方须自持引用（intern 内部先 add+1 再 lookup，安全）。已加注释。
- **② 性能**：find(ref) 朴素 O(nm)（短 needle，region_eq 首字节即退，已注）；add 在锁**外**算 hash（缩短临界区）；intern=add+lookup 两次锁/两次 find（非热路径，留简洁，标为可优化）。
- **③ 规范**：内部 .cc 用 std::mutex（允许）；hashmap 用 HYLE SwissTable 非 std::unordered_map；hash 用 XXH3 非 murmur。**瑕疵**：string_ref.h 经 hash.h 间接拉入 `<string_view>`（因 `Hash<string_view>` 特化）——string_view 是视图非拥有型，不破"禁 std 拥有型"硬线，但与"内部连 string 都不靠 std"理想有距离，标可议（hash.h 重构时再去）。
- **④ clay 覆盖**：见上 diff，类型核心全覆盖 + 当场补 4 个漏；高层工具/全局单例登记 06b。
- **⑤ 残留脚手架**：string_ref 落地后**应回填** Status message / thread name 的 `std::string_view` 顶替——分层顾虑（error P0 / platform P1 依赖 container P3 目录头），**单列评估，给主决策**（见对话 task#1），未擅自改核心 ABI。
- **⑥ 过度复杂**：无缝合怪；Entry/Impl 结构直白。

## 双平台 + TSan

- **Win x64 MSVC 19.50**：container 95 case 全过，零 warning（/W4，含修掉的 C4251/C4273/LNK2019——首版构造/析构漏标导出宏 + friend 链接不一致，改为方法级导出 + inline `allocator()` 去 friend）。
- **Linux x64 GCC14.2 Release**：95 case 全过，零 warning。
- **WSL TSan（setarch -R）**：StringBuf 13 case（含 8 线程并发 intern 同/异内容）零 data race——锁路径真验证（Windows 跑不了的闸门）。

## 交付物

- `include/hyle/container/string_ref.h`（视图类型 + 零拷贝算法 + Hash 特化 + `_sr` 字面量）
- `include/hyle/container/string_buf.h` + `src/container/string_buf.cc`（引用计数驻留池，pimpl）
- `include/hyle/container/swiss_table.h` 的 `for_each`（填 container-04 缺口：持有自有指针 value 的表需迭代释放）
- `tests/container/string_ref_test.cc`（20）+ `string_buf_test.cc`（13）

## 参考

- clay `sys/cstring.h`(1033) / `sys/string_buf.h`(141)：ptr+len+null_terminated 位、interning、make/destroy 跨 DLL。
- conventions §7（string_ref null_terminated 位是相对 std::string_view 的真实增量）、§3.5（导出宏放方法不放类）、§1（禁 std 流/垃圾容器）。
