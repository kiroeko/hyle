# container-06b：string_algo 字节串算法

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | container/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；container +12 test）|

> **范围**：吸纳 clay `utils/string_utils.cc` + `sys/cstring.cc` 中 **clay 真正实现了的** 字节串算法。string_ref-in / buffer-fill-out,无 std 流。
> **cloc**：**生产 175 行**(string_algo.h) + **测试 92 行**。

## 重大诚实发现:coverage-map 之前的 06b 范围基于"声明"而非"实现"

之前 coverage-map/里程碑写 06b = `score_similarity/path_match/format_si/find_path_matches/capitalize`。**核对 clay `src/` 后发现**:`format_si`、`find_path_matches`、`cscore_path_match`、`cstr_score_similarity` **clay 全部只在头文件声明、`src/` 中无任何实现**(与 sprintf 假 UTF-8 截断同款"README 美化")。**无可吸纳**——它们不是"漏做",是 clay 本就没做。若 HYLE 要模糊路径匹配 = 净新设计,登记 container-06c(非吸纳)。

## clay 真实现的 → HYLE 去向

| clay 物(真有实现) | HYLE 去向 |
|---|---|
| `capitalize`(cstring.cc,in-place C-ident→UI) | ✅ `capitalize(char*)`,含 "Ui"→"UI" 缩写特例 |
| `hex_encode`(string_utils.cc) | ✅ `hex_encode(data,len,out,cap)` buffer-fill |
| `escape`(string_utils.cc,→std::string) | ✅ `escape(string_ref,out,cap)` buffer-fill |
| `utf8_decode`/`utf8_length`/`utf8_icmp` | ✅ `Utf8Decoder`(string_ref 上)/`utf8_length`/`utf8_icmp` |
| `is_hex_digit`/`hex_to_int`/`is_whitespace`/`isprint` | ✅ constexpr 分类器(locale-free)|
| `tolower`/`toupper`(std::string,locale) | ✅ `to_lower/to_upper`(ASCII buffer-fill 无 alloc/locale)|
| `str`/`make_str`/`join`/`to_string` | ♻️ `std::format`(conventions §8)|
| `stoi`/`stod`/`stoll`(Android polyfill) | ♻️ `std::from_chars`/`std::stoi`(C++23,非 Android)|
| `replace_all`/`split`/`trim`(std::string) | ♻️ trim/split 已在 string_ref(container-06);replace_all 走 string_buf/format |
| `to_utf8`/`from_utf8`(wchar_t,Windows)| 📋 登记 platform-03(宽字符↔UTF-8 是 OS 边界,随 path 一起)|

## 改进 / 设计

- **hex_value 修 clay bug**:clay `hex_to_int` 对非 hex 字符**静默返 0**(藏坏数据);HYLE 返 **-1** 显式信号。
- **全 buffer-fill / string_ref**:零 std 流、零分配(clay 全程 std::string + ostringstream)。snprintf 截断语义(返完整长度)。
- **Utf8Decoder 移植**:clay 基于 `std::string::const_iterator`,HYLE 基于 string_ref(ptr 范围),逻辑等价(前导 1 计数 + 续字节校验 + 非法标记)。

## 对抗式自审（六类）

- **① 正确性**:hex_value 非法返 -1 真测、case 转换截断真测、capitalize "Ui"→"UI"+下划线真测、escape 截断报完整长度真测、UTF-8 多字节(héllo/你好 6 字节 2-5 码点)+ 非法(孤立续字节/截断序列)真测、utf8_icmp 限长+前缀真测。
- **② 性能**:全 inline,buffer-fill 零分配;分类器 constexpr;ASCII fold 无 locale 调用(clay `::tolower` 走 locale 慢)。
- **③ 规范**:**零 std 流/string**(clay 的 epicenter,HYLE 彻底切干净);string_ref-in/buffer-fill-out;头-only inline。
- **④ clay 覆盖**:真实现的逐项吸纳;声明未实现的(format_si 等)诚实标记为 clay 未做+登记 06c,**非略过**。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:每个函数单一职责;Utf8Decoder 是 clay 算法的忠实精简。
- **"比 std 快"**:to_lower/to_upper ASCII 无 locale(std `std::tolower` 走 locale 表慢);UTF-8 std 无(`<codecvt>` 已废弃),是 std 缺的能力。

## 交付物

- `include/hyle/container/string_algo.h`（分类器 + case + capitalize + hex + escape + UTF-8）
- `tests/container/string_algo_test.cc`（12）

## 参考

- clay `utils/string_utils.cc`、`sys/cstring.cc`(真实现部分)。conventions §1(禁 std 流)、§8(std::format)。container-06(string_ref 已含 trim/split)。
