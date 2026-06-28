# platform-03a：path 路径字符串操作

| 字段 | 值 |
|---|---|
| **阶段** | P1（伞 ID platform-03 的字符串子步） |
| **模块** | container/（**非 platform**，见下"模块归属决策"） |
| **状态** | ✅ 完成（Win x64 MSVC 19.50 + Linux x64 GCC14.2 双平台；container +18 test，114→132）|

> **范围**：吸纳 clay `sys/path.cc` 的 **`path_api_t` vtable 表面**——纯字符串路径操作（join/dirname/basename/extension/normalize/split/is_absolute），**零 syscall**。clay path/os 的 fs/env/os_info syscall 半边留 platform-03。
> **cloc**：**生产 232 行**（path.h 124 + path.cc 108）+ **测试 166 行**。

## 模块归属决策（为何 path 字符串落 container 而非 platform）

coverage-map 把 `sys/path.*` 标 platform-03。但 clay `path_api_t` 的 16 个函数**全部**返回/接受 `string_ref`、用 `Allocator`、产 `carray`——零 syscall。而 `string_ref`/`carray`/`Allocator` 在 `hyle::container`/`hyle::mem`。

CMake 依赖实测两条独立链：`platform`→只依赖 `error`；`container`→`mem`（mem 不依赖 platform）。**若 path 放 platform 又用 string_ref，platform(P1) 反向依赖 container(P3)**——破坏分层，且 clay 里 path.cc 本就是 cstring 的邻居（`#include "clay/sys/cstring.h"`）。

**决策**：path **纯字符串逻辑落 `container/path.{h,cc}`**；clay path/os 的 **fs/env/os_info**（getcwd/absolute/temp_dir/directory_entries/宽字符↔UTF-8）才是真 syscall，留 **platform-03**。这正合"path 纯字符串先做、fs/env 单列"的拆分建议。伞 ID 仍记 platform-03a 以对齐 coverage-map。

## clay `path_api_t`(16) → HYLE diff（逐个核对，CHECKLIST#2）

| clay vtable 函数 | HYLE 去向 |
|---|---|
| `extension` / `extension_cstr` | ✅ `extension(string_ref)`——**语义升级**(见下)；C 串变体合并(string_ref 隐式从 `const char*` 构造) |
| `strip_extension` | ✅ `strip_extension`(splitext 配套) |
| `replace_extension` | ✅ `replace_extension(path,ext,alloc)`——**无扩展名时改为追加**(clay 返 null,无用) |
| `base` / `base_cstr` | ✅ `base`(""→"."、纯斜杠→"/") |
| `directory` | ✅ `directory`——**修 clay OOB**:全斜杠前缀(`//x`)时 clay 反扫下标下溢后先解引用再判越界；HYLE 先判界 |
| `parent_dir` | ✅ `parent_dir`=`dup(directory(p))`(clay 单列只因其 directory 返 view、parent_dir 返 owned C 串;HYLE 一个 view op+dup 覆盖) |
| `pop_first` / `pop_last` | ✅ 同名 |
| `split_all` | ✅ `split_all`→`CArray<string_ref>`(元素是 path 内 view,零串拷) |
| `join` | ✅ `join`(单 `/`,平台无关) |
| `join_array` | ✅ `join_array`(buffer-fill snprintf 语义,折叠内部分隔符) |
| `normalize` | ✅ `normalize`——吸纳 `path_normalize_native`(纯 `.`/`..`/`//` 折叠);blend `//` 前缀 + Win 盘符/UNC 大小写折叠**登记 platform-03**(耦合 fs 域) |
| `is_relative` | ✅ **升级为 `is_absolute`**(见下) |
| `is_unc` | ✅ `is_unc`(`\\` 前缀) |

**clay path.cc 全 1372 行已读 → 无"声明无实现"美化函数**（与 sprintf 假截断 / format_si 不同，path 这块 clay 诚实）。vtable 外 ~1100 行是平台/域特定，按表分流：

| clay vtable 外的实现 | 去向 |
|---|---|
| `path_contains`/`path_cmp_normalized`(PATH_MAX 缓冲 + tolower) | 📋 platform-03(依赖 native sep + 大小写域) |
| `path_rel`(Blender `//` blend-相对) | ⛔ blend 域,不吸纳 |
| `path_normalize_unc`/`path_unc_to_short`/`alloc_utf16_from_8` | 📋 platform-03(宽字符↔UTF-8 是 OS 边界) |
| `path_make_safe_filename_ex`(Win con/prn 保留名净化) | 📋 platform-03(Win 文件系统域) |
| `path_sequence_decode`/`encode`(帧号 `####`) | ⛔ 渲染域(slate 特有),不吸纳 |
| `path_is_win32_drive*`/`path_slash_native`/`path_suffix` | 📋 platform-03(随 native sep 处理一起) |

## 三处语义升级（clay 有不一致或域特定行为，登记）

1. **extension/strip_extension 用 Python `os.path.splitext` 语义**：clay **自身就有两套不一致实现**——`path_api_t.extension`(L13) 是朴素反扫(".bashrc"→".bashrc")，而 `path_extension_or_end`(L1076) 是 splitext(".bashrc"→无扩展)。选 splitext：**更健壮**(strip 永不产生空名),且与 clay 自己内部用的那套一致。
2. **`is_absolute` 替 clay `is_relative`**：clay `path_is_rel`(L234) 实为测 Blender `//` blend-相对前缀(`path[0]=='/' && path[1]=='/'`)——blend 域约定,通用 infra 无用。HYLE 给**真·绝对路径判定**(POSIX `/` / Win 盘符 `C:\` / UNC)。
3. **`normalize` 用 `/` 确定性折叠**：clay 在 native sep 上折叠且耦合 Win 盘符/UNC 大小写。HYLE 用 `/`(两平台同构 → 可双平台单测),Win 盘符/UNC 折叠登记 platform-03。normalize 改 clay 的就地 memmove 批处理为**组件走查**(读游标+写游标+floor),逻辑更清晰、结果等价;额外按 Python normpath 去尾斜杠。

## 对抗式自审（六类，CHECKLIST#4）

- **① 正确性**:18 测覆盖 16 API + 边界——空串/纯斜杠(`///`)/`..` 还原(相对保留不可约 `../`,绝对根吞 `/..`)/UNC/盘符(`C:` 非绝对)/join_array 截断报完整长度/directory 的 `//x` OOB(clay bug)真测;normalize 表驱动 14 例。`string_ref` size 为 uint32_t,反扫用 `i < size` 判下溢(借 string_ref.h 既有模式)。
- **② 性能**:查询族(extension/base/directory/is_*)全 constexpr inline,可编译期求值;零分配(仅 join/split/replace 显式 alloc 注入);normalize 单遍+局部回扫(路径深度小,实际 O(n))。
- **③ 规范**:path.h 只 include carray/string_ref/macros/allocator——**零 std 拥有型容器/流**(§1);返回 `CArray`/`string_ref` 全 HYLE 物;allocator 显式注入(§4)。
- **④ clay 覆盖**:vtable 16 函数逐个 diff(上表);vtable 外按域分流登记,**非略过**。
- **⑤ 残留脚手架**:无(未临时借 std)。
- **⑥ 过度复杂**:每函数单一职责;normalize 组件走查是 clay 算法的等价精简,非新发明。
- **跨 DLL**:`split_all` 返 `CArray<string_ref>`,析构走 DLL 内 `carray_free_internal`(allocator 一致,无 CRT 堆错配);MSVC 无 C4251。

## 预测 vs 真实 LOC

预测(README 校准)path clay 1372→HYLE 0.6–1k。**真实生产 232**。差距解释:1372 的 ~1100 行是 vtable 外的 fs/blend/Win 平台特定(已分流登记,非略过),**真正对标的 vtable 实现约 250–300 行 → HYLE 232 ≈ 1:1**(不可约真逻辑,符合"大件不缩水"预期)。剩余平台特定部分落 platform-03 时会再补可观行数。

## 交付物

- `include/hyle/container/path.h`(16 API:extension/strip/base/directory/parent_dir/pop_first/pop_last/is_absolute/is_unc/is_separator + join/join_array/split_all/replace_extension/normalize)
- `src/container/path.cc`(normalize/join/join_array/split_all/replace_extension)
- `tests/container/path_test.cc`(18)

## 参考

- clay `sys/path.h` + `sys/path.cc`(vtable 部分真实现)。conventions §1(禁 std 容器/流)、§4(allocator 注入)、§7(string_ref)。container-06(string_ref dup/concat/substr 复用)、container-03(CArray)。
