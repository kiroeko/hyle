# platform-03b：fs/env/os_info syscall

| 字段 | 值 |
|---|---|
| **阶段** | P1 |
| **模块** | platform/ |
| **状态** | ✅ 完成（Win x64 MSVC 19.50 + Linux x64 GCC14.2 双平台 fs 14/14；platform 套件 19→33）|

> **范围**：吸纳 clay `sys/os.h` 的 `file_system_api_t`(13 fn) + `os.shared.cc` 的可移植 helper(make_directory_recursive/touch/traverse) + `os_info_api_t`。这是 clay path/os 的 **syscall 半**;纯字符串半已在 platform-03a(container/path.h)。
> **cloc**：**生产 379 行**(fs.h 59 + fs.cc 320) + **测试 162 行**。

## 关键设计决策：buffer-fill + visitor → 保 fs 在 platform(P1)

clay fs 每个返回路径的 API 都走 `Allocator*` + `const char*`,目录列表返回 `strings_t*` blob,遍历用 `std::function`。**若 HYLE 照搬,fs 要依赖 mem(P2)/container(P3)——platform(P1) 反向依赖**(与 path 同款分层陷阱)。

**解法**:
- **所有路径返回走 buffer-fill** `(char* dst, size_t cap)` → snprintf 语义(NUL 终止 + 返完整长度,>= cap 即截断)。无 Allocator。
- **目录遍历走 visitor** `directory_iterate(path, fn_ptr, ctx)`(type-erased 核 + 头内 `for_each_entry` 模板 wrapper 收任意 lambda)。无 `strings_t` 分配、无 `std::function`。
- 结果:**fs.h 只 include macros/visibility,零 mem/container 依赖,fs 留在 platform(P1)**。比 clay 更精简(无隐藏分配),且对齐 HYLE buffer-fill 哲学(string_algo/path join_array 同款)。

## clay `file_system_api_t`(13) + helper → HYLE diff

| clay | HYLE | 备注 |
|---|---|---|
| `stat` | ✅ `stat`→`FileInfo` | exists/is_directory/size/mtime;缺失返 {exists=false} 不崩 |
| `directory_entries`(→strings_t blob) | ✅ `directory_iterate`+`for_each_entry` | **visitor 替分配式 blob**;排除 `.`/`..` |
| `make_directory` | ✅ `make_directory` | **改进:幂等**(已存在为目录也返 true,clay 裸 mkdir EEXIST 失败) |
| `remove_file` | ✅ `remove_file` | unlink/DeleteFile |
| `remove_directory` | ✅ `remove_directory`(空目录) + `remove_directory_recursive`(rm -rf) | clay 的 remove_directory 直接递归;HYLE 拆显式两个,递归版建在 visitor 上(可移植,替 clay Linux 专有 nftw) |
| `rename` | ✅ `rename_path` | 避与 POSIX `rename` 撞名 |
| `copy_file` | ✅ `copy_file` | **改进:可移植 read/write 循环**(替 clay Linux 专有 `sendfile`);Win 用 CopyFile |
| `getcwd` | ✅ `current_directory`(buffer-fill) | |
| `chdir` | ✅ `change_directory` | |
| `is_absolute` | ♻️ path-03a `is_absolute` | 纯字符串,已在 container/path.h |
| `absolute` | 📋 登记(组合) | = `current_directory` + `path::normalize`,跨 P1/P3 的组合,调用方拼;不在 fs 单列 |
| `temp_directory` | ✅ `temp_directory`(buffer-fill) | **改进:认 $TMPDIR/$TMP/$TEMP**(clay 硬编码 P_tmpdir 忽略环境);Win GetTempPath |
| `app_folder` | ✅ `home_directory`(buffer-fill) | Win %USERPROFILE% / POSIX $HOME→getpwuid |
| `SYS_make_directory_recursive` | ✅ `make_directory_recursive` | 逐段建,mkdir -p |
| `SYS_file_touch` | ✅ `touch` | 创建或更新 mtime |
| `SYS_traverse_folder`(std::function) | ✅ `for_each_entry`(模板,可递归调用方拼) | 净化:函数指针+ctx 替 std::function |
| `SYS_ensure_parent_dir_exists`/`copy_directory_recursive` | 📋 组合 | = directory(path)+make_directory_recursive / iterate+copy,调用方拼,不单列 |
| `os_info_api_t.num_logical_processors` | ♻️ `std::thread::hardware_concurrency()` | std 已有,不自写 |
| `os_info_api_t.avx_support` | 📋 simd §10 | CPUID 探测属 simd dispatch,登记不在此 |
| **新增** `get_env_var` | ✅ | clay os.h 无显式 env 读取 API;HYLE 补(区分 unset=kEnvNotSet vs set-but-empty=0) |

**clay os.linux.cc 全 830 行已读 → fs 原语全是真 syscall**(stat/opendir-readdir/mkdir/unlink/nftw/rename/sendfile/getcwd/chdir/getpwuid),os.shared.cc 209 行 helper 也真。无美化空函数。`file_io_api_t`(open/read/write)+ `mmap_*` 属文件 IO 域 → **io-05**,非本步。`socket_*` → io-04。`thread_api_t`/`time_api_t` 已 platform-01/time-01。

## 对抗式自审（六类）

- **① 正确性**:14 测在真实 temp 目录建唯一树,真做 mkdir/recursive/touch/stat/rename/copy/iterate/recursive-remove,断言后清理。visitor 排除 `.`/`..` 真测、早停真测、缺失目录返 false、get_env_var unset vs set、buffer 截断。双平台 14/14。
- **② 性能**:buffer-fill 零分配;visitor 函数指针(非 std::function);copy_file 64KiB 缓冲。POSIX readdir 用 d_type 快路径,DT_UNKNOWN 才 stat 回退。
- **③ 规范**:fs.h 零 mem/container/std 容器依赖(P1 干净);buffer-fill+visitor 范式;`HYLE_PLATFORM_API` 导出;无 std 流。
- **④ clay 覆盖**:file_system 13 fn 逐个 diff(上表);file_io/mmap/socket 按域分流 io-04/05,os_info 分 ♻️/simd,absolute/ensure_parent 标组合登记,**非略过**。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:recursive 操作建在 visitor 原语上(不重复遍历逻辑);join_path 小 helper 替 snprintf(避 GCC -Wformat-truncation 同时 bounded)。
- **改进 vs clay**:mkdir 幂等、temp 认环境变量、copy 可移植(非 Linux 专有 sendfile)、remove_directory 拆空/递归两个显式、无隐藏分配。

## 真编真测踩坑（诚实记录）

- GCC `<cerrno>` 漏 include → `errno`/`EEXIST` 未声明编译失败。补 `<cerrno>`。
- GCC fortify `-Wformat-truncation` 警 `snprintf(buf[4096],"%s/%s",path,name)`(path 可能满 4096)→ 改 bounded `join_path` helper(无 snprintf,放不下返 false)。本地 -Werror off 不致命,但 CI -Werror on 会挂,提前修。
- MSVC C4996 `fopen` 弃用 → fs.cc/fs_test.cc 顶 `_CRT_SECURE_NO_WARNINGS`(无全局处理)。

## 交付物

- `include/hyle/platform/fs.h`(FileInfo + stat/exists/is_directory/file_size + mkdir(+recursive)/remove_file/remove_directory(+recursive)/rename_path/copy_file/touch + current/temp/home_directory + get_env_var + directory_iterate/for_each_entry)
- `src/platform/fs.cc`(Win Win32 API 分支 + POSIX 分支 + 可移植 helper)
- `tests/platform/fs_test.cc`(14)

## 参考

- clay `sys/os.h`(file_system_api_t/os_info_api_t)、`os.linux.cc`(POSIX 真实现)、`os.shared.cc`(可移植 helper)。conventions §1(禁 std 流)、§4(buffer-fill 替隐藏分配)、§10(CPUID→simd)。platform-03a(path::normalize 供 absolute 组合)。
