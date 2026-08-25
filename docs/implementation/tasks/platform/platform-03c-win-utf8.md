# platform-03c：Win UTF-8 路径（...A→...W）+ 长路径 \\?\ + copy_file 内核零拷贝

| 字段 | 值 |
|---|---|
| **阶段** | P1 |
| **模块** | platform/ |
| **Tier** | 2 |
| **LOC 估** | 0.1k |
| **真实 LOC** | fs.cc 320→475 |
| **骨架?** | 否 |
| **状态** | ✅ 完成（双平台真测；UNC 长路径实跑待网络共享） |

## 前置依赖

- 硬前置：`platform-03b`（fs/env syscall 层，本步在其 Win 分支上修）。

## 目的（为什么做这一步）

platform-03b 的 Win 分支全程走 `...A`（ANSI）API。**ANSI 按活动代码页解码**——非 ASCII 的 UTF-8 路径（CJK、emoji）会被误解，而 03b 的 ASCII 测例恰好掩盖了这点。HYLE 路径对外是 UTF-8 字节串（POSIX 一致），Win 侧必须 UTF-8↔UTF-16 转换走 `...W` API 才正确。顺带补两项 Win 文件系统的现实缺口：>260 字符长路径、大文件拷贝的内核零拷贝。

## 在架构中的位置

纯 platform 层内修复，**不引入 mem/container 依赖**（保 P1 分层不反向依赖 P2/P3）——转换缓冲用栈 + RAII 堆兜底，不经分配器。

## 具体事务（做什么）

- **`WidePath`**：UTF-8 `const char*` → UTF-16（`MultiByteToWideChar`，栈 512 + 堆兜底 RAII），自包含、零模块依赖。
- **`fill_w`**：UTF-16 → UTF-8 buffer-fill（`WideCharToMultiByte`，`snprintf` 语义）。
- **12 个 `...A` 全转 `...W`**：stat / make_directory / remove_directory / remove_file / rename / copy_file / current_directory / change_directory / temp_directory / home_directory / get_env_var / directory_iterate。
- **长路径 `\\?\` 扩展前缀**：>260 路径绕过 MAX_PATH（drive 路径测过；UNC `\\?\UNC\` 形式写了但未验）。
- **Linux `copy_file_range`**：内核零拷贝，失败回退到用户态读写循环。

## 交付物

- `src/platform/fs.cc` Win 分支重写（320→475 行，含 UTF-8 转换 + 长路径 + 零拷贝）。
- `tests/platform/fs_test.cc`：UTF-8 非 ASCII 路径、长路径、大文件拷贝测试。

## 验收标准

- **编译**：✅ Win MSVC 19.50 + Linux GCC14.2。
- **正确性**：✅ Win 36（含 `Utf8NonAsciiPath`：CJK+emoji 路径）+ Linux 18（POSIX 字节透明）。
- **诚实边界**：UNC 长路径（`\\?\UNC\`）写了未实跑——待网络共享环境验。

## 对抗式自审（红队 UTF-8 测试揪出真 bug）

红队的 CJK+emoji 路径测试揪出 `touch()` 的**双重 ANSI 漏洞**：它在可移植段用 `std::fopen`（Win 上 ANSI）+ `CreateFileA`，两条都绕过 `WidePath`。修：Win 改 `CreateFileW`（`OPEN_ALWAYS` 创建或打开，无 fopen）+ `SetFileTime`。测试先红（揪出 bug）、修后转绿——证明测试真有牙。

## 参考

- `platform-03b`（被修的 fs/env 层）。
- Win API：`MultiByteToWideChar`/`WideCharToMultiByte`、`\\?\` 长路径前缀；Linux `copy_file_range`。
