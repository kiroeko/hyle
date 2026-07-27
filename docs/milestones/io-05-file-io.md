# io-05：file io（+ 跨进程共享内存待）

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-02（真 op）|
| **状态** | ✅ 文件 io + named shmem（能力 C5）。Windows/WSL/macOS 真测 |

> **范围**：给 proactor 补文件元数据 op（`submit_open/close/statx/fsync/fallocate`，三后端）+ `file.h/cc` 的 `File` RAII + co_await read/write/stat/sync/allocate/close/open_file，shard-per-core 绑核。co_await 契约与 socket 完全一致。**shmem（mmap 跨进程）未做**，留 io-05 后续。
> **cloc**：文件 op 三后端 +~250 / `file.h`+`cc` 264（+ `file_test` 190）。

## 为什么是它
文件是「北极星① 推理引擎」加载权重、读配置的基本面。而**常规文件 IO 是 io_uring 真正打赢 epoll 的唯一场景**——reactor 无法 async 一个常规文件（`epoll_ctl` 返 EPERM），tokio 为此把文件 IO 卸载到线程池；io_uring 的 `openat/read/write/fsync` 是真异步 SQE。所以文件层同时是 proactor 完成式抽象「对三后端统一」价值的最好展示。

## 核心决策
- **对上一套统一 co_await 契约，三后端差异藏在 proactor**：
  - **io_uring**：`openat/close/fsync/fallocate` 真异步 SQE（`submit_one` 通用填 opcode/fd/addr/len/off/flags）。**fallocate 字段映射反直觉**——addr=len·len=mode·off=offset，照 liburing `io_uring_prep_fallocate` 核对。初版 `statx` 曾走同步 fstat + posted；2026-07-20 已改成真实 `IORING_OP_STATX`，稳定 native storage 在 CQE 时转换为公共 POD。
  - **IOCP**：全同步执行 + `iocp_post_meta` 合成（`kFileMeta` IoOp 带 64 位 `syn_result`——open 返 HANDLE 不被 post 的 DWORD 截断；FILETIME→signed Unix seconds/nanoseconds 精确换算；reap 认 kind 跳过 `GetOverlappedResult`）。
  - **epoll**：全同步 + posted；regular-file 的 `submit_read/write` 因 `epoll_ctl` 返 EPERM 回退**同步 pwrite/pread（尊重 offset）** + posted（epoll 不能异步 regular file，tokio 同理线程池卸载）。
- **中性 `FileStat`**：size/modified_time/mode/is_dir——modified_time 复用 canonical signed Unix POD，1970 前不 wrap、保留纳秒；其余为跨平台 statx 交集（一个文件服务器需要的）。
- **中性 `OpenFlags`**：kRead/kWrite/kCreate/kTruncate/kAppend，映射到 POSIX `O_*` / Windows `CreateFileW` disposition。
- **独立于 C2**：文件 op 这里不带 deadline（open/stat/fsync 极少需要；慢磁盘不是死 peer）。超时/取消是 io-06。

## 硬点：path 与 buf 生命周期
`submit_open` 的 `path` **必须活到完成**（io_uring openat 在 op 运行时才读 path，不是 submit 时）——`OpenAwaitable` 把 path 留在协程帧上。`File::close()` 的 `submit_close` **接管 fd 所有权**：io_uring 经 async SQE 关、IOCP/epoll 已在 submit_close 内同步 `::close`——故 `await_suspend` 无条件清 `fd_ = -1` 防析构双关。

## 对抗式自审（正确性 / TSan / 残留）
- **① 正确性**：open→write→fsync→read→stat→close 往返 + open missing 失败 + fallocate 扩展，三后端（Win IOCP / Linux io_uring / forced-epoll）。
- **④ TSan**：WSL 净。
- **⑤ 残留（诚实）**：shmem（mmap 跨进程）**未做**，明确登记 C5 后续。

## 交付物
- `include/hyle/io/file.h`（FileRead/Write/Stat/Sync/AllocOp + File RAII + OpenAwaitable/open_file）
- `src/io/file.cc`；`src/io/proactor.cc`（submit_open/close/statx/fsync/fallocate 三后端）
- `include/hyle/io/proactor.h`（FileStat / OpenFlags + 五个 submit 签名）
- `tests/io/file_test.cc`

## 参考
- io-design「① JIT 延后」io-05 文件/shmem 节。liburing `io_uring_prep_fallocate`（字段映射）；tokio 文件线程池卸载（epoll 无法 async regular file 的同款理由）。相关里程碑：io-02（read/write 底座）、io-04（同款 shard-per-core + co_await 契约）。
