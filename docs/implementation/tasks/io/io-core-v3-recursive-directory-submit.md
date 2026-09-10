# IO Core v3：Windows 递归目录提交语义

<!-- MODULE CLOSURE RECORD: CLOSED -->

## Closure card

- component / owner / round：`io_core` / IO core maintainers /
  `io-core-v3-recursive-directory-submit`；kind=`REOPEN`；canonical checkout=
  `C:\public\project-hyle\hyle`；baseline=
  `6c0b877bab4ff574a6073d8f336423f8f94418dd`；前轮 terminal record 是
  [`io-core-v2-file-watch-submission`](io-core-v2-file-watch-submission.md)。
- trigger 与证据：正在 `SCOPED` 的 `io_watch` G05 源码预审证明，Windows 已关闭 detail seam 把
  `ReadDirectoryChangesW` 的 `bWatchSubtree` 固定为 `FALSE`，不能承载已准入的显式 recursive-tree
  target。绕过 Proactor 会复制 IOCP operation/cancel/completion 生命周期；前轮记录已把“io_watch
  证明 seam 不足”列为合法 reopen trigger。
- 正常 consumer 任务：`io_watch` 的 owner-pinned Windows pump 使用同一个 IOCP operation，在 direct
  target 传 `FALSE`、recursive-tree target 传 `TRUE`，buffer/token 继续借到 terminal completion。
- 主 / 反偏置纵切：主纵切是递归 model/asset tree；反偏置是既有 exact-file/direct-child target，必须
  保持原提交、完成与取消语义。
- scope：只让 Windows-only installed-detail `submit_directory_changes` 接受明确的 subtree 语义并透传给
  OS；补 true/false 直接机制测试与调用方迁移。不得增加另一种 operation、线程、队列或公开产品 API。
- non-goals：事件解析、相对路径、Linux inotify 树、macOS FSEvents、watch policy、公开 Native FFI、
  G03/G04/G06、其它 IO 算法或公共性能工具。
- 直接依赖 / consumer：中央依赖不变；唯一产品 consumer 是 `io_watch`。`io_core` 不得反向依赖它。
- 性能 tier：`P0`。新增值只是一次 native submit 的已有 syscall 参数；普通 IO operation、slab、对象 layout
  和 non-Windows binary 均不得变化。源码账单足以裁决，不运行孤立纳秒 benchmark。
- 适用机器 / 验证：Windows MSVC direct true/false mutation、rearm/cancel/admission；既有 io_core
  regression；Linux/macOS 条件编译；changed-header；exact source/static/shared IO consumer；P0、红队、
  closure/docs/diff。sanitizer 只覆盖可移植既有路径，不冒充 Windows syscall 插桩。
- 调查 / 纠正预算：一次源码与 Win32/libuv 契约审计，一个产品批次；若需要改变 completion layout、普通
  submit 成本或公开 IO API，立即 `BLOCKED`。
- stop：上述有限 seam 及调用方不能完整表达两种语义，或 Windows 递归需要第二 runtime 时 `BLOCKED`；
  否则通过预声明门后恢复 `CLOSED`。

## 审计与有限改动集（`AUDITED`）

Win32 合同与 libuv `3e606bc30a84` 的 Windows backend 均把递归性直接映射为
`ReadDirectoryChangesW` 的 `bWatchSubtree` 参数；它不改变事件 buffer、one-shot rearm 或 completion
机制。HYLE 当前 `proactor.cc` 已经拥有唯一 IOCP operation/slab/cancel 路线，差距只在把该参数硬编码为
`FALSE`。故冻结最窄 detail API：

```cpp
bool submit_directory_changes(std::intptr_t directory, void* buffer,
                              std::uint32_t length, bool watch_subtree,
                              std::uint64_t user_data) noexcept;
```

裸 `bool` 在这里是 Win32 已定义的二值 syscall 语义，不是用户档位或可扩展 policy；引入 detail enum
只会再造名字。`io_watch` 从 root 的 target 集合预计算一次值，rearm 不重复扫描 targets。

| ID | 已证实问题 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `IOCORE-R01` | subtree 被硬编码为 `FALSE` | detail declaration/definition 增加 `bool watch_subtree` 并原样透传 | false 不报告孙级；true 报告孙级；既有 completion/rearm/cancel |
| `IOCORE-R02` | 唯一 consumer 尚不能传语义 | `io_watch` Windows awaitable 从 root 的预计算 bit 传参；迁移 direct tests | exact/direct 不变，recursive journey 后续由 G05 验证 |

### P0 源码账单

- 两档均为一个 slab acquire、一个 in-flight link、一个 `ReadDirectoryChangesW`、一个 terminal IOCP
  completion；allocation/lock/wake/copy 数完全相同。
- 只增加一个调用参数和 root 上由 G05 所有的预计算 bit；不增加 operation/Proactor layout、runtime branch、
  syscall、线程或 map。Win32 内核是否遍历 subtree 是请求语义本身，不能与 direct-child 冒充同合同计时。
- Linux/macOS 声明和 object 不含该方法；所有其他 IO submit 路径字节级不变。没有源码未决机制，因此 P1
  `NOT_APPLICABLE`。

### 七问

1. 无产品第三方；Win32 与 libuv 仅作定义/成熟 peer 证据。
2. 本轮完整表达 native direct/recursive 两档，不拥有 event 产品语义。
3. 复用既有 Proactor slab、IOCP、cancel/completion；没有复制 runtime。
4. P0 如上，时间复杂度及用户请求的 syscall 数不变。
5. detail C++ seam，不建立 C/Native FFI。
6. 仍随既有 io_core source/static/shared object 交付，无新 target/dependency/export macro。
7. 普通用户不可见；参数名、注释和 true/false 直接测试防止误用。

## 验证与裁决

`IOCORE-R01--R02` 已按冻结形状实现：detail declaration/definition 只增加一个 `watch_subtree` 参数，
Windows caller 从 root 上的预计算 tree bit 传入；新增 IOCP 机制测试直接验证 `true` 可见孙级文件，原
`false` rearm/cancel/admission 测试保持。候选前 Windows MSVC 与 WSL GCC 14 的 io_core/io_watch + changed
header 各 `15/15 PASS`。

不可变产品候选为 `03fe0a1f8ef52bed09bc4e1b0f3d85be7200056d`。后续 Linux `io_watch` 修正和测试
fixture 提交均未改变 `io_core` 的公开头、实现、layout 或成本结构。

| 风险 / 候选绑定 receipt | 结果 |
|---|---|
| Windows MSVC 19.50 / Release / WERROR | io_core + io_watch 定向集合 `15/15 PASS`，包含 `FALSE/TRUE` 的孙级可见性、rearm、cancel 与 admission |
| Linux GCC 14.2 / Release / WERROR | exact product tree 的 io_core + io_watch 定向集合 `15/15 PASS` |
| macOS arm64 / AppleClang 21 / Release / WERROR | Git archive 的 io_core + io_watch 定向集合 `15/15 PASS` |
| sanitizer | WSL ASan 的完整定向集合 `15/15 PASS`；UBSan/TSan 的 watcher/header 各 `2/2 PASS`，没有 sanitizer report |
| source / shared / static delivery | Windows `io` selected-product smoke 全部通过，包含源码嵌入、relocated shared SDK 与 relocated static SDK consumer |
| P0 | 参数只原样进入既有 Win32 syscall；allocation、lock、wake、copy、operation layout 与 non-Windows object 均不变；P1 `NOT_APPLICABLE` |

候选后源码红队反查了 `submit_directory_changes` 的全部声明、定义和调用者：唯一产品调用者仍为
`io_watch`，两档都沿用同一 Proactor slab、in-flight、cancel 和 terminal completion 路线；没有第二
operation/runtime，也没有公开 API 或 ABI 扩张。ASan 暴露的 signal fixture shutdown 顺序错误已在纯测试
提交 `77513172cebf18bd09fb821e5902cc486084105e` 中纠正，产品候选没有随测试修复变化。

七门全部 `PASS`，`io_core` v3 裁决为 **`CLOSED`**。只有 Win32 目录提交合同、唯一 consumer、公开
IO API/layout、普通 submit 成本或 source/static/shared 交付发生变化，或者出现可复现 completion/cancel
缺陷时才可重开；递归事件产品语义仍由 `io_watch` 拥有。exact candidate 的 LOC：全库 `539,025` 行、
自实现不含测试 `94,884` 行、`io_core` 实现 `11,072` 行。

<!-- MODULE CLOSURE RECEIPT: CLOSED -->
