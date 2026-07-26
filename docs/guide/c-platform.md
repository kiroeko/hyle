# HYLE C ABI：platform

> **发布面**：`include/hyle/c/platform.h`  
> **实现**：`src/platform/c_platform.cc`（链 `hyle_platform` + `hyle_error`）
> **稳定性**：未发布，无 ABI 版本门（纯 `extern "C"` 自由函数导出）  
> **命名**：全部 `hyle_*` / `HYLE_*`

跨语言 / 跨编译器只许绑本头（及 `hyle/c/error.h`）。不要 include `hyle/platform/*.h`。

所有可恢复失败都直接返回拥有型 `hyle_status`：

| 返回面 | 诊断 |
|---|---|
| Parker / Thread / DynLib / MappedFile / VM / NUMA / fs / env | 返回 `hyle_status`，用 `hyle_status_get_code` / `hyle_status_message_cstr` |

`hyle_status` 由调用方用 `hyle_status_free` 释放。platform 不再维护模块私有 TLS
last-error；C 与 C++ 共用 error 的 code、message 和所有权规则。

---

## 1. 链什么

```text
#include "hyle/c/platform.h"   /* pulls hyle/c/error.h for hyle_status */
// link: hyle_platform (+ hyle_error)
```

纯 C 验收：`hyle_platform_c_test`。

---

## 2. 子系统与所有权

| 块 | 入口 | 所有权 |
|---|---|---|
| wait | `hyle_wait_on_address*` / `hyle_wake_*` | 调用方提供 `uint32_t` 字 |
| Parker | create / destroy / park / unpark / park_timeout | create→destroy |
| Thread | spawn → join | join **释放**句柄 |
| DynLib | open / get / current / close / symbol / open_symbol | close 释放 |
| VM 自由函数 | reserve/commit/decommit/release/protect | reserve 配对 release |
| VM region | `hyle_vm_region_*` | destroy 释放 |
| hugepage/THP | `hyle_vm_advise_hugepage` 等 | soft 建议，无句柄 |
| NUMA | `hyle_numa_*` / `hyle_vm_reserve_numa` / `hyle_vm_bind_numa*` | reserve 配对 release |
| fs/env | buffer-fill / visitor | 无堆句柄 |
| fs 权限/链接/临时文件/env 写 | `hyle_fs_get_permissions` / `hyle_fs_*_symlink` / `hyle_fs_create_temp_file` / `hyle_env_set` | temp file 返回 `hyle_file_handle`（`hyle_fs_close` 释放） |
| MappedFile | `hyle_mapped_file_*` | destroy 释放 |
| cpu | `hyle_cpu_features_get` 等 | 值返回 POD |

---

## 3. 导出符号 1:1

与 `include/hyle/c/platform.h` **逐符号对齐**。改头必须同步改本节。

### 3.1 wait / Parker（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_wait_on_address(addr, expected)` | 地址等待；允许伪唤醒 |
| `hyle_wait_on_address_for(addr, expected, timeout_ns)` | 0=超时，1=唤醒/变化/伪唤醒 |
| `hyle_wake_one` / `hyle_wake_all` | 唤醒 |
| `hyle_parker_create(out)` / `hyle_parker_destroy` | 不透明 Parker；create 返回 Status 并先清空 `*out` |
| `hyle_parker_park` / `unpark` | token 模型（Rust std） |
| `hyle_parker_park_timeout` | 1=消费 token，0=超时/无 token |

### 3.2 Thread（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_thread_spawn(out, entry, ctx, opts)` | 成功设 `*out`；失败返 `hyle_status`（调用方 free） |
| `hyle_thread_spawn_opts(out, entry, ctx, name, stack, affinity, priority)` | FFI 便利入口；不用构造 opts/has 字段，0/-1 是默认 sentinel |
| `hyle_thread_join` | join + **destroy** 句柄 |
| `hyle_thread_joinable` | 1/0 |
| `hyle_hardware_concurrency` | 系统在线逻辑 CPU 数 ≥1；worker 数优先用下文 `hyle_cpu_available_count` |
| `hyle_this_thread_id` | 稳定线程 id |
| `hyle_this_thread_yield` | |
| `hyle_this_thread_set_name` | 返回 Status；name 可 NULL→空串 |
| `hyle_this_thread_set_affinity` / `_affinity_cpus` / `set_priority` | 调用线程逐项配置；单 CPU 或 CPU 数组；失败可观察 |

**头内 inline / 宏**（无 DLL）：

| 符号 | 说明 |
|---|---|
| `hyle_thread_opts` | 首字段 `struct_size`，其后为 name/stack/单 CPU 或 CPU 数组 affinity/priority 与 `config_policy` |
| `hyle_thread_priority` | `LOWEST`/`LOW`/`NORMAL`/`HIGH`/`HIGHEST`（-2..2）；Linux 提升需权限，macOS 映射 QoS |
| `hyle_thread_config_policy` | 默认 `BEST_EFFORT`；`REQUIRED` 在 entry 前同步配置，失败不运行 entry |
| `HYLE_THREAD_OPTS_INIT` | 默认初始化宏；会写 `struct_size`，不是全零字节 |
| `hyle_thread_opts_default` | 返回默认 opts |
| `hyle_thread_opts_set_name` / `set_stack_size` / `set_affinity` / `set_affinity_cpus` / `set_priority` / `set_config_policy` | setter；数组是 view，只需活到 spawn 返回 |

### 3.3 DynLib（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_dynlib_open(out, path)` | 加载；成功写拥有句柄，失败返 `hyle_status` |
| `hyle_dynlib_get(out, path)` | **仅**附着已加载，不加载缺失模块；返回 Status |
| `hyle_dynlib_current(out)` | 当前映像；返回 Status |
| `hyle_dynlib_close` | 释放句柄 |
| `hyle_dynlib_valid` | 1/0 |
| `hyle_dynlib_symbol(out, lib, name)` | 成功写符号地址；Status 的成功态可表达合法的 NULL 值符号 |
| `hyle_dynlib_extension` | `"dll"` / `"so"` / `"dylib"` |
| `hyle_dynlib_open_symbol(out_lib, out_sym, path, name)` | open+resolve；返回 Status，成功后 `*out_lib` 由调用方 close |

### 3.4 Virtual memory（DLL + inline）

| 符号 | 说明 |
|---|---|
| `hyle_protect` | `NONE` / `READ` / `READ_WRITE` |
| `hyle_page_size` | 页大小 |
| `hyle_page_align_up` | **inline**：上取整；0→0 |
| `hyle_vm_reserve(out, size)` / `commit` / `decommit` / `release` / `protect` | 全部返回 `hyle_status`；reserve 成功写 `*out` |
| `hyle_vm_alloc_rw(out, size)` | 一次 OS 操作得到可立即使用的 RW mapping（size 上取整）；返回 Status |
| `hyle_vm_free_rw(p, size)` | 释放 `alloc_rw` 的块（size 为原始请求）；返回 Status |
| `hyle_vm_alloc_numa_interleaved(out, placement, size, nodes, count, chunk)` | 已提交的 selected-node/chunk 连续 RW span；整段 `hyle_vm_release`，不能 decommit/recommit |
| `hyle_vm_region_reserve(out, size)` / `reserve_guarded` | Status+out；guarded 低 1 页 NOACCESS |
| `hyle_vm_region_reserve_numa(out, size, node)` | 偏好 NUMA `node`（`HYLE_NUMA_ANY`==普通 reserve）的托管 region |
| `hyle_vm_region_alloc_numa_interleaved(out, size, nodes, count, chunk)` | 上述 selected allocation 的 opaque owner；destroy 释放，decommit 返回 `FAILED_PRECONDITION` |
| `hyle_vm_region_adopt(out, base, size)` | 收编裸预留；只有成功后才转移所有权，wrapper OOM/参数错误时调用方仍拥有 base |
| `hyle_vm_region_destroy` | |
| `hyle_vm_region_valid` / `data` / `size` / `usable` / `usable_size` / `has_guard` / `is_numa_interleaved` / `numa_placement` | 查询 |
| `hyle_vm_region_commit` / `commit_usable` / `decommit` / `protect` | 子区间，返回 Status |

### 3.4b Hugepage / THP（DLL + inline，v3）

soft 建议，绝不影响正确性。Linux 用 madvise；Windows/macOS 匿名区无等价时返回
`HYLE_STATUS_UNIMPLEMENTED`，映射保持可用。

| 符号 | 说明 |
|---|---|
| `hyle_large_page_size` | 大页字节（未知/不支持→0） |
| `hyle_large_page_align_up(n)` | **inline**：上取整到大页（0 回退普通页）；n==0→0 |
| `hyle_vm_advise_hugepage(ptr, size)` | 空范围→OK；ptr 须页对齐；返回 Status |
| `hyle_vm_advise_no_hugepage(ptr, size)` | 同上，反向建议 |

### 3.4c NUMA 拓扑 + 放置（DLL + 常量，v3）

| 符号 | 说明 |
|---|---|
| `hyle_numa_node`：`HYLE_NUMA_ANY` | 具名 enum；`(-1)` 无 NUMA 约束 |
| `hyle_numa_policy` | `PREFERRED`(软) / `BIND`(硬,Linux) / `INTERLEAVE`(全节点条带) |
| `hyle_numa_placement` | `APPLIED`（OS 已接受 policy，不是物理驻留证明）/ `NOT_NEEDED` / `UNAVAILABLE`（软 preferred 降级） |
| `hyle_numa_capabilities` | runtime feature bitset；用户不要围绕 `HYLE_OS_*` 编译分叉 |
| `hyle_numa_nodes(NULL,0)` / `(dst,cap)` | 查询总数/复制实际 node id；id 可稀疏，稳定到进程结束 |
| `hyle_numa_node_count` | `hyle_numa_nodes` 的数量，≥1；不能据此推导 id |
| `hyle_numa_current_node` | 调用线程 home 节点；失败时回退 nodes 列表首项 |
| `hyle_numa_node_of_cpu(cpu)` | 逻辑 CPU 所属节点；越界/失败时回退 nodes 列表首项；配单 CPU affinity 放置 worker arena |
| `hyle_vm_bind_numa(out_placement, ptr, size, node, policy)` | 返回 Status；placement 出参可 NULL；Linux BIND 是严格集合，INTERLEAVE 是 placement policy |
| `hyle_vm_bind_numa_interleave(out_placement, ptr, size)` | 全节点条带（INTERLEAVE 别名） |
| `hyle_vm_reserve_numa(out, out_placement, size, node)` | Status+out；preferred 失败仍可成功分配并报告 `UNAVAILABLE` |
| `hyle_vm_alloc_numa_interleaved(...)` | `nodes==NULL,count==0` 选择全部；显式 id 必须可用、唯一且按条带顺序；`chunk==0` 表示一页 |

capability 是启动期可发现性，操作的 `hyle_status` 才是权限、配额和实时 OS 状态的最终裁决。
单节点机器上 selected allocation 正常成功并报告 `NOT_NEEDED`；Linux/Windows 多节点会安装
条带 policy，但目标节点压力下可回退，因此 `APPLIED` 不是逐页 residency 证明；macOS 当前公开
逻辑 node 0。裸 selected allocation 只允许整段 release；需要自动所有权和防误用
decommit 时用 `hyle_vm_region_alloc_numa_interleaved`。

### 3.5 fs / env（DLL）

| 符号 | 说明 |
|---|---|
| `hyle_file_type` / `hyle_file_info` | `type` / `size` / `modified_time` / `is_symlink`；stat 成功本身就表示存在，不保留可矛盾的 `exists` 字段 |
| `hyle_fs_stat(out, path)` | Status+out；缺失是 `NOT_FOUND`，不返回伪造的全零 stat |
| `hyle_fs_exists(out, path)` / `hyle_fs_is_directory(out, path)` | Status+out；缺失是成功的 false，权限/I/O 失败仍是 Status |
| `hyle_fs_file_size(out, path)` | Status+out；空文件是成功的 0，缺失与目录分别是 `NOT_FOUND` / `FAILED_PRECONDITION` |
| `hyle_fs_make_directory*` | Status；已存在目录为 OK，已存在非目录为 `ALREADY_EXISTS` |
| `hyle_fs_remove_directory*` / `hyle_fs_remove_file` | Status；递归删除保留首个失败，且不跟随 symlink/junction |
| `hyle_fs_rename` / `hyle_fs_copy_file` / `hyle_fs_touch` | Status；保留原生 operation/code/message |
| `hyle_fs_current_directory(out_size,dst,cap)` / `temp` / `home` | Status+out buffer-fill；`*out_size>=cap` 表示截断，`dst==NULL` 可只查长度 |
| `hyle_fs_change_directory(path)` | Status；缺失、权限、非法 UTF-8 分开报告 |
| `hyle_env_get(out_size,name,dst,cap)` | Status+out；未设置为 `NOT_FOUND`，已设置为空是成功且 size=0 |
| `hyle_fs_visit_fn` | `(name, hyle_file_type, ctx)→int`；0 停止；类型描述 entry 自身，不跟随 symlink |
| `hyle_fs_directory_iterate(out_completed,...)` | Status+out；无分配快速遍历，常见文件系统不逐项 stat；visitor 提前停止是成功且 completed=0 |

### 3.5b fs 权限 / 符号链接 / 临时文件 / env 写（DLL，v4）

| 符号 | 说明 |
|---|---|
| `hyle_fs_get_permissions(out,path)` | Status+out；低 9 位权限，合法 0000 不再与失败混淆；Windows 由 READONLY 合成 |
| `hyle_fs_set_permissions(path, mode)` | Status；POSIX chmod；Windows 只认写位→READONLY |
| `hyle_fs_is_readonly(out,path)` / `hyle_fs_set_readonly(path, ro)` | Status(+out)；跨平台可靠子集 |
| `hyle_fs_create_symlink(target, link, kind)` | Status；`AUTO` 适合已存在 target，创建可移植的 dangling link 必须显式 `FILE/DIRECTORY`；Windows 仍可能 `PERMISSION_DENIED` |
| `hyle_fs_create_hard_link(target, link)` | Status；硬链接（同卷、仅文件） |
| `hyle_fs_is_symlink(out,path)` | Status+out；自身是否符号链接（不跟随），缺失为 false |
| `hyle_fs_read_symlink(out_size,path,dst,cap)` | Status+out buffer-fill；非链接为 `FAILED_PRECONDITION` |
| `hyle_file_handle` / `HYLE_FILE_HANDLE_INVALID` | 一字宽 native file resource；可容纳 POSIX fd / Windows HANDLE，不暗示 POSIX 语义 |
| `hyle_fs_create_temp_file(out, out_path_size, dir, prefix, dst, cap)` | Status+out；原子创建并返回**打开的** handle；完整路径长度即使 dst 截断也可见，避免 TOCTOU |
| `hyle_fs_open(out, path, flags)` | Status+out；`flags`=`HYLE_OPEN_READ/WRITE/CREATE/TRUNCATE/APPEND` OR；Windows 用 UTF-16 API；无需 mmap 的普通 I/O |
| `hyle_fs_read(out_moved, file, buf, n)` / `hyle_fs_write(out_moved, file, buf, n)` | Status+out；读 0=EOF，短读/短写是成功，调用方可继续 |
| `hyle_fs_close(file)` | 返回 Status；关闭 create_temp_file / open 返回的 handle；INVALID 安全 |
| `hyle_env_set(name, value)` / `hyle_env_unset(name)` | Status；进程级，与并发读非线程安全；Windows 写入也按 UTF-8 转 UTF-16 |

### 3.5c MappedFile —— 文件内存映射（DLL，v4）

| 符号 | 说明 |
|---|---|
| `hyle_map_mode` | `READ_ONLY` / `READ_WRITE` / `COPY_ON_WRITE`（对标 memmap2） |
| `hyle_map_advice` | `NORMAL` / `RANDOM` / `SEQUENTIAL` / `WILL_NEED` / `DONT_NEED` |
| `hyle_mapped_file_open(out, path, mode, offset, length)` | length=0→到 EOF；支持非页对齐 offset（内部扩大/对齐 OS view）；返回 Status |
| `hyle_mapped_file_destroy(m)` | unmap + 关句柄 |
| `hyle_mapped_file_valid` / `hyle_mapped_file_data` / `hyle_mapped_file_size` | 查询 |
| `hyle_mapped_file_flush(m)` | 回写脏页（RW；RO/COW 为 OK no-op）；返回 Status |
| `hyle_mapped_file_advise(m, advice, offset, length)` | 映射范围 hint；length=0→到结尾；不支持返回 UNIMPLEMENTED |
| `hyle_mapped_file_lock` / `unlock` / `locked` | 显式 page residency；幂等；失败可见，不影响 owner 有效性 |

Windows 只有 `WILL_NEED` 有直接 hint 对等物；POSIX 支持五档 `madvise`。hint 返回
`HYLE_STATUS_UNIMPLEMENTED` 时 mapping 仍可正常使用。`lock` 受系统权限和 memlock quota
约束，调用方必须检查 Status；它不是 prefetch 的别名。打开的 mapping 在 Windows 也允许
路径 rename/delete，已有 opaque owner 继续持有原文件对象。

### 3.6 CPU（DLL + 常量）

| 符号 | 说明 |
|---|---|
| `hyle_cpu_features` | 与 C++ `CpuFeatures` **全 bool 字段 1:1**（C 用 int 0/1）；不混入拓扑或 ISA 派生值 |
| x86 字段 | `sse2`…`avx10_2`（见头） |
| arm 字段 | `neon`…`arm_bf16` |
| RISC-V 字段 | `rvv`…`rvv_zvfhmin` |
| `hyle_isa`：`HYLE_ISA_*` | 具名 enum；`SCALAR`/`SSE2`/…/`RVV` 粗粒度档（与 `Isa` 数值一致） |
| `hyle_cpu_features_get` | 值返回 |
| `hyle_cpu_best_isa` | 当前进程可安全执行的最高粗粒度 ISA 档 |
| `hyle_cpu_supports_isa(isa)` | 本进程能否跑该档（同架构比较） |
| `hyle_cpu_isa_name(isa)` | 稳定英文名（日志） |
| `hyle_cpu_cache_line_size` | L1 数据缓存行字节数（主流 64；对齐/填充用；探测失败回退 64） |
| `hyle_cpu_cache_info` | 值返回 `hyle_cache_info`：L1D/L1I/L2/L3 容量（字节；未知级为 0） |
| `hyle_cpu_topology_get` | 缓存的系统视图：`system_logical_cpus`、performance/efficiency CPU 数与 `is_hybrid` |
| `hyle_cpu_available_count(out_count)` | 动态进程视图的 count-only 便利入口；创建 worker / 选择 shard 数优先用它 |
| `hyle_cpu_available_ids(out_count, dst, cap)` | 动态进程视图；返回 Status，`*out_count` 是完整数量，最多写 cap 个可用于 thread affinity 的 id；`dst==NULL` 可只查数 |
| `hyle_cpu_available_info(out_count, dst, cap)` | 每个可用 CPU 的 id、单机单调 `performance_rank` 与 `hyle_cpu_kind`；rank 0 = 未知 |
| `hyle_cpu_available_ids_ordered(out_count, order, dst, cap)` | 保留全部可用 CPU，按 OS 默认/强核优先/能效核优先排列；未知时不猜 |

拓扑只采用 OS/CPU 的明确分类（Windows `EfficiencyClass`、macOS
`hw.perflevelN`、Linux `cpu_capacity`）。若平台没有可靠信息，会保守返回
`performance_cpus == system_logical_cpus`、`efficiency_cpus == 0`、`is_hybrid == 0`；
这表示“未确认异构”，不会根据最高频率猜核型。

创建 worker 或绑核应使用 `hyle_cpu_available_ids`，不能把系统总数当进程配额，也不能假定
可用 id 是 `[0, count)`：Linux cpuset/affinity 可产生稀疏 id。该查询不缓存，以观察运行期
限制变化；Linux 返回调用线程的 mask（其新建线程会继承）。它是初始化/重配控制面，不应在
每 task 热路径重复调用。

“优先”与“只允许”要分开：`HYLE_CPU_ORDER_PERFORMANCE_FIRST` 是软排序，不会丢掉能效核；
硬限制时过滤 `HYLE_CPU_KIND_PERFORMANCE` 后传给 `hyle_thread_opts_set_affinity_cpus`。
若 kind 全是 `UNKNOWN`，不能声称获得了强核集合。Windows rank 来自 `EfficiencyClass`，Linux
只用内核 `cpu_capacity`；macOS 不提供可靠的 logical-id 硬绑映射，按线程 QoS 表达软偏好。

### 3.7 导出形态

纯 `extern "C"` 自由函数 + opaque 句柄（Parker/Thread/DynLib/VmRegion/MappedFile），FFI 直接绑符号。**无 `hyle_platform_api` 函数指针表、无 `api_get` 版本协商**（版本协商是分发独立编译插件才需要的件；HYLE 单一 `.so`、未发布，不背此负担）。设计推理见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md) §3。

**header-inline（无 DLL 符号）**：`hyle_page_align_up`、`hyle_large_page_align_up`、thread opts helpers / `HYLE_THREAD_OPTS_INIT`。

### 3.8 有意不导出

| C++ | 原因 |
|---|---|
| `Unparker` 类型 | 用 `hyle_parker_unpark` 同一句柄 |
| `DynLib::symbol_as<Fn>`（模板） | 用 `hyle_dynlib_symbol` + 调用方 cast |
| `FileHandle`（RAII） | C 走 `hyle_file_handle` + `hyle_fs_close`；不把 Windows HANDLE 伪称 POSIX fd |
| `VmRegion::release` / `MappedFile::reset`（提前释放） | 用 `hyle_vm_region_destroy` / `hyle_mapped_file_destroy` |
| path 词法 / container 字符串 | 不在 platform C 面 |
| 模板 `for_each_entry` | C 用 `hyle_fs_directory_iterate` |
| 测试钩子 | 非发布面 |

---

## 4. 常用写法

### Parker

```c
hyle_parker_o* p = NULL;
hyle_status st = hyle_parker_create(&p);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return; }
hyle_status_free(&st);
hyle_parker_unpark(p);
hyle_parker_park(p);
hyle_parker_destroy(p);
```

### 普通文件 I/O

```c
hyle_file_handle file = HYLE_FILE_HANDLE_INVALID;
hyle_status st = hyle_fs_open(
    &file, "config.bin", HYLE_OPEN_READ);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return; }
hyle_status_free(&st);

unsigned char buf[4096];
size_t got = 0;
st = hyle_fs_read(&got, file, buf, sizeof buf);
if (hyle_status_is_ok(st)) { /* consume buf[0..got), got==0 means EOF */ }
hyle_status_free(&st);
st = hyle_fs_close(file); /* report close/flush failures */
hyle_status_free(&st);
```

### Thread

```c
void worker(void* ctx) { *(int*)ctx = 1; }
int flag = 0;
hyle_thread_o* t = NULL;
hyle_thread_opts opts = HYLE_THREAD_OPTS_INIT;
hyle_thread_opts_set_name(&opts, "worker");
hyle_thread_opts_set_stack_size(&opts, 2u * 1024u * 1024u);
hyle_thread_opts_set_config_policy(&opts, HYLE_THREAD_CONFIG_REQUIRED);
hyle_status st = hyle_thread_spawn(&t, worker, &flag, &opts);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return; }
hyle_status_free(&st);
hyle_thread_join(t);
```

FFI 不方便构造结构体时可直接：

```c
hyle_status st = hyle_thread_spawn_opts(
    &t, worker, &flag, "worker", 0, -1, HYLE_THREAD_PRIORITY_NORMAL);
```

`spawn_opts` 是默认 best-effort 的简写；需要 required 或需要“不设置 priority”时使用完整
`hyle_thread_opts`。required affinity 在 macOS 返回 `HYLE_STATUS_UNIMPLEMENTED`。已经在
entry 内时可调用 `hyle_this_thread_set_*` 并按普通 `hyle_status` 规则检查、释放。

若线程应在一个应用分区内迁移，传数组而不是为每个 CPU 重复造一套 option：

```c
unsigned cpus[2] = {cpu_a, cpu_b}; /* id 来自 hyle_cpu_available_ids */
hyle_thread_opts_set_affinity_cpus(&opts, cpus, 2);
/* cpus 只需保持有效到 hyle_thread_spawn 返回。 */
```

单 CPU 适合 per-core shard/NUMA owner；数组适合性能核、NUMA-local 或应用保留核集合。
Windows 一个硬 affinity 集合不能跨 processor group，跨组会返回
`HYLE_STATUS_INVALID_ARGUMENT`；应由上层线程池拆 worker。macOS 没有公开硬 affinity，
required 返回 `UNIMPLEMENTED`，best-effort 仍启动线程。轮转、one-of 和超订属于 sched 策略。

### DynLib / open_symbol

```c
hyle_dynlib_o* lib = NULL;
hyle_status st = hyle_dynlib_get(&lib, "kernel32.dll"); /* or libm.so.6 */
if (hyle_status_is_ok(st)) {
  void* sym = NULL;
  hyle_status sym_st = hyle_dynlib_symbol(&sym, lib, "GetCurrentProcessId");
  if (hyle_status_is_ok(sym_st)) { /* use sym */ }
  hyle_status_free(&sym_st);
  hyle_dynlib_close(lib);
}
hyle_status_free(&st);

hyle_dynlib_o* plug = NULL;
void* entry = NULL;
hyle_status plug_st = hyle_dynlib_open_symbol(&plug, &entry, "plugin.so", "entry");
if (hyle_status_is_ok(plug_st)) {
  /* use entry */
  hyle_dynlib_close(plug);
}
hyle_status_free(&plug_st);
```

### VM

```c
void* p = NULL;
hyle_status st = hyle_vm_alloc_rw(&p, 4096);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return; }
hyle_status_free(&st);
/* ... */
hyle_status free_st = hyle_vm_free_rw(p, 4096);
hyle_status_free(&free_st);

hyle_vm_region_o* r = NULL;
st = hyle_vm_region_reserve_guarded(&r, 64 * 1024);
if (!hyle_status_is_ok(st)) { hyle_status_free(&st); return; }
hyle_status_free(&st);
char* lo = (char*)hyle_vm_region_usable(r);
size_t n = hyle_vm_region_usable_size(r);
hyle_vm_region_destroy(r);
```

### fs

固定上限明确时直接给栈/arena buffer；结果长度达到 `cap` 就表示截断：

```c
char tmp[512];
size_t full = 0;
hyle_status st = hyle_fs_temp_directory(&full, tmp, sizeof tmp);
if (!hyle_status_is_ok(st)) {
  fprintf(stderr, "platform: %s\n", hyle_status_message_cstr(st));
}
hyle_status_free(&st);
if (full >= sizeof tmp) { /* truncated: retry with a larger buffer */ }
```

需要任意长度 owning 字符串时做 count→allocate→fill。C ABI 刻意不偷偷选择
`malloc`、语言 runtime 或 HYLE allocator；下面的 storage policy 完全属于调用方：

```c
#include <stdint.h>
#include <stdlib.h>

char* tmp = NULL;
size_t cap = 0;
int complete = 0;
for (unsigned attempt = 0; attempt != 4; ++attempt) {
  size_t full = 0;
  hyle_status st = hyle_fs_temp_directory(&full, tmp, cap);
  if (!hyle_status_is_ok(st)) {
    hyle_status_free(&st);
    free(tmp);
    return;
  }
  hyle_status_free(&st);
  if (full < cap) { complete = 1; break; } /* includes empty string */
  if (full == SIZE_MAX) { free(tmp); return; }
  cap = full + 1;                     /* room for NUL */
  char* grown = (char*)realloc(tmp, cap);
  if (grown == NULL) { free(tmp); return; }
  tmp = grown;
}
if (!complete) { free(tmp); return; } /* process state kept changing */
/* use tmp; free(tmp) when done */
```

cwd/home/env/symlink 和 CPU ids/info 使用同一模式；数组的 `cap` 是元素数，不额外加 NUL。
状态可能在 count 与 fill 之间变化，所以完整 owning helper 必须在返回数量超过容量时重试，
不能把 partial buffer 当完整结果。C++ 用户可直接包含 `hyle/platform/convenience.h` 使用已封装
的 owning 重载；它是 header-only source API，不是缺失的 C ABI 能力。

---

## 5. 稳定性

HYLE 未发布，C ABI 面无版本门/协商——发布前按需自由演进，追求干净正确。发布后如何冻结 ABI 见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md) §3。

导出面覆盖：wait/Parker、Thread（priority 5 档）、DynLib、VM 自由函数 + VmRegion（含 `reserve_numa`/selected interleave/`adopt`）、hugepage·THP、NUMA（实际 node 列表 + runtime capability + preferred/bind/all/selected interleave）、fs/env（权限/符号链接/硬链接/原子临时文件/env 写）、MappedFile（RO/RW/COW + advice/residency）、cpu 全字段。

> mem 的 `StatsAllocator` / `LeakTracker` 不属于 platform 导出面；其 C ABI
> 位于 [`c-mem.md`](./c-mem.md) §2.2。
> NUMA/hugepage 是 platform 用户能力，故在本模块直接导出。

---

## 6. 验收

```bash
cmake --build <build> --target hyle_platform_c_test
./hyle_platform_c_test   # all checks passed
```
