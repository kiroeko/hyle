# 性能机器合同 v1

本页冻结“哪台机器可以回答什么问题”。机器合同与 benchmark 统计分层：模块只能选择已登记的
machine/instrument profile，不得复制或修改 host probe、formal eligibility 或统计核心。可执行的唯一
机器登记表是 [`scripts/perf-machines.json`](../../../scripts/perf-machines.json)，公共探测器是
[`scripts/hyle_perf/machine.py`](../../../scripts/hyle_perf/machine.py)。

## 角色矩阵

| Profile | 执行环境 | Correctness | Engineering / Closure | Formal release |
|---|---|---|---|---|
| `W1-WIN` | i9-13900K / Windows / MSVC | Windows ABI、MSVC、clang-cl、IOCP、CUDA | 默认 x86 工程性能机 | 可签限定为 Windows/MSVC 的证书 |
| `W1-WSL` | 同一 W1 上的 WSL2 Ubuntu / GCC | ELF、GCC、TSan、io_uring、CUDA portability | Linux/compiler sensitivity、WSL 专项和大回归 | **禁止 native-Linux 与独立硬件证书** |
| `M1-MAC` | MacBook Air M4 / macOS / AppleClang | arm64、NEON、kqueue、Apple ABI、MoltenVK | Apple/ARM 平台复核（显式 unbound） | **当前不可签 Formal**；采集并登记稳定主机摘要后才能启用 |
| `L1-LINUX` | 独立 bare-metal Linux | provision 后启用 | provision 后启用 | 当前 `UNPROVISIONED`；native-Linux 正式声明需要它 |

Correctness 只证明行为、平台路径和交付闭包，不产生性能数字。普通模块的 Closure 默认只在
`W1-WIN` 使用 `closure-perf-v1` 作工程裁决；只有平台专属路径或真实消费者缺口才增加其他环境。
`formal-release-v1` 是独立发布任务，不阻塞模块产品收口。

## 主机身份与基线绑定

命名 profile 不是调用者可以自报的标签。performance probe 会生成
`hyle-perf-physical-host-identity-v1` 记录，并把以下内容绑定在一起：

- registry 中非敏感的 `physical_host_id`（例如 `W1`）及其域分离 SHA-256 key；
- 操作系统稳定机器标识的 SHA-256 与来源类型，不记录 MachineGuid、machine-id 或
  IOPlatformUUID 原文；
- 登记的 CPU/架构/逻辑处理器数、安装内存、主板、OS build 和 BIOS 等必须逐项相等的 host facts；
- baseline、filesystem、power、scheduler 与 performance toolchain 合同的规范 JSON 摘要。

`W1-WIN` 当前登记为 `REGISTERED_MATCH`，engineering 和 Formal 都要求稳定机器摘要及全部 host facts
匹配。相同 CPU 型号、相同 Windows 版本或相同 hostname 都不能冒充 W1；稳定摘要正确但 CPU/内存/
主板/BIOS/OS 漂移也会失败。编译器、SDK、链接、ISA/LTO 等实际 build facts 仍由统一 runner 的 build contract
单独核验；profile 摘要把该工具链合同纳入证据身份，不能用机器 probe 代替编译产物核验。

`M1-MAC` 尚未在 Windows 权威仓库中登记实机的 platform-UUID 摘要，因此其
`host_binding.status` 是 `unregistered`。它仍可产生带醒目 warning 的 engineering 数据，用于移植和
定位，但任何 `--role formal` 请求必须 fail closed，不能让首次运行结果反向“登记自己”。只有在受控
会话中核对真实 M1、将摘要和 exact host facts 明确写入 registry 并评审后，才可开放 macOS Formal。
`W1-WSL` 同样是显式 unbound engineering 环境；它共享 W1 物理资源但 Linux guest machine-id 不是
Windows 物理身份凭据。`L1-LINUX` 未 provision，所有性能角色均保持关闭。

证书组合器应直接使用 `physical_host_identity_record(profile, probe)` 的返回值；不得从 hostname、IP、
CPU 名称拼接另一套主机 ID，也不得把原始平台标识复制进 manifest。`registered_profile_contract_sha256`
一旦变化，旧证据与新 profile 不再是同一基线。

## WSL 裁决

`W1-WIN` 与 `W1-WSL` 的 `physical_host_id` 都是 `W1`。WSL 是独立执行环境，不是独立物理机器：

- guest 只能固定虚拟 CPU，不能证明 Hyper-V 将其固定在 13900K 的某个物理 P/E core；
- guest 没有可验证的物理 cpufreq/governor，电源和 thermal 状态由 Windows host 管理；
- Windows 与 WSL 共享 CPU、memory、NVMe 和电源状态，必须使用同一个物理主机互斥锁；
- WSL2 的 syscall/跨线程唤醒地板不能外推 bare-metal Linux。

因此 WSL 可以提供 correctness、engineering 和明确标为 `WSL2` 的诊断结果，但不能进入
native-Linux 或“第二台独立机器”的 formal gate。以后若把 WSL 本身定义为用户交付合同，可以建立
单独的 WSL-scoped 发布结果；它仍不替代 `L1-LINUX`。

## 固定控制

### `W1-WIN`

- 单线程默认 logical CPU 2 / CPU set 258，属于最高 efficiency class；SMT sibling CPU 3 保持空闲；
  priority 为 `above-normal`。
- 多线程合同可选择 `cross-core-pair`：caller 保持 logical CPU 2，runner 从实时 CPU Set 选择下一颗
  同为最高 efficiency class 且 physical-core 不同的逻辑 CPU（当前为 CPU 4 / set 260）；进程 mask、
  caller/worker thread pin 和 child telemetry 三层都必须一致。一物理 P-core 一线程；SMT、E-core、
  mixed-core 是独立合同，不混合汇总。
- 固定规模的真实并发 replay 可选择 `n-worker-physical-cores` 并声明 worker count。runner 从实时
  CPU Set 派生同 group、同最高 efficiency class、每 physical core 一个 logical sibling 的有序集合；
  当前 8-worker 选择为 `0,2,4,6,8,10,12,14`，但这些编号只是本机 probe 事实，不是模块硬编码。
  parent mask、singleton masks、CPU Set/core ids、priority 及逐 worker observed CPU 都进入证据。
- 外部 caller 不参与 worker 工作时，选择 `caller-plus-n-worker-physical-cores`。worker count
  只计实际 worker；另为 caller 保留一个完整物理核，总资源为 N+1 核。当前两 worker 派生为
  caller=2、workers=0,4、parent mask=0x15；caller 的 SMT sibling 也不能作为 worker。
- 使用交流电与登记的“卓越性能”scheme；source、peer、build、output 均位于本地 fixed NTFS。
- 原生 SDK 自管辅助线程时，可选 `native-runtime-pair`：复用实时双 P 核选择，process mask 只开放
  caller 与另一物理 P 核，caller 仍 singleton pin。SDK helpers 在该集合内由 runtime 调度；不声明
  worker 数、逐 helper pin 或 helper 总能独占另一核。记录两颗核、CPU Set id、实际 mask 与 caller
  telemetry；不能用它签有明确逐 worker placement 要求的并发实验。
- 工程性能工具链为 MSVC 19.50.35728 / v145 / Windows SDK 10.0.26100.0；clang-cl 只作
  correctness/codegen 复核。Release 为 `/O2 /Ob2 /DNDEBUG`，ISA/linkage/LTO 由 workload 明示。

### `W1-WSL`

- source、peer、build、output 全在 WSL 原生 ext4；`/mnt/c` 不产生性能证据。
- guest affinity 只改善重复性，不获得物理 P/E-core 声明。
- 与 Windows 采样必须错开；Windows power contract 同时适用。
- 当前工具链为 GCC 14.2 / binutils 2.42 / glibc 2.39。

### `M1-MAC`

- 使用本地 internal APFS、交流电、`lowpowermode=0`。
- macOS 没有受支持的 hard affinity；默认合同是 `interactive` QoS，probe 记录当前 process QoS 与
  expected QoS 是否一致。`utility` 只在真实 workload 属于 utility 时作为另一合同，不能冒充
  E-core pin。
- 当前工具链为 AppleClang 21.0.0 / SDK 26.5 / ld 1267，Release 为 `-O3 -DNDEBUG`。

### `L1-LINUX`

准入条件是 bare metal、本地 ext4、可验证物理 topology/affinity、performance governor、固定
BIOS/microcode/memory 与 exact GCC 14 compiler/linker/stdlib。机器 provision 前不得发布宽泛的
native-Linux 性能结论。W1 双启动可以签 Linux OS/toolchain 合同，但仍不是独立硬件复证。

## Probe 与隐私

从仓库根运行：

```powershell
python scripts/hyle_perf/machine.py --machine W1-WIN --role engineering `
  --path source=. --path build=build-bench
```

Formal 必须明确 claim，例如 `--role formal --formal-claim windows`。M1 完成身份登记后，macOS 还传
`--apple-qos interactive`；当前 M1 仍会因未登记身份而拒绝 Formal。该参数在 probe 作用域内应用、
验证并恢复当前线程 QoS；benchmark child 仍会独立应用并上报同一 QoS，不能只信父进程声明。输出包含
filesystem、power、topology/QoS 与 eligibility verdict。不传参数而让 process 保持 `default` 的
engineering/formal probe 会 fail closed。
Windows 的 engineering 与 formal verdict 都会证明登记的 CPU set 258 / logical CPU 2 仍存在且属于
最大 efficiency class；probe 不要求父 Python 已绑定。`expected_child_policy` 明确返回 affinity
`0x4`、`above-normal` 和空闲 sibling 3，由 benchmark child launcher 实际应用并写入运行记录。
`cross-core-pair` 合同则基于同一 host probe 派生并记录完整双核 mask 与两个 CPU Set id，且要求
benchmark 回报 caller/worker 的实际角色 pin；`n-worker-physical-cores` 同理派生并记录 N 个不同
physical core 的 worker 集与逐 worker singleton pin。独立 caller 拓扑额外记录 N+1 个保留物理核，
并校验 caller CPU telemetry。这些 topology 都不修改默认单线程机器合同。

## 采样互斥锁

机器 probe 只验证环境；任何 engineering/formal 采样还必须在启动 benchmark child 前取得持久锁：

```python
from hyle_perf import (
    get_machine_profile,
    load_registry,
    sampling_mutex,
)

registry = load_registry()
profile = get_machine_profile(registry, "W1-WIN")
with sampling_mutex(profile) as lease:
    manifest["sampling_mutex"] = lease.manifest_record()
    run_benchmark_children()
```

`W1-WIN` 的 `C:\hyle-perf-locks\W1.sampling.lock` 与 `W1-WSL` 的
`/mnt/c/hyle-perf-locks/W1.sampling.lock` 是同一个 NTFS marker。创建使用操作系统原子的
exclusive-create；因此 Windows 与 WSL 任一方持锁时另一方机械失败。`M1-MAC` 使用独立的
`/Users/Shared/HYLE/perf-locks/M1.sampling.lock`，防止远程任务重叠。

正常 context 退出只在 schema、host/key/profile/environment 与随机 owner token 全部仍匹配时删除
自己的 marker；manifest 只保存 owner token 的 SHA-256。进程崩溃、断电、marker 损坏或被替换后均
**fail closed**：后续采样报告 `SamplingMutexBusy`/`SamplingMutexOwnershipError`，不得按文件年龄、
PID 或“看起来 stale”自动删除，也不得清理未知锁。

人工恢复步骤固定：

1. 停止该 physical host 的所有采样入口；W1 必须同时检查 Windows 和 WSL，M1 检查所有 SSH/session。
2. 只读检查 exact marker 的 `mutex_key`、`profile_id`、`environment`、`pid` 与
   `acquired_at_utc`。PID 只用于定位，不能单独证明 stale，因为 PID 可能复用。
3. 确认登记环境中的 owner 已结束，并再次确认另一执行环境没有 sampler。
4. 人工删除 registry 指定的**单个 exact marker**：W1 可从 Windows 执行
   `Remove-Item -LiteralPath C:\hyle-perf-locks\W1.sampling.lock`，或从 WSL 删除它的 `/mnt/c` 映射；
   M1 删除登记的 APFS marker。不要递归删除锁目录。
5. 重跑 machine probe，再开始一个全新的实验；崩溃前的部分样本不继承。

Windows `MachineGuid`、Linux machine-id 和 macOS platform UUID 只在进程内读取并立即 SHA-256；
manifest 只记录 digest、来源类型和 `raw_identifier_recorded=false`，绝不保存原值。hostname 同样只
记录 digest。IP 地址不是机器身份。

环境、CPU/BIOS/microcode、OS/kernel、memory speed、power scheme、filesystem、compiler/linker/
stdlib/SDK 或最终 flags/linkage/ISA 改变时，旧性能 baseline 失效并建立新实验；这不自动重开产品
设计。Correctness 支持范围与 exact 性能工具链是两份合同，不能把最低版本范围当性能身份。
