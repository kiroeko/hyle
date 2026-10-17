# 验证环境与测量限制

运行前读取实际硬件与工具链，优先复用可用缓存。下面是已有验证环境的事实索引，具体版本以本次
运行记录为准；不需要机器登记或资格审批。

| 环境 | 已知硬件 | 适合验证的内容 | 解释限制 |
|---|---|---|---|
| Windows | i9-13900K、RTX 4070 | MSVC/clang-cl、Windows ABI、IOCP、CUDA | P/E 核、SMT、功耗和线程落点会影响结果 |
| 同机 WSL2 | 同一 Windows 主机 | GCC/ELF、TSan、io_uring、CUDA | 是独立执行环境，不能称为第二台物理机或 bare-metal Linux |
| macOS | MacBook Air M4 `Mac16,12` | AppleClang、ARM/NEON、kqueue、Metal | 不支持硬 affinity，使用并记录 QoS |

当前 Apple 验证使用 `ssh kiro@172.22.10.198` 与 Windows OpenSSH 默认身份。Mac mini 当前不可用。
产品源码以 Windows 工作树为准，远程 checkout 与 archive 是验证副本。

## 记录与控制

- 保存 CPU/GPU、内存、OS/kernel、driver、编译器/SDK、ISA、优化与链接配置；可记录产物 hash，
  无需采集机器私密标识。
- 使用本地文件系统：Windows 本地 NTFS、WSL 原生 ext4、Mac 内部 APFS。WSL 的 `/mnt/c` 可搬运
  源码，不适合重构建、压力和性能测量；build 与依赖 cache 留在 ext4。
- 单线程 CPU 比较保持同类物理核、同优先级、同资源数量；并发比较记录各角色实际 CPU 与工作量。
  当前 Windows 常用 logical CPU 2，SMT sibling 3 保持空闲，但编号要从本次拓扑核对，不写死到产品。
- Windows 与同机 WSL 共享 CPU、GPU、内存、电源和磁盘，采样错开运行；远程同机任务也不要重叠。
- 记录交流电、电源模式、温度/频率与后台负载。Windows 的历史性能配置是“卓越性能”；
  Mac 使用交流电、`lowpowermode=0`。这些设置需要在运行时核实。
- macOS 的 `interactive`、`utility` 等 QoS 是不同条件，不能宣称它们等于固定 P/E 核。
- WSL guest affinity 只固定虚拟 CPU，不能证明 Hyper-V 将其固定到某个物理 P/E 核；
  WSL syscall、唤醒和设备数据不直接外推原生 Linux。

硬件、OS、驱动、工具链或配置改变时，明确新旧环境差异，必要时重测相应结果；不会因此自动要求
重审产品。跨平台数据分别报告，不混成一个“全平台”速度比。
