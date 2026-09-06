# Platform v10：NUMA 节点内存事实

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由冻结基础模块的真实推理消费者回放触发。当前仅为 `REOPENED`：本页保存可复现产品事实与
> 最小影响面；closure card、能力账本和有限 `FIX` 集必须分别在后续 `SCOPED/AUDITED` 冻结，之前
> 不修改 Platform 产品源码。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`platform` / host NUMA topology facts /
  `platform-v10-numa-node-memory-facts` / `REOPEN`；baseline
  `5c529802cd2974b222856608dcaf4cdd0fcf240b`，tree
  `fe5e3841d5de46c39ce01b7dae6c0d14e2d53f38`；
- trigger ID：`PLATFORM-NUMA-NODE-MEMORY-FACT-2026-09-06`。锁定 vLLM
  `569adb5a9780f9c02d22a6b29826acf711512356` 的
  `vllm/utils/cpu_resource_utils.py:get_memory_node_info` 读取每个可见 NUMA node 的物理总量和当前可用量，
  `vllm/platforms/cpu.py:get_device_total_memory` 与 CPU worker 内存状态直接用于启动预算；锁定 SGLang
  `5bebe7a033ab57751fac3c9d8db219da62ca8c72` 的
  `python/sglang/srt/utils/numa_utils.py:init_threads_binding` 按 rank/node 绑定 CPU 与内存，并在 rank 多于
  node 时明确无法预先确定每 rank 可用内存。两条独立推理旅程共同证明节点容量/压力是通用 host
  planning fact，不是某个模型或 benchmark 的形状；
- HYLE 当前 `numa_nodes/numa_node_of_cpu/vm_*numa*` 能发现、选择和放置 node，
  `host_memory_info()` 只返回整机/进程约束，公开面没有任何 per-node memory query。消费者因此能选 node，
  却不能在分配模型权重、KV 或 worker arena 前对选定 node 做容量预算；这命中 Platform v9 的
  “named consumer reproduces a missing memory fact” reopen trigger；
- 权威 OS 机制存在但语义不应被抹平：Linux stable sysfs
  `/sys/devices/system/node/nodeX/meminfo` 给出 node distribution/utilization；Windows
  `GetNumaAvailableMemoryNodeEx` 给出 node 的 free+zero pages（多 node 时不含 standby）；macOS 没有公开
  multi-NUMA topology，HYLE 的逻辑 node 0 应组合现有 host snapshot。OS 不公开 total 时必须表达 unknown，
  不能从 available、型号、启动时采样或全机均分猜总量；cgroup 是进程级约束，也不能伪造为某一 node
  的物理容量；
- 受影响范围只含：NUMA node memory value shape/query、Windows/Linux/macOS 实现与 parser/system
  tests、Platform selected component 的 source/static/shared 交付、Platform architecture/user guide、
  一个直接 host-planning consumer journey，以及旧 Platform host P1 路线的有限 holdout；
- 明确不重开 `platform_thread`、`platform_fs`、Mem 或 Mem-VM：thread/rank policy、GPU-to-NUMA mapping、
  allocator reservation、model/KV sizing 与 cgroup budget composition 仍由各自 owner 决定。也不重做 CPU/
  cache/frequency、NUMA placement algorithm、Native FFI/C API、公共性能仪器或 Formal；
- 新 query 是启动/重规划控制路径，按 P0 验证每次至多一个 bounded OS/file snapshot、无缓存、无 allocation
  热路径状态；现有 host-planning P1 journey 只作一次未受影响 holdout，不为节点查询发明纳秒 benchmark；
- `CLOSED -> REOPENED` 前 registry snapshot：
  `C:\public\project-hyle\_validation\platform-v10-numa-node-memory\module-status-before-reopened.json`，
  SHA-256 `A3E938CDDE79B4A5CA09D2C37FFF30970ED3CDD1AEB17016F56DA4BA67748B00`。

## 初步停止边界

- Platform 只报告可由当前 OS 公开机制证明的 node 物理事实；它不替用户选 rank、node、利用率比例或
  allocation size；
- node id 必须来自同一 `numa_nodes()` 稀疏 id space；无效/不可见 node 返回结构化错误，不回落到 node 0；
- total 与 available 的 unknown/平台差异必须在类型中可见；零是合法观测值，不兼任 unknown sentinel；
- 审计若发现必须改变 placement、其它 Platform factor、Mem owner 或公共仪器，先登记相邻任务，不在本轮
  顺手扩 scope。
