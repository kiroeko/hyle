# HYLE 预发布就绪度与模块冻结规则（2026-07-19）

## 结论

HYLE 当前不是“功能只完成一半”的骨架，也还不是可直接发给任意语言用户的完整
发行版。按可验证的层次读：

| 层次 | 当前判断 | 最大剩余项 |
|---|---|---|
| 基础模块机制 | **高完成度** | error/platform/mem/time/simd/util/obs/container 已多轮强收口 |
| sched/io/compute/graph 机制 | **主体存在，不是 stub** | sched/io 尚需同等级模块收口；compute/graph 仍有能力尾 |
| 稳定外部产品面 | **部分完成** | 上述八模块已有纯 C ABI；sched/io/compute/graph、install/export、跨编译器和 FFI 门未完 |
| 完整北极星能力 | **约 70–80% 判断带** | graph-02 后两步、compute 后端/算子尾、统一发布验收 |

`cloc src include` 当前约 **67.2k**；tests **29.9k**；bench **13.2k**；
docs **17.5k**。LOC 只说明资产体量，不作为功能或 SOTA 的替代证据。

## 预发布冻结门

模块同时满足下列六项，才标成“预发布冻结”：

1. 功能清单与参考实现/设计逐项对账，缺项要么实现，要么写明不做及理由；
2. 红队覆盖所有权、OOM/overflow、并发、失效、ABI 和平台差异；
3. Windows x64、Linux x64、macOS arm64 的相关测试通过，适用时加
   ASan/UBSan/TSan 和特殊 ISA 验证；
4. 性能按**同契约强对手**对拍，已知输项有测量证据和边界，不能只比弱 std；
5. 纯 C ABI 机械覆盖、纯 C 测试和用户指南齐全，或明确说明该模块还不进入稳定面；
6. 架构、维护红线、验证命令和剩余触发器有权威文档。

冻结不是声称“永远没有更快写法”，而是停止无证据的重复全审。冻结后只有下列
事件可以重开：

- correctness/security/ABI 缺陷；
- 新支持平台、ISA 或后端使原合同不成立；
- 按既定方法论确认的性能回归，或强对手显示复杂度档/持续显著落后；
- 已排期的真实消费者暴露缺失能力或难用接口；
- C ABI/install/发布工程需要调整模块边界。

以下**不能**单独触发重开：亚纳秒点值波动、没有 workload 的猜测 API、新库出现、
命名/风格偏好、为了增加 op/LOC 数而扩面。

## 当前模块判定

| 模块 | 判定 | 后续方式 |
|---|---|---|
| error/platform/mem/time/simd/util/obs | **预发布冻结** | 只按上述触发器重开；SVE/RVV 真机证据仍是平台到位后的专项补证 |
| container | **预发布冻结** | StringBuf 复杂度/争用/batch/三平台强 peer 已收口；已知 Linux/M4 lifecycle 常数边界写入证书 |
| sched/io | **下一批强收口** | 功能/下游/红队/测试性能/C ABI/指南六项一次完成 |
| compute/graph | **机制成立、产品能力未封口** | 按北极星补能力，不能称“只有架子”，也不能提前冻结 |
| close-01/发布 | **部分** | 八个底层模块完成；上层四模块、install/export、跨编译器和 FFI 冒烟待做 |

## 冻结证据入口

- platform/mem：[`platform-audit-followup.md`](./platform-audit-followup.md)、
  [`mem-audit-followup.md`](./mem-audit-followup.md)；
- time/obs/util/container：[`time-module-closure.md`](./time-module-closure.md)、
  [`obs-module-closure.md`](./obs-module-closure.md)、
  [`util-module-closure.md`](./util-module-closure.md)、
  [`container-module-closure.md`](./container-module-closure.md)；
- SIMD：[`simd-24-foundation-completion.md`](./simd-24-foundation-completion.md)、
  [`simd-capability-fill-list.md`](./simd-capability-fill-list.md) 和
  [`simd-23-cxx-facade-and-perf-closure.md`](./simd-23-cxx-facade-and-perf-closure.md)；
- 稳定 C 面及用户入口：[`close-01-c-abi-shim.md`](./close-01-c-abi-shim.md) 和
  [`../guide/README.md`](../guide/README.md)。error 的证据收在 close-01 与
  `c-error` guide 中，不为凑文档数再复制一份 closure。

## 预期施工顺序

1. sched、io 依次做同等级模块收口，同时补稳定 C ABI 与指南；
2. compute/graph 以北极星缺口为分母补能力，不按 LOC 或里程碑数量凑完成度；
3. 完成 close-01 的 install/export、跨编译器和最小 FFI 冒烟，才进入候选发布版。

## 2026-07-19 冻结增量

- container 的 StringBuf 例外已完成：resident hash 缓存把 rehash/erase 从字符串
  字节长度中解耦；自适应 shard、slab/slot reuse、lazy permanent ID 和四类
  batch API 已落地。
- Windows/WSL/M4 三平台 Release 通过；WSL ASan/UBSan/TSan 通过。TSan 同时
  揪出并补上 mimalloc 未插桩的仪器盲区，启用 vendor `MI_TSAN` 后 188 项全绿。
- 结论不是“StringBuf 三平台全赢”：长字符串增长三平台赢 strongest available
  floor，8-thread 距 Absl 约 4–5%；Linux/M4 lifecycle 仍落 14–22%。这些是冻结
  边界，不能因为一个点值不绿就继续无限改架构。

权威进度入口仍是 [`README.md`](./README.md)；本文件定义“什么时候停止继续优化”的
验收纪律，路线拓扑见 [`../architecture/roadmap.md`](../architecture/roadmap.md)。
