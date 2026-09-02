# container_intrusive v4 能力扩展

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`REOPENED`。本页先保存击穿 v3 negative space 的新事实；在进入 `SCOPED`、完成两遍能力
> 审计并冻结有限 `FIX` 集以前，不写产品实现、不开始计时。权威状态仍是
> [`module-status.json`](../../module-status.json)。

## Reopen trigger

- component / owner / round：`container_intrusive` / Container owner /
  `container-intrusive-v4-capability-extension`；
- trigger ID：`INTR-CAPABILITY-EXTENSION-2026-09-02`；
- baseline：`b4915574020cc7da857d2871c545a3489c75288b`；
- 新事实：owner 明确要求把 v3 已记录的模块化常用负空间重新作为产品能力裁决；全局能力账本的
  `INTR-A-06/07/10`、推理 watchlist 的 `INTR-WATCH-01/05`、ggml 的一指针 object chain，以及当前
  Folly/Boost/oneTBB 的 counted/slist/list-algorithm 实际消费共同证明这些不是 benchmark-only symbol；
- 旧结论受影响范围：v3 对 counted list、单向链/hlist 和完整非 owning sequence 算法的 `DEFER`，以及
  “只有双向 member-hook primitive”这一公开能力边界；v3 已签的现有双链 correctness 不被无界重审，
  但默认 layout/codegen 必须作为回归门。

锁定 peer identity（本轮审计可从中为每个 family 最多选择两个）：

- Boost `78505eab9b203e3e26e83893d0685f160133fa0f`：Intrusive list/slist、constant-time-size 与
  sequence algorithms；
- Folly `176c088840724715b2c3c6211b2b347611676002`：counted waiter、Arena slist、atomic publish 和
  runtime 消费形状；
- oneTBB `b03a694ee0fe4b188bc77daa8a567551e3e3ebfc`：runtime counted intrusive list；
- ggml `d4716378882593333721eb33f153144b6885caf2` 与 llama.cpp
  `b96806d96061049a5b574269b049bf6241d63d46`：北极星基础/真实消费差异输入。

## 进入 `SCOPED` 前的有界审计

待冻结的四个能力族如下；名称和操作面仍须由普通 C++ journey、生命周期与成本合同验证，不能从 peer
symbol 或旧 benchmark 直接复制：

1. **可选 O(1) membership count**：独立 counted safe-list，不给现有 uncounted list/hook 增字段；
   single/whole-list mutation O(1)，range splice 明确区分扫描计数与 caller-provided count 的 O(1)
   路线。
2. **一指针 forward membership**：arena object chain、单 owner free/deferred-work/FIFO chain；节点只付
   一个 pointer，显式 retirement，不能虚假承诺 arbitrary-known-node O(1) unlink 或 auto-unlink。
3. **hlist membership**：`next + prev_next`、head 一指针、已知节点 O(1) unlink/replace；与完整双向 list
   的区别必须来自 bucket/head footprint 与状态机，不以“更快”命名。
4. **非 owning sequence algorithms**：`reverse/remove_if/unique/merge/sort` 等只有在稳定性、异常策略、
   iterator/lifetime 和无分配复杂度完整时才进入 public surface；不复制 base/value/fancy-pointer 等
   Boost policy matrix。

审计同时扫描 Mem/Sched/IO 的私有链结构。较高 factor 可直接增加
`PUBLIC/PRIVATE container_intrusive` 边；不得因依赖顺序复制同义 primitive。以下特化先作为反例保留：

- freed slot 上覆盖 next 的 pool free-list，不能让 live object 为公共 hook 付费；
- waiter 摘链后复用同一 storage 的锁外 wake chain；
- Win32 kernel completion/inflight/free-state 复用的 IOCP link。

cancellation hlist、arena/deferred-work forward chain 和普通 counted membership 必须在新类型成立后重新
裁决，能自然迁移才增加真实直接边；不能为了证明新 API 有消费者而强行迁移状态机。

## 初始施工边界（尚未冻结）

- 初始范围：上述四族的 capability map、自然 C++ API、header-only 实现、复杂度/layout、correctness、
  直接消费者回审、Container façade/source/static/shared 交付、user/architecture 文档和适用 P1 证据；
- 初始非目标：owning storage、allocator、内部同步、lock-free publication/reclamation、intrusive
  tree/hash/heap、base/function/value traits、fancy/offset pointer、Native FFI/C API、Sched/IO policy、
  Formal 证书与公共性能仪器重写；
- 依赖目标：仍只 `PUBLIC core`；新增 header 继续由 `container_intrusive` factor 拥有并由
  `hyle::container` façade 聚合；
- tier：`P1`；热 mutation/遍历/批处理需保持零分配、无锁、无 syscall，并在同语义成熟路线存在时用
  `closure-perf-v1` 的 5% material margin；没有同合同 peer 的 hlist 只签预冻结 consumer/resource
  budget，不制造 SOTA 声明；
- 验证平台：W1-WIN 为工程性能机，W1-WSL/M1-MAC 为 correctness、sanitizer、ABI 和可移植性门；
- 停止规则：审计后只允许一个有限 `FIX` 集；每族最多两个 strong peer；P1 异常只允许协议规定的一次
  复现、一次机制诊断和一个产品修改。公共仪器缺口单独阻塞，不在本模块扩建仪器。

## 当前裁决

`REOPENED`，尚未 `SCOPED`。允许的下一步只有：完成四族自顶向下 journey、现有 public surface 反向
追踪、私有重复/直接消费者扫描，冻结 API/cost sketch、`FIX` 集、测试和性能 workload。尚无 candidate，
旧 v3 性能证据不能覆盖新增能力或变化后的产品 identity。

