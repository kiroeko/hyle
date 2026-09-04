# Mem-reclaim 冻结确认 — 2026-09-05

<!-- MODULE FREEZE CONFIRMATION: FROZEN -->

> 这是对既有 `CLOSED` factor 的只读确认，不是无触发重开，不建立新 product candidate，也不替换
> [`mem-reclaim-v2-inference-resume`](mem-reclaim-v2-inference-resume.md) 的 terminal record。当前状态真相
> 仍是 [`module-status.json`](../../module-status.json)；其中的 `freeze_confirmation` 把本次确认绑定到
> 明确仓库快照和本文证据。

## 绑定身份与范围

- confirmation：`mem-reclaim-freeze-2026-09-05`；marker：`FROZEN`；
- reviewed repository commit / tree：
  `aeb1156f6a973f37b1745434e5b5c146a3635794` /
  `859593f66e2a277607921de6ec8ea1a54485a028`；
- frozen v2 candidate / tree：
  `0f6ef8e83ee290ab565c0d53081a0acb75a46675` /
  `8cb5ba0024cdb20af65347144f09e16f26147632`；
- candidate 到 reviewed commit 的 `include/hyle/mem/reclamation.h` 与 `src/mem_reclaim` 产品实现为
  zero diff。后续中央组件图只增加了与本 factor 无关的 `QUEUED` 身份预留；Mem package sample 的变化
  扩大了既有 Mem 路线覆盖，没有改变 reclamation API、算法或正常成本；
- 本次 confirmation 没有产品修改。Obs v4 只在自己的 closure card 内裁决将该已冻结 factor 用作
  registry 短快照读者保护；这不把 Obs 的 callback 生命周期或 schema 所有权下沉到 Mem-reclaim。

## 质因数与能力边界

`mem_reclaim` 是当前内部图中的 read-mostly lifetime **质因数 factor**：直接且仅 `PUBLIC core`。
它拥有 guard、grace period、intrusive retire record 与同步回收语义；不知道 container node、Obs sink、
模型、allocator、executor 或 GPU fence。普通用户从 `hyle::mem` façade 取得该能力，内部 Container/Obs
可以直接依赖 factor，避免把整个 allocator/mapping 产品闭包带入底层 registry。

## 七问终答

### 1. 第三方产品路径

`PASS`。产品实现没有第三方源码、object、runtime 或公开类型，直接闭包只有 Core；C++ atomic、mutex
与 TLS 是工具链机制。Folly RCU 只作同合同 peer，不链接进 HYLE 产品。

2026-09-05 复核的最新 Folly commit 为
`dea9962e1a7d7cbbbeab5d6d4fe12d72b72ddabb`；相对冻结 peer
`46afdbec8f58fc21445741278f14d53ec7144676`，`folly/synchronization/Rcu.h` 只有四处注释拼写修正，
没有 API、算法或成本变化。推理输入锁定为 llama.cpp
`8b4b3558f1459c13e4aa38d5c94d306a00dc6acd` 与 ggml
`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`；它们仍有长寿命 backend/model/registry metadata，且没有
提供可替代的公共 reclamation primitive。

### 2. 功能超集与推理消费

`PASS`。当前自然能力路线为：

| 用户任务 | 当前路线 | 完整边界 |
|---|---|---|
| 极热读取、低频替换的不可变 snapshot | `read()` + move-only guard + `protect()` | 先进入后 acquire-load；guard 同线程、短生命周期 |
| 不分配地延迟释放 caller-owned 对象 | intrusive `RetireRecord` + raw/typed callback | record/payload/domain 均由 caller 拥有；callback 同步、`noexcept` |
| writer 不等待地尝试减压 | `try_reclaim()` | O(slots+stripes+ready records)，未过 grace period 返回 0 |
| 明确 grace period 与静止 teardown | `synchronize()` / `drain()` | 前者等待既有读者；后者要求外部禁止新 read/retire |
| 普通共享域与隔离慢读者 | default / explicit domain | 默认域为 process-image lifetime；隔离由 owner 显式选择 |
| 停顿读者独立、线程/fence 亲和释放 | hazard / executor / GPU owner | `ROUTED`；没有已准入 common contract，不伪装成 RCU 半项 API |

Container 的持续增长 table hand-off 是现有直接消费者；Obs 的固定双快照 publication 是新的同合同
消费者，但只复用短 reader grace period，callback quiescence 仍由 Obs 自己的 per-sink active reference
负责。当前未处置 `MUST` 为 0。

### 3. 现有模块与加速设施复用

`PASS`。

| Needed mechanism | Existing candidate | Verdict | Reason |
|---|---|---|---|
| compiler/cache-line/fatal invariant | Core | `REUSE` | 直接使用统一 assert、cache interference size、compiler 与 visibility 合同 |
| guard/grace-period/retire queue | Mem-reclaim | `OWN` | 本 factor 的唯一语义 authority；Container/Obs 不应复制 epoch protocol |
| recoverable error | Error | `NOT_APPLICABLE` | 资源已由 caller 提供；合法 read/retire 没有可恢复失败，合同误用 fail-fast |
| allocator/container | Mem/Container | `NOT_APPLICABLE` | read 与 retire node 均不分配；内部固定数组/侵入链不需要通用容器或 backing |
| clock/topology/SIMD | Time/Platform/SIMD | `NOT_APPLICABLE` | 算法不读钟、不选择 CPU/ISA，也不做 bulk data transform |

没有第二套 runtime reclamation authority。Obs v4 必须在 callback 外的短 snapshot-copy 区间使用 guard；
把 guard 延长到任意 sink callback 会让慢 callback 阻塞全域回收并使 callback 内注销产生错误自等待，
因此不属于正确复用方式。

### 4. 拟真、公平性能

`PASS`，但只签冻结命题。P1 正常任务是 warmed immutable pointer read；HYLE 与 Folly 都在 read guard 内
执行 acquire-load 和相同 payload observation，registration/retire/synchronize 不计入 read phase。
保留 bundle 当前由 `hyle-perf verify` 返回 `QUALIFIED / EXACT / CURRENT`、`paths_verified=true`；四个
placement/order strata 均通过 5% non-inferiority 且为明确 `WIN`，最低 99.26% CI lower 为 1.3358。

该证据不外推 writer-heavy、长 reader、超过 64 个同时 live thread 的 fallback、callback work、跨平台
排名或 Formal SOTA。`try_reclaim` 是固定 64 slots + 8 stripes 的有界扫描，`retire` 为 O(1) enqueue；
没有为了 benchmark 增加 payload metadata、隐藏 allocation 或 fast/light 公共分叉。

### 5. 旧 C ABI / Native FFI

`CLEAN`。没有公开 C header/schema、bridge source、未装饰 C export、managed sample、专用 option/target/
package metadata，也没有只为 FFI 支付的 layout 或 indirection。shared library 暴露的是同工具链 C++
decorated symbols；raw callback 是 C++ 内部类型擦除形状，不是 Native FFI 产品。

### 6. Source / static / shared 构建与导入

`PASS`。当前 Windows MSVC 19.50、WSL GCC 14.2 与 macOS AppleClang 21 均从 reviewed commit 验证精确
`mem_reclaim` factor（resolved=`core,mem_reclaim`，20/20 tests）。三端还分别验证了 `hyle::mem`
source embedding、relocated shared SDK consumer 与 relocated static SDK consumer；安装树包含
`reclamation.h` 与真实 `hyle_mem_reclaim` artifact。中央 product/factor graph 的 Mem façade包含该 factor，
内部 factor 仍可独立选择，不制造伪 façade 依赖。

### 7. C++ API 易用性与文档

`PASS`。[`user/mem`](../../../user/mem.md) 先按 read-mostly snapshot 任务选择该类型，再给出可复制的
guard/publish/typed-retire recipe，并明确 RCU 与 mutex/shared_ptr/hazard 的选择、domain/record/payload
lifetime、短读区、同线程退出、callback 线程和 blocking self-wait 陷阱；
[`mem design`](../../../architecture/mem-design.md) 记录 factor 边界、消费者和复杂度。RAII guard、显式
domain 与 caller-owned intrusive record 让 ownership 和成本可见，没有 benchmark/internal seam 泄漏。

## 独立红队终审

`PASS`。本次只攻击冻结范围，不借红队扩张 API 或 benchmark：

- 重新核对了 epoch flip 后迟到 reader、nested guard、同线程离开、double-retire、pending record 析构、
  callback 递归 retire、collector serialization、domain teardown、默认域 image lifetime、64-slot 租约
  回收与 65+ reader fallback；现有边界/death/concurrency tests 与候选 sanitizer 证据覆盖这些风险；
- `RetireRecord::pending_` 在 callback 前清除，使 callback 可复用 record；collector mutex 不在 callback
  期间递归取得，threshold callback 只 enqueue 后由 `drain` 后续轮处理，没有自死锁；
- domain 的固定 reader table 是显式 owner control storage，不进入 payload 或每条 retire metadata；默认域
  只适合短读者，长读者隔离规则已在 guide 暴露；
- candidate 到 reviewed commit 的产品实现为 zero diff；最新 Folly RCU 没有机制变化，保留 P1 bundle
  的 candidate、contract、peer、机器、仪器和 provenance 仍完整；
- 精确 factor、三平台头/并发测试、Mem 三交付和 C/FFI 负向边界均未发现遗漏。

剩余 medium-or-higher finding、未处置 `MUST`、correctness/lifetime blocker、material consumer regression
与重复 reclamation authority 均为 0。

## 当前快照验证

Windows 权威工作树的无关既有 Container/performance 文档改动没有进入 reviewed commit；WSL/macOS
使用该提交的 disposable native copy，远端内容没有回写产品。

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| exact `mem_reclaim` factor | resolved=`core,mem_reclaim`；20/20 | same；20/20 | same；20/20 | `PASS` |
| Mem product source/shared/static relocated | 1/1 each | 1/1 each | 1/1 each | `PASS` |
| current component graph/header closure | selected factor + façade | selected factor + façade | selected factor + façade | `PASS` |
| retained read-side P1 | verifier `QUALIFIED/EXACT/CURRENT` | not a timing claim | not a timing claim | `PASS` |
| candidate sanitizer evidence | retained WSL ASan/UBSan/TSan | candidate product zero diff | same product identity | `PASS` |

## 冻结裁决与 LOC

没有发现新的准入能力缺口、错误复杂度、material 性能缺陷、benchmark-shaped API、内部复用缺口、
易用性/文档错误、旧 C ABI 或 build/package integration 缺陷。`mem_reclaim` 保持 `CLOSED`，并在 reviewed
repository snapshot 上附加 `FROZEN` 标记；只有 registry 中既有 reopen trigger 的新事实才能解除。

`python scripts/loc-report.py --module mem_reclaim --commit
aeb1156f6a973f37b1745434e5b5c146a3635794 --self-check --json`：

| Metric | code |
|---|---:|
| 全库代码 | 439,904 |
| 自实现不含测试 | 70,651 |
| `mem_reclaim` factor 实现 | 227 |

