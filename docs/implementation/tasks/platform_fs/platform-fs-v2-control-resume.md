# Platform-fs v2：控制稳定窗口恢复

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 本页是 `platform_fs` 从 v8 性能证据 blocker 恢复的独立 `RESUME` 轮。产品能力、API 与实现已经由
> [`platform-v8-os-resource-contract`](../platform/platform-v8-os-resource-contract.md) 冻结；本轮只重绑
> 未变化的产品候选并取得一次合法的工程性能裁决。

## Closure card（`SCOPED`）

- component / owner / round ID：`platform_fs` / Platform-fs owner /
  `platform-fs-v2-control-resume`；
- task kind：`RESUME`；
- 前一 round / blocker：`platform-v8-os-resource-contract`；第二候选的 r3 因 5% known-delta control
  失格，唯一重跑 r4 又因 A/A control 失格，因此产品行即使接近 1 也不能签成 P1 `PASS`；
- 前一 resume condition：在新的无竞争 W1-WIN 窗口，保持候选、合同和 5% margin 不变，先证明 A/A、
  5%/10% known-delta 可稳定裁决，再一次性生成完整八 strata 产品 bundle；旧 r3/r4 不拼接；
- baseline commit：`27e6128683fbfdf371310dc433ab9abf68d13e9a`；复用的产品候选是
  `c39df6f94c636ab8fa94cb700dd263763a42d680`、tree
  `37889346564f3a5cdc895f26c9d610883076ba02`；baseline 以后没有 Platform-fs 产品/API/layout/算法/
  依赖或正常路径成本修改；
- 进入原因与 evidence ID：`PLATFORM-FS-CONTROL-WINDOW-2026-09-06`。W1-WIN 没有 compiler/build
  进程、登记 sampling mutex 可取得；15 个独立进程 × 双顺序的 controls-only 预检已经同时通过 A/A、
  5% 与 10% controls；
- 真实 consumer task / normal workload：打开一个 warmed 4 MiB 模型文件，完整顺序读入 caller-owned
  buffer，校验全部 bytes 并关闭；
- 范围：使用未改的 `platform-fs-p1-v2`、Folly peer identity、W1-WIN 和 5% margin，对同一候选执行
  一次 fresh `qualify-platform`；深验 manifest，记录全部 controls、八 strata 与固定方向点值；
- 非目标：任何 Platform-fs 产品/API/benchmark/peer adapter/contract/margin 修改；把 WSL 或 macOS
  结果代替 Windows/NTFS；拼接旧样本、挑样、扩大样本或失败后再次运行；公共仪器重构；Formal、
  Native FFI、IO 或相邻模块审查；
- 直接依赖 / 直接消费者：公开 `core,error,time`，私有 `platform`；旧轮的 Mem/Mem-VM/SIMD/
  Container/Sched/IO exact direct-consumer 回放保持有效，本轮不修改这些模块；
- 待复用机制：现有统一 `closure-perf-v1`、`platform-fs-p1-v2`、W1 sampling mutex、固定 P-core/
  above-normal policy、artifact/build validators；不建立模块私有统计器；
- tier 与理由：`P1`；同步模型/配置/cache 文件读取是正常 consumer journey，当前轮只续签工程非劣；
- material margin / resource budget：5% non-inferiority；每侧一次 open、完整 cursor read、全 buffer
  checksum 与一次 close，独立等大预分配 buffer；计费阶段和错误/EOF/close 语义相同；
- 总调查 / 性能预算：controls-only 前置检查一次；随后最多一次 fresh n=15 qualification，并且只有
  统一仪器既有停止规则才可自动进入完整 n=31；不允许产品修改或第二次资格化；
- machine / instrument：`W1-WIN` / `closure-perf-v1`；`W1-WSL` 与 `M1-MAC` 的 correctness、sanitizer
  和三交付沿用未漂移候选的 v8 exact evidence，不进入 Windows 点值；
- strong peer：Folly `65749da412d92fe27d87ae45a8ef670720712858`，只提供冻结的 File-shaped direct-OS
  mechanism provenance；实际 peer adapter 与 HYLE 在同一 executable、toolchain 和 OS ABI 中运行；
- provenance：`NORTH_STAR_JOURNEY / replay`，consumer key
  `inference/model-loader/synchronous-read`；comparison class 为 `SAME_CONTRACT`；
- 适用 gates：产品 identity/no-drift、P1 qualification/performance、固定方向点值复核、文档/registry/
  closure consistency。能力、correctness、delivery、易用性与红队只核对未漂移，不重复三平台施工；
- 停止条件：唯一完整 bundle 为 `QUALIFIED` 且 performance gate `PASS` 时进入 `VERIFIED/CLOSED`；任一
  required control 或产品 NI 未通过则直接再次 `BLOCKED`，不运行第二包；Formal 保持 `PENDING`。

## Resume 前置证据

controls-only 预检位于
`C:\public\project-hyle\_validation\platform-fs-v2-control-resume\preflight-01\preflight.json`，SHA-256
`F53EEE379D617CFADE5A3355F9892C8AB4C082482F2AE2F38FF5F042C508FF15`。它使用 r4 留存的 exact
common-control executable（SHA-256
`889765B5DF744CE3C7BA517B4797642CD87DEB8E047108ABE36F50EBA08A56F5`），在登记的 W1 mutex、逻辑核 2
和 above-normal child policy 下运行；30 份 raw log 保留且只重放一次，没有在看到结果后补样。

| control | candidate-first | peer-first | verdict |
|---|---|---|---|
| A/A | median 0.997743；99.26% CI `[0.957755, 1.034376]` | median 1.006165；CI `[0.987895, 1.020996]` | PASS |
| known delta 5% | median 1.047587；CI `[1.023728, 1.067925]` | median 1.047509；CI `[1.034450, 1.061063]` | PASS |
| known delta 10% | median 1.097444；CI `[1.059727, 1.138311]` | median 1.082093；CI `[1.065684, 1.119977]` | PASS |

预检脚本在采样完成并释放 mutex 后才请求 lease manifest，因而出现一次纯记账异常；30 份日志完整，
随后只读重放得到上表。该预检只证明新的采样窗口满足恢复条件，不冒充 product evidence；正式 bundle
仍须自行通过 A/A、5%/10%、timer-floor、placement 与全部 artifact/host controls。

`BLOCKED -> SCOPED` 前态 registry snapshot：
`C:\public\project-hyle\_validation\platform-fs-v2-control-resume\transitions\module-status-before-scoped.json`，
SHA-256 `361F60B57A5A21A2870A7AF9988429B005692B177507C427A9FC3AD2E6409FB1`。

## Audit 与冻结 FIX ledger（`AUDITED`）

- `git diff c39df6f9..1360c1e0` 对 `include/hyle/platform`、`src/platform`、`src/platform_thread`、
  `src/platform_fs` 与中央 component graph 为空；冻结产品候选及其依赖产品路径没有漂移；
- 同一范围对 `bench/platform_fs` 与 `bench/contracts/platform-fs-p1-v2.json` 也为空；当前合同、peer
  adapter、八 strata、5% margin 与 runner mapping 没有在看到旧结果后改变；
- v8 的 capability、correctness、delivery、API/usability、内部复用与 red-team 结果仍绑定同一产品
  identity；本轮没有新的 consumer、缺陷或成熟 peer 事实触发产品重审；
- preflight 的 A/A 与 5%/10% 双顺序 controls 全部通过，满足前一 blocker 唯一恢复钥匙；它不参与正式
  product summary，也不与随后 bundle 拼样；
- 冻结 `FIX` 集为空。允许的工作只有 fresh build、一次 qualification、deep verify、结果文档和合法
  registry transition；任何产品、benchmark、contract 或公共仪器修改都使本轮停止并退回新审计。

`SCOPED -> AUDITED` 前态 registry snapshot：
`C:\public\project-hyle\_validation\platform-fs-v2-control-resume\transitions\module-status-before-audited.json`，
SHA-256 `2381C8042A552B9CA175D928FC0535C44BA533930AE566BE1E0FA3F26BD1F6DF`。

## Candidate

- product commit：`c39df6f94c636ab8fa94cb700dd263763a42d680`；tree：
  `37889346564f3a5cdc895f26c9d610883076ba02`；
- 该 identity 直接复用 v8 第二候选，没有生成替代产品 commit；当前 clean source 位于
  `C:\public\project-hyle\_validation\platform-v8-c39df6f9-perf-src`；
- `c39df6f9..7be19402` 的 Platform/Thread/FS 产品、组件图、FS benchmark 与 FS P1 contract diff 均为空；
- 公开合同、ownership/lifetime/concurrency/failure、算法、复杂度、常态 metadata/layout 和直接消费者
  均无变化；
- harness/control-plane commit：`7be19402912d98fbeb710f6a145869a027e512b9`。它只新增本 RESUME
  记录和 registry transition，不进入被测产品或 benchmark；
- `AUDITED -> CANDIDATE` 前态 registry snapshot：
  `C:\public\project-hyle\_validation\platform-fs-v2-control-resume\transitions\module-status-before-candidate.json`，
  SHA-256 `8FFF1F673C711C3C5BE7887971A6DCFFA92838CC504B077D235637286B15724A`。

## API / usability freeze

- 状态：`FROZEN / UNCHANGED`；
- `FileHandle` owning convenience、caller-owned buffer、`Result/Status`、EOF/short transfer、close 和 EINTR
  语义全部沿用 v8；
- benchmark、环境变量、Folly adapter 与 control 不进入公开 API；
- 本轮没有允许的内部产品优化边界。任何产品变化都终止本 RESUME，并回到新的 `AUDITED` 轮。

## SAME_CONTRACT 等价矩阵

| Dimension | HYLE | peer adapter | verdict / charged note |
|---|---|---|---|
| work/result | 完整读取同一 warmed 4 MiB file 并校验所有 bytes | 相同 | `EQUAL` |
| ownership/lifetime | move-only file owner；独立 caller buffer | native handle；独立等大 caller buffer | `ADAPTER_EQUALIZED` |
| failure/EOF/close | open/read/EOF/close 全检查 | 相同阶段全部检查 | `ADAPTER_EQUALIZED` |
| ordering/capacity | cursor 顺序读满固定容量 | cursor 顺序读满相同容量 | `EQUAL` |
| thread/progress | 单线程同步阻塞 | 单线程同步阻塞 | `EQUAL` |
| setup/teardown | allocation/warmup 不计时；open/read/checksum/close 计时 | 相同 | `EQUAL` |
| memory/delivery/codegen | static source route、MSVC 19.50、LTO off | 同一 executable/TU/toolchain | `EQUAL` |

额外工作/保证方向为 `none`；execution order、buffer allocation order 和 function placement 都由八个冻结
strata 覆盖。预检不能修改这张矩阵。

## 唯一正式资格化与固定方向复核（`VERIFIED`）

- fresh evidence：
  `C:\public\project-hyle\_validation\platform-fs-v2-resume-evidence-r1`；统一仪器按预声明的
  `[15, 31]` power stages 自动进入 31 个独立进程，本轮没有第二个正式 bundle，也没有人工补样、删样或
  与 preflight/r3/r4 拼样；
- deep verifier：`QUALIFIED / EXACT / CURRENT`，`inputs_stable=true`、`artifacts_stable=true`、
  `host_stable=true`，consumer evidence 与 provenance gate 均 `PASS`；MSVC 19.50.35728、Release、static、
  LTO off、VS 18 x64、Windows SDK 10.0.26100；
- controls：A/A 双顺序 median 0.999988 / 1.006608，CI 都在等价带内；5% known-delta 为
  1.043835 / 1.059100，10% 为 1.101676 / 1.095109；timer-floor 与 primary/placement 双 variant
  也都 `PASS`；
- product：八个 strata 全部 exact `PARITY` 且 5% non-inferiority `PASS`。advantage ratio 为
  0.996402、0.999517、1.012026、0.998621、0.987503、1.005656、0.994822、1.002090；对应 CI 全部
  包含 1。5 行点值低于 1、3 行高于 1，并随 execution order、allocation order 和 placement 翻转；最小
  点值约低 1.25%，没有跨 strata 的固定方向，也没有可归因的产品机制；
- 因而旧 r3/r4 暴露的是当时 controls 未取得资格，不是三台机器共同证明产品回归。本轮满足同合同、控制
  灵敏度和功效后，产品结论只能是 `PARITY / PASS`，不宣称 WIN、跨机器 SOTA 或 Formal certification；
- contract-lock / manifest / summary / observations / controls SHA-256 分别为
  `4324C5FD0A97E9B63D2868290B9988A175DBFA344254A52F0F2A79C40F50BE06`、
  `6BC5AE79776C0A4B561326297F0910E00BB3C44CD9D405368369A1C18E655C14`、
  `6AE27668D8DC940535E7F1495525D21F7AC67346E14DC26BFCDFAE4E551E9A19`、
  `C71F9EFD171F99092FF71EB97602579FAC076E4BF5DFADC6B0C5CE5EEE0B35A1`、
  `D994AF76428673F48EDD2E92610E478265B4AB9264725882E22B54D1594FC4F4`；canonical contract SHA-256 为
  `c72d5c879273dffcb805892bc0acc746af61b1841a45e57d332411bf82e83adc`。

`CANDIDATE -> VERIFIED` 前态 registry snapshot：
`C:\public\project-hyle\_validation\platform-fs-v2-control-resume\transitions\module-status-before-verified.json`，
SHA-256 `9DE5D7B9BE03B09640C3F76B19A0C3A08B9F4064743B49C5D4DC274C19AAE5E2`。

## Current decision

`CLOSED / FROZEN`。未变化的 v8 第二候选已经取得唯一一份合法的 fresh P1 receipt；全部适用 gate 为
`PASS`，旧 blocker 已由预声明的恢复条件解除。本轮没有产品、benchmark、contract、margin 或仪器修改。

## Closure decision 与 terminal receipt

| gate | verdict | basis |
|---|---|---|
| scope | PASS | 只恢复既有 exact candidate 的一次 P1 裁决，没有范围漂移 |
| architecture | PASS | 产品/API/ownership/复杂度和依赖保持 v8 已验收形状 |
| correctness | PASS | 同一 candidate 的 W1-WIN、W1-WSL、M1-MAC exact v8 correctness/sanitizer 证据未漂移 |
| delivery | PASS | source/static/shared、component/direct-consumer 与符号证据绑定同一产品 identity 且无产品 diff |
| usability/docs | PASS | `FileHandle` C++ 旅程和性能语义未改；当前证据页已区分 qualified 与 diagnostic |
| red team | PASS | 不拼样、不重跑到绿、不改合同/阈值；固定方向、苹果/橘子与 provenance 均复核 |
| engineering performance | PASS | 唯一 fresh bundle 为 `QUALIFIED / EXACT / CURRENT`，8/8 PARITY、NI PASS、controls PASS |

已知限制：

- 该 factor 是同步、buffered 文件 owner/namespace 路线；async/direct I/O 仍由 IO owner 承担；
- POSIX 只对可安全重启且无已知副作用的操作重试 `EINTR`，close-like/可能已产生副作用的操作故意不盲重试；
- 数值 receipt 只覆盖 W1-WIN 上 warmed 4 MiB model-file open/read/checksum/close 同合同旅程，不外推为
  冷 cache、全部文件系统、跨机器或 Formal SOTA；
- W1-WSL 与 M1-MAC 提供正确性、sanitizer 和交付覆盖，不与 Windows/NTFS 点值混合。

reopen triggers：

- 真实 consumer 复现同步文件 correctness、ownership、EOF/short-transfer、EINTR 或 close 语义缺陷；
- 公开 API、复杂度、native-call 成本/语义、直接依赖、交付或受测 product bytes 改变；
- 合格的同合同正常 consumer 证据证明超过 5% 的可重复 material regression，或成熟 peer 证明更优复杂度；
- 支持的 OS/SDK 改变这里依赖的文件 handle 或同步 I/O 合同。

LOC 绑定 candidate `c39df6f94c636ab8fa94cb700dd263763a42d680`；`loc-report.py --module platform-fs`
加 `--commit <candidate> --self-check --json` 为 `PASS`：全库代码 451,412；自实现含测试 116,939；
自实现不含测试 74,308；本 factor 实现 2,712 行。公开共享头由 Platform conventional boundary 计数，
本数只统计 `src/platform_fs`，不能与 base 数值相加冒充互斥拆分。

本提交保存 terminal task、receipt 与 candidate identity；它的完整 commit 将作为 registry 的
`end_commit`。后续第一个真实保存 `platform_fs=CLOSED` 的提交作为 `registry_anchor_commit`，不在本页
伪造尚未生成的自身 hash。
