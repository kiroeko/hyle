# Platform-fs v2：控制稳定窗口恢复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

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

## Current decision

`AUDITED`。恢复条件、产品/合同零漂移和空 `FIX` 集已经冻结；尚未产生本轮正式性能 verdict。
