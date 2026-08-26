# error v3 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页只记录 `error-v3` 这一轮的冻结合同、事实和证据。权威状态是
> [`module-status.json`](../../module-status.json)；旧 error task 与性能页均为历史输入，不能替代本轮
> 结论。

## Closure card（`SCOPED`）

- component / owner / round ID：`error` / error owner / `error-v3`
- task kind：`REFACTOR`
- baseline commit：`8b37eb42635fcf89505b7a2d9cad006d57664ba1`；tree
  `666e7c8542c6670c7ec2d19399e5ed6f3fa97ca9`
- 进入原因与 demand/defect ID：`ERR-CARRIER-01`（按真实 C++ 消费者与成熟库重审 `Status/Result`
  能力、形状和不变量）、`ERR-NATIVE-01`（重审 OS/backend 原生错误分类与保真）、
  `ERR-ADOPT-01`（核对底层模块是否正确使用统一错误通道）、`ERR-PERF-01`（旧性能点值未绑定当前
  candidate/统一仪器资格）、`ERR-DELIVERY-01`（build/package v2 后重签组件交付）、
  `ERR-DOC-01`（architecture/user/evidence 仍标为旧轮或含过时结论）。
- 真实 consumer task / normal workload：platform 的文件、虚拟内存、动态库、线程与 NUMA 失败；mem
  的映射/大页/fiber stack 与统计查询；obs sink、util registry、sched/io 控制面以及 compute/graph
  叶子转换。常态是 OK/code-only `Status`、`Result<T>` 成功/失败逐层传播；诊断路径是 owning message、
  一个 native domain/code、语义 context、首错聚合与最终格式化。
- 范围：公开 `Status`、`Result<T>`、传播宏、格式化和 errno/Win32/HRESULT adapter；ownership、
  lifetime、OOM、UTF-8、并发读取、复杂度与 one-word carrier；直接 `core` 依赖和代表性直接消费者；
  C++ source/static/shared delivery；当前 error benchmark contract、architecture/user/performance 文档。
- 非目标：日志/trace/crash reporting、异常层级、取消状态机、RPC/跨进程序列化、任意 payload/cause/
  backtrace/source-location registry、公共 MultiError、完整 WinSock/第三方 category adapter、完整
  `Result<T&>` 替代实现、下游模块自身收口、Native FFI/C API、Formal release certification。
- 直接依赖 / 直接消费者：`PUBLIC core`；直接消费者为 `platform`、`mem`、`obs`、`util`、`sched`、
  `io`、`compute`、`graph`，本轮只验证代表性编译/运行与语义采用，不为未收口模块签署结论。
- tier 与理由：`P1`；值 carrier、成功检查和 `Result` 传播处于基础运行时高频路径，message/context
  虽是冷失败路径，也必须保持同合同成熟实现的正常复杂度。
- material margin / resource budget：同合同 Abseil peer 的 required comparable workload 使用冻结的
  5% material/non-inferiority margin；64 位 `Status` 一个 word；OK/code-only 零分配零原子；message/
  native 单块 allocation；copy/move 与已知层级传播保持其公开复杂度合同。
- 总调查预算 / 性能预算：人工审计与归因最多 120 分钟；性能最多一次 `probe`、一次 fresh n=15
  `qualify-platform`，只有统一仪器停止规则允许时自动进行全新 n=31，不靠新增 benchmark 姿势延长。
- 最多复现、机制诊断、产品修改次数：一次正常运行；合同内 order/placement strata；一次 codegen/
  profile 机制诊断；至多一个同时有消费与机制证据的性能产品假设及重跑。
- machine profile：`W1-WIN` 是 P1 engineering 裁决机；`W1-WSL`、`M1-MAC` 只承担适用 correctness/
  sanitizer/portability 复核，不池化性能样本。
- instrument profile：P1 使用 `closure-perf-v1`；correctness/delivery 使用本轮风险矩阵与现有
  component/package 合同。
- strong peers：Abseil commit `81c993af4fb88b3a51829a796df02251fa27102f`（`Status/StatusOr`、
  taxonomy、OS mapping 和同合同 P1 peer，local source `C:/public/project-hyle/abseil-cpp`）；Folly
  commit `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`（`Expected`/error carrier/API 与实现查漏，
  local source `C:/public/project-hyle/folly`）。每项 capability 最多使用这两个预选 peer；C++23
  `std::expected` 作为标准合同，不计第三个性能 peer。
- 无同合同 peer 时的 consumer latency/throughput/resource 预算及禁止声明：native detail、固定 caller
  buffer 和 HYLE 特有 OOM 降级没有完整同合同 peer，只验证复杂度、分配次数和真实消费者行为；不得把
  Abseil arbitrary payload 或较弱的 `std::error_code` 对拍写成同能力胜负。
- correctness gates：normal、OK/empty/invalid code、size/overflow/OOM、copy/move/swap/destroy/lifetime、
  shared read/destroy concurrency、context alias/COW/growth、native mapping/UTF-8/NUL/truncation、Result
  error-arm invariant/monadic/macros、property/differential、Debug misuse、ASan/UBSan/TSan（适用时）、
  三编译器和 public-header self-containment。
- delivery gates：component graph/direct target edge；error-only source build；relocated static/shared SDK
  consumers；代表性直接消费者；没有 C schema/bridge/Native FFI symbol 或只为 FFI 存在的 option。
- performance gates：`bench/contracts/error-p1-v1.json` 冻结后所有 required comparable workload、
  strata 与公共 controls；timer-floor 必须按 executable 声明的真实 measurement batch 通过，合同、peer、
  candidate、build 和 artifact identity 全部绑定。
- 预期输出：有限 capability/gap ledger 与冻结 `FIX` 集；candidate commit/tree；correctness、delivery、
  docs/usability、独立 red-team 与 P1 manifest；四项 LOC、限制和 reopen triggers。
- 停止条件 / 可接受限制：全部适用工程门通过则 `CLOSED`；真实产品 blocker 或统一仪器不能产生 P1
  `PASS` 则带点名恢复条件进入 `BLOCKED`。没有具名消费者时，payload/cause/source location、
  MultiError、reference-result、WinSock/任意 category adapter 和 Native FFI 均可接受地留在 negative
  space，不因参考库 API 数量或 LOC 扩面。

## Capability / gap ledger

`SCOPED` 后只把消费者、peer 和标准审计得到的有限裁决填入这里；在 ledger 冻结并进入 `AUDITED`
前不修改产品实现。

## Candidate

尚未形成。

## Verification manifest

尚未运行 candidate 验证。

## 独立红队

尚未对 candidate 签署。

## Closure decision

当前为 `SCOPED`，不得提前写入 candidate 或 terminal 结论。
