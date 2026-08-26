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

## Capability / gap ledger（`AUDITED`，已冻结）

| ID | requirement class | demand kind | exact consumer / peer / standard source | normal shape | owner | ownership/failure/concurrency | complexity/resource target | verdict | severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|---|---|
| ERR-STATUS-01 | MUST | capability + direct consumer | platform/mem/graph/compute 的 `Status`；Abseil `Status` | canonical code；OK/code-only；可选 owning message | error public | immutable value；调用方拥有每个对象；不同 copy 可跨线程 | 64 位一个 word；OK/code-only O(1)、零分配零原子 | `KEEP` | P0 | code/factory/predicate/size/property；不增加常态 metadata |
| ERR-RESULT-01 | MUST | capability + C++23 standard | platform/mem/compute 的 `Result<T>`；`std::expected`；Abseil `StatusOr`；Folly `Expected` | value 或非 OK Status；`Result<void>`；monadic observers；传播宏 | error public | checked composed error arm 不暴露 mutable `Status&`；lvalue copy/rvalue move | expected-shaped layout/branch；深度 d 传播 O(d) | `KEEP` | P0 | invariant death tests、monadic/macros/codegen；不另造 expected |
| ERR-CONTEXT-01 | MUST | capability + real propagation | mem/compute 失败边界；Abseil `StatusBuilder` | 只在增加语义时 rvalue prepend；最终扁平文本 | error public | unique chain 可复用 prefix；shared/lvalue COW；OOM 保留原值 | unique chain 总 O(F)、O(log F) 增长分配；普通 Status 无 capacity tax | `KEEP` | P0 | alias/COW/OOM/growth/property；显式保留历史的 lvalue 链可 O(k²) |
| ERR-NATIVE-01 | MUST | capability + backend recovery | errno/Win32/HRESULT、Vulkan、CUDA/NVRTC consumers | canonical code + 最多一个 `{domain,int64}`；OS cold factory | error public + owning backend mapper | message 不作机器协议；不同 domain 不混猜；OOM 至少保 code | classification O(1)；message/native 单 block O(m) | `KEEP` | P0 | mapping table、cross-host、UTF-8、OOM；WinSock/第三方 category 不伪 round-trip |
| ERR-FORMAT-01 | MUST | user/obs boundary | caller-buffer `ToString` 与 opt-in C++ `to_string`；Abseil `ToString` | allocation-free borrowed output或 owning C++ string | error public | borrowed message 随 Status；caller 管 buffer；byte truncation 可切 UTF-8 | O(min(F,cap))；size-only O(1)；owning O(F) | `KEEP` | P0 | null/zero/truncation/NUL/long message；不引入 iostream/log coupling |
| ERR-OOM-01 | MUST | foundation failure contract | platform/mem allocation failure；Abseil/Folly 实现对照 | optional diagnostic 分配失败时退化为 code-only | error public | `noexcept`；不因格式化失败伪装 OK | 保留 canonical code；不抛异常；不要求 emergency TLS heap | `KEEP` | P0 | forced-OOM 全入口；下游 factory policy 归各 owner |
| ERR-LIFE-01 | MUST | concurrency + shared delivery | cross-thread copy test；Abseil intrusive rep | immutable heap rep、inline retain、DLL final release | error public/build | 匹配 header+artifact；同一对象并发写仍需外部同步 | copy/move/destroy O(1)；unique destroy 避免 atomic RMW | `KEEP` | P0 | thread/stress/ASan/TSan、shared SDK；不声称稳定 C++ ABI |
| ERR-DEFAULT-01 | MUST | usability defect | C++23 `expected` 与 Abseil 默认 `StatusOr` 的不同选择 | `Result<T>{}` 是成功且包含 value-initialized `T` | error docs/tests | 不是 empty/unknown；非默认构造 T 时本就不可用 | 无产品成本 | `FIX` | P0 | guide pitfall + explicit test；不改变标准 expected 语义 |
| ERR-DELIVERY-01 | MUST | build/package | component graph 与 fresh error-only SDK smoke | `error PUBLIC core`；source/static/shared C++ | build/error | allocation/final free 留在同一 artifact；无 C schema | exact selected closure；无 Native FFI symbol/option | `KEEP` | P0 | relocated source/static/shared consumer + export inspection |
| ERR-INCLUDE-01 | MUST | direct-consumer hygiene | `platform/convenience.h`、`sched/fiber.h` 直接写 `Result` | 公开头直接 include 自己使用的声明 | platform/sched integration | 不依赖传递 include；不改变模块语义 | compile-time only | `FIX` | P0 | direct includes + header self-containment；不替下游模块结案 |
| ERR-WORDING-01 | MUST | current policy mismatch | `sched/sync.h` 的旧 FFI factory；`util/random.h` 的 C bridge 字样 | 按 C++ callback/native-FFI N/A 的真实边界描述 | sched/util integration docs | 不复活旧 bridge 或兼容承诺 | 无运行时成本 | `FIX` | P0 | stale-term scan |
| ERR-PERF-REPLAY-01 | MUST | P1 consumer evidence defect | `graph::TaskGraph::evaluate` 的 dirty-node function-pointer sweep | 64 个 stable node 的一次 evaluate；全成功与末节点 code-only 失败；HYLE/Abseil 对称 carrier | error benchmark | callback/context/loop/最终观察对称；一次 graph evaluation 是自然计费任务 | 5% NI；batch=1 evaluation；必须达到同 batch 20x floor | `FIX` | P1 | 替换会被 DCE 的 OK/OK-copy 与无资格短传播行；双 order/placement/A-A/delta/floor |
| ERR-PERF-NORMAL-01 | MUST | P1 normal diagnostic path | 当前 29B message、shared copy、context 1/3、2-frame message Result、54B owning format、fresh-prvalue Update | 真实普通诊断创建/传播/消费，不交叉 size/depth 矩阵 | error benchmark | HYLE/Abseil 同 code/message/context 与 ownership；allocator order 可诊断一次 | 每项 5% NI 且独立过 20x floor | `FIX` | P1 | contract 只保留这些 normal rows；任一无资格即按冻结停止条件 BLOCKED，不换姿势 |
| ERR-PERF-BOUNDARY-01 | MUST | invalid evidence classification | 7/256B construct、16-layer context、512B format、runtime-code primitive、native/errno rows | correctness/机制/边界诊断，不冒充 P1 consumer 总分 | error tests/diagnostic bench | native 与 Abseil payload 非同合同；lvalue x16 是刻意保留历史 | 结构与绝对成本可记录，不形成同能力胜负 | `FIX` | P0 | 从 required P1 contract 移出；保留 bounded diagnostic coverage |
| ERR-PERF-VALIDATOR-01 | MUST | instrument gap | unified runner 与旧专用 runner 的窄结构检查 | actual-object DCE/call-edge、placement authenticity、semantic setup oracle | error benchmark plugin | validator 不改变 workload/统计；SourceLocation 为空；输出语义/长度锁定 | fail closed；contract hash/peer/product/artifact 全绑定 | `FIX` | P1 | module command validators；不复活旧 3% runner/统计核心 |
| ERR-DOC-01 | MUST | stale current documentation | architecture/user/performance pages | 当前能力选择、成本、pitfalls、negative space、peer pins 与本轮证据 | docs | architecture 讲设计，user 讲用法，evidence 只讲当前结果 | 不复制历史 task 数字 | `FIX` | P0 | 去掉 `NOT_REVIEWED_V2`；旧页明确历史或替换为当前证据 |
| ERR-REF-01 | GENERAL | mature-library gap scan | Abseil `StatusOr<T&>`；Folly legacy `Expected` 拒绝引用、新 `result` 有 deep-const | 当前用 `reference_wrapper`/pointer/span/string_view | future error | borrowed value 不延长 owner lifetime | 只有具名 non-null borrow consumer 才重开 | `DEFER` | P2 | 不为 API 数量另造 `expected` |
| ERR-DETAIL-01 | GENERAL | backend gap scan | future Metal/HIP/foreign category | 新稳定 domain 或第二 typed detail | owning backend/future error | enum 防碰撞；跨 DLL category pointer 不进入 carrier | 不预装 map/第二 block | `DEFER` | P2 | 真实 backend/两个同时恢复字段触发 |
| ERR-SOURCE-01 | GENERAL | Abseil breadth scan | Abseil message Status location chain | HYLE 保持显式语义 context；未来可 opt-in builder | future obs/error | 不给每个 error 增静态地址、传播噪声或 vector | 无常态 source metadata | `REJECT` current / `DEFER` opt-in consumer | P2 | source/backtrace 属诊断 owner |
| ERR-PAYLOAD-01 | GENERAL | Abseil/Folly breadth scan | Abseil arbitrary payload；Folly rich error/provenance | 一个 native detail + caller-owned aggregate 已覆盖当前需求 | external/future | payload schema、序列化、DLL lifetime、OOM 责任未成立 | 不增加线性 payload map/常态 allocation | `REJECT` | P2 | reference API 数量/LOC 不构成需求 |
| ERR-EXCEPTION-01 | GENERAL | Folly breadth scan | `exception_wrapper`、rich error、Abseil builder logging | operational failure 继续走值；异常在拥有第三方/callback 的边界转换 | owning module | 不跨 noexcept/模块边界传播 exception ABI/RTTI | 不增加隐式分配/重抛路径 | `REJECT` | P2 | error 与 obs 保持解耦 |
| ERR-FFI-01 | MUST | delivery policy | `native-ffi-policy.md`；当前零具名 foreign consumer | C++ source/static/shared；Native FFI=`N/A` | build/product | 不导出模板/STL/Status layout；未来按 journey 类型擦除 | 零伪 bridge/symbol/option | `KEEP` | P0 | export/product-tree scan |
| IO-ADOPT-01 | GENERAL | downstream adoption debt | io proactor/net/dns/shared-memory/TLS/TCP/UDP 的 bool/sentinel/native errno 协议 | 热 completion 可保留紧凑 native value；cold factory/control seam 在 io 轮裁决 Result/Status adapter | io | 不在 error carrier 中吸收 async state machine | 由 io 正常 workload 决定 | `DEFER` | P1 | error 本轮不得宣称全库统一；登记 io closure MUST |
| SC-ADOPT-01 | GENERAL | downstream adoption debt | sched cancellation/channel 与 owning container cold factory 的 invalid/OOM 形状 | 在各模块轮裁决 `create()->Result<T>`、try leg 或 fail-fast | sched/container | 协议 empty/full 与 operational OOM 分开 | 不给热结构强加 Status | `DEFER` | P2 | 不跨 owner 顺手重构 |

- 公开入口反向追踪：`StatusCode`/factory/predicate、message/native/context、copy/move/swap/equality/
  `Update`/`IgnoreError`、两种格式化、三组 OS adapter、`Result<T>`/`Result<void>`、`make_unexpected`
  与两个宏均落入上述 MUST 行；没有无 owner 的 public entry。
- 语法消费者 / 语义重复扫描：底层已审消费者正确使用 C++ Status/Result；`io` 以及少数 sched/
  container cold factory 仍有独立协议，只登记为对应模块 backlog。正常 empty/full/pending/native
  completion 不机械改成 Status。
- public/internal/higher-layer 裁决：error 只拥有 immutable failure value、公共 taxonomy 和最小 adapter；
  日志/trace、异步状态机、backend-specific mapping、聚合排序和异常捕获属于上层 owner。
- 本轮冻结 `FIX` 集：`ERR-DEFAULT-01`、`ERR-INCLUDE-01`、`ERR-WORDING-01`、
  `ERR-PERF-REPLAY-01`、`ERR-PERF-NORMAL-01`、`ERR-PERF-BOUNDARY-01`、
  `ERR-PERF-VALIDATOR-01`、`ERR-DOC-01`。**Status/Result 产品 API、layout 与实现算法的 FIX 集为空。**
- 相邻模块 backlog：`IO-ADOPT-01`、`SC-ADOPT-01`；不得据此替尚未收口模块签署 adoption。
- 审计身份：HYLE baseline 如 closure card；Abseil/Folly pins 如 closure card；三路只读审计分别覆盖
  peer/能力与复杂度、全库消费者/交付/FFI、性能合同/仪器。Boost superproject 的 System/LEAF/
  Outcome 子模块未初始化，未冒充本轮可读证据。

## Candidate

当前产品候选固定为：

- commit：`462db32c9b6a364f233a89a2da541c2daca37bea`
- tree：`9e886667c861a43bd9b9e6c83cf055c16a3ed54a`

候选没有修改 `Status` / `Result<T>` 的产品 API、layout 或算法；peer/复杂度审计没有找到需要产品
修改的 MUST 缺陷。它只实施冻结 FIX：明确并测试 `Result<T>{}` 的标准成功语义；补齐公开头直接
include 和过时 FFI/C bridge 措辞；补回 UTF-8、embedded-NUL、native detail、context 与 caller-buffer
格式化的完整 C++ 字节旅程；重写 architecture/user/performance 当前文档。

性能合同从会被 DCE 或落入 timer floor 的常量/原语矩阵，收缩为 9 个正常消费者 replay：64 节点
active function-pointer stream 的全成功/末节点 code-only failure，以及 29B message construct/copy、
context 1/3、两 frame message Result、54B owning format、fresh-prvalue Update。旧 short/boundary/native
行仍可作诊断，不进入 P1 verdict。窄 validator 对同一 primary/placement binary 的实际 replay 函数执行
64-call/checksum/语义/source-location oracle，并验证 noinline identity 与 placement 真实改变；统一统计
核心没有分叉。

候选前的 dirty probe 只证明合同可运行，不是候选证据：9/9 workload 通过 batch=1 的 20x timer-floor
门（最小 55.43x，最大 391.17x），semantic/call-topology/placement validator 通过；n=1 的 A/A 与 delta
控制按设计保持 FAIL/DIAGNOSTIC。正式性能判决必须从上述 clean candidate 重新运行。

## Verification manifest

尚未运行 candidate 验证。

## 独立红队

尚未对 candidate 签署。

## Closure decision

当前为 `AUDITED`；只允许实施上述冻结 `FIX` 集，不得提前写入 candidate 或 terminal 结论。
