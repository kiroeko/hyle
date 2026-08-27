# error v3 重构与收口记录

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

<!-- MODULE CLOSURE RECEIPT: BLOCKED -->

> **记录性质：`HISTORICAL TASK EVIDENCE`。** 本页只记录 `error-v3` 这一轮的冻结合同、事实和证据，
> 不单独声明当前状态。权威状态是
> [`module-status.json`](../../module-status.json)；旧 error task 与性能页均为历史输入，不能替代本轮
> 结论。本轮已在 candidate `9595724a` 终止为 `BLOCKED`；下文 `SCOPED/AUDITED/CANDIDATE`
> 标题保留的是阶段历史，不表示当前仍活动。
>
> **2026-08-27 post-terminal evidence correction：**后续审计发现本轮把固定 backing 的一次
> retain/release microbench 错标为 consumer replay，并在另一 required row 中给予 HYLE rvalue move、
> Abseil lvalue copy 的不对称 ownership。`raw samples/statistics retained`；原合同据此派生的 comparison/
> product verdict rejected。当前 blocker 是缺少有效的首席 source-consumer 性能合同，而不是已证明产品
> 慢 8%。

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

## Capability / gap ledger（历史 `AUDITED` 合同；性能分类已被下文纠正）

本表保留 terminal round 当时冻结的判断，供追溯产品改动来源；其中 `ERR-PERF-*` 的 consumer/
replay 资格不是现行结论，必须连同文末 post-terminal correction 阅读。功能、API、正确性与交付行
仍有效。

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
| ERR-PERF-RESULT-SETUP-01 | MUST | candidate red-team defect | `result-message-error` 宣称 callback 外预建 error，首个候选却通过函数内 lazy static accessor 取得 | 两侧 owning error 在 measured callback 外显式构造，并沿两 frame 以 const reference 传入 | error benchmark | 不计费 lazy initialization、guard 或隐藏 accessor；两侧参数拓扑对称 | 不改变产品，只校正 charged phase | `FIX` | P1 | replacement validator/构建；拒绝首个 candidate 与其已启动样本 |
| ERR-PERF-TAG-01 | MUST | original-contract qualified consumer + peer/codegen defect（历史） | shared owning-message copy；固定 Abseil low-bit inline representation | OK/code-only 都是 low-bit tagged inline rep，aligned heap pointer 的 low bit 为 0 | error public | 不改变一字 carrier、ownership、分配或公开能力；内部 C++ ABI 尚未发布 | retain/release 判别 O(1)，MSVC 正常复制循环与 peer 使用同形 bit-test | `FIX` | P1 | original-contract computed：replacement n=15 四 strata 均约慢 8%；汇编为两次 range-compare，对手为 bit-test；采用成熟 tagged-pointer 不增加 benchmark metadata |
| ERR-PERF-TRANSFER-01 | MUST | original-contract qualified direct-consumer/codegen defect（历史） | `TaskGraph::evaluate` 的局部 `Status` 在失败即返回 | 生命周期结束的局部错误显式 move；外部/member lvalue 仍由通用宏复制 | graph integration + replay | 不改变 `HYLE_RETURN_IF_ERROR` 的 lvalue-copy/rvalue-move 合同；失败后不再读取局部 | success loop 与直接 `if/return` 同形；heap error 不做无意义 retain/release | `FIX` | P1 | original-contract computed：replacement success 四 strata 约慢 6%；MSVC 133B lvalue 形状改为 104B move 形状，与 direct 相同；同步真实 TaskGraph 与 replay |
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
  `ERR-PERF-VALIDATOR-01`、`ERR-PERF-RESULT-SETUP-01`、`ERR-PERF-TAG-01`、
  `ERR-PERF-TRANSFER-01`、`ERR-DOC-01`。setup 项来自首个 candidate 的独立红队；后两个产品项是原合同
  当时按 n=15 computed output 与唯一 codegen 诊断加入的历史施工来源，范围限于成熟 low-bit tag 和直接
  消费者的局部所有权转移，不授权新增 API、metadata 或 workload，也不代表当前产品性能裁决。
- 相邻模块 backlog：`IO-ADOPT-01`、`SC-ADOPT-01`；不得据此替尚未收口模块签署 adoption。
- 审计身份：HYLE baseline 如 closure card；Abseil/Folly pins 如 closure card；三路只读审计分别覆盖
  peer/能力与复杂度、全库消费者/交付/FFI、性能合同/仪器。Boost superproject 的 System/LEAF/
  Outcome 子模块未初始化，未冒充本轮可读证据。

## Historical terminal candidate

原轮唯一产品性能假设当时形成的 terminal candidate：

- commit：`9595724a0a284cbc3502f1c9e9e3ec9c9cf9e2b9`
- tree：`a828fb4cdf1cd7f72aaea098b44fd42e0d966d2d`

它保持公开 API、8B carrier、heap metadata、错误能力和 9 项性能合同不变，只实施两个由原合同当时的
computed failure 与唯一 codegen 诊断触发的修复：借用成熟 peer 的 low-bit inline tag（OK/code-only 为
odd inline、aligned heap pointer 为 even），并在真实 TaskGraph 与 replay 中显式移动失败即返回的局部
`Status`。0 继续只是内部 allocation failure sentinel，`StatusAccess::adopt(0)` fail-fast；moved-from
仍为 code-only `Unknown`。这是 header-inline semantic ABI 变化，必须使用匹配版本 header/artifact 并
全量重编，不继承 replacement 的 build/package artifact。

该产品实现保留：low-bit tag 是成熟的 aligned-pointer 不变量，移除了“heap 地址必然大于某个小整数”
这一非语言保证，并使常态判别更简单；TaskGraph 的显式 move 则准确表达失败即返回局部值的所有权。
原合同的后续 raw observations 显示 fixed-fixture shared-copy ratio gap 仍存在；它不证明该差距对首席
consumer material。因而保留架构修复，同时不把它写成性能修复成功，也不再授权第二个产品猜测。

## Historical candidate verification manifest（原合同 raw/computed 记录）

本节冻结 exact artifact、raw samples/statistics 与 original-contract computed fields，供 hash replay 和
审计。raw samples/statistics retained；由旧 workload 分类派生的 comparison/product verdict rejected，
不得把下表单独摘作当前资格或产品 `FAIL`。

- identity：HYLE `9595724a0a284cbc3502f1c9e9e3ec9c9cf9e2b9` / tree
  `a828fb4cdf1cd7f72aaea098b44fd42e0d966d2d`；Abseil
  `81c993af4fb88b3a51829a796df02251fa27102f` / tree
  `b8e8bdc41d9f13b8894b4540b910ed8dc2ced66c`；两侧 clean。
- W1-WIN / MSVC 19.44：exact candidate 的 Release/Debug WERROR error CTest 各 2/2；header
  self-containment 通过。MSVC 19.50 的 P1 static artifacts 由下述 manifest 绑定。
- W1-WIN / clang-cl 20.1.8：shared Release WERROR 完整 19/19 CTest 通过。
- W1-WIN delivery：selected-component source/shared/static relocated consumer 3/3；shared DLL SHA-256
  `947313e90b284fb35e295c82f8246c1fbb29bb26df3e949dde1b08d0f6d983fc`，只含 18 个 decorated
  C++ exports，旧 C error/FFI exports 为 0。
- W1-WSL / GCC 14.2：Release/Debug WERROR 各 2/2，完整 error 69/69，Debug misuse 3/3（4 个
  death assertions），ASan/UBSan 各 2/2，TSan 在 `setarch x86_64 -R` 下实际执行 1/1；source/shared/
  static relocated 3/3；graph build 与三个 `evaluate/evaluate_node` failure-propagation tests 3/3；
  warning、sanitizer diagnostic、skip 均为 0。
- M1-MAC / AppleClang 21：Release/Debug WERROR 各 2/2，完整 error 69/69，source/shared/static
  relocated 3/3；graph focused 3/3、完整 graph 19/19；shared exports 18 个 C++、旧 C/FFI 为 0；
  warning、skip、failure 均为 0。
- 文档/API：architecture 与 user guide 已覆盖能力选型、`Result<T>{}`、lvalue-copy/rvalue-move、
  context/OOM/native/格式化、ownership/concurrency、low-bit 交付 ABI 限制和 Native FFI=`N/A`；
  public entry 反向追踪完整。

W1-WIN P1 retained evidence 位于
`C:/hyle-scratch/error-v3-final-win-perf-evidence`。manifest SHA-256 为
`4cd3627d134be6d44835c7ac976554a3f65d963532af155fd28b21044a728b85`，summary SHA-256 为
`68dc98addc69a00007a1be0f88a605a495d3c4fa120a666646fa1c5b817740c6`；contract SHA-256 为
`82cf325480a85d7ea76e17fcf526d12c36619b7089ec81d4c947fa56aea215d7`。primary/placement
artifact SHA-256 分别为 `43a9e28636a24a1a6179bd47e6d90bb772ab5a7ae8cab08ad4ca7b3298686926` /
`df54840477ee6b5c1f0f146ed01d2114244c89493d5ee25ffeb541e47e31e3fa`。统一 verifier 返回
`paths_verified=true`、`qualification=QUALIFIED`；A/A、delta 5/10、timer-floor、placement、
artifact/build/semantic/codegen controls 全部 `PASS`。这证明 raw samples 忠实执行了原合同，不证明原合同
具备 consumer/materiality 产品资格；以下 WIN/LOSE/PASS/FAIL 均为 original-contract computed output。

| 原合同声明的 workload | HYLE / peer 中位 ns | original-contract computed exact | original-contract computed 5% NI |
|---|---:|---|---|
| `taskgraph-active-success-64` | 79.119 / 76.388 | `INCONCLUSIVE` | `UNRESOLVED`（4 strata 中 2 PASS、2 UNRESOLVED） |
| `taskgraph-active-last-error-64` | 78.644 / 77.716 | `PARITY` | `PASS` |
| `status-message-29` | 23.821 / 48.054 | `WIN` | `PASS` |
| `status-message-copy` | 11.841 / 10.923 | `LOSE` | `FAIL`（4 strata 全 FAIL，ratio 0.920057/0.925041/0.920034/0.920214） |
| `status-context-1` | 41.891 / 265.869 | `WIN` | `PASS` |
| `status-context-3` | 74.750 / 879.920 | `WIN` | `PASS` |
| `result-message-error` | 10.892 / 21.971 | `WIN` | `PASS` |
| `status-format-54` | 34.599 / 43.811 | `WIN` | `PASS` |
| `status-update-prvalue` | 48.148 / 54.826 | `WIN` | `PASS` |

原 runner 的最小 timer-floor multiple 为 53.96x，因此 fixed-fixture raw gap 不是 timer-floor 噪声。
按原合同 stopping rule，当时 computed `FAIL` 也禁止从 n=15 自动升级 n=31；这两项都不构成当前
产品 `FAIL`。

## Historical rejected candidates

第二个（replacement）candidate 为：

- commit：`a4ba5a14cdd46a7b9c6e33c06211ce2d6c7a760f`
- tree：`4a2ca7977241080a41b5eb5ab79a6fd37d71db04`

它在首个候选的冻结产品形状上，只修正 `ERR-PERF-RESULT-SETUP-01`：两侧 owning error 在 measured
callback 外显式构造，沿两个 noinline frame 以 `const&` 传入；validator 使用内容不同的外部 sentinel，
证明 frame 消费传入 fixture，而不是隐藏 static/accessor。产品代码、公开 API、layout 与性能合同均未改。
原合同当时由 W1-WIN n=15 computed output 驳回该 candidate：9 项中 7 项 WIN、末节点 code-only error
通过非劣门，但
`taskgraph-active-success-64` 四 strata 的候选优势中位比为 0.940--0.942，
`status-message-copy` 为 0.917--0.923，均未通过 5% non-inferiority gate。A/A、delta、timer-floor、
placement、semantic 与 error-codegen controls 全部通过，说明 raw samples 不是仪器随机噪声；旧分类所
派生的 candidate/product verdict 已被文末 correction 拒绝。

首个候选已被拒绝：

- commit：`462db32c9b6a364f233a89a2da541c2daca37bea`
- tree：`9e886667c861a43bd9b9e6c83cf055c16a3ed54a`

该候选没有修改 `Status` / `Result<T>` 的产品 API、layout 或算法；peer/复杂度审计没有找到需要产品
修改的 MUST 缺陷。它只实施冻结 FIX：明确并测试 `Result<T>{}` 的标准成功语义；补齐公开头直接
include 和过时 FFI/C bridge 措辞；补回 UTF-8、embedded-NUL、native detail、context 与 caller-buffer
格式化的完整 C++ 字节旅程；重写 architecture/user/performance 当前文档。

原合同当时把 9 行标作“正常消费者 replay”：64 节点 active function-pointer stream 的全成功/末节点
code-only failure，以及 29B message construct/copy、context 1/3、两 frame message Result、54B owning
format、fresh-prvalue Update。post-terminal correction 后，copy 行只具 mechanism-micro 资格，Result 行
存在 ownership 不对称；其余 synthetic rows 也没有自动取得首席 consumer provenance/materiality。窄
validator 忠实执行原合同声明的函数并验证 64-call/checksum/语义/source-location oracle、noinline identity
与 placement 真实改变；它没有验证上述 consumer/ownership 合同，统一统计核心本身没有分叉。

原合同下候选前的 dirty probe 只证明合同可运行，不是候选证据：9/9 workload 通过 batch=1 的 20x timer-floor
门（最小 55.43x，最大 391.17x），semantic/call-topology/placement validator 通过；n=1 的 A/A 与 delta
控制按设计保持 FAIL/DIAGNOSTIC。

原轮独立红队随后发现 `result-message-error` 虽在合同中声明 prebuilt error 位于 callback 外，首个实现却由
measured leaf 内的函数局部 lazy static accessor 取得；首次初始化和每次 guard/accessor 都没有列入
charged phases。即使一次初始化会摊薄、两侧也近似对称，这仍是证据合同违约，不能用性能点值解释掉。
已启动的 clean n=15 立即终止且全部作废；它不消耗 replacement candidate 的唯一 fresh qualification。
replacement 必须显式在 callback 外构造两侧 error，并以对称 const reference 沿两 frame 传入。

## Historical rejected replacement verification manifest（原合同 raw/computed 记录）

本节的 `QUALIFIED`/`PASS`/`FAIL` 是原合同字段，不是当前产品资格；raw samples/statistics retained，
derived comparison/product verdict rejected。

- identity：HYLE `a4ba5a14cdd46a7b9c6e33c06211ce2d6c7a760f` / tree
  `4a2ca7977241080a41b5eb5ab79a6fd37d71db04`；Abseil
  `81c993af4fb88b3a51829a796df02251fa27102f` / tree
  `b8e8bdc41d9f13b8894b4540b910ed8dc2ced66c`；两侧 clean。
- W1-WIN：contract `82cf325480a85d7ea76e17fcf526d12c36619b7089ec81d4c947fa56aea215d7`；
  `qualification_gate=PASS`、`performance_gate=FAIL`；统一 verifier 返回 `QUALIFIED`。这些是
  original-contract computed fields，只证明 bundle identity/integrity 与声明的执行门；不证明产品未通过
  当前工程性能门。primary artifact `8d9179d608976f02582c414e0d40f77439bd1786423ea3ec19cded04322b71a1`，
  placement artifact `85def6f9e3018a8b0e38bdf8566f3f03f604f942d2827ba9b5a66babe963d533`。
- M1-MAC：同一 identity 的 Release/Debug WERROR、68 gtests、header 与 source/shared/static relocated
  consumers 通过；shared DSO 只有 18 个 C++ exports、旧 C error symbols 为 0；P1 semantic/placement
  validator 通过，未采性能数据。
- W1-WSL：因 W1-WIN 是唯一性能裁决机，Windows 机制诊断期间主动停止并发 sanitizer/build；新
  candidate 形成后重跑，不继承未完成产物。
- 唯一 codegen 诊断：shared-copy 循环只有 HYLE `cmpq $0x41` 与 Abseil `testb $1` 的标签判别差异；
  low-bit inline tag 的临时编译使两循环逐指令同形。TaskGraph 局部 lvalue 传播为 133B，并在 heap
  failure 做 retain/release；显式转移该生命周期结束的局部后为 104B，与直接 `if/return` 同形。

## Original red-team finding（已被 post-terminal correction 取代）

以下红队结论准确重放了原 artifact/raw statistics，但遗漏 consumer provenance/materiality 与 Result
value-category 对称性；只作历史审计记录。产品/API、correctness、delivery 与文档审计通过。性能证据
审计重算并核对 candidate/peer/contract/
artifact identity、90/90 raw logs、summary/observations/controls/validator hashes，结果一致；统一 verifier
按原合同仍为 `QUALIFIED`。红队当时确认 low-bit tag 与 TaskGraph ownership transfer 都不是
benchmark-only 形状，
但汇编循环同形后 shared-copy 仍约 0.92，只能证明先前 codegen 假设不足，不能推出第二个产品缺陷，
也不能在 controls 全过时宣布仪器失效。

原合同没有辨识动态 heap backing 的实际 allocation 地址、cache-line placement 与 allocation-order
sensitivity。这是原红队当时提出的下一轮仪器建议，不授权本轮改变 copy/destruct charged phase、公开
API/layout/metadata、workload 或 5% margin；现已由 `ERR-PERF-CONSUMER-KEY-01` correction 取代，不是
当前 resume condition。

## Original closure decision（已被 post-terminal evidence correction 取代）

以下是原合同下当时实际作出的流程决定，不能作为当前产品性能结论；现行 blocker、owner 与恢复条件
以文末 correction 和中央 registry 为准。

`BLOCKED`，从 `CANDIDATE` 终止，绑定 `9595724a` / `a828fb4c`。

- 原合同当时记录为通过：范围/功能/API/架构、三平台 correctness/sanitizer、直接消费者、source/static/
  shared 交付、用户与架构文档、独立红队；9 个 P1 workload 的 original-contract computed output 为
  6 个 `WIN`、1 个 `PARITY`。
- 原 blocker `ERR-PERF-NORMAL-01 / status-message-copy`：四个 order/placement strata 的 raw ratios 稳定，
  原合同 computed `LOSE` 且 5% NI `FAIL`；`taskgraph-active-success-64` 另有两个 strata `UNRESOLVED`。
  “因此产品不能签署工程性能 `PASS` 或 SOTA”是已被 correction 拒绝的 original-contract product verdict。
- blocker owner：独立 performance-instrument owner；只有下述新证据成立后，future error resume owner
  才能接手产品工作。
- 恢复条件：先在独立仪器 task 中，在不改变 shared copy/destruct charged phase、产品 API/layout/
  metadata 的前提下，加入 fail-closed 的实际 heap address/cache-line 与 allocation-order sensitivity
  诊断并形成因果分类。只有它证明 placement/instrument confound 并冻结新版合同，或点名一个具体产品
  机制并允许新 round 的一次范围化修复，才能创建 `error-v4` 从 `SCOPED` 恢复。单独跑 n=31、换 workload/
  顺序、放宽 margin 或第二次产品猜测都不满足恢复条件。
- 已知限制：low-bit header-inline semantic ABI 仍要求匹配 header/artifact 并全量重编；当前没有稳定 C++
  ABI 或 Native FFI；io 与少数 sched/container cold factory 的 adoption 仍由所属模块裁决。
- 四项 LOC（candidate `9595724a` immutable tree）：全库 293,909；自实现含测试 122,027；自实现不含
  测试 82,135；error 实现 1,350。
- reopen/resume trigger：上述窄性能因果证据；或新的可复现 correctness/security/consumer/API/交付
  缺陷、直接依赖/编译器改变 carrier 语义、强 peer 证明更好复杂度档。新 API 数、LOC、命名偏好或
  单次 benchmark 点值不构成 trigger。

本轮在这里停止；保留候选产品改进，但不把 `BLOCKED` 写成 `CLOSED`，也不继续用 error 消耗无边界
调研预算。

## Post-terminal evidence correction（2026-08-27）

- correction red-team finding / review date：`ERR-PERF-CONSUMER-KEY-01` / `2026-08-27`
- finding class / disposition：`METHOD_GAP / ACCEPTED`；raw samples/statistics retained，derived comparison/
  product verdict rejected。

产品 owner 明确首席消费者是未来推理引擎在同一 build graph 中直接编译 HYLE 源码。按
[`benchmark-evidence-standard`](../../../developer/performance/benchmark-evidence-standard.md) 的
“消费者 workload / 组件 replay / 机制 microbench”分层和机制、消费两把钥匙重新审计后，发现原
P1 合同虽然统计与 artifact identity 合格，但不具备产品裁决资格：

1. `status-message-copy` 在每个进程内反复约 2,000,000 次复制同一个预建、cache-hot backing；它没有
   真实 consumer trace、调用频率、冷热/并发/lifetime 分布或端到端贡献预算。HYLE source 始终先
   allocation、Abseil 后 allocation；双执行顺序不反转 fixture 构造，placement control 也只改变代码
   布局，没有记录或平衡 heap address/cache-line/page offset。该行是 mechanism microbench，却在
   `error-p1-v1` 中被自报为 `replay` 并错误获得产品否决权。
2. `result-message-error` 的 peer ownership 不对称。HYLE 宏从即将销毁的临时 `Result` 移出 error；
   Abseil replay 使用 `return result.status()` 从命名 lvalue 复制，尽管固定 peer 已提供
   `StatusOr::status() &&`。absolute/raw samples retained；原合同 computed 约 2x `WIN` 及其 comparison/
   product verdict rejected，不能作为真实传播优势。
3. validator 证明了输出语义、call topology、静态 placement、DCE 与统计控制，却没有验证 workload
   的 consumer provenance、material contribution 或 value-category/retain-release 对称性。合同冻结
   只阻止测后换姿势，不能把测前定义错误变成产品事实。

流程根因不是缺少书面标准，而是标准没有进入机器合同：当首席 source consumer 尚无可核对 trace 时，
施工流程用“常见 API 操作”填补了 consumer evidence 空位；`tier=replay` 又是 workload 自报字段，
runner 对所有 `COMPARABLE` 行一律聚合，因而严谨地执行了一个语义上无资格的合同。正确的 fail-closed
动作本应在采样前就是 `INCONCLUSIVE/BLOCKED`。

修正后的裁决如下：

- candidate 产品/API/correctness/delivery 结论不变，不回滚 low-bit representation 或 TaskGraph move。
- `status-message-copy` 的四 strata 约 0.92 只保留为“固定 fixture 上稳定、机制未知”的诊断；它既不是
  随机噪声，也未证明产品缺陷或 compact representation 的必要取舍。
- `result-message-error` 的 absolute/raw samples retained；ownership 不对称使 original-contract computed
  `WIN` 与 comparison/product verdict rejected，不得继续引用约 2x 产品优势。
- v3 engineering performance 从 `FAIL` 更正为 `INCONCLUSIVE`。模块仍 `BLOCKED`，但 blocker 改为
  `ERR-PERF-CONSUMER-KEY-01`：缺少以首席 source consumer 为中心、ownership 对称且有 material
  contribution 的有效性能合同，而不是“产品已证慢 8%”。
- 恢复不要求先改 Status 产品。新性能合同必须绑定同一父 build graph 的 source-primary profile、自然
  ownership、真实或保真 trace 的分布/lifetime/冷热/并发及贡献预算；mechanism microbench 只能解释
  原因，不能独立 PASS/FAIL 产品。若新 consumer 证据显示 material regression，才重新进入产品归因与
  有限 FIX。

这是 terminal evidence 字段的审计纠正，不倒写原始 raw manifest；raw samples/statistics retained，
derived comparison/product verdict rejected。它也不改变 product candidate、
receipt/end commit 或 registry product-identity anchor。原 `Closure decision` 记录当时按错误合同作出的
流程裁决；本节与中央 registry 是当前解释。
