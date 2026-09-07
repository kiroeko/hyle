# Container v6：Container Text 身份与 P1 证据恢复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`AUDITED`）

- components / owner / rounds：
  - `container` / Container / `container-v6-identity-p1-resume`；
  - `container_concurrent` / Container / `container-concurrent-v2-p1-resume`；
  - `container_queue` / Container / `container-queue-v2-p1-resume`；
  - `text -> container_text` / Container / `container-text-v2-identity-p1-resume`；
- task kind：`RESUME`；前轮为 `container-v5-product-factorization`、
  `container-concurrent-v1-factor`、`container-queue-v1-factor`、`text-v1-factor`；
- baseline：`d04710a33319e4f1088d81608bc5a7bd031b8eef`。产品实现仍是前轮第三候选
  `6fae3c11ebce9a0741f45780e63c9896e08bb2a2` / tree
  `8025ee111380c2600f804cf979ed59d289a8be76`；其 correctness/delivery 证据只在产品边界未变时复用；
- previous blocker：LRU、concurrent membership、unbounded MPMC、Unicode 四个 P1 cell 没有先等价化
  allocator、OOM、rollback、result 等合同，旧数字只是 `NATURAL_ROUTE` diagnosis；
- previous resume condition：逐 cell 冻结 `EQUAL/ADAPTER_EQUALIZED/INTENTIONALLY_DIFFERENT` 矩阵，
  用具名 adapter 真正等价后在 immutable candidate、W1-WIN、5% margin 下取得 qualified verdict；
- new owner decision `CONTAINER-TEXT-IDENTITY-01`：内部 factor 的真实 owner 是 Container，预发布时将
  `text/hyle_text` 完整迁移为 `container_text/hyle_container_text`，不保留 compatibility alias；受控
  registry 迁移合同见
  [`Container Text factor 身份迁移`](../build/container-text-factor-identity-migration-v1.md)；
- scope：迁移内部 factor/component/target/source/test/package detail 与所有直接边；保持公开
  `hyle/container/*`、`hyle::container` 和用户 `hyle::container` façade；修正四条 benchmark adapter 和
  comparison metadata；重放受影响 correctness、direct consumers、source/static/shared、symbols、docs、
  P1 与独立红队；
- non-goals：新 Container 能力、公开 API/layout/算法/默认 allocator、常态 metadata、推理策略、
  tokenizer/KV/scheduler/IO service、Native FFI/C API、稳定 C++ ABI、公共性能仪器或 margin 修改、Formal；
- direct dependencies：sequential `PUBLIC core,mem,container_intrusive / PRIVATE platform`；concurrent
  `PUBLIC core,mem,mem_reclaim,container`；queue `PUBLIC core,mem`；container_text
  `PUBLIC core,mem,container / PRIVATE platform`；
- direct consumers：Container façade；`util -> container_text`；`io -> container_text`；Sched 私有使用
  `container/container_queue`；package sample 与四 factor tests；
- frozen inference demand identities：llama.cpp
  `3ad1ba7336986d98592d3e28cafd1a406715351f`、standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`、ds4
  `9ab705347c1775e7599ede7eb81a6255ec7dccb5`、vLLM
  `52358e6e192aeae73dd3764046c98fc4b156ea83`、SGLang
  `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110`。本轮只确认前轮 capability map 没有因身份迁移改变，
  不重新求 API 并集；
- strong peers：LRU 为 Abseil `38f8c1535de9ebc12c6340e5c1c8ebd72afde560`；membership/queue 为
  oneTBB `e00c68dcac0e83947c9b54f6d5304c0616c9639f`；Unicode mechanism 为 HYLE dependency
  lock 对应的 simdutf `2d4e249dcfd978b39dd567b156f20a43db97377b`。每个 cell 只有一个 peer；
- tier / machine / instrument / margin：P1；W1-WIN primary，W1-WSL/M1-MAC correctness/delivery；
  `closure-perf-v1`；5%；
- budget：identity/audit 45 分钟，迁移与 build 45 分钟，P1 90 分钟，三平台/交付 90 分钟，红队
  30 分钟；性能最多一次正式 qualification、一次机制检查、一次内部产品 correction 和一次重跑；
- stop：任一 comparison 仍非同合同、qualification 无 verdict、material regression 无法在一次机制检查/
  correction 内解决，立即以具名 blocker 终止；不得扩建第五个 workload、修改公共仪器或用有利
  natural-route 点值覆盖失败。

## 冻结 normal workloads 与语义矩阵（`FROZEN`）

四条 primary workload、输入规模、线程数、operation mix、计时 phase、双顺序/placement 与 checksum
沿用前轮 workload identity；本轮只修 adapter/metadata 中被红队证明不等价的格。以下矩阵于
2026-09-07 冻结；任何输入规模、operation mix、线程角色、计时 phase、产品 API 或产品算法变化都必须
退回 `SCOPED/AUDITED`，不得用新的有利姿势替代旧 blocker。

| Cell | comparison class | candidate minimum contract | peer adapter 必须补齐 | 不计入 steady phase |
|---|---|---|---|---|
| entry+weight LRU hit / insert+evict | `ADAPTER_EQUALIZED` | recoverable allocation result、失败不改变 cache、相同 touch/evict/checksum | 两侧使用 `SystemAllocator` 路由与同一 `StableHash`；peer 只捕获 `bad_alloc`，在 index failure 后回滚 list，并返回与 HYLE 同构的 stored/OOM/evicted 结果 | object/resource setup、reserve、input generation、final checksum |
| read-mostly membership | `ADAPTER_EQUALIZED` | concurrent contains/insert/erase result、相同 key/load 生命周期 | HYLE 与 oneTBB set 都使用同一 `StableHash`、equality 与 `SystemStdAllocator`；reserve/populate/warmup 相同且在计时外 | table/resource construction、thread creation、final verification |
| unbounded typed MPMC | `ADAPTER_EQUALIZED` | try push/pop、recoverable OOM、FIFO conservation、相同 producer/consumer participation | 两侧使用 `SystemStdAllocator`/`SystemAllocator` 的同一底层 owner；oneTBB throwing push 只捕获 `bad_alloc` 并映射为 false；相同 setup/warmup/drain | queue/resource construction、worker creation、final conservation |
| safe UTF-8→UTF-16 caller buffer | `ADAPTER_EQUALIZED` | validate、typed status/input offset/output count、exact caller buffer、相同 valid/error mix | peer 固定到 vendored simdutf 同一 revision，返回同构 status/input offset/output count 而不 abort；validator 覆盖 malformed 与 output-too-small；steady valid corpus 仍执行相同三趟 safe route | corpus construction、buffer allocation、final checksum |

若 peer 的 allocator template 或 API 使同一 resource 无法合法注入，该格必须写
`INTENTIONALLY_DIFFERENT` 并把 cell 降为 `NATURAL_ROUTE`；此时本轮没有预冻结独立 consumer budget，
工程 P1 不能因此通过。

## Capability-family carry-forward

前轮已经把四组推理需求和成熟 peer 差异映射为 sequential sequences/maps/hash/bitset/LRU、concurrent
map/set/vector、role-specific/bounded/unbounded queues、borrowed text/path/interner/Unicode。所有已准入
`MUST` 在第三候选为 `COMPLETE` 或完整 `ROUTED`，终局红队没有发现新缺口。本轮身份更名不改变
semantic owner、公开 surface、生命周期、复杂度或负空间；若审计发现任何新的 `PARTIAL/MISSING`，必须
在进入 `AUDITED` 前明确加入有限 FIX，否则本轮只保留：

| ID | class | coverage | action | verification |
|---|---|---|---|---|
| `CONT6-F01` | build/product identity | `PARTIAL`：factor owner 正确但内部 ID/target 过宽 | `FIX`：完整迁移为 `container_text`，无 alias | component graph、headers、direct consumers、三交付、symbols |
| `CONT6-F02` | P1 comparison contract | `PARTIAL`：产品完整，peer adapter/metadata 不等价 | `FIX`：按四行矩阵等价化；不改产品 API | independent pre-sample review、qualification、manifest |

当前没有第三个产品 FIX 授权。

## Audit verdict（`AUDITED`）

Top-down public trace 与 bottom-up ownership scan 得到以下冻结结论：

- `path.h`、`string_algo.h`、`string_interner.h`、`string_view.h` 的公开 include 与全部类型仍位于
  `hyle/container`、`hyle::container`；错误只存在于内部 factor/target/directory/package identity。更名
  不准移动公开头、改 namespace、API、layout、算法或 provider；
- 真实直接消费者是 Container façade、`util` 的 Registry 和 `io` 的 HTTP/TLS surface。中央图必须把
  `util/io -> text` 改为 `util/io -> container_text`；其他源码消费者只通过公开头或 façade，不新增边；
- 当前 LRU peer 已使用相同 hash 与 System allocator，但其 throwing insert 缺 OOM 映射和 index-failure
  rollback；membership peer 还使用不同 hash/allocator；queue peer 使用 oneTBB 默认 allocator 且把
  throwing push 无条件当成功；Unicode peer 终止而非返回同构 typed result，且 task 一度错误记录了
  本地 simdutf HEAD 而非产品 lock revision。这些都是比较器合同缺陷，不是产品缺陷；
- frozen workloads 的输入、角色、计时 phase 和产品实现不变。`CONT6-F02` 只允许 benchmark-only adapter、
  新版本 contract/shape metadata 与 validator；不得以异常处理、allocator 或 hash 差异继续做
  `SAME_CONTRACT` 声明，也不得因数字修改产品 API/layout/metadata；
- factor identity 执行完整一对一迁移：`text -> container_text`、`hyle_text -> hyle_container_text`、
  `src/text -> src/container_text`、`tests/text -> tests/container_text`、`HYLE_TEXT_* ->
  HYLE_CONTAINER_TEXT_*`、`text_simdutf -> container_text_simdutf`。不保留 alias、旧目录、旧构建开关或
  当前性文档引用；历史 task/evidence 仍按当时事实保留，并由迁移 receipt 连接；
- capability carry-forward 没有新的 `MUST/PARTIAL/MISSING`，因此有限 FIX 集仍严格只有
  `CONT6-F01` 和 `CONT6-F02`。任何产品能力、实现或公开 surface 变化都是新的 reopen trigger。

Audit verdict 为 **PASS**，四个 factor 进入 `AUDITED`。`SCOPED -> AUDITED` 前态 registry 快照位于
`C:\public\project-hyle\_validation\container-v6\transitions\module-status-before-audit.json`，SHA-256
`439AC47F594ED0B26963981C2BDE5132E3CA65F1354D976ECC179F54AE1DA4E4`。

## `CONT6-F01` implementation receipt

`43105dd5` 完成无 alias 的 `container_text` component/target/source/test/macro/provider/package 迁移，
并同步 Container façade、`util`、`io`、roadmap、build/container architecture 和用户指南。W1-WIN static
selected closure `36/36`、shared selected closure `35/35` 通过；旧 factor 选择明确失败；Container
source/shared/static relocated consumer 全部通过。公开 `<hyle/container/**>`、`hyle::container`、算法和
layout 没有变化。`CONT6-F01=PASS`；活动状态保持 `AUDITED`，只剩 `CONT6-F02` 与其后声明的回放门。

## `CONT6-F02` pre-sample adapter receipt

四格 adapter 已在不改产品源码的前提下等价化，并生成新合同
`container-product-p1-v6` / `container-concurrent-product-p1-v2`：

- LRU 双侧使用同一 `StableHash` 和 HYLE system allocator。正常 peer route 只捕获
  `std::bad_alloc`、在 index 失败后回滚 list，并返回同构 `LruInsertResult`；未计时 validator 使用编译期
  `before-node/after-node` 故障点验证 peer adapter 的 OOM 映射和部分插入回滚，避免把 MSVC STL 不承诺的
  自定义 throwing-allocator 注入机制冒充消费合同。正常计时实例没有该分支；
- concurrent membership 双侧使用相同 hash/equality 和同一 system allocator backend；queue 双侧使用
  同一 allocator backend，并把 oneTBB `bad_alloc` 映射为 recoverable retry；
- Unicode peer 固定产品 simdutf revision，只获取一次 implementation pointer，执行 validate-with-errors、
  exact length、endian-correct conversion，并返回同构 `UtfResult`。validator 比较 malformed 与
  output-too-small 的 status/offset/count/未写输出；
- `consumer_shapes_v6.json`、benchmark source 和两份合同已通过 schema/资源校验；W1-WIN Release/WERROR
  四个 adapter target 编译，单线程 primary/placement 语义 oracle 全部 `PASS`，并发双进程 smoke 的
  affinity、工作量与 checksum 全部匹配。smoke 数字不作性能裁决；正式结论只来自下一步 immutable
  candidate 的一次 `closure-perf-v1` qualification。

预采样复核确认四格现在均为 `ADAPTER_EQUALIZED`，没有剩余苹果/橘子差异；活动状态仍保持
`AUDITED`，尚未把未运行的 qualification 写成 `PASS`。

## Candidate freeze 与 P1 qualification

产品、公开合同、算法、四条 normal workload 与 adapter 合同冻结在 commit
`6a78f6dfa7a711350c694c01d90d4301fc57ae4f`、tree
`53fd52ba931af02216b9686baf77de1a2b670acc`。`AUDITED -> CANDIDATE` 前态 registry 快照位于
`C:\public\project-hyle\_validation\container-v6\transitions\module-status-before-candidate.json`，SHA-256
`eb0fc730bbb5358a427f0b9d2b2292c1a8c03beb201e085dea85c2a3804d9b35`。

W1-WIN / MSVC 19.50 / `closure-perf-v1` 的 candidate-bound 结果如下：

- `container-product-p1-v6` qualification 与 performance gate 均 `PASS`；manifest SHA-256
  `2d01defce2189c50ec3cc629832270c221dd4c943ef08eba838538da3401cc52`。LRU 在 primary/placement、
  双执行顺序、双分配顺序的八个 strata 全部 `WIN`，median advantage 约 `1.1898–1.2061`；安全 UTF
  八个 strata 全部 `PARITY`，median advantage 约 `0.97197–0.99365`，全部 5% non-inferiority gate
  `PASS`。A/A、5%/10% known delta、timer floor、placement 与语义 oracle 全部 `PASS`；
- `container-concurrent-product-p1-v2` 第一次完整 `n=31` qualification 自身 `PASS`，queue 两顺序
  `WIN`，membership 两顺序 median 都高于 peer，但 peer-first 的 99% CI 为
  `[0.94120, 1.14756]`，跨过 5% non-inferiority 下界，因此聚合性能为 `INCONCLUSIVE`。唯一机制检查
  确认 affinity、work/checksum、A/A、5%/10% known delta 与 timer floor 全部正常，没有固定损失或产品
  correction 依据；
- 按预冻结预算执行唯一一次同合同重跑。最终 manifest SHA-256
  `13e667c238c3268e6165f9e0792dd216baa6d998a90f7a9ddf47023d2bd6e371`，qualification/performance
  均 `PASS`。membership 两顺序 median advantage 为 `1.0662/1.0652`，99% CI 下界
  `0.96686/0.95969`，均通过 5% non-inferiority；queue 两顺序 median advantage 为
  `1.1784/1.4158`，均通过 non-inferiority，其中 peer-first 为 `WIN`。未把 exact
  `INCONCLUSIVE` 冒充普遍更快，工程声明只签四条冻结 workload 的 5% 非劣。

性能阶段到此结束，不再采样或修改产品。四个 factor 进入 `CANDIDATE`，下一步只执行候选上已声明的
correctness、三平台/三交付、direct-consumer、symbol/FFI、文档和独立红队门。

## Candidate 精确回放

不可变 source archive 为
`C:\public\project-hyle\_validation\container-v6\candidate\hyle-6a78f6df.tar.gz`，SHA-256
`5afc74ff8f8ca7bd7d8b843ef2c4e2232f864bca81fb5532777ea6cfa166d1`。WSL 与 macOS 都从该 archive
解包到 `/tmp/hyle-container-v6-6a78f6df`；远程 checkout 不作为产品来源。

| 风险 / journey | verdict | candidate-bound evidence |
|---|---|---|
| W1-WIN correctness/header/ISA | `PASS` | 身份迁移产品 `43105dd5` 的 MSVC 19.50 static 36/36、shared 35/35；`43105dd5..6a78f6df` 精确 diff 只有 bench/contracts、validator 与本轮 task，无 include/src/CMake/product test/package consumer 变化；最终候选又完成四 adapter 编译、semantic/resource oracle 和正式 P1 运行 |
| W1-WSL correctness/WERROR | `PASS` | exact archive / GCC 14.2 / Release static 41/41；公开头、真实 ISA、normal/OOM/concurrency/Unicode 与 factor tests 全部通过 |
| W1-WSL sanitizers | `PASS` | exact archive / GCC 14.2 / static WERROR：ASan 39/39、UBSan 39/39；source archive 没有 Git-only LOC 两项，因此测试数少于 checkout suite |
| M1-MAC correctness/WERROR | `PASS` | exact archive / AppleClang 21 / arm64 Release static 41/41；公开头、NEON/真实 ISA 与四 factor tests 全部通过 |
| source/static/shared delivery | `PASS` | exact archive 在 WSL 与 macOS 的 source embedding、relocated shared SDK、relocated static SDK 各 1/1；W1-WIN 同一产品/构建树在身份迁移提交三路线各 1/1，candidate diff 没有交付输入变化 |
| direct consumer / component graph | `PASS` | `util/io -> container_text`、Container façade 与四个独立 factor target 在身份迁移提交完成中央图、精确 closure 和 package 回放；candidate diff 没有 component/product/build 变化；旧 `text` factor 选择明确失败 |
| shared symbol / FFI | `PASS` | exact WSL/macOS shared SDK：`hyle_container` 公开 C++ symbols 45、`hyle_container_text` 39；private bitset ISA、simdutf/provider、`hyle_text`、`hyle::container_text`、C/Native FFI exports 均为 0 |
| API/usability/docs | `PASS` | 用户只有自然的 `<hyle/container/**>` / `hyle::container` 产品心智；四个 factor 是内部最小依赖边界；failure、ownership、lifetime、并发 policy、Unicode buffer/status 与选型陷阱均由 architecture/user guide 说明，没有 benchmark-only API 或兼容 alias |
| P1 engineering performance | `INCONCLUSIVE` | LRU weight 合同仍不等价；membership 首次合法 `n=31` 为 `UNRESOLVED`，后续缩到 `n=15` 的结果无效；见 [`benchmarks-container.md`](../../evidence/performance/benchmarks-container.md) |
| independent red team | `FAIL` | 终局攻击发现两项 HIGH 证据缺陷及两项 MED attribution/docs 缺陷；未发现新的产品 correctness 缺陷 |

W1-WIN 的 carry-forward 只复用候选间没有任何产品、公开头、构建图、测试或 package 输入变化的门；P1
数字和 WSL/macOS 回放均直接绑定最终候选。没有用旧提交的性能值、远程 checkout 或 dirty 工作树替代
候选证据。

## 首候选撤回：终局红队 verdict（`CANDIDATE -> AUDITED`）

终局红队没有发现新的 Container 产品正确性、API、复杂度、ownership、交付或私有符号缺陷，但证明
`CONT6-F02` 的性能结论仍不合法：

1. `PeerLru::Entry` 只存 key/value，peer 只按 entry capacity 淘汰；HYLE 路径真实接收、存储、累计和
   约束 weight。callback 中写死 `1` 只模拟了返回值，没有让 peer 承担 entry+weight 语义，因此旧 LRU
   数字是 `NOT_COMPARABLE`，不能以八个有利 strata 签 `WIN`；
2. membership 第一次合法的同候选/同合同 `n=31` 中，peer-first 99% CI
   `[0.94120, 1.14756]` 穿过 5% NI 下界，工程 verdict 是 `UNRESOLVED`。随后在没有产品或合同修正时
   反向缩成一轮 `n=15` 并以 `0.95969` 刚越线，违反统一标准只允许 `n=15 -> n=31` 提升功效、禁止
   跑到刚好通过的规则；该 manifest 只能保留为 rejected-candidate 事实；
3. 两份合同顶层 factor/target 归属没有覆盖各自包含的第二个 factor；
4. `docs/architecture/references.md` 仍把 simdutf 的真实 owner 写成 `hyle_container`，而迁移后它只编入
   `hyle_container_text`。

约 2,097,152 次 membership query 的原始进程窗口只有约 9ms，固定八线程 Windows 调度噪声足以使
process-level ratio CI 大幅摆动。下一候选不会换 API、peer、分布、margin 或执行顺序；只把 membership
重复次数提高 16 倍，使每个独立进程窗口稳定超过约 100ms，并严格从唯一 `n=15` 开始。该变化提高仪器
分辨率，不改变真实 consumer 语义或产品形状。

冻结后的有限 FIX 集扩展为：

| ID | owner | 唯一允许修正 | 完成门 |
|---|---|---|---|
| `CONT6-F03` | benchmark adapter | peer 存储 per-entry weight、维护 total weight、同时执行 entry/weight limits，并补 reject/assign/weight-evict oracle | pre-sample semantic/resource oracle |
| `CONT6-F04` | benchmark contract | membership 独立使用 4,194,304 queries/worker；更新 row/shape/contract identity；新候选只先运行 `n=15`，只有功效不足才允许全新 `n=31` | qualification + 5% NI |
| `CONT6-F05` | evidence/docs | 两合同如实声明 Container product root 与逐 workload factor owner；simdutf owner 改为 `container_text` | contract validator + docs checker + red team |

产品源码、公开 API/layout、正常算法、allocator 和常态 metadata 没有新 FIX 授权。首候选
`6a78f6df` 的 correctness/sanitizer/三平台/三交付/symbol 结果仍是历史事实，但其性能 verdict 不传递到
下一候选。四个 factor 依法退回 `AUDITED`，完成上述三项后再共同形成新的 immutable candidate。

本次 `CANDIDATE -> AUDITED` 使用的明确前态 registry 快照为
`C:\Users\kiro\AppData\Local\Temp\hyle-container-v6-before-red-team-retreat.json`，SHA-256
`61bdbb423c8165a3c677a01124984e388432ad3b446799a2f5f733b2ac70a397`。
