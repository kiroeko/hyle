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

## 第二候选预采样修复（`AUDITED`）

`CONT6-F03..F05` 已在不改产品源码、公开 API/layout、allocator 或正常算法的前提下完成：

- 新 `PeerLru` 与 HYLE 一样存储 per-entry weight、维护 total weight，并同时按 entry/weight limits
  循环淘汰。未计时 oracle 覆盖 weight-driven eviction、assignment 改重后淘汰、overweight rejection、
  callback、size、weight、value 和 membership 等价；原有 node/index OOM/rollback oracle 保留；
- membership workload 保持同一 32768-entry set、八个物理核 worker、50% hit、hash/equality/allocator、
  setup 与双执行顺序，只把每 worker 查询数由 262144 提至 4194304。W1-WIN smoke 的 HYLE/oneTBB
  独立进程窗口为约 133.7/137.0ms，已消除原约 9ms 窗口的分辨率缺陷；queue scale 完全不变；
- 当前语义矩阵升为 `consumer_shapes_v7.json`，SHA-256
  `78b6ba01743c0566f9828098a7e5a583c2b7ef09f3eea231230c4e607afe8b60`；单线程合同升为
  `container-product-p1-v7`（SHA-256
  `3009a1294fdc7004b365e8af79c7a5d13f237e9ffaa540c722f105cc7b71d5ef`），并发合同升为
  `container-concurrent-product-p1-v3`（SHA-256
  `0af5617ca1c4b978a8ad65ace62764efc2e792642939465110796c333f4e6389`）。旧 v6/v2 合同只存在于
  被撤回候选的 Git 历史与 manifest 中；
- 两合同现在以真实 `hyle_product_container` / `hyle::container` façade 为产品根，并逐 workload 声明
  `container`、`container_text`、`container_concurrent`、`container_queue` factor owner；simdutf 的真实
  compiled owner 文档也改为 `hyle_container_text`；
- W1-WIN Release/WERROR 四个 adapter target 已编译；新 v7 validator 的 primary/placement artifact
  分离、LRU/OOM/weight/Unicode semantic oracle 全部 `PASS`；并发 HYLE/oneTBB smoke 的 affinity、工作量、
  checksum 与结构化 row identity 全部匹配。smoke 点值不是性能证据。

三项 FIX 的预采样门均为 `PASS`。状态仍为 `AUDITED`：下一步只能先提交并冻结新的 immutable candidate，
然后对 v7/v3 各执行一次 `n=15` qualification；只有合法 verdict 为 `UNRESOLVED` 时才可按合同执行唯一
一次全新 `n=31`，不得缩样本、追加或挑选有利结果。

## 第二候选冻结

修正后的 benchmark、语义矩阵、合同、validator 和归属文档冻结在 commit
`111c28eab99daff6257427473a2a1b11ac1cfd0c`、tree
`a2e578165582cd4f34c23d5350bfd7d019ca900b`。该候选相对首候选没有 `include/`、`src/`、产品测试、
组件图或 package consumer 的变化；候选新增事实只影响性能证据合法性。

`AUDITED -> CANDIDATE` 的明确前态 registry 快照位于
`C:\public\project-hyle\_validation\container-v6\transitions\module-status-before-second-candidate.json`，
SHA-256 `60a8ad7aca27092ef43cbc753e509366b7661c696b55e5e2a381ecec9fc6d80d`。四个 factor 共同进入
`CANDIDATE`，不得再改变候选树；下一步严格运行两份新合同各自的首轮 `n=15`。

## 第二候选 P1 与 `container_text` 定向退回

`container-product-p1-v7` 自动按合法 `n=15 -> n=31` 路径完成 qualification，manifest 位于
`C:\public\project-hyle\_validation\container-v6\qual-single-v7-evidence\manifest.json`，SHA-256
`9e277f514978582be8fe023e091db20f279f02af124e826f7a4ab94d9a1ee785`。weighted-LRU 8/8 strata 为
`WIN`，median advantage `1.2276–1.2693`；safe UTF-8→UTF-16 8/8 strata 为 `PARITY` 且 5% NI
全部 `PASS`，但 median advantage 稳定为 `0.9666–0.9854`，每个 99% CI 上界都低于 1。

`container-concurrent-product-p1-v3` 在首轮 `n=15` 直接 qualification，manifest 位于
`C:\public\project-hyle\_validation\container-v6\qual-concurrent-v3-evidence\manifest.json`，SHA-256
`f16b135ff6a1b0974055ae63940e88df44f2ffe44c71841d6417930eb06eb196`。membership 双顺序为
`PARITY`、5% NI `PASS`，median `0.9882/0.9777` 且 CI 跨 1；queue 双顺序为 `WIN`，median
`1.2888/1.2614`。因此 `container`、`container_concurrent`、`container_queue` 的本轮 P1 证据合法，
不因相邻 factor 的后续修正撤销。

对 Unicode 唯一机制检查比较了 MSVC 19.50 Release/LTO-off 的 candidate 与 peer object code：两个
`UtfReplay` 循环除 call relocation 外逐指令同形，peer adapter 也未被 inline；差异位于产品
`validation_result()`。HYLE 对常见 `simdutf::SUCCESS` 仍执行 enum bounds check 和间接 jump table，
peer 则先直接 fast-success，只在错误时映射状态。这是同语义正常调用路径的确定成本，不是 placement
或顺序幽灵。

有限 FIX 集增加且只增加 `CONT6-F06`：令 `validation_result()` 对 success 直接构造结果，错误路径继续
调用既有映射；不得改变公开 API/layout、错误语义、provider、workload 或 margin。修后只验证 affected
Unicode correctness/oracle、三平台 Container 选择测试和新 candidate 的 v7 performance；未改变产品字节
的 factor 继续复用已绑定证据。

只有 `container_text` 从 `CANDIDATE` 退回 `AUDITED`；其前态 registry 快照为
`C:\public\project-hyle\_validation\container-v6\transitions\module-status-before-unicode-correction.json`，
SHA-256 `3a5d1d3df110e72285eb2e49bd953b37458c9469720ca954f6187c76f8504bff`。其他三个 factor 保持
`CANDIDATE`，不重新运行已经合法且不受影响的 concurrent qualification。

## `CONT6-F06` implementation receipt

`validation_result()` 现在先判断 `simdutf::SUCCESS` 并直接返回 typed success，只有错误结果进入既有
`from_simdutf()` 映射。公开 API/layout、provider、错误码、三遍 safe-conversion 语义和 benchmark workload
均未改变。

W1-WIN Release/WERROR 的 `container_text` header/self-contained、ordinary 和 ISA-variant 三项定向测试
全部通过，新 v7 primary/placement semantic oracle 也全部 `PASS`。修后 MSVC object code 在 simdutf
validation 返回后先 `test` 并直接进入 sizing/conversion，错误码 jump table 已离开 success path。唯一
非证据 smoke 中 Unicode HYLE/peer 为约 `190.85/192.63 M units/s`（ratio `0.9908`），相对第二候选
正式样本的 `0.9666–0.9854` 方向改善；该单点不用于性能裁决。

`CONT6-F06=PASS`。下一步只把 `container_text` 的这一产品提交冻结为新 candidate，并对不可变候选运行
一次正式 v7 qualification；没有产品字节变化的 concurrent/queue 结果继续 carry-forward。

## 第三候选冻结

`container_text` 的 fast-success 产品修正冻结于 commit
`6f07b9ed023daa72c346bc65e4613d65e3cd20d5`、tree
`c25e36393175dba6a7f5118564a18377de53f30d`。`AUDITED -> CANDIDATE` 前 registry 快照为
`C:\public\project-hyle\_validation\container-v6\transitions\module-status-before-container-text-third-candidate.json`，
SHA-256 `04393c89bcb7770e9c853c6c52adb18542779e0e6defea32d9b5062836f64eb6`。

这一状态提交不改变 candidate 产品树，不运行 P1。下一项且唯一允许的性能动作是从干净 detached
worktree 对该不可变候选运行一次 v7 qualification；其余 factor 的合法证据继续复用。

## 第三候选 P1 与终态构造修正

第三候选的唯一正式运行位于
`C:\public\project-hyle\_validation\container-v6\qual-single-v7-fast-evidence\manifest.json`，
manifest SHA-256 `ba4fa20dd6556134d5c2d800244da398474c5cfe846c85148e532e20c9575f99`；合同 SHA-256
`3009a1294fdc7004b365e8af79c7a5d13f237e9ffaa540c722f105cc7b71d5ef`。资格化与 5% engineering
performance gate 均 `PASS`，LRU 8/8 `WIN`；但 safe Unicode primary median 为
`0.9666–0.9728`，placement 为 `0.9829–0.9906`，其中 6/8 的 98.9% CI 上界仍低于 1。这个结果不允许
被聚合 `PASS` 隐藏，`container_text` 不能据此冻结。

同一构建的 MSVC Release/LTO-off object code 证明，candidate 与 peer 的 simdutf provider、验证、精确
sizing、valid conversion 和外层 replay 语义一致；剩余差异来自 HYLE 的公共实现组合方式。六种 safe
conversion 都先令 `validate_*_core()` 构造 `UtfResult`，检查成功后又令 `output_result()` 构造第二个
`UtfResult`。MSVC 未消去这些中间聚合对象，成功路径仍包含额外状态/填充字节搬运、分支和寄存器保存；
peer 则直接检查原始 `simdutf::result.error`，只在错误或最终返回时构造 typed result。

有限 FIX 集增加且只增加 `CONT6-F07`：对全部六种 safe conversion 使用相同的 terminal-result 形状——
参数检查后直接检查 provider 的原始验证结果，错误时映射，成功时进入 sizing/conversion，且只在
output-too-small 或最终成功时构造 `UtfResult`；scalar fallback 语义保持原样。不得改变公开 API/layout、
错误码、完整写入保证、provider、workload 或 margin。这是公共 Unicode 家族的成本结构修正，不是只为
`utf8_to_utf16` benchmark 添加分支。

只有 `container_text` 从 `CANDIDATE` 返回 `AUDITED`；前态 registry 快照为
`C:\public\project-hyle\_validation\container-v6\transitions\module-status-before-container-text-terminal-correction.json`，
SHA-256 `99843515ce8ef08861392dd1f35390e391acbf8813582f2314618beb1b3335ae`。其余三个 factor 保持
`CANDIDATE`，所有不受影响的证据继续 carry-forward。

## `CONT6-F07` implementation receipt

六种 safe conversion 已统一为 terminal-result 实现：simdutf provider 的原始 validation result 在错误时
才映射为 `UtfResult`，成功路径直接进入精确 sizing 和 valid conversion；容量不足与最终成功各只构造一次
公开结果。旧 `output_result()` 中间聚合 helper 已移除。公开 API/layout、provider、错误枚举、完整写入和
scalar fallback 合同不变，空的 UTF-16/UTF-32 输入增加等价的直接成功路径。

W1-WIN Release/WERROR 只构建受影响的 `hyle_container_text_tests`、普通测试与 ISA-variant 目标，三项
定向测试全部通过；v7 primary/placement semantic oracle 也为 `PASS`。MSVC Release/LTO-off object code
在 provider validation 返回后直接按 `error == SUCCESS` 跳入 sizing；中间 success-result 搬运与随后状态
检查已经消失，错误 jump table 只可由错误分支到达。未运行 exploratory performance sample。

`CONT6-F07=PASS`。下一步提交这一产品字节，再把精确提交冻结为新的 `container_text` candidate；只有该
不可变候选需要最后一次 v7 qualification，其他 Container factor 不重放。

## 终态候选冻结

`container_text` 的 terminal-result 产品实现冻结于 commit
`3ba50461ec7c8f387f4d953f1e499a1f6d449870`、tree
`bde13e7f012dd8616eaf75fdb2f2e39fc6782e1f`。`AUDITED -> CANDIDATE` 前 registry 快照为
`C:\public\project-hyle\_validation\container-v6\transitions\module-status-before-container-text-terminal-candidate.json`，
SHA-256 `ac8ad50f3fde7cdeab79143773e3b0a257dd4608204664104fdd07dd304ca248`。

这一状态提交不改变产品树，因此不运行 P1。下一项只允许对这个干净不可变候选运行一次 v7
qualification；若预注册判据仍证明固定产品劣化，本轮进入 `BLOCKED` 而不是继续追加试验或 FIX。

## 终态候选验证

最后一次且唯一一次终态 v7 qualification 位于
`C:\public\project-hyle\_validation\container-v6\qual-single-v7-terminal-evidence\manifest.json`，manifest
SHA-256 `c4793256dfe376d82feb4beeb3b5d9789f90666bd9c981bd4a9323d108f4ea5a`，contract SHA-256
`3009a1294fdc7004b365e8af79c7a5d13f237e9ffaa540c722f105cc7b71d5ef`。n=15 已足够裁决，未升级为
n=31；qualification/performance 均 `PASS`。LRU 八个 strata 全部 `WIN`，median advantage
`1.2223–1.2728`。safe Unicode 八项全部 `PARITY` 且 5% NI `PASS`，median `0.9863–1.0102`；七项 CI
跨 1，余下一项 CI 完全高于 1。第三候选的 6/8 固定劣化已消失，没有通过换顺序、缩 margin 或挑样掩盖。

精确 source archive 为
`C:\public\project-hyle\_validation\container-v6\candidate\hyle-3ba50461.tar.gz`，SHA-256
`ce6718997233e0997845d09761d209adf7e17d26d69561d70c510a4855c9f916`。只重放受影响的
`container_text` 风险：

- W1-WIN / MSVC 19.50：Release/WERROR 三项定向 test `3/3 PASS`，v7 primary/placement semantic oracle
  `PASS`，正式候选构建与 P1 通过；
- W1-WSL / GCC 14.2：Release/static/WERROR + UBSan，header、ordinary 与 ISA-variant `3/3 PASS`；
- Mac mini `Mac16,10` / M4 / macOS 26.6.2 / AppleClang 21：static `3/3 PASS`，shared
  header+ordinary `2/2 PASS`；
- MacBook Air `Mac16,12` / M4 / 同 OS/toolchain：static `3/3 PASS`；相同 Apple ABI 的 shared 路线由
  Mac mini 覆盖，不重复构建；
- 没有 include、公开 layout、component graph、package metadata 或 symbol visibility 变化；此前精确
  source/static/shared relocated consumer、direct consumer、symbol/FFI 证据按未变化风险复用。

两个 macOS 环境是不同硬件，不把 static/shared 两种 linkage 误计为两台机器。两者只提供 correctness /
portability 事实；当前 performance profile `M1-MAC` 仍只登记 MacBook Air 且未绑定 Formal 主机身份。

## 终局独立红队与 VERIFIED 回执

独立只读红队逐项攻击 `3ba50461` 的六种 safe conversion、候选身份和两份性能合同，结论为 `PASS`：

- 参数、null/empty、完整 validation、错误 code-unit offset、exact sizing、output-too-small 不写和最终完整写入
  语义保持一致；UTF-16 按 host endian 选择 provider 路线，UTF-32 保持 native code point 语义；
- 公开头、API、`UtfResult` layout、符号签名和正常 metadata 没有变化；修正覆盖六种转换，不是只服务
  UTF-8→UTF-16 workload 的 benchmark 特化；
- v7 contract、consumer shape、adapter、候选与 manifest provenance 为 `EXACT`，verifier 为
  `QUALIFIED` 且 `paths_verified=true`；Unicode 两侧使用同一 provider、validate、size、convert 和 typed
  result 合同；LRU 两侧使用相同 hash、allocator、OOM/rollback 与 entry/weight 双预算；
- manifest 的 LRU 最小 median 是 `1.2223276`，文档已据此校正为 `1.2223–1.2728`；这是事实笔误修正，
  没有重跑测试或 P1。

四个 factor 的 scope、architecture、correctness、delivery、usability/docs、red-team 与 engineering
performance 门均为 `PASS`，可由 `CANDIDATE` 进入 `VERIFIED`。迁移前 registry 快照为
`C:\public\project-hyle\_validation\container-v6\transitions\module-status-before-terminal-verified.json`，
SHA-256 `9cc4537856ce2d7685377034886877f948a3654241a808e50f50e0ae36b2b024`。

不可变候选 LOC 由 `loc-report.py --self-check` 计算且均为 `PASS`：sequential `container` 7,962、
`container_concurrent` 1,600、`container_queue` 1,101、`container_text` 2,084；四个活动 factor 合计
12,747。加上已经冻结的 `container_intrusive` 1,590，消费者所见 Container 产品实现合计 14,337。
