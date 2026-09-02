# Build/package v9：factor 迁移后的 current-tree 复检

<!-- PROJECT UNIT REVIEW: VERIFYING -->

> Build/package 是横向开发单元，不伪造成可链接 factor，也不进入
> `module-status.json`。本轮由 v8 关闭后中央 build/package boundary 与实际
> factor/target 投影发生变化触发，只复检当前构建、依赖与交付合同，不替产品模块签功能或运行时性能。

## Closure card（VERIFYING）

- unit / owner / round / kind：`build-package` / build-package maintainers /
  `build-v9-current-tree-replay` / `REOPEN`；
- reviewed baseline：`7ae3de400b5d7b598972ebf4913d336b60517483` / tree
  `b6c04d0e97c93724a9136c34e4ea55b6e4449092`；前一轮为 v8 Core compile-contract isolation；
- replay input：`8cbea4278e83f445dce54cfda98bad0596d8b673` / tree
  `b8f4e7ccd9da35880f6e6d266096a7d6097d5139`；
- trigger：v8 后新增 split-header manifest projection、`timer_queue` factor、Time product roots、
  closure-registry `QUEUED` insertion 与相应 source/test target；这些命中 build/package 的 target、
  component graph、selected headers 与交付重开条件；
- normal consumers：以源码直接嵌入 HYLE 的量化推理引擎父工程；选择单个产品或 all-products 的开发者；
  relocated static/shared CMake SDK 消费者；离线依赖准备与首次 configure；并行 factor 维护者；
- scope：中央 factor/product manifest、真实 target/link edges、selected header ownership、option/cache
  语义、source/static/shared 与 relocation、共享锁定源码缓存和严格离线、source package、增量/no-op、
  CI 投影及面向用户/维护者的当前文档；
- non-goals：重审模块功能/API/运行时算法或性能；完成 Compute/Graph；升级依赖；增加 Native FFI、
  Bazel/Meson/pkg-config、monolithic DSO、发布签名/SBOM；扩大 CI 额度或重写公共 closure/perf 仪器；
- direct affected factors：`core,container_intrusive,util_core,time,timer_queue,container,util,sched,io`；
  其他 factor 只进入完整图与交付回归，不借本轮改产品；
- performance tier：`P0 STRUCTURAL`。timing peer comparison `NOT_APPLICABLE`；硬门是 closure 解析
  `O(V+E)`、selected source scan `O(S_selected)`、no-op 零重编、leaf 变化不扩散、cache hit 不下载，
  不以一次 configure 毫秒点值塑形；
- machine/tool：W1-WIN / MSVC 19.50 为本地主门；W1-WSL / GCC 14 与 M1-MAC / AppleClang 21
  回放可移植性；中央 component、dependency-cache、offline、package/source smoke 为现有统一仪器；
- mature peer input：沿用 v7 已锁定的 Folly、oneTBB、ggml target/product 分层结论；本轮不因没有新的
  产品形状重新扩张 peer 调研；
- budget：总计 3 小时，其中静态审计 45 分钟、执行门 90 分钟、红队/文档 30 分钟；最多一次
  correctness 复现、一次机制定位、一组有限修复与一次受影响回归；
- stop：当前合同全部 `PASS` 且无未处置 material finding 时 `CLOSED`；可点名的平台、工具、依赖或
  artifact blocker 写 `BLOCKED`；不得用“继续研究”保持活动。

## 初始变化边界

v8 candidate 到 replay input 的 canonical build/package LOC boundary 只修改：

- `cmake/HyleComponents.cmake`；
- `cmake/HyleCiComponentContract.cmake`。

实际 target/交付影响还包括 `src/tests` 下的 split factor 与 Time/TimerQueue 迁移；它们必须与中央
manifest、registry、公开头 owner 和 product façade 一起验证，不能因为不计入 build/package LOC
边界而漏审。同期 `check-module-closure.py` 的新 factor transition 和 `loc-report.py` 的模块边界变化属于
相邻 project tooling，本轮只验证它们与 manifest/registry 的接口，不把其实现行数误记为 build/package。
工作树中既有 Container/Mem 性能证据改动不属于本轮，也不进入 candidate。

## 审计与 FIX ledger

| ID | requirement | initial state | action / verification |
|---|---|---|---|
| `BUILD-V9-GRAPH-01` | 17 factor / 14 product、拓扑、visibility、registry 与真实 target 一致 | `KEEP` | 三平台中央 schema/negative fixtures、final target graph、all-products configure 均 `PASS` |
| `BUILD-V9-SPLIT-02` | split header 只有一个 owner，并统一驱动选择、安装和 include audit | `KEEP` | Intrusive/Util/TimerQueue exact factor 与 Container/Util/Time product 正负 smoke 均 `PASS` |
| `BUILD-V9-DELIVERY-03` | source/static/shared 使用同名 product façade，selected header/artifact 可搬迁 | `KEEP` | Windows all-products static/shared 全前缀搬迁及逐产品 consumer 均 `PASS` |
| `BUILD-V9-DEPS-04` | lock、cache、offline、package/license 仍由同一事实投影 | `KEEP` | dependency cache 并发/生命周期、strict offline、package registry 与 source-package 均 `PASS` |
| `BUILD-V9-OPTIONS-05` | option 只在 owner/gate 下出现，嵌入父工程与 fresh cache 语义正确 | `KEEP` | owner-negative、fresh cache、source embedding、target-platform 正负合同均 `PASS` |
| `BUILD-V9-INCREMENTAL-06` | selected configure 不读未选源码，no-op 零重编，leaf 变化不扩散 | `KEEP` | Ninja no-op 零 action；Time leaf 只编译一个 TU 并重链一个 DSO |
| `BUILD-V9-CI-07` | PR/main 与 manual matrix准确表达当前合同且不过度消耗额度 | `FIX` | 修复 hosted Linux 当前真实 `-Werror` blocker；四个 checkout step 同步到当前 major v7 |
| `BUILD-V9-DOC-08` | architecture、developer、user 与 task index 没有过时状态/入口 | `FIX` | v8 index 从 `ACTIVE` 更正为 `CLOSED`；新增本轮 task 与 build structural evidence |

本表在静态审计后冻结为有限 `KEEP/FIX` 集；性能结果不能新增产品 option、target、header owner 或
常态构建层。

## 有限修复

### 当前 Linux CI 阻塞

GitHub Actions run `33598368691` 在当前 replay input 上失败于 GCC 对
`SmallVector<int, 2>::move_from()` 的 fortified `memcpy` 诊断。失败调用是 heap/inline `swap`；
`inline_storage()` 分支由类不变量保证 `size_ <= N`，但 GCC 跨过程路径合并把 heap 分支的 size 带进
inline object-size 推导，分别可报 `-Warray-bounds` 或 `-Wstringop-overflow`。这不是允许忽略任意
Container warning：修复只在该行周围平衡 push/pop，并只压制这两个 GCC 诊断。

曾尝试把长度钳位到 `min(size_, N)`，虽可编译，却在同一 noinline probe 中生成额外 `cmov` 和可变复制，
因此拒绝。最终修复前后 GCC 14 `-O3 -DNDEBUG` 汇编 SHA-256 都是
`0f7d6c497fae1a5018620205260eb219bfe1244271fd831eab2f9f04726c7c5a`，即代码生成逐字节一致；
W1-WSL WERROR 全树、ASan Container `336/336`、Windows 与 macOS 全树也通过。它不改变 API、layout、
算法、常态 metadata 或运行时成本，故不借 build/package 卡重开 Container 产品收口。

workflow 的四个 checkout step 从 `actions/checkout@v6` 一致更新到当前 major `v7`；cache 保持当前
`actions/cache@v5`。PR/main 仍只有一个 Linux 高信号门，Windows/macOS、交付和 feature 矩阵仍仅在
`workflow_dispatch` 展开，没有借修复扩大额度。

## Verification manifest（候选绑定前）

| Gate | Verdict | Evidence / limit |
|---|---|---|
| static graph / option / dependency audit | `PASS` | 中央 component contract 在三平台通过；17 factor、14 product、真实 target 与 direct edge 一致 |
| W1-WIN component and affected selected journeys | `PASS` | MSVC 19.50 Release/WERROR all-products build；`70/70` tests |
| W1-WIN all-products source/static/shared | `PASS` | source build；static/shared 安装、整前缀搬迁、14 product 与 Mem consumer 全部通过 |
| dependency cache / strict offline / package registry / source package | `PASS` | 并发/lifecycle/offline/registry 合同通过；clean archive 完成 manifest/digest/license 与脱离 Git 构建 |
| reconfigure / no-op / leaf invalidation | `PASS` | Windows/WSL Ninja 均 `no work to do`；M4 Make 零编译；`time/interop.cc` 只触发 1 compile + 1 link |
| W1-WSL portability | `PASS` | GCC 14.2 Release/WERROR build；`69/69` 适用 tests；ASan Container `336/336` |
| M1-MAC portability | `PASS` | Apple M4 / AppleClang 21 / deployment 14.4；Release/WERROR build；`69/69` tests |
| docs / diff hygiene / LOC | `PENDING` | 等待不可变候选上的 docs/closure/diff 与 LOC |
| independent red team | `PASS` | hosted Linux 真实失败与 fresh 三平台 scratch 相互印证；未发现第二项 material finding |

WSL 最初使用有意不携 `.git` 的源码副本，唯一失败是维护者专用 `hyle_loc_report_test` 无法解析
`HEAD`；补入临时 Git metadata 后该项通过。全部 68 个不依赖仓库身份的测试在第一次运行已经通过，
源码发行本来以 `HYLE_BUILD_TESTS=OFF` 构建，因此不把这个测试环境前置条件伪报成产品缺陷。

当前处于 `VERIFYING`：下一步只绑定包含上述有限修复的不可变候选，重放 source-package 与最终
项目检查，然后按 stop rule 裁决；不得继续扩张构建功能。
