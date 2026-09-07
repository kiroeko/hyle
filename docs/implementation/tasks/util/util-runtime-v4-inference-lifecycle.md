# util-runtime v4：推理 factory 生命周期与新标准收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen intake 与有限 closure card

- component_id：`util`，语义名称 **util-runtime**；owner：runtime utility owner；round：
  `util-runtime-v4-inference-lifecycle`；kind：`REOPEN`。`util_core` 是另一个已 FROZEN 因子；
  `hyle::util` 是聚合两因子的用户产品 façade，三者不能混称。
- canonical checkout：`C:\public\project-hyle\hyle`。开始时 clean，main、origin/main 和只读查询到的
  远端 main 均为 `ded05c9e36fffe8cdc54515d7413097cfdec230e`；本轮 baseline 固定为该提交。
- 权威输入：完整 AGENTS、README、registry、closure methodology、capability-completeness、任务模板、
  engineering-charter、coding-conventions、roadmap、build-system、util-design，以及已 CLOSED 的
  frozen inference/performance audits。另读取 registry 当前指向的 util-runtime-v3 terminal 和
  change-impact selector v2 使用合同；不重做三个已完成项目，不继承其微性能问题为实施任务。
- trigger `UTIL-V4-DEPENDENCY-AND-DELIVERY-2026-09-07`：旧 candidate
  `b2b7bd679293d748d996662414c01c4cbcefcbda` 后，`796b654d2791a0472565f19a2ca2fdd311347728`
  已把 Registry 的公开 allocator/storage 改为 `AllocatorRef/Arena`；中央图又引入
  `container_text` 直接边和产品 façade。当前 `samples/package/util/cpp/CMakeLists.txt` 仍断言
  不得出现 `_factor_container_intrusive`，与 `util -> container -> container_intrusive` 的真实
  闭包相反。满足旧 registry 第 2 条 dependency/delivery 及第 3 条 correctness/usability trigger。
  标准升级、旧卡缺 FROZEN 字样本身不是产品 trigger。
- 真实任务：owner 在启动时选择并登记少量已编译或已装载 backend，校验版本/能力，按名解析 factory，
  构造可失败的对象；服务前 freeze，随后缓存解析结果。不同 backend 可有自己的 descriptor/image
  状态；注册失败必须回滚，创建的对象及 callback 不得越过定义它们的 image 生命周期。诊断线程
  把 runtime C++ symbol 写入 owning string 或 caller buffer。
- 范围：两份 runtime public headers、`src/util`、`tests/util`、util package consumer、相关四类文档与
  仅 `module=util` 的状态/投影。审计自然注册/查询/优先级/发布、错误、所有权、复杂度、依赖与交付。
- 非目标：修改或重开 util_core/其他 FROZEN 依赖；Sched/IO/Compute/Graph 产品；通用 ModuleManager、
  unload/unregister/drain 服务、版本求解器、设备选择、Native FFI/C API、稳定跨工具链 C++ ABI、
  benchmark-only API、公共 checker/构建基础设施/性能仪器修改。
- 依赖：`PUBLIC core,error,mem,container,container_text`；没有当前产品直接依赖 util-runtime。
  util façade 与 `samples/package/util` 为本轮真实 C++ 交付消费者；后续 owner 只接收输入。
- tier / instrument：`P0 / closure-correctness-v1`，Formal `NONE`。五个参考的相关操作均为初始化、
  发现或冷诊断；没有 Registry token/kernel 热路径 profile 或计时主张。P1/P2、计时样本预算 **0**；
  timing peer、5% margin、固定输项评测均 `NOT_APPLICABLE`，不能写成数值 PASS。
- 成本预算：register 事务式、owned 字节线性复制、索引 expected-amortized O(1)；freeze 为 O(1)、
  零分配、发布同一索引；frozen lookup 为 O(key bytes)+expected O(1)，无查询隐藏分配/锁；creator/
  visitor 在锁外。仅真实 ownership 需求允许登记必要 metadata，不以旧 benchmark 布局为先验。
- 预算：审查 45 分钟、实现一个有限批次 60 分钟、取证 75 分钟（含候选后 20 分钟独立第二遍红队）；
  审查与取证合计不超过两小时，总计三小时。最多两遍审计、一个冻结 FIX 批次、一次具名纠正与
  受影响重跑；不反复跑已通过门。计时从本轮入口 2026-09-07 08:50 UTC 计，11:50 UTC 硬停止。
- 机器：W1-WIN MSVC 19.50、W1-WSL GCC 14.2、M1-MAC AppleClang 21；只读工具探测三处均可达。
  WSL/macOS 使用 Windows candidate 的独占验证副本；不回传产品，不改既有 checkout。工具链、stdlib
  文件/include roots、flags、machine 与 candidate tree 由 selector context/receipt 绑定。
- 适用门：selector 选择的 util factor correctness、util runtime 两头 self-contained、util 产品
  source/static/shared（搬迁后安装消费）；没有 util-runtime 反向产品消费者，不跑其他 factor 测试。
  另预声明同一 util 测试集的 WSL ASan/UBSan/TSan，三平台 compiled demangle symbol 检查，以及
  P0 release codegen/成本检查。默认 shipping layout；若需 fault seam，只留 util 测试的私有 adapter。
- correctness：正常/empty/invalid、长度与 overflow、每分配点失败、重复/priority、owned/borrowed
  生命周期、capture 与 image 保活、concurrent register/read/freeze、release/acquire、callback 重入、
  锁外执行、typed Result/nullable、demangle required size/截断/双向 overlap/异常与 ABI failure。
  任一适用风险未运行记 BLOCKED；同步名字表的异步取消/排空服务语义不适用。
- 验收：所有准入 MUST COMPLETE/ROUTED；有限 FIX 全部落实；适用门真实通过；文档命名/ownership/
  依赖校准；没有其他 factor 产品/状态变化；immutable candidate、终态 receipt、提交与 push 可追溯。
- 停止：满足上述条件才 CLOSED 并登记新 FROZEN；否则 BLOCKED，列出 owner、已知失败与精确
  resume_condition。不为取得 FROZEN 把未知门改为 PASS 或继续扩大任务。
- LOC：按本轮用户要求只统计全库、自实现不含测试、util-runtime 不含测试。旧 schema 强制保留的
  `self_with_tests` 字段只携带旧值与独立 provenance，不重新统计、不作为本轮候选规模、不对外报告；
  不在模块任务中改 checker 或 LOC 工具。

每次 registry 状态变化之前把原字节保存到 checkout 外
`C:\public\project-hyle\_validation\util-runtime-v4\state-before-<phase>.json`，运行
`check-module-closure.py --previous <snapshot>`，并更新受检 status 投影。状态只以
[`module-status.json`](../../module-status.json) 为真；本卡的 audit/candidate/receipt 是轮次证据。

## 审计进入条件

本卡先登记实际依赖/交付 trigger，再经 `REOPENED -> SCOPED`。在 capability-family map、public trace、
owner 七问与 API/有限 FIX 集完成并进入 AUDITED 前，不修改产品或测试实现。

## AUDITED：能力、所有权与有限 FIX

2026-09-07 只读审计已完成。当前 scope 不增加 component、依赖或第三方产品输入。

### 固定来源与能力族

五个 engine 的本地 clean revision 沿用已完成 audit，2026-09-07 再只读确认 HEAD/工作树与远端：

| 来源 | 本地完整 commit | default branch / 本轮查询远端 HEAD |
|---|---|---|
| llama.cpp | `3ad1ba7336986d98592d3e28cafd1a406715351f` | master / `5202104b59ada9005db079eea43882a2b7bf5802` |
| standalone ggml | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` | master / `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` |
| SGLang | `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110` | main / `8ae962021dbb5eb8ee408c578e772c13e2c934fe` |
| vLLM | `52358e6e192aeae73dd3764046c98fc4b156ea83` | main / `392db567b260cb4d1e5f91547b65d527038fe641` |
| ds4 | `9ab705347c1775e7599ede7eb81a6255ec7dccb5` | main / `b6af0adf8ca97c89145c9f9c15be70c9fd6c4507` |

origin 与准确消费点继承 [frozen capability audit](../project/frozen-inference-capability-audit-2026-09.md)。
只补读 U01 所需的 ggml factory descriptor 和 SGLang/vLLM registry 文件，没有重新扫描五仓全部能力。
llama 随仓 ggml backend-reg 与独立 ggml 为同一实现来源，不算两个独立 peer。

成熟基础库只作功能/算法参考，不作性能 peer、不引入产品依赖：Folly clean
`65749da412d92fe27d87ae45a8ef670720712858` 的 `folly/Singleton.h:322–336` 用 owning
`std::function` 保存 creator，`:188–196` 区分注册与使用阶段；Boost.DLL clean 子仓
`de905f331abdfac9aabe895b2bbbf1fbca71a002` 的 `include/boost/dll/import.hpp:40–61,82–85`
通过 aliasing shared owner 把 callable 与 library 生命周期绑定。Boost superproject
`cb94f33fcb13e90922905c2d8105508e71fa6db4` 有既存 `libs/filesystem` 修改，本轮保留且不读取其 dirty
内容作机制证据；DLL 子仓独立 clean identity 为准。demangle 参考同一 Folly `Demangle.cpp:149–194` 的
ABI 临时内存/owning route 和 HYLE 已采用的 compiler ABI/DbgHelp，保留 fallback 语义。

| Family / requirement | 完整旅程与证据 | 当前覆盖 / 动作 | 生命周期、边界、复杂度与负空间 |
|---|---|---|---|
| F01 MUST 显式命名 factory | ggml backend-reg.cpp:186/220/325；SGLang attention_registry.py:40–49、mem_cache/registry.py:53–81/创建路线；vLLM utils/registry.py:9–53、renderers/registry.py:36–76 | COMPLETE / KEEP | register→find/create→freeze→借用→quiescent destruction；empty/null、OOM、dup、length-aware key 明确；expected O(1) 索引 |
| F02 MUST 每项 factory 状态所有权 | ggml backend-impl.h:176–224 的 device/context + init_backend(device,params)，backend.cpp:610–612；Folly owning creator 与 Boost.DLL callable/image | PARTIAL / FIX-1 | 注册后的 callable 必须能绑定不同 descriptor/image，而不要求调用者再维护同 key 的第二张 map 或全局 trampoline；接管成功、失败不接管、旧 snapshot 和销毁均验证 |
| F03 MUST 版本与优先级 | ggml backend-reg.cpp:245–253 在注册前验证 API version；当前三档 RegistryPriority | ROUTED + COMPLETE / KEEP | owner 校验 typed version/score，再 register；并存版本可用显式不同 key 或不同 registry；更高替换、同级失败、更低 no-op；不增加泛用 semver/score 求解或每项版本 metadata |
| F04 MUST lookup-only / nullable / Result | 当前 find/create；SGLang get_radix_cache_factory→factory(ctx)；vLLM load_renderer_cls→class(config,tokenizer) | COMPLETE / FIX-1 适配 callable | missing find=nullopt；nullable create=empty；Result create=NotFound/原样 factory error；结果由 caller 拥有，lookup 不构造、不分配 |
| F05 MUST 发布与枚举 | 当前 mutable/frozen 公开承诺；Folly registrationComplete 作为阶段参考 | COMPLETE / KEEP | 一张表，mutex mutation，release/acquire 发布；freeze O(1) 零分配；visitor/factory 锁外且可重入；cached view 借用 |
| F06 MUST symbol presentation | 当前全部 demangle overload；Folly ABI/owning；平台 ABI | COMPLETE / KEEP，FIX-3 补证 | 普通诊断线程、full owning 或 snprintf buffer、type_info、空/非法符号 fallback、required-size、重叠、截断；允许 ABI malloc/串行，非 async-signal-safe |
| F07 MUST C++ source/static/shared | 当前 façade、factor、sample 与新闭包 | PARTIAL / FIX-2 | util-runtime 两头/compiled demangle；util_core 仅由 façade 聚合；当前 sample 的禁止 intrusive 断言错误 |
| F08 MUST 可发现性与文档 | user/architecture/roadmap 命名混淆、runtime ownership 新形状 | PARTIAL / FIX-4 | 产品 Util 与因子 util-runtime（ID util）分开；不修改 util_core 能力合同 |
| F09 GENERAL 动态模块/服务生命周期 | U01 的 unload、S02 的 start/stop/drain；ggml backend-reg.cpp:266–287；SG15/V12/D04 | OUT_OF_SCOPE / REJECT 下沉 | DynLib 是 Platform；版本/device/object/worker/late callback 是 owner；名字表不能证明 unload quiescence，不提供 unregister/unfreeze |
| F10 GENERAL signal-safe/Rust/整栈 symbolizer | Folly 的可选 libiberty/Rust branch | OUT_OF_SCOPE / DEFER | 需具名 crash/foreign-runtime 消费与独立跨平台合同；当前 diagnostic fallback 已完整，不复制功能并集 |

F02 的准入不是“库里有 std::function 就必须复制”：U01 的不同 runtime descriptor 无法作为普通
function pointer 的捕获，`Args...` 是调用时输入，无法独自保存每个注册项各自的绑定。用第二份名字索引、
全局变量或预分配 trampoline 槽解决会重复 Registry 的查找、失败与生命周期。C++ owning callable 是
模型无关的自然形状；只在 Registry 内存储它，不在 util_core 新增另一种 owning function 通用产品。

### API / usability freeze

状态 **FROZEN**，冻结于 AUDITED；普通 C++ 形状为：

```cpp
Registry<Result<Backend>, const Config&> factories;
HYLE_RETURN_IF_ERROR(factories.register_creator(
    name, [owner = std::move(owner)](const Config& config) noexcept {
      return owner->create(config);  // Backend 也持有它需要的 image owner
    }, RegistryPriority::kPreferred, help));
auto view = factories.freeze();
auto entry = view.find(name);
if (!entry) return Status::NotFound("backend is absent");
return entry->creator(config);
```

- `register_creator` 接受可无异常构造/销毁、通过 `const F&` 无异常调用的 callable（包括 move-only
  capture 与普通 noexcept function pointer）。构造/销毁的特殊成员不得重入 Registry；注册成功前
  所有可能失败的内存准备先完成，失败或低优先级 no-op 不移动具名 rvalue callable。
- `Registry::Creator` 从裸 function pointer 重塑为可缓存的 **借用 callable handle**；call/显式 bool/
  handle identity 可用，默认空 handle 不可调用。它不是独立 owner，也不再承诺与原函数地址比较。
  不保留错误兼容别名；factory 类型由 register 推导，不要求用户手造 type-erasure record。
- 每次被接受的登记拥有不可变 entry、callable、key/help。替换只交换索引指针，旧 entry/capture
  留到 Registry 析构，确保已返回 Creator/EntryView 和进行中的调用不悬空。三档 priority 限制每 key
  最多三份被接受的 lifetime record；相同/较低优先级不分配。
- 回调调用与 visitor 均在锁外，可重入并自行同步捕获的共享可变数据。`const` 不自动让 referent
  thread-safe。Registry allocator 必须活过 Registry；并发枚举临时快照的锁外释放要求 allocator
  支持该并发，或 caller 外部串行化。Arena 与 views 均不许提前 clear/reset。
- freeze 后只能读/调用/枚举；register/reserve 失败，freeze 幂等；没有单项删除、unfreeze 或隐式
  worker/image drain。所有 handle/view/操作先 quiesce，再销毁 Registry；返回对象的 image owner
  可以活得更久，由该对象独立持有。仅捕获裸 DynLib pointer 不延长其生命周期。
- compiled demangle API 不变：ABI failure（包括 ABI allocation failure）fallback 原名，不另发错误码；
  buffer required size 属最终 spelling，NUL 不计入；owning string 分配异常仍向 caller 传播。
  Windows mutex 只覆盖同一 util runtime image 的 HYLE DbgHelp 调用；多份 static image/直接 DbgHelp
  用户需宿主统一串行化。不得声称它自动协调进程所有 DbgHelp 客户。
- 成本：以一个稳定 entry pointer 索引 owning record，销毁链是 callable teardown 的必要 metadata。
  普通 factory 的 erasure 调用可能多一层 trampoline；P0 显式计费，不按旧微基准要求保留重复热字段。
  frozen lookup 自身没有 allocation/lock/refcount；factory 自己的工作另计。freeze 不复制 callable。

### 冻结 FIX 集（唯一实现批次）

1. **FIX-1 / F02**：仅重塑 Registry 的 stored callable/borrowed Creator 和事务式 entry replacement；
   支持每项 owning state，保留同一索引、优先级/Result/发布合同。测试 capture、move-only/overalign、
   OOM 不接管、替换旧 handle 保活、并发发布与锁外重入。预计一头和 owner tests；无新依赖。
2. **FIX-2 / F07**：util package sample 承认真实 Container intrusive 闭包；普通 C++ sample 验证
   captured factory、façade foundation+compiled runtime、错误/版本 owner route。只改 util sample，
   不改公共 smoke/CMake 基础设施或中央图。
3. **FIX-3 / F06、红队**：补足 owner 风险测试，包括 freeze race、枚举 OOM、demangle cap=0/1、
   两方向/截断 overlap、真实 ABI 长名字、并发 demangle、失败 sink。仅 util tests 的私有故障
   adapter 可注入 platform demangler OOM/错误；不加公共测试 API。校正 DbgHelp 锁范围注释。
4. **FIX-4 / F08**：校准 runtime headers 注释、architecture/user/roadmap/build-system 的名称与依赖、
   当前性能页和 task/status。已 FROZEN util_core 的产品及能力合同不变；旧任务保留历史语义。

以上之外的新 material MUST 只按方法论裁决，不追加模块或进行无界实现。

### Owner 七问（审计阶段）

1. **第三方**：Util 无直接第三方。Registry 组合 Core、Error Result/Status、Mem AllocatorRef/Arena、
   Container FlatHashMap、Container-text StringView/hash。mimalloc、Highway/xxHash/wyhash/simdutf 为
   依赖闭包自己的锁定实现，semantic/decision authority 各留在 Mem/Container，不在 Util 再选 allocator/
   ISA。Folly/Boost 仅只读需求参考；demangle 直接用 compiler ABI/Windows DbgHelp。
2. **五个引擎**：llama/独立 ggml 的 U01 名字/版本/descriptor/image 由 F01–05、F09 处置；SGLang
   factory(ctx) 与 vLLM renderer/extension class 属同一通用创建旅程，Python import/GIL/模型策略不下沉；
   SGLang SG15、vLLM V12 与 ds4 D04 的 stop/drain/join 交 Sched/IO；ds4 没有新的通用 runtime Registry
   要求，其诊断与 typed lifetime 可用现有能力。当前未解决 MUST 只有冻结 FIX 所列，不声称五仓已采用 HYLE。
3. **内部复用**：`rg` 的产品 include/call 扫描未发现 util-runtime 直接消费者。Obs sink registry 有
   unregister/quiescence/dispatch 保活，不能替成一向冻结表；Mem/Sched/Graph 的 util_core helpers
   不等于 runtime 重复实现。SIMD/NUMA/reclamation/clock/wait/pool 不适用于这个短 startup 索引；
   现有 flat hash/string/Arena/标准 mutex 是同合同路线。teardown 单链只是私有 owning record 销毁顺序，
   不为它新增 intrusive factor 公开边；没有重复 runtime decision authority。
4. **性能**：P0；find→cache 的启动消费没有新 timing claim；核验 complexity、allocation、锁与
   release call chain。P1/P2、qualified ratio、fixed loss 均不适用，不续签旧点值。
5. **FFI**：无已准入 slice；两个 demangle decorated C++ exports 为 runtime 编译边界，Registry 是
   C++ template。所有新增测试私有 fault adapter 不进入安装产品；没有原生 C 产品承诺。
6. **三交付**：util-runtime 独立 factor；Util façade 聚合 util_core+util，完整闭包是
   core/error/platform/mem/container_intrusive/container/container_text 加两 util roots。source/static/
   shared、逐头、export/staging/relocation 将实际验证；本节尚不签 PASS。
7. **易用性/docs**：按注册/解析/创建/发布/销毁写 recipe，版本 gate 和 image owner 明确；产品与
   factor 三层命名校准，runtime capture 不借用 FunctionRef 冒充 owner。实现后检查旧函数指针假设、
   不适用 snapshot、旧 CLOSED/FROZEN 和当前性能文案；本节尚不签最终 PASS。

### 预声明取证矩阵

三平台 selector 同一 immutable candidate：只选 docs/closure、configure(util)、factor:util、
headers:util、delivery:util；其余 factor 测试、全库 component contract 和 P1 均跳过。
测试包含 stateless/captured C++ 例子与 allocator fault sweep、版本拒绝/优先级事务、creator/visitor
重入及 register/read/freeze race。WSL 同一 candidate 另跑 util-only ASan、UBSan、TSan；不启动
依赖单测、benchmark 或全库 CTest。demangle 平台 failure 注入为独立 util-owned test target，借私有
链接/编译 adapter，不能为测试改变生产 API/metadata。release codegen 和 exports 只读实际产物。

目前各 runtime 验证均 **PLANNED**，没有把历史 v3 PASS 复制到新候选。候选锁定后再逐项回填。

## 实现批次与候选准备

四项 FIX 已完成一次实现批次：Registry 以 immutable owning entry 保存 const noexcept callable，
Creator 是借用 handle，替换保留旧 entry；版本拒绝和 image/object 双重保活由 util package consumer
演示。独立 util 私有故障 executable 编译真实 demangle 源码，仅替换平台调用/C allocator，覆盖
平台错误、OOM、失败 sink 的释放与同 image DbgHelp 串行化；另在该隔离 executable 验证真实 public
owning string 的 `bad_alloc` 传播。未向产品增加 hook，也未修改公共测试基础设施。

selector 对 AUDITED baseline `a8f6ab8c` 的影响计划只选择 docs、closure、configure、factor:util、
headers:util、delivery:util（61 门跳过）；其中测试 CMake 只归 util，公开 Registry 头没有实际反向
产品消费者。实现后文档/closure checker 和 `git diff --check` 已通过；native 仍待不可变候选。

产品 commit/tree 先独立落盘，再用仅 card/registry/projection 的状态提交绑定 CANDIDATE。
三平台验证使用这个干净的状态提交作为 selector execution commit/tree；其 runtime/test/sample
输入与绑定的产品 commit 完全相同。终态分别记录产品身份与 execution tree，不把后续文档树冒充
已编译的产品变化。不存在 P1 claim 或新计时样本。

### 唯一具名纠正：UTIL-V4-TEST-PORTABILITY-01

首次 W1-WSL 编译暴露 GTest assertion 在裸 `if` 下触发 `-Wdangling-else`；W1-WIN 暴露测试
`strcpy` 的 C4996、始终 throw 的测试 visitor 在 template return 处触发 C4702。只修正 util 测试：
加 braces、按已知 NUL 长度 memcpy、按实际 entry 条件注入 visitor 异常，并固定 Windows 私有
adapter 的 windows.h/DbgHelp include 顺序。没有改变产品/API/算法/normal workload，CANDIDATE
产品身份仍为 `0fff17ef3abe6830dba58a344c3878199dfb21ea` / tree
`365f633cdf44153bb865526b3854cd9c1f04ec58`；测试修正后的 execution tree 单独绑定。

首次 Windows configure 本身完成，但用户同时提交 roadmap 文档 `d9e23b02`，selector 正确因树漂移
拒绝签证；此失败不算产品失败。保留该用户提交，改用 Windows canonical commit 的只读验证副本，
与 WSL/macOS 一样不作为产品来源。M1-MAC 首次 configure 因机器缺 Ninja 失败，补齐外部工具后
重新执行原选择门；不修改 HYLE 构建系统。以上初始失败证据保留，不写成 PASS。
