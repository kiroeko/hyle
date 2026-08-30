# SIMD v4：统一 expert provider SDK

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只修复 Highway/xsimd expert 能力在三种 C++ 交付之间断裂的问题。上一轮
> [`simd-v3`](simd-v3-provider-routing.md) 的 ordinary API、387-operation 迁移账本、422 个 route scope、
> 实现选择和 P1 性能结论不在本轮重做。

## Reopen fact

- trigger ID：`SIMD-V4-DELIVERY-01`；
- baseline：`38dd0be938a19d7989e80daeb859139ebad14e52`；
- 可复现事实：源码消费者可经 `hyle_target_simd()` / `hyle_add_fat_kernel()` 使用锁定 Highway/xsimd，
  但 installed static/shared SDK 明确不导出 expert target、helper 或 vendor headers。需要 HYLE ordinary
  API 尚未包装的 provider 能力时，三种受支持交付没有同一条用户路线；
- 击中的旧结论：公开交付形状发生变化，且真实 owner 明确要求 static/shared C++ 用户获得与源码用户
  相同的 provider-header 开发能力；这满足模块方法的 delivery/API-usability reopen 条件；
- 受影响范围：只包括 expert header capability 的 CMake target、fixed/fat helper、锁定第三方开发资产、
  package relocation/offline、示例和文档。

## Closure card

- component / owner / round / kind：`simd` / SIMD + build/package / `simd-v4-expert-sdk` / `REOPEN`；
- 真实 consumer task：C++23 用户在 HYLE 没有 ordinary 语义 API 的场景中编写一个 pointer/POD kernel，
  直接使用锁定 Highway 或 xsimd；同一 CMakeLists 必须能消费源码、installed static 或 installed shared
  HYLE，并能选择固定 ISA 或生成 HYLE 管理的 fat kernel；
- 范围：三种交付共同的 `hyle::simd_expert` build target；安装时自包含的精确 Highway/xsimd header
  资产；可搬迁的 `hyle_target_simd()` / `hyle_add_fat_kernel()`；selected-component、离线、许可证和
  source/static/shared consumer 验证；architecture/user/build 文档；
- 非目标：把 Highway/xsimd 全部函数改名为 `hyle::simd::*`；改变 ordinary SIMD API、387/422 route、
  kernel 算法、运行时 dispatch 或性能；承载必须编译上游 `.cc` 的 thread pool/VQSort/image/hash 产品；
  兼容 `hyle::simd_source_expert` 旧拼写；Native FFI/C API、稳定跨工具链 ABI、GPU/tensor/模型策略；
- 直接依赖：`core`、`platform`，以及中央锁定 Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c`
  与 xsimd `3b3151ff6655967c4d783d0534dde68fbb7b258f`；
- 直接消费者：独立 expert kernel fixture；普通 `hyle::simd` 用户不自动继承 vendor include/define；
- tier：`P0 / closure-correctness-v1`。本轮不改变运行时代码，性能要求是 fixed/exact 最终编译仍为直接
  provider code、fat path 仍只有既有 batch-boundary selection，不增加运行时 wrapper、allocation、lock
  或 metadata；timing comparison `NOT_APPLICABLE`；
- 总预算：90 分钟；实现 45 分钟、验证 35 分钟、红队与记录 10 分钟；不新增性能实验；最多一次产品
  修复重跑；
- machines：`W1-WIN` 为本地完整门，`W1-WSL` 与 `M1-MAC` 验证可用时运行相同的 source/static/shared
  expert consumer；不可达平台如实 `BLOCKED`，不能写成 `PASS`；
- strong peers：锁定 Highway/xsimd 本身；本轮只借用其原生 header API，不建立第三套 register peer；
- correctness/delivery gates：固定 ISA 编译与运行、fat kernel 编译与运行、vendor 类型不进入 HYLE
  binary ABI、source/static/shared、relocation、selected component、offline cache、许可证、负向缺资产；
- 预期输出：三种交付使用同一 target/helper/示例；ordinary 与 expert 成本隔离；完整 owner/升级边界；
- 停止条件：三种 journey 全通过且没有运行时产品漂移则关闭；若可搬迁 installed helper 无法在预算内
  重建准确 ISA/fat contract，进入 `BLOCKED`，不退回 source-only 假完整。

## 当前状态

`VERIFIED`。上述 closure card 与有限 `FIX` 集已落实到下列不可变产品身份；三平台、三交付、P0
结构成本、文档和最终红队门均已完成。产品 candidate 后只增加验证与 terminal record；若公开合同、
算法或正常运行时代码再变化，退回 `AUDITED`。

## Capability-family coverage map

| Family | admitted user journey / evidence | required variants and boundary | current route | coverage / action | negative space |
|---|---|---|---|---|---|
| ordinary semantic SIMD | 普通连续数组、推理 row/activation/dtype/search/sort；simd-v3 | source/static/shared 的 pointer/count/POD API，私有 provider route | `hyle::simd` compiled target | `COMPLETE / KEEP` | 不因 expert delivery 重做 API、算法、387/422 ledger 或 P1 |
| native provider header access | 用户需要 HYLE 尚未准入的 Highway/xsimd primitive/contrib header | 两个锁定版本；native namespace/type；只在 consumer TU；升级需重编 | 只有 source build 的 `hyle_simd_source_expert` | `PARTIAL / FIX` | 不把 vendor 函数逐个改名到 HYLE；不承诺上游 `.cc` library |
| fixed-target expert kernel | 部署 ISA 已知，希望宏/模板直接内联且无 runtime selector | scalar/SSE4/AVX2/AVX512/NEON/SVE/RVV；准确 compiler flags；不静默降档 | `hyle_target_simd()` 只接受 source target | `PARTIAL / FIX` | 生成物要求对应 ISA；classic cl.exe AVX3 继续 fail closed |
| distributable custom fat kernel | 未知客户 CPU 的自定义 pointer/POD kernel | 每 ISA TU；唯一 batch-boundary selection；provider 类型不跨 TU | `hyle_add_fat_kernel()` 只接受 source target | `PARTIAL / FIX` | 不支持逐元素 callback、第二层 vendor dispatch 或任意寄存器 ABI |
| self-contained development SDK | headers+static/shared 用户离线编译上述 kernel | provider header、HYLE expert header、CMake helper、license、relocation | installed package 明确删除这些资产 | `MISSING / FIX` | 不在 package consumer configure 时联网，也不共享 dependency binary state |
| compiled Highway contrib | thread pool、完整 VQSort/image/hash 等需要 upstream `.cc` 的产品 | 各自语义 owner、二进制/异常/线程/交付合同 | 未由 expert header target 构建 | `OUT_OF_SCOPE / REJECT` | 后续按 Sched/Container/SIMD owner 的真实 demand 单独准入 |

所有本轮 `MUST` 都集中在同一 delivery journey。ordinary API 为 `COMPLETE`，compiled contrib 明确
`OUT_OF_SCOPE`；三个 `PARTIAL` 与一个 `MISSING` 必须由下列有限 FIX 变为 `COMPLETE`，否则本轮阻塞。

## Capability / gap ledger

| ID | class / source | exact gap | owner | cost/complexity target | action | verification |
|---|---|---|---|---|---|---|
| `SIMD-V4-TARGET-01` | `MUST` / real delivery | build target 名称和存在性按 source/static/shared 分裂 | SIMD semantic；build/package delivery | INTERFACE only；普通用户零 compile/runtime 传播 | 将旧 target 替换为三种交付共有的 `hyle::simd_expert` 并导出 | 三种 consumer 查询/链接同名 target |
| `SIMD-V4-ASSET-01` | `MUST` / dependency delivery | installed SDK 不含锁定 Highway/xsimd headers | vendor artifact；SIMD/build delivery | development package only；普通 `hyle::simd` 不继承 include | 安装精确锁定 `hwy/` 与 `include/xsimd/`，通过 expert target 的 relocatable include path 投影 | relocated package 在断网环境直接 include 两家 header；license registry 通过 |
| `SIMD-V4-HELPER-01` | `MUST` / usability | installed config 不加载 ISA/fat helpers，函数还显式拒绝 package | SIMD/build | fixed path 零 runtime；fat path维持一次边界选择 | 安装并加载同一 `HyleIsa/HyleDispatch` 实现；helper 统一链接 expert target，记录 package target/exact facts | fixed codegen guard；fat sample run；错误 ISA/缺 target 失败关闭 |
| `SIMD-V4-HEADER-01` | `MUST` / public journey | `target.h`/`fat_kernel.h` 被 installed selected-header 排除 | SIMD | header-only；无常态 metadata | 将两头作为 expert public headers 安装；`provider/*` 继续 internal | installed header inventory、self-contained compile、provider directory absent |
| `SIMD-V4-DOC-01` | `MUST` / discoverability | architecture/user/sample 声称 expert source-only，且容易把 422 当 provider 全集 | SIMD/build docs | N/A | 明确两层 API、三交付同形、原生 namespace、compiled contrib 负空间与升级重编 | `check-docs.py` + sample commands |

## Public-surface trace 与 owner 裁决

- `hyle::simd` ordinary headers：仍映射到 simd-v3 已准入的完整用户任务，`KEEP`；不传播 expert target，
  普通用户不会为未使用 vendor header 支付编译或运行成本。
- `hyle_simd_source_expert` / `hyle::simd_source_expert`：名称把交付偶然性写进产品，且只存在 build tree；
  `DELETE`，预发布项目不保留兼容 alias。
- `hyle_target_simd()`：自然 fixed-target 任务，`FIX` 为三种交付同实现；定义
  `HYLE_SIMD_EXPERT=1`，不再传播 source-only 宏。
- `hyle_add_fat_kernel()` 与 `HYLE_SIMD_SELECT_FAT_KERNEL`：自然 unknown-CPU custom-kernel 任务，
  `FIX` installed helper/header/link closure；原复杂度和 dispatch contract 不变。
- `provider/*` 与 route manifest：HYLE 产品实现内部，ordinary route ledger 使用；不因用户能直接取得
  Highway/xsimd 而安装或成为 expert API。
- Highway/xsimd 原生类型：有意的 expert interop capability，只能停留在 target-bound consumer TU；
  HYLE ordinary/shared ABI 继续只用 pointer/span/POD。

第三方 owner：Highway/xsimd 拥有 native header 语义和实现；HYLE SIMD 拥有 expert 使用边界、ISA helper
与 fixed/fat journey；build/package 拥有锁、安装、离线、许可证和 relocation。用户显式选择 native provider；
HYLE 不为任意未准入 operation 冒充自动 provider decision authority。若某任务成为 ordinary API，才另行
建立 HYLE semantic contract 与有限 route。

## 冻结 FIX 集

只允许：重命名并导出 expert target；增加其 install interface；通过现有 dependency package registry 安装
锁定 provider development tree；安装/load 同一 helper；补 expert HYLE headers；把现有 SIMD sample 和
selected-package smoke 扩成三种交付同一 journey；更新当前 architecture/user/build 文档。禁止修改
`src/simd/**/*.cc`、ordinary public declarations、route ledger、性能 harness 和中央 closure 工具。

## Candidate

- product commit / tree：`1e543dec85ad6cef411f4499f7449294a2507b1c` /
  `4e4e82e5e22e09f55c7af165b45de9d4dfbb4896`；
- `hyle::simd_expert` 在源码、installed static、installed shared 中同名存在，只向显式消费 target 投影
  `HYLE_SIMD_EXPERT`、锁定 Highway/xsimd include 与 C++23；ordinary `hyle::simd` 不链接或传播它；
- installed development assets 是锁定 Highway 的完整 88 个 `.h` 与 xsimd `include/xsimd`，不携带
  Highway 的测试、benchmark 或 compiled `.cc`；`target.h` / `fat_kernel.h` 安装，`provider/*` 保持内部；
- 安装后的 `HyleConfig.cmake` 加载与源码相同的 `HyleIsa.cmake` / `HyleDispatch.cmake`，fixed helper 仍是
  compile-time direct provider code，fat helper 仍在 pointer/POD whole-kernel 边界选择一次；
- `src/simd/**/*.cc`、ordinary declaration、387-operation / 422-route ledger、算法、layout、常态 metadata
  与性能 harness 均未变化，所以本轮 timing peer comparison 为 `NOT_APPLICABLE`；P0 只验证零新增运行时
  wrapper/allocation/lock/dispatch 和现有 codegen gate；
- Windows candidate-bound 预验证：SIMD 5/5 correctness/codegen；source、relocated shared、relocated
  static expert consumer 各 1/1，均直接编译运行 Highway + xsimd fixed object 和 custom fat kernel；文档、
  closure checker 与 route verifier 通过。跨平台和最终红队 verdict 见后续 verification manifest。

## Verification manifest

验证始终绑定 product commit `1e543dec85ad6cef411f4499f7449294a2507b1c` / tree
`4e4e82e5e22e09f55c7af165b45de9d4dfbb4896`。用于 WSL/macOS 的 clean source archive SHA-256 为
`EEE0092EA1DA0EB5BE1525731B3363883EE83A81B4DCBF91D84B0441575EBA69`；candidate 到本次验证提交之间，
`CMakeLists.txt`、`cmake/`、`include/`、`src/`、`samples/` 与 `scripts/` 的产品 diff 为零。

| machine / toolchain | source consumer | relocated shared SDK | relocated static SDK | verdict |
|---|---:|---:|---:|---|
| `W1-WIN` / MSVC 19.44 + Ninja | 1/1 | 1/1 | 1/1 | `PASS` |
| `W1-WSL` / GCC 14.2 + Ninja | 1/1 | 1/1 | 1/1 | `PASS` |
| `M1-MAC` / AppleClang 21 + Unix Makefiles | 1/1 | 1/1 | 1/1 | `PASS` |

每条 journey 都使用同一 package sample，直接编译 Highway 与 xsimd fixed-target object，生成并运行
HYLE custom fat kernel，并在断网可用的 relocated SDK 中查找 package。Windows 另运行 SIMD
correctness/provider route/provider codegen/provider winner/ordering 5/5；route verifier 保持 387 operation、
422 route scope、零 pending。installed inventory 含 Highway 88 个 `.h`、零 `.cc`、完整 xsimd public
header tree、两个 helper 与 expert HYLE headers；不含 `include/hyle/simd/provider`。文档检查为 330 个
Markdown、934 条本地链接全通过。

风险矩阵：固定 ISA 直接 code、custom fat dispatch、空/尾部与 provider 组合由同一真实 sample 和既有
SIMD tests `PASS`；source/static/shared、relocation、离线依赖、许可证、selected component、普通/expert
隔离和 public-header inventory 均 `PASS`。本轮不新增对象状态、allocation、OOM、锁、线程、取消、销毁
或 C/FFI 边界，相应生命周期风险 `NOT_APPLICABLE`；运行时 SIMD correctness 与 sanitizer 结论继承未变的
simd-v3 candidate，本轮没有重写算法来伪造新结论。

P0 engineering performance verdict 为 **`PASS / STRUCTURAL`**：fixed path 只产生 consumer 自己的 provider
模板/宏代码，fat path 保持既有 whole-kernel batch-boundary selection；没有新增运行时 wrapper、逐元素
provider decision、allocation、lock、syscall 或常态 metadata。timing peer comparison
`NOT_APPLICABLE`，本轮不宣称新的 operation winner 或 SOTA。

## Final red team

| finding | disposition / evidence |
|---|---|
| 安装整个 Highway tree 会把测试、benchmark 和 compiled contrib 混入 SDK | `FIXED BEFORE CANDIDATE`：只安装递归得到的 88 个 `.h`；三平台 inventory 均无 `.cc` |
| ordinary `hyle::simd` 可能无意传播 vendor include/define | `DISPROVED`：package configure 的负向 target-property assertion 与三种 journey 通过；只有显式 helper/target 取得 expert surface |
| provider register/type 跨 shared/static ABI | `DISPROVED`：fixed object 与每 ISA TU 在 consumer 侧编译，HYLE binary 边界仍为 pointer/count/POD；export target 为 `INTERFACE` |
| installed helper 可能不知道 producer processor/exact target | `DISPROVED`：两项 facts 作为 exported target property 搬迁，x86_64 Windows/WSL 与 arm64 macOS 均生成正确有限 ISA 集 |
| header target 可能暗示完整 compiled Highway contrib 受支持 | `FIXED BY CONTRACT`：architecture/user guide 明确 thread pool、完整 VQSort/image/hash 等 `.cc` 产品不属于本轮，须按真实 owner 另行准入 |
| vendor 资产缺 license 或 configure 时联网 | `DISPROVED`：安装 HYLE notices 与 Highway license；relocated consumers 只读 SDK 自带头，无下载步骤 |
| 为交付缺口顺手改变 ordinary route/算法/性能 harness | `DISPROVED`：candidate 后产品 diff 为零；本轮 candidate 本身未修改 `src/simd/**/*.cc`、公开 ordinary declaration、route ledger 或性能工具 |
| WSL 默认选到 GCC 13，反复在 configure 才失败 | `OUT_OF_SCOPE BUILD ENTRY DEFECT`：顶层按合同 fail closed；同一 clean candidate 显式绑定 GCC 14.2 后三条 journey 全通过。统一 Linux/WSL 工具链入口由独立 build/package task 处理，不在 SIMD 候选中扩建构建系统 |

红队没有发现未处置的 SIMD `MUST`、correctness/security/ownership/交付缺陷或 material runtime 成本。
`CANDIDATE -> VERIFIED` 的 pre-transition registry snapshot 为
`C:\hyle-scratch\simd-v4-expert-sdk\module-status-before-verified.json`，SHA-256
`6AD2DE92C29304A85735EEF2DC0704C85400E8DCC346BEB6C9A8FCF31687722C`。

## Owner 六问（VERIFIED）

1. **第三方：**产品 expert 路径直接使用锁定 Highway
   `f503c485cec6b4f2f596fa60d83484797c9eeb5c` 与 xsimd
   `3b3151ff6655967c4d783d0534dde68fbb7b258f` public headers；两家拥有 native 语义/实现，HYLE 拥有
   fixed/fat 组合与 ordinary route 决策，build/package 拥有锁、安装、离线与许可证。ordinary 产品仍按
   simd-v3 的逐 route owner 结论执行。
2. **功能：**ordinary 387-operation/422-scope 结论不变；其外的 provider header 能力现通过原生 namespace
   完整 `ROUTED`，而不是复制一套不完整 HYLE wrapper。需要上游 `.cc` 的产品、tensor/quant/GEMM/
   attention/RoPE/fusion/GPU 仍分别归未来真实 owner，不冒充 SIMD primitive 缺口。
3. **性能：**本轮为 P0 delivery-only；结构成本 `PASS`，timing peer comparison `NOT_APPLICABLE`。没有
   因 package 形状重测或改写既有 route winner，也没有新的 SOTA 声明。
4. **旧 C ABI / Native FFI：**产品树没有 SIMD C schema/header/bridge/export/sample/option；provider 类型
   只在 C++ consumer TU。Native FFI 仍为 `N/A`。
5. **构建与导入：**Windows、WSL、macOS 的 source embedding、installed static、installed shared、
   relocation、selected component 与 offline consumer 均 `PASS`；三种形状使用同一
   `hyle::simd_expert` 和 helper。
6. **C++ API 与文档：**普通任务继续选 `hyle::simd`；缺少语义 wrapper 时显式选 expert target，直接用
   `hwy::` / `xsimd::`，fixed/fat 示例可复制。文档明确版本升级需重编、类型/ownership 边界、普通用户
   零传播和 compiled contrib 陷阱。
