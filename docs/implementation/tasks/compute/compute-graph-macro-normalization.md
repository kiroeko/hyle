# Compute + Graph 宏规范专题

<!-- MODULE REVIEW: SCOPED -->

2026-10-08 用户要求加入任务，使全部 Compute + Graph 的宏使用规范。
这是当前 [CUDA goal](cuda-final-closure-plan.md) 的附加横切任务，执行时按实际文件 owner
合法定界；本登记不自动重开 CLOSED／BLOCKED、不恢复其它路线施工队列。

## 范围与要求

- 覆盖 `include/hyle/compute`、`include/hyle/graph`、全部 `src/compute*`／`src/graph*`，
  以及直接决定这些源码条件的 owning CMake；测试同步必要的宏用法和语义检查。
- OS、目标架构、编译器事实使用已有 `hyle/core/environment.h`，不得各模块重复定义检测。
  `HYLE_OS_WINDOWS`／Linux／macOS 和 `HYLE_ARCH_*`／`HYLE_COMPILER_*` 是恒定义的 0/1，
  使用值条件 `#if`，不能用 `#ifdef`／`defined` 判真。
- 区分 OS 与 compiler／ABI：`_MSC_VER` 原条件可能包含 clang-cl；迁移时按实际语义使用
  `HYLE_COMPILER_MSVC` 或 `HYLE_COMPILER_MSVC || HYLE_COMPILER_CLANG_CL`，不机械换名字。
- 路线／第三方 feature、静态／共享与导出宏按真实配置合同核对：宏缺失、0/1、numeric version、
  presence-only 不混用；检查公开头、编译 TU、安装 consumer 是否一致。必要 include 要显式、可独立编译。
- CMake 的 `WIN32`／`CMAKE_SYSTEM_NAME` 是构建时目标事实，与 C++ HYLE 宏分层处理；
  不把 C++ 宏直接写入 CMake，也不根据维护宿主替代目标平台。
- CUDA host/device、NVCC、SM/AOT、Metal／Vulkan shader 和第三方版本条件按原本语义保留；
  先判断项目是否已有同等表达，没有则保留具名原条件，不能把 device 宏误替换成 host 架构。
- 不修改借入第三方原数学和 vendor 的原宏系统；HYLE owning adapter／补丁中的平台 glue 属本次审计。
  不新建一套平台宏，不增加通用配置框架，不顺手改算法、路由资格或底层 Core 已闭合同。

## 执行顺序与有限预算

1. 当前 N1 产品批次完成集中构建／验收后，一次扫描上述完整范围，分类 OS／arch／compiler／ABI、
   feature／visibility、GPU／vendor，形成路径／owner／旧条件／正确条件／风险清单。
2. 只对实际需要改变的 owner 绑定 baseline、scope 与有限 FIX；用户明确新要求是重定界事实。
   CPU 数学／性能结论不重开；公共 Graph 旧 Vulkan 性能 blocker 不自动解除。
3. 批量迁移，保留精确逻辑、默认、include/lifetime/API 和 source/static/shared 行为；一次稳定候选验证。
4. predeclare：平台条件的 Windows/Linux/macOS 编译与 headers、clang-cl ABI 条件静态核对／适用实际编译、
   CPU-only／GPU-OFF 负空间及四路线 owning 配置、直接 consumer／三交付宏传播；
   复用与 tree/contract 身份匹配的既有证据，不能用本机编译代签跨平台。
5. 运行 selector／docs／closure／diff／统一 LOC。纯逻辑等价宏／include 规范变化不启动 P1；
   如发现真实 feature 或 ABI 错误，先记录具体反例并单独固定语义修复范围，再选适用验证。

预算为一次扫描、一个批量修复、至多一次具名纠正；不对本专题展开全库算法审查。
完成须有全范围分类账和实际适用验收，未运行项不签 PASS；本页存在不计作已完成工作。

## 初次只读扫描

扫描当前全部上述C++／GPU源码／测试条件，110个文件含437条条件／定义：OS48、架构10、
编译器／ABI11、feature/version81、visibility154、其它133；未见将HYLE恒定义整数fact用presence判真。
这只是原条件分类，不签规范完成；CMake及借入补丁接缝还须按具体语义审计。
原inventory保留于 [N1归档](../../../../evidence/correctness/cuda-final-u3-artifact-carrier/README.md)。
N1还已提供实际feature传播反例：Provider私有边不能自动让普通route consumer看到其组装能力，
现已在 owning route修复；此专题要核对其它路线同类合同，不能用宏名替换代替传播检查。

## Closure card：宏与现有目标事实的统一

- baseline：`a3e6046b053c7a40dc52c703ea049a38ec7795d5`；Windows工作树干净，后续修改保留。
- owner：compute（route_policy公开平台事实）、compute_provider_native、compute_provider_ggml、
  compute_ggml_runtime、compute_provider_ggml_cuda、compute_provider_ggml_vulkan、compute_provider_iree_vulkan。
- trigger：用户明确要求全部Compute/Graph宏规范；扫描找到下表具体原生OS／compiler／ARM重复条件。
  已闭CPU owner仅为逻辑等价宏／include规范重定界，不重审数学、清退或性能，不解除公共Graph或Vulkan旧blocker。
- 已活动runtime／ggml CUDA保留 [U3 G1/G2](cuda-final-u3-mature-bindings.md) 全部有限FIX；本专题只增加宏规范。
  ggml Vulkan保留 [新增格式明确拒绝](cpu-quant-mtp-c3-format-bindings.md) 范围和原限制；
  IREE当前candidate只因其HYLE编译器条件修改退回AUDITED，原能力／资源／性能限制不变。
- 主consumer：同一Compute/Graph源代码经Windows／Linux／macOS源码／static／shared消费时的平台、
  provider availability、符号可见性一致；反偏置：CPU-only／GPU-OFF及clang-cl的MSVC ABI。
- Scope：目标事实、明确include、必要owning CMake include路径、同等编译条件与验证分类账。
  API／layout／新数学／provider顺序／限值／driver资格／队列、资源、完成和错误行为不变。
- deps：只复用已有Core environment／visibility；不改Core，不新定义事实。所有GPU原版vendor条件KEEP。
- P0：预处理条件与已有0/1事实相同，不增加运行字段、分支、分配或查询；无P1。
  原有限性能资格条件包括`_MSC_FULL_VER`精确版本继续保留，不扩大任何证书。
- 验证：全范围条件账、真实Windows MSVC及clang-cl事实、Linux GCC／Mac ARM Clang平台与公开头；
  selected owning factors和直接consumer、四路线feature／GPU-OFF公开头、source/static/shared宏传播，
  CPU独立AVX2 program真实export/load。未运行项UNVERIFIED，等价宏不重跑P1。
- API/owner/成本冻结：常态二进制语义不变，只有预处理事实来源与私有导出拼写统一；
  include路径只补CPU program直接消费既有HYLE visibility的必要头。
- 停止：有限FIX与适用编译／条件／交付验证完成后关闭本专题；新能力问题单独具名，不无限扩张。

## 有限 FIX / KEEP 账

| ID | 文件／owner | 当前事实 | 动作 |
|---|---|---|---|
| M1 | include/hyle/compute/route_policy.h；compute | 公开build platform重复识别三OS | 显式Core environment；HYLE_OS值条件；枚举不变 |
| M2 | native/aten_cpu_provider.cc、torch_cpu_program.cc、default_rules.cc | `_MSC_VER`含clang-cl；默认证书精确Windows／release／MSVC版本 | 显式environment；MSVC或clang-cl warning条件，OS／build事实统一；版本常数KEEP |
| M3 | ggml/ggml_cpu_provider.cc | ARM F16内核不是F32累加，需要保留准入拒绝 | HYLE_ARCH_ARM64同target条件；数学与拒绝KEEP |
| M4 | ggml_runtime/cpu_program.cc、cpu_execution.cc | Windows loader／Win32错误域 | HYLE_OS_WINDOWS，loader与错误不变 |
| M5 | ggml_runtime/cpu_program_export.cc、cpu_program.cmake | 私有同revisionCPU程序独立export重复平台拼写 | 借HYLE_EXPORT；owning target补既有include目录；ABI／visibility相同 |
| M6 | ggml_cuda/ggml_cuda_provider.cc | Windows getenv_s与原DLL timer初始化 | HYLE_OS_WINDOWS，SDK动作原样 |
| M7 | ggml_vulkan/native_device.cc | 原平台loader文件名 | HYLE_OS_WINDOWS/MACOS，三个原字符串不变 |
| M8 | iree_vulkan/runtime_internal.h | 原C头warning范围包含MSVC兼容前端 | 显式environment；MSVC或clang-cl；原警告范围KEEP |
| K1 | 所有HYLE_STATIC／*_BUILD／feature宏 | owning CMake仅选中时定义；factory有显式guard或disabled返回invalid合同 | KEEP presence-only，核对定义／消费／安装传播；不把关闭未定义改成定义0 |
| K2 | GPU／vendor／原dtype／版本条件、WIN32_LEAN_AND_MEAN／NOMINMAX | SDK或device编译合同，不是目标事实重复实现 | KEEP；原数学／SM／shader／SDK阈值不改 |
| K3 | CMake WIN32／MSVC／CMAKE_SYSTEM_NAME／ISA flags | CMake目标OS／ABI与源码facts分层 | KEEP目标条件；无新增宿主探测或统一宏系统 |

上述owner原record的能力与限制都仍有效，只有本轮有限宏变化获得施工范围；旧terminal不自动重新签绿。
新scope先独立提交，AUDITED后才写产品。
