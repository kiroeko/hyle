# SIMD-06：Scalable backend、dispatch 与交付

- ticket state：`DONE`
- kind：product review/refactor
- depends on：SIMD-05
- budget：1 工程日；性能预算 60 分钟
- input commit：`6ac76122d5be832b0ad1bb5e0184cab0a7ef441e`

审查 SVE/RVV sizeless 合同、Runtime 表、exact/native/fat-kernel、Platform CPU facts、初始化/并发和
source/static/shared 组件交付。当前 public `Runtime` 函数指针表改为 private implementation table；普通
公开面只保留实际 selected target/features 的小型只读诊断，不再从 `hyle::simd` 转发 Platform detector。
Highway 是 SIMD kernel target 的唯一 decision authority，Platform 只拥有通用 CPU/OS facts。

真机只对现有机器签性能；SVE/RVV emulator 只签 correctness。runtime 比较必须计入双方等价 dispatch
成本，source-primary 入口必须独立可用。初始化一次构造整张表，不允许每个 op 分别探测或缓存
未初始化的 Highway trampoline。

不得新增 per-ISA DLL、稳定跨工具链 C++ ABI 或 Native FFI；不得把 internal selector 暴露为普通用户面。

## 资格化审计

- `hyle_target_simd()` 已能给源码 fixed-target TU 提供锁定 Highway 头，但仍复用了 HYLE legacy backend
  的 ISA flags。GCC/Clang 下 Highway 的 AVX2/SSE4 baseline 还要求 AES/PCLMUL/BMI 等完整 feature
  chain；缺位会让 `HWY_STATIC_TARGET` 静默降档。仅验证数值正确不能发现该问题，必须在生成 TU 对实际
  vendor target 做编译期断言。
- `hyle_add_fat_kernel()` 当前只是 `hyle_add_dispatched_kernel()` 的完整-tier校验包装；它生成的真实 fixture
  仍 include `native/algorithm.h` 和旧 HYLE D-tag backend，没有把 build-tree-only Highway include、
  `HWY_COMPILE_ONLY_{SCALAR,STATIC}` 或 vendor target proof 带入每个 TU。因此用户文档中的 vendor-first
  unknown-CPU journey 尚未成立。
- `Runtime` 仍是 installed public function-pointer table，公开了每个 family 的内部函数布局、table version
  与 `runtime_for()` selector；仓库外没有真实语法 consumer，仓库内普通用户样例只读取 selected ISA。
  `dispatch.h` 还把 Platform 的 `cpu_features/best_isa/supports/isa_name` 整体转发进 SIMD namespace，造成
  capability facts 与实际 compiled/selected kernel 两套公共入口。
- 现有 SVE/RVV QEMU 门能证明 legacy product variant，但 fat fixture 仍手写旧 backend；它没有证明 direct
  Highway 的 sizeless `ScalableTag` 在多个 VL/VLEN 中由同一 pointer/POD body 正确生成和执行。没有真硅，
  本票不签 scalable 性能。
- installed static/shared SDK 不交付 Highway register headers 或 build-tree helper；这是有意边界，不是缺失
  journey。预编译普通 API 只保留每 batch 一次 private table 间接调用；源码 fixed route 零派发，源码 fat
  route只允许缓存前一次选择。

## 冻结裁决与有限 FIX 集

1. 在中央 ISA helper 中区分 HYLE legacy flags 与 Highway static-target flags；fixed/fat Highway TU 使用完整
   vendor baseline，并由生成代码断言 `HWY_STATIC_TARGET` 等于声明 tier，禁止“名为 AVX2、实际 128-bit”。
2. 将 `hyle_add_fat_kernel()` 收敛为 build-tree-only direct-Highway seam：每个生成 TU 先冻结 scalar/static
   target，再模板内联同一份 pointer/POD body；不得形成 vendor shared ABI、逐元素 callback 或第二层
   Highway dynamic dispatch。经典 cl.exe 无 AVX3 时显式把该 consumer 的最高生成档裁为 AVX2，并让选择
   宏不引用不存在的 AVX-512 symbol；clang-cl/GCC/Clang 保留真实 AVX-512。
3. 将函数指针 `Runtime` 和 `runtime_for()` 移入 `src/simd` 私有面；installed public 面只保留
   `selected_target(family)` 的小型只读诊断。移除 `dispatch.h` 对 Platform detector 的转发；需要原始 CPU
   facts 的代码明确调用 `hyle::platform`，产品 family 继续共用一次构造的 private table。
4. 把 fat fixture 改为直接 Highway 模板 body，覆盖 Windows source fat、WSL source fat、macOS
   scalar/NEON，以及 SVE/RVV 多 VL/VLEN correctness；static/shared ordinary consumer 只验证 HYLE batch
   和小型诊断，不虚构 vendor-header journey。
5. 性能只验证机制：fixed/fat target codegen 与同 TU direct Highway 等价、ordinary 每 batch 恰有一次间接
   调用。若代表 row 在同 ISA/同 body 下无 material regression 即停止；不在本票改 dtype、activation、
   search/scan、bitpack/random、ordering body，也不接纳对应工作树草稿。

本票的停止线是上述五项及适用交付/正确性门完成；SVE/RVV 真硅性能、各 family 算法迁移和 legacy
backend 删除分别留给具名后续票，不得以“继续研究 dispatch”越界。

## 实现与验证结果

- `hyle_target_simd()` 与 `hyle_add_fat_kernel()` 现在都使用中央 Highway target flags 和生成 TU 内的
  `HWY_STATIC_TARGET` 编译期证明；固定目标直接实例化 vendor 模板，源码 fat 路线对整段 kernel 只选择并
  缓存一次函数指针。经典 cl.exe 明确裁到 AVX2，显式 AVX-512 请求 fail-fast，不再静默冒充目标。
- public function-pointer `Runtime` table 已移到 `src/simd/runtime_internal.h`；installed C++ 面只保留
  `selected_target(family)` 诊断。SIMD 不再转发 Platform 的 CPU detector，原始机器事实与实际 kernel
  选择各有唯一 owner。shipping DLL 的符号检查没有再导出 `runtime/runtime_for` 或 Platform aliases。
- Windows MSVC 19.50、WSL GCC 14、macOS AppleClang 21 的 direct Highway source-fat fixture 均编译运行；
  Windows fixed AVX2、exact AVX2 artifact、source/static/shared selected-SDK consumer 和 installed-SDK
  source-fat fail-fast 均符合声明的交付合同。
- 同一 direct Highway `ScalableTag` body 在 QEMU SVE VL 128/256/384/512/2048 和 RVV VLEN
  128/256/512/1024 下通过 correctness。旧 full scalable harness 先被 SIMD-08 拥有的迁移前 math symbols
  阻断，已作为具名后续事实登记；本票没有把未运行的全 family gate 写成 PASS，也没有声明模拟器性能。
- 受 public Runtime 收窄影响的六个 SIMD benchmark target 均完成接口迁移；其中五个成功编译，
  `hyle_math_scaling_bench` 的 direct Highway math include/target 缺口已路由 SIMD-08，未借本票修改 math
  产品。文档合同、closure contract 与 staged diff check 均通过。

## 有界性能 verdict

本票签 `PASS / STRUCTURAL`，不制造跨算法 SOTA 点值：固定目标和 source-fat 的最终 AVX2 loop 已反汇编为
直接 `vaddps/vmaxps/vmovups`，循环体内没有 HYLE wrapper、per-element callback 或第二次 vendor dispatch；
fixed route 为零 runtime dispatch，fat route 和 ordinary route 分别只有一次整-buffer 函数指针调用。该机制
与同 TU direct Highway 等价，已经排除“引用第三方导致内层抽象开销”。各 family body 的同合同 P1 结论仍
由 SIMD-07～11 分别签署，SVE/RVV 真硅性能保持未声明。

## 终止 receipt

- product output commit：`3fb8e5a2b923df3b27bb98e04ae7f2f5f5ca72c3`。
- 四项 LOC 由上述 immutable commit 以 `cloc 2.08`、
  `python scripts/loc-report.py --module simd --commit 3fb8e5a2 --self-check --json` 复算，`self-check=pass`：
  全库代码 **310,189**；自实现包含测试 **121,167**；自实现不包含测试 **79,583**；当前 SIMD 实现
  **33,900** 行。
- known limits：classic cl.exe source-fat 最高 AVX2；installed SDK 不交付 vendor register headers；
  scalable ISA 只签 correctness；legacy family backend 要等其直接消费者在 SIMD-07～11 迁走后由
  SIMD-12 删除。
- reopen：编译器/Highway target 语义使静态目标证明失效，source/fat/ordinary 路线出现循环内二次派发，
  三种交付不再满足声明合同，或真机同合同 workload 证明该 dispatch 机制有 material regression。
