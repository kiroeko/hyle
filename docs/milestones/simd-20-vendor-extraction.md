# simd-20: vendor 摘除——库零 vendor 依赖,bench 引用外部 highway/xsimd

> **状态**:X0/X1/X2 已落地(2026-07-16); X3~X4 待执行。**动机**(用户):库里只有少数专项 kernel 借
> highway/xsimd,却把整库拖进 `third_party/` 当依赖,不应该;借用的代码移植进 HYLE 后，Highway/xsimd
> 仅作为外部 benchmark 对照。X0 已将 bench 接到外部 pin 目录；X1 bitpack、X2 math 已迁为 HYLE；
> 产品侧仅剩 vqsort 借 `third_party/highway`（X3）。
> **前提修正(盘点见 `docs/architecture/simd-vendor-surface.md`)**:初始借用面是 4 个——①超越函数
> ②vqsort ③bitpack ④simdutf(可选)。X1 后 bitpack 不借；**X2 后 math 不借**。`include/` 始终
> 零泄漏，移植只使用 HYLE D-tag 原语。

## 一、端态

1. `hyle` 库(include/ + src/)**零** highway/xsimd 依赖;移植代码带原始 license 头 + 出处注释
   (Apache-2.0 / BSD-3 保留在各移植文件头；项目已获授权，不另设根级 notices 文件)。
2. `third_party/highway`、`third_party/xsimd` 从仓库删除;bench 目标经 CMake 缓存变量
   `HYLE_BENCH_VENDOR_DIR`(Windows 默认 `C:/public/project-hyle`,Mac `~/vendor`,WSL 自定)引用
   外部副本,**只有 bench 目标**挂这些 include。
3. 可复现性:`cmake/vendor-versions.cmake` 记 pinned SHA;`scripts/fetch-bench-vendors.sh`(+ ps1)
   按 SHA clone 到外部目录。防漂移闸门:`scripts/check-no-vendor.sh` 断言 include/+src/ 无
   `#include.*hwy/|xsimd` 命中,进常规验证流。
4. 性能不回退:每个移植 kernel 与原借核 A/B(op_parity 框架,借核走外部 vendor 编译)——**摘除
   期间旧借核天然还在,就是现成对照组**。

## 二、分阶段(X0/X1/X2 已落地，X3 每卡常绿)

**X0 — 构建管线先行(已落地 2026-07-16)**
- bench 的 Highway/xsimd include 全部改根到 `HYLE_BENCH_VENDOR_DIR`：Windows 默认
  `C:/public/project-hyle`，macOS 默认 `~/vendor`，WSL/Linux 必显式指定。`cmake/vendor-versions.cmake`
  是唯一 pin 源；CMake 同时校验 checkout 的路径、`HEAD` 与 clean `git status`。`scripts/fetch-bench-vendors.sh`
  与 `.ps1` 读同一 manifest，拒绝覆写非 Git 或脏 checkout。`bench` preset 使用隔离的 `build-bench`，
  `win`/`wsl` preset 显式复位 `HYLE_BUILD_BENCH=OFF`，防止一次 bench 配置让随后的产品 build 意外
  依赖外部目录。
- `simd-19-workorders.md` 的 Mac 同步明确分开产品 `third_party/` 与含 `.git` 的外部 bench 对照；各平台
  setup/bench 文档使用同一变量与 fetch 流程。产品 `third_party/{highway,xsimd}` 此时仍只服务
  `src/simd/**` 借核，故外部目录不会进入库编译。
- 验证：Windows MSVC 编译 `hyle_simd`、math/bitpack/sort/parity bench，`hyle_simd_test` 331 PASS +
  1 个无 BF16 硬件 SKIP、AVX2 221 PASS；WSL GCC14 同为 331 PASS + 1 SKIP、AVX2 221 PASS；
  Apple Silicon 编译 NEON、math/bitpack/sort/parity bench，`hyle_simd_test` 328 PASS + 1 SKIP。三端
  均以 pin checkout 配置成功；Windows/WSL `hyle_simd_bench` 与 Mac `HYLE_BENCH_ONLY=Add`
  `hyle_op_parity_bench_neon` 均真实运行。WSL 另以 `HYLE_BUILD_BENCH=OFF` 与不存在的
  `HYLE_BENCH_VENDOR_DIR` 构建 `hyle_simd`，证明产品库不读取外部 benchmark vendor；反向以
  `HYLE_BUILD_BENCH=ON` 配置不存在的根，CMake 在 `Highway/hwy/highway.h` 处按预期拒绝。
- `cloc --diff` 对 067e0fa 的 X0 机制净增 **215 code LOC**：CMake 85、Bourne shell 72、PowerShell 58。

**X1 — bitpack 移植(已落地 2026-07-16)**
- `src/simd/bitpack/bitpack_hyle-inl.h` 将 Highway `Pack8<1..8>`、`Pack16<1..16>`、
  `Pack32<1..32>` 逐行翻为 HYLE D-tag 原语；三份 `bitpack_hyle_{avx2,avx512,neon}.cc` 以各自
  ISA 编译，原 Highway bitpack TU 已删除。dispatcher、ABI、`count < 128` scalar cutoff 与 scalar
  tail 均未改变。
- `bitpack_hyle_u16-inl.h` 保留 11/13-bit 修正布局：13-bit rawF 用 `0x8000` scatter mask，两个
  inverse 保持先选源向量再 shift/mask。产品 bitpack 源目录无 `hwy`/`hn::`/`HWY_*`。
- benchmark 保留外部、pin 校验的 Highway 直接行；11/13 另保留**正确**的 pre-X1 Highway-primitive
  reference，和已知错误的 Highway upper-bound 分开，后者不作正确性或性能判定。
- Apple 的同 ISA 对拍揭示 HYLE generic `Xor3` 曾降成两个依赖 EOR，而 Highway 使用 SHA3 `EOR3`；
  NEON full-vector `Xor3` 现以 `__ARM_FEATURE_SHA3` 门控 `veor3q_u8`，objdump 确认 bitpack 内核
  发出 `eor3.16b`，D-band 仍走原两 Xor 实现。
- 验证：Windows MSVC `hyle_simd_test` 331 PASS + 1 BF16 SKIP；WSL GCC14 同为 331 PASS + 1 SKIP，
  并以 SDE `-future` 跑 AVX-512 bitpack 10/10 PASS；Apple Silicon `hyle_simd_test` 328 PASS + 1
  SKIP、`hyle_simd_neon_test` 276 PASS。三端均构建并运行 `hyle_bitpack_bench`，外部 vendor SHA/clean
  gate 通过。
- 同 ISA 对拍（12 次、`n=65536`）：通用 u16 10-bit pack 的 HYLE public / Highway direct 为 Windows
  1.015（TIE）、WSL 0.995（TIE）、macOS 1.016（WIN）。修正 13-bit unpack 对正确 pre-X1 reference：
  Windows 1.156（WIN）、WSL 1.006（TIE）；macOS 在 `n=1048576` 为 1.004（TIE），11-bit 为 1.010
  （TIE）。
- 对 X0 后的 scoped tree 执行 `cloc --diff`：新增 **2,562** code LOC、修改 **217**、删除 **38**；主体是
  2,474 行 HYLE Pack8/16/32 port，另有三 ISA TU 与 benchmark correct-reference control。

**X2 — 超越函数移植(已落地 2026-07-16)**
- Highway 算法侧：`math_hyle-inl.h` 移植 `hwy/contrib/math/math-inl.h`（Apache-2.0，SHA
  `83eff393…`）；f64 exp/log **融合 P2-1**，继续走 `table_f64.h` N=128 表核（不回退 `hn::Exp/Log`）。
  ISA TU `transcendental_highway_{avx2,avx512,neon}.cc` 改 HYLE D-tag + `HYLE_SIMD_TARGET_*`。
- xsimd 算法侧：`math_xsimd_hyle-inl.h` + `math_xsimd_body.inc` 移植产品赢家
  （exp/exp2/exp10/tanh/erf/erfc/tgamma/hypot_in_range/fdim/fmod 等，BSD-3，SHA `5d2490f…`）；
  `transcendental_xsimd_{avx2,avx512,neon}.cc` 与 `complex_hyle-inl.h` 去掉 `xsimd::`。
- best-of dispatch **保留**：符号名仍为 `*_highway_*` / `*_xsimd_*`，只是两侧都是 HYLE 代码。
  `lgamma`/`remainder`/`nextafter` 仍 scalar。
- 产品 CMake：`hyle_simd` **不再** include `third_party/xsimd`；`third_party/highway` 仅 vqsort（X3）。
- 支撑修：`ConvertTo` f64↔i64（avx2/neon/avx512）、`RebindMask` 源类型、`DemoteToNearestInt` partial-tag、
  `[[maybe_unused]]` 属性序（GCC）。
- 验证（真编真测）：
  - Win MSVC：`Math.*` 24/24；`hyle_simd_test` 331 PASS + 1 BF16 SKIP；P2-1 `exp≤2` / `log≤1` ULP
  - WSL GCC14：同 24/24 Math + 331 PASS + 1 SKIP
  - Mac Apple Silicon NEON：24/24 Math；`hyle_simd_test` 328 PASS + 1 SKIP
- `cloc` 新增移植核 **2,231 code LOC**（`math_hyle-inl.h` 1,712 + `math_xsimd_hyle-inl.h` 510 +
  body.inc 9）；既有 TU/后端/CMake 为接线与支撑改动。

**X3 — vqsort 移植(排序面不可由 radix 替代)**
- `hwy/contrib/sort` 体量大(排序网络 + traits + 分区)。R5 已落地 radix sort/argsort，但它只在大规模
  整数键排序有明确优势；小 n、浮点排序、partial/select 与完整 KV 面仍依赖 vqsort。因此 radix 是
  独立算法件，不能作为不移植 vqsort 的理由；X3 按 X2 模式逐模块 port。
- 闸门:正确性穷举(现有 sort 测试)+ bench vs 旧借核(sort 是招牌件,持平才准换)。

**X4 — 摘除与收口**
- 删 `third_party/{highway,xsimd}`;`check-no-vendor.sh` 进闸;三平台全绿(gtest + bench 锚点);
  `simd-vendor-surface.md` 改写为"库零 vendor,bench 外部对拍"终态;cloc 增长写回本文。
- simdutf:默认关的可选项,**本里程碑不动**,单独决策(要么同模式移植 UTF-8 计数核,要么保留
  可选外部依赖并同样挪出仓库)。

## 三、工单化(交中上 AI 的部分)

- X1/X2 逐函数卡片完全复用 `simd-19-workorders.md` 的铁律/闸门/台账机制,追加两条专属铁律:
  1. **移植 = 语义逐行对照翻译,禁止"顺手改进"算法**;发现疑似 vendor bug → STOP 上报(前例:
     Pack16、xsimd lgamma),不准静默绕。
  2. **license 头与出处注释是卡片验收项**(文件头:origin 文件 + SHA + 原 license 标识)。
- 卡片正文(每函数 before=vendor 源码摘录 / after=hs:: 目标形 / oracle 命令 / bench 命令)由
  高能力方在 X0 后生成,追加到本文档。
- X3 决策、winner 表复核、X4 删除与收口 = 高能力方。

## 四、风险

- **xsimd 翻译的语义陷阱**(batch 语义与 hs:: 细节差,如 NaN 传播/舍入)→ 靠精度 oracle 逐 ULP
  兜底,卡片要求贴特殊值测试输出(±0/±Inf/NaN/denormal)。
- **移植后性能回退** → 每卡 bench A/B 持平方准合入;回退即 revert 归因(证据阶梯照矩阵)。
- **外部目录漂移**(有人手动改 benchmark 副本)→ CMake 配置同时拒绝错 SHA 与脏 checkout，fetch
  脚本从 `vendor-versions.cmake` 的唯一 pin 源恢复。

## 五、验收

1. 库零 vendor:`check-no-vendor.sh` 绿;外部目录改名后库构建仍成功。
2. 三平台 gtest 全绿;math 精度 oracle 全绿(ULP 与现状持平)。
3. 移植 kernel bench A/B vs 原借核:全部 TIE 以上(校准仪器判定)。
4. `third_party/{highway,xsimd}` 已删;fetch 脚本可从零重建外部目录;移植文件头 license/出处齐。
