# simd-20: vendor 摘除——库零 vendor 依赖,bench 引用外部 highway/xsimd

> **状态**:X0 已落地(2026-07-16); X1~X4 待执行。**动机**(用户):库里只有少数专项 kernel 借
> highway/xsimd,却把整库拖进 `third_party/` 当依赖,不应该;借用的代码移植进 HYLE 后，Highway/xsimd
> 仅作为外部 benchmark 对照。X0 已将 Windows `C:\public\project-hyle\{highway,xsimd}`、Mac
> `~/vendor/{highway,xsimd}` 与 WSL 自选目录接入 bench；产品源码仍暂用 `third_party/`，直到每项 port
> 通过对照闸门。
> **前提修正(盘点见 `docs/architecture/simd-vendor-surface.md`)**:借用面不是"几个 op",是 4 个面——
> ①超越函数(最大,~25-30 函数 ×2 精度,逐 op best-of 借 Highway contrib/math 与 xsimd)②vqsort
> (`hn::VQSortStatic` 族)③bitpack(`hn::Pack8/16/32`)④simdutf(编译期可选,默认关)。
> 幸运的是 `include/` 零泄漏(vendor 全关在 `src/simd/**` TU),且 **HYLE op 面镜像 Highway API,
> "copy"实际是 `hn::`→`hs::` 移植,大半机械**——适合工单化交给中上 AI。

## 一、端态

1. `hyle` 库(include/ + src/)**零** highway/xsimd 依赖;移植代码带原始 license 头 + 出处注释,
   仓库根加 `THIRD_PARTY_NOTICES`(Highway Apache-2.0、xsimd BSD-3,允许复制,须署名)。
2. `third_party/highway`、`third_party/xsimd` 从仓库删除;bench 目标经 CMake 缓存变量
   `HYLE_BENCH_VENDOR_DIR`(Windows 默认 `C:/public/project-hyle`,Mac `~/vendor`,WSL 自定)引用
   外部副本,**只有 bench 目标**挂这些 include。
3. 可复现性:`cmake/vendor-versions.cmake` 记 pinned SHA;`scripts/fetch-bench-vendors.sh`(+ ps1)
   按 SHA clone 到外部目录。防漂移闸门:`scripts/check-no-vendor.sh` 断言 include/+src/ 无
   `#include.*hwy/|xsimd` 命中,进常规验证流。
4. 性能不回退:每个移植 kernel 与原借核 A/B(op_parity 框架,借核走外部 vendor 编译)——**摘除
   期间旧借核天然还在,就是现成对照组**。

## 二、分阶段(X0 先行可独立落地,X1~X3 每卡常绿)

**X0 — 构建管线先行(已落地 2026-07-16)**
- bench 的 Highway/xsimd include 全部改根到 `HYLE_BENCH_VENDOR_DIR`：Windows 默认
  `C:/public/project-hyle`，macOS 默认 `~/vendor`，WSL/Linux 必显式指定。`cmake/vendor-versions.cmake`
  是唯一 pin 源；CMake 同时校验 checkout 的路径、`HEAD` 与 clean `git status`。`scripts/fetch-bench-vendors.sh`
  与 `.ps1` 读同一 manifest，拒绝覆写非 Git 或脏 checkout；根目录 `THIRD_PARTY_NOTICES` 已就位。`bench`
  preset 使用隔离的 `build-bench`，`win`/`wsl` preset 显式复位 `HYLE_BUILD_BENCH=OFF`，防止一次 bench
  配置让随后的产品 build 意外依赖外部目录。
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
- `cloc --diff` 对 067e0fa 的 X0 机制净增 **215 code LOC**：CMake 85、Bourne shell 72、PowerShell 58；
  `THIRD_PARTY_NOTICES` 的许可证正文不计入代码 LOC。

**X1 — bitpack 移植(最小面,先证明移植模式)**
- `hn::Pack8/16/32<kBits>` 的用法翻译成 `hs::` 等价(HYLE op 面已镜像所需原语;`bit_pack-inl.h`
  的 Pack 实现本身写在 hn:: 泛型 op 上,逐函数 port);`bitpack_hyle_u16-inl.h` 已是 hn:: 原语编排,
  纯改前缀。
- 闸门:现有 round-trip oracle 全绿(曾抓到 Highway Pack16 bug 的那套)+ bench A/B vs 旧借核
  (三 ISA:avx2/avx512/neon)。

**X2 — 超越函数移植(最大面,逐函数工单)**
- Highway 侧(f64 几乎全部 + f32 的 log/sin/cos/…):`hwy/contrib/math/math-inl.h` 逐函数 port 到
  hs::(系数表/Estrin/range-reduction 原样搬,api 调用改前缀;特殊值 mask 路径逐条对照)。
- xsimd 侧(f32 exp/tanh/erf/… 9 个赢家 + f64 erf/tgamma):xsimd::batch 惯用法→hs:: 翻译,
  判断密度更高,卡片要贴双侧源码。
- 每卡闸门:精度 oracle(vs libm/双精度参考,ULP 阈值与现状持平——现有 math 测试即闸)+ bench A/B
  vs 旧借核持平以上 + 逐精度逐 ISA winner 表复核(移植后 best-of dispatch 保留,只是两侧都成了
  HYLE 代码)。
- `lgamma/remainder/nextafter` 本就 scalar(vendor bug),不移植。

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
4. `third_party/{highway,xsimd}` 已删;fetch 脚本可从零重建外部目录;NOTICES 齐。
