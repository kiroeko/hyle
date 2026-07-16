# simd-20: vendor 摘除——库零 vendor 依赖,bench 引用外部 highway/xsimd

> **状态**:规划(2026-07-15)。**动机**(用户):库里只有少数专项 kernel 借 highway/xsimd,却把整库
> 拖进 `third_party/` 当依赖,不应该;借用的代码移植进 HYLE,highway/xsimd 挪出仓库到
> `C:\public\project-hyle\{highway,xsimd}`(已挪),只供性能对拍(op_parity 的 `hn::` 侧)引用。
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

**X0 — 构建管线先行(高能力方,独立收益,不等移植)**
- bench 目标的 vendor include 切到 `HYLE_BENCH_VENDOR_DIR`;fetch 脚本 + SHA pin + NOTICES;
  Mac/WSL 同步流程加"vendor 目录单独同步"(Mac:`~/vendor/highway`;同步命令写死进
  `simd-19-workorders.md` §3 同款格式)。
- 此步后 `third_party/{highway,xsimd}` 只剩 `src/simd/**` 借核在用,依赖面立刻可见、可度量。
- 闸门:三平台 bench 目标构建 + 锚点 PASS;库目标构建证明与 vendor 目录无关(临时改名外部目录再
  构建库,应成功)。

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
- **外部目录漂移**(有人手动改了 C:\public\project-hyle 副本)→ fetch 脚本 SHA 校验,bench 输出头
  打印 vendor SHA(可加进 engine 打印,X0 顺手)。

## 五、验收

1. 库零 vendor:`check-no-vendor.sh` 绿;外部目录改名后库构建仍成功。
2. 三平台 gtest 全绿;math 精度 oracle 全绿(ULP 与现状持平)。
3. 移植 kernel bench A/B vs 原借核:全部 TIE 以上(校准仪器判定)。
4. `third_party/{highway,xsimd}` 已删;fetch 脚本可从零重建外部目录;NOTICES 齐。
