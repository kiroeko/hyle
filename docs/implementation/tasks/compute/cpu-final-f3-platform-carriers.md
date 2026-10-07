# F3 三平台成熟CPU承载

<!-- MODULE REVIEW: SCOPED -->

Native原ATen/Torch程序owner的F3范围重新定界；baseline `2792b79f`（实施前记录完整revision）。
此SCOPED只冻结范围，旧F1 AUDITED不授权本批新产品；独立审计固定下表后再绑定registry至本卡。
总合同[CPU最终收口](cpu-final-closure-plan.md)，以[F2固定矩阵](cpu-final-demand-carriers.json)为输入。
优先复用ggml原CPU、可移植成熟ATen/BLAS/量化程序及已有借入组合，补Mac ARM64和实际平台缺口。
保留原BF16/F16/FP8/FP4/编码与state精度、selected expert/cold转换成本；CPU执行不调用Metal代签。
每个采用库先锁版本/hash/许可证/ABI，完整source/static/shared runtime交付；同进程core归属按F1冻结。
一次批量接线，两入口/正确性/资源/失败/OOM/并发/生命周期/三交付在本包完成；单承载P0，不造赢家P1。
没有合格成熟承载时列精确端口/反例与有限取舍，请求必要决策，不能自己扩数学或删除固定需求。
写产品前真实owner SCOPED/AUDITED，公共Graph BLOCKED不自动恢复；只改有具体反例的合法owner。

## Mac CPU有限承载批

真实trigger：平台门只允许Win/Linux x86_64，已借原ATen/codec程序未在Mac CPU交付，固定模型必需合同因此缺失。
官方ARM64 LibTorch2.8原archive已从Windows取得并传Air验证SHA；build-version为`2.8.0`，
archive SHA `be0642231d2008c7b90071fc0898d82a0c002d47f38350ebf3ed779fc01972d7`、71,787,762字节。
Air实际Mac16,12/macOS27.0.1/AppleClang21/CMake4.3.4，产品源码仍只由Windows前向同步。
otool发现CPU dylib的libomp依赖硬编码Homebrew路径：不能靠本机已安装brew冒充可搬移SDK。
原包也有weak MPS/Metal系统链接；HYLE借用程序的张量与操作全部CPU，不启用HYLE Apple路线或以GPU代签。

| FIX | 有限动作 | 本批验收 |
|---|---|---|
| T1 | 锁Darwin-aarch64官方SDK原archive/版本及许可证，标准CPU AUTO取得而非静默关闭 | CPU preset实际包含原ATen/codec，offline本地archive/hash miss明确失败 |
| T2 | 平台SDK key/精确version校验同时服务source与installed consumer；Win/Linux+cpu条件保持 | 外部root错误版本/目标拒绝，Mac2.8.0只用于ARM64 CPU |
| T3 | 只在有owner marker的managed SDK staging修正libomp为loader-relative，保留原archive/SHA与派生内容manifest | 不改用户外部SDK；搬移/无Homebrew依赖仍装载、真实CPU执行 |
| T4 | 注册torch/torch_cpu/c10/libomp真实dylib闭包，RPATH复用现helper；编译现原程序不写新数学 | 两入口、dtype/packing/状态/失败/资源/并发、source/static/shared集中验证 |

普通API/ownership/失败/并发/完成与准备/热路径不改变。tensor逻辑和执行精度按原程序保持，
全模型CPU math明示、shadercompiler/Metal context不作为本批依赖。无本批赢家P1，单承载P0与正确性/资源为主。
预算：一个锁定SDK、一次批量接线/构建及至多一批具名工具链/loader纠正；不换版本寻找通过。
C1/F2合同全部仍适用；缺数学、模板/SDK ABI无法适配时记录具体反例，不缩范围、不开自研。
本卡不等于F1已完成；已稳定F1事实按精确身份复用，最终交付在本批完成不首次留给F6。
