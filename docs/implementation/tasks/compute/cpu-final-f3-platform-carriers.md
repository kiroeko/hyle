# F3 三平台成熟CPU承载

<!-- MODULE REVIEW: AUDITED -->

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

## 审计完成与实施冻结

baseline完整revision `2792b79fabd9439341517edf3f44b1762faa22c0`，T1–T4已AUDITED。
原代码不含x86-only数学/ISA要求；既有ATen/借入量化程序按CPU Tensor执行，原SDK ARM64提供所用接口。
runtime闭包为torch/torch_cpu/c10/libomp四dylib。SDK原压缩包SHA固定，managed stage独占地修正
CPU dylib的绝对libomp依赖为`@loader_path/libomp.dylib`，必要ad-hoc重签只作用于该派生stage；
原archive永不改、派生文件重新进完整manifest。外部用户root不修改，必须符合可装载/版本/架构合同。
installed shim同样按平台key判断版本，Win/Linux2.8.0+cpu条件不放宽；CUDA Torch始终拒绝。
现runtime helper只有Linux RPATH分支，T4有限补Darwin路径，保持Linux专用linker flag不传Apple。
公开API/对象/原数字程序不改；Native owner重新审计后合法绑定本卡，不继承旧F1有限FIX授权。
F2/Mac缺口由此实际施工，shared/static/source批量验收与负空间均需运行，不以SDK获取签覆盖。

## T5组合oracle有界纠正（测试，不改数学/误差门）

Mac原C5 Direct653项中652通过；Pro0813 Q8的mHC→BF16 head失败。
定向同输入探针证明mHC所有原输出门已通过，最大f32差异4.768e-7，但一个值跨BF16 RNE阈值；
原head actual使用observed mHC、oracle却使用另一reference mHC，两个不同输入不能签head数学失败。
actual输入的原codec/dot独立对照通过；不据诊断PASS签原完整批，原失败/全部数值差异保留。
T5只修测试组合：上游mHC仍独立reference、原误差与实际state反馈全验；下游head保持原BF16转换/权重/真实K/V分片，
独立oracle消费相同observed上游f32值。没有阈值放宽、不换dtype、不预计算actual输入、不删除分支/反馈/全head。
Float32允许的实现误差不承诺跨BF16舍入阈值后与另一reference bit-identical；这是逐精度边界组合验收。
单case纠正重验+Graph全部C5首次验；Direct其它652复用实际通过XML，原aggregate FAIL不重写PASS。

## 当前实际验收（shared/source部分）

Windows不可变产品`42898c37ef90005944b73f4bf174f9b7f1070550`前向同步至Air，
CPU配置中Apple/CUDA/Vulkan关闭；managed原ARM64 SDK身份/版本与loader修正通过。
原Provider owning组通过，普通Direct12/Graph11均无skip；本地Qwen原权重投影fixture已绑定。
C5 Direct原652项通过，T5候选`6359804abc6e0543f3fef9a4b6b622f01a8b98db`具名单case完整head重验通过；
Graph全部653项首次运行通过、无skip。原Direct aggregate FAIL及诊断边界保留。
[原日志/XML/来源与receipt](../../../../evidence/correctness/cpu-final-f3/README.md)归档，不改原误差门。
installed static/shared/搬移、Linux当前版本及其它完整旅程仍待运行，本包不关闭、不签新性能WIN。
统一Compute+Graph实现61,242行/251文件，新增自研数学0。

## Mac三种C++普通交付已验

候选`9a9936e8271fc834f4b090316bc0a5d7f9b25982`由Windows现有source-package脚本生成完整带manifest/SHA的源码包，
Air只消费前向副本。两个公开consumer按每种交付共用一个producer：source嵌入2项、
shared搬移后Direct/Graph各1项、static搬移后各1项均通过，无skip；shared出口检查16库/1262出口通过。
唯一环境纠正为SSH非登录PATH缺Ninja，原失败保留，只续失败步骤，不重建已通过producer。
四原ARM64 CPU dylib的lipo/otool已审；torch_cpu的libomp边为loader-relative。
机器SDK仍按installed shim单独解析，不冒称LibTorch随HYLE SDK完整打包。
[交付receipt/日志](../../../../evidence/correctness/cpu-final-f3/README.md)有原archive、manifest、runtime与consumer身份。
此为普通两入口交付，不签全部C5在installed形态重跑；Linux当前升级候选集中验证中。
